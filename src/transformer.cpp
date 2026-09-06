#include "miniinfer/transformer.h"
#include "miniinfer/kernels.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace miniinfer {
KVCache::KVCache(size_t layers, size_t context, size_t kv_heads, size_t head_dim)
    : context_(context),
      keys_(layers, Tensor({context, kv_heads, head_dim})),
      values_(layers, Tensor({context, kv_heads, head_dim})) {
  if (!layers || !context || !kv_heads || !head_dim)
    throw std::invalid_argument("KV cache dimensions must be non-zero");
}

void KVCache::clear() { length_ = 0; }

void KVCache::append(size_t layer, const Tensor& key, const Tensor& value) {
  if (layer >= keys_.size()) throw std::out_of_range("KV cache layer");
  if (length_ >= context_) throw std::runtime_error("context length exceeded");
  const std::vector<size_t> expected = {keys_[layer].dim(1), keys_[layer].dim(2)};
  if (key.shape() != expected || value.shape() != expected)
    throw std::invalid_argument("KV tensor shape mismatch");
  for (size_t head = 0; head < key.dim(0); ++head) {
    for (size_t dim = 0; dim < key.dim(1); ++dim) {
      const size_t cache_index =
          (length_ * keys_[layer].dim(1) + head) * keys_[layer].dim(2) + dim;
      const size_t value_index = head * key.dim(1) + dim;
      keys_[layer][cache_index] = key[value_index];
      values_[layer][cache_index] = value[value_index];
    }
  }
  if (layer + 1 == keys_.size()) ++length_;
}

Tensor forward_token(const Model& model, const Backend& backend, int token, size_t position,
                     KVCache& cache, ForwardTrace* trace) {
  const Config& config = model.config;
  if (token < 0 || static_cast<size_t>(token) >= config.vocab)
    throw std::out_of_range("token ID is outside the vocabulary");
  if (position != cache.length())
    throw std::invalid_argument("token position does not match KV cache length");

  Tensor x({config.hidden});
  for (size_t i = 0; i < config.hidden; ++i)
    x[i] = model.embedding.value(static_cast<size_t>(token) * config.hidden + i);
  if (trace) {
    trace->embedding = x;
    trace->layers.assign(config.layers, LayerTrace{});
  }

  const size_t head_dim = config.hidden / config.heads;
  const size_t query_heads_per_kv = config.heads / config.kv_heads;
  for (size_t layer_index = 0; layer_index < config.layers; ++layer_index) {
    const LayerWeights& layer = model.layers[layer_index];
    Tensor normalized = rmsnorm(x, layer.attn_norm, config.eps);
    Tensor q = backend.linear(layer.q, normalized);
    Tensor k = backend.linear(layer.k, normalized);
    Tensor v = backend.linear(layer.v, normalized);
    if (trace) {
      trace->layers[layer_index].attn_norm = normalized;
      trace->layers[layer_index].q = q;
      trace->layers[layer_index].k = k;
      trace->layers[layer_index].v = v;
    }

    Tensor q_heads({config.heads, head_dim});
    Tensor k_heads({config.kv_heads, head_dim});
    Tensor v_heads({config.kv_heads, head_dim});
    for (size_t i = 0; i < q.size(); ++i) q_heads[i] = q[i];
    for (size_t i = 0; i < k.size(); ++i) {
      k_heads[i] = k[i];
      v_heads[i] = v[i];
    }

    for (size_t head = 0; head < config.heads; ++head) {
      Tensor vector({head_dim});
      for (size_t dim = 0; dim < head_dim; ++dim)
        vector[dim] = q_heads[head * head_dim + dim];
      rope(vector, position, config.rope_theta);
      for (size_t dim = 0; dim < head_dim; ++dim)
        q_heads[head * head_dim + dim] = vector[dim];
    }
    for (size_t head = 0; head < config.kv_heads; ++head) {
      Tensor vector({head_dim});
      for (size_t dim = 0; dim < head_dim; ++dim)
        vector[dim] = k_heads[head * head_dim + dim];
      rope(vector, position, config.rope_theta);
      for (size_t dim = 0; dim < head_dim; ++dim)
        k_heads[head * head_dim + dim] = vector[dim];
    }
    if (trace) {
      trace->layers[layer_index].q_rope = q_heads;
      trace->layers[layer_index].k_rope = k_heads;
    }

    cache.append(layer_index, k_heads, v_heads);
    const size_t attended_tokens = position + 1;
    Tensor attention({config.hidden});
    for (size_t query_head = 0; query_head < config.heads; ++query_head) {
      const size_t kv_head = query_head / query_heads_per_kv;
      Tensor scores({attended_tokens});
      for (size_t sequence = 0; sequence < attended_tokens; ++sequence) {
        float dot = 0.0f;
        for (size_t dim = 0; dim < head_dim; ++dim) {
          const size_t key_index =
              (sequence * config.kv_heads + kv_head) * head_dim + dim;
          dot += q_heads[query_head * head_dim + dim] *
                 cache.key(layer_index)[key_index];
        }
        scores[sequence] = dot / std::sqrt(static_cast<float>(head_dim));
      }
      Tensor probabilities = softmax(scores);
      for (size_t dim = 0; dim < head_dim; ++dim) {
        for (size_t sequence = 0; sequence < attended_tokens; ++sequence) {
          const size_t value_index =
              (sequence * config.kv_heads + kv_head) * head_dim + dim;
          attention[query_head * head_dim + dim] +=
              probabilities[sequence] * cache.value(layer_index)[value_index];
        }
      }
    }

    Tensor projected_attention = backend.linear(layer.o, attention);
    x = add(x, projected_attention);
    if (trace) {
      trace->layers[layer_index].attention = attention;
      trace->layers[layer_index].projected_attention = projected_attention;
      trace->layers[layer_index].after_attention = x;
    }
    normalized = rmsnorm(x, layer.ffn_norm, config.eps);
    Tensor gate = backend.linear(layer.gate, normalized);
    Tensor up = backend.linear(layer.up, normalized);
    Tensor gated = mul(silu(gate), up);
    Tensor down = backend.linear(layer.down, gated);
    x = add(x, down);
    if (trace) {
      trace->layers[layer_index].ffn_norm = normalized;
      trace->layers[layer_index].gate = gate;
      trace->layers[layer_index].up = up;
      trace->layers[layer_index].down = down;
      trace->layers[layer_index].output = x;
    }
  }

  const WeightTensor& output = config.tie_word_embeddings ? model.embedding : model.lm_head;
  Tensor final_norm = rmsnorm(x, model.final_norm, config.eps);
  Tensor logits = backend.linear(output, final_norm);
  if (trace) {
    trace->final_norm = final_norm;
    trace->logits = logits;
  }
  return logits;
}

namespace {
void validate_prompt(const Model& model, const std::vector<int>& prompt) {
  if (prompt.empty()) throw std::invalid_argument("prompt must contain at least one token");
  if (prompt.size() > model.config.context)
    throw std::invalid_argument("prompt exceeds the model context length");
  for (int token : prompt)
    if (token < 0 || static_cast<size_t>(token) >= model.config.vocab)
      throw std::out_of_range("prompt contains an invalid token ID");
}

Tensor rmsnorm_rows(const Tensor& values, const Tensor& weight, float epsilon) {
  if (values.shape().size() != 2 || values.dim(1) != weight.size())
    throw std::invalid_argument("batched RMSNorm shape mismatch");
  Tensor result(values.shape());
  for (size_t row = 0; row < values.dim(0); ++row) {
    float sum_squares = 0.0f;
    for (size_t column = 0; column < values.dim(1); ++column) {
      const float value = values[row * values.dim(1) + column];
      sum_squares += value * value;
    }
    const float inverse = 1.0f / std::sqrt(
        sum_squares / static_cast<float>(values.dim(1)) + epsilon);
    for (size_t column = 0; column < values.dim(1); ++column)
      result[row * values.dim(1) + column] =
          values[row * values.dim(1) + column] * inverse * weight[column];
  }
  return result;
}
}  // namespace

Tensor prompt_logits(const Model& model, const Backend& backend,
                     const std::vector<int>& prompt, ForwardTrace* trace) {
  validate_prompt(model, prompt);
  KVCache cache(model.config.layers, prompt.size(), model.config.kv_heads,
                model.config.hidden / model.config.heads);
  Tensor logits;
  for (size_t position = 0; position < prompt.size(); ++position)
    logits = forward_token(model, backend, prompt[position], position, cache,
                           position + 1 == prompt.size() ? trace : nullptr);
  return logits;
}

Tensor prompt_logits_batched(const Model& model, const Backend& backend,
                             const std::vector<int>& prompt) {
  validate_prompt(model, prompt);
  const Config& config = model.config;
  const size_t sequence_length = prompt.size();
  const size_t head_dim = config.hidden / config.heads;
  const size_t query_heads_per_kv = config.heads / config.kv_heads;
  Tensor states({sequence_length, config.hidden});
  for (size_t position = 0; position < sequence_length; ++position)
    for (size_t hidden = 0; hidden < config.hidden; ++hidden)
      states[position * config.hidden + hidden] = model.embedding.value(
          static_cast<size_t>(prompt[position]) * config.hidden + hidden);

  for (const LayerWeights& layer : model.layers) {
    Tensor normalized = rmsnorm_rows(states, layer.attn_norm, config.eps);
    Tensor q = backend.linear(layer.q, normalized);
    Tensor k = backend.linear(layer.k, normalized);
    Tensor v = backend.linear(layer.v, normalized);
    for (size_t position = 0; position < sequence_length; ++position) {
      for (size_t head = 0; head < config.heads; ++head) {
        Tensor vector({head_dim});
        const size_t offset = position * config.hidden + head * head_dim;
        for (size_t dim = 0; dim < head_dim; ++dim) vector[dim] = q[offset + dim];
        rope(vector, position, config.rope_theta);
        for (size_t dim = 0; dim < head_dim; ++dim) q[offset + dim] = vector[dim];
      }
      const size_t kv_width = config.kv_heads * head_dim;
      for (size_t head = 0; head < config.kv_heads; ++head) {
        Tensor vector({head_dim});
        const size_t offset = position * kv_width + head * head_dim;
        for (size_t dim = 0; dim < head_dim; ++dim) vector[dim] = k[offset + dim];
        rope(vector, position, config.rope_theta);
        for (size_t dim = 0; dim < head_dim; ++dim) k[offset + dim] = vector[dim];
      }
    }

    Tensor attention({sequence_length, config.hidden});
    const size_t kv_width = config.kv_heads * head_dim;
    for (size_t position = 0; position < sequence_length; ++position) {
      for (size_t query_head = 0; query_head < config.heads; ++query_head) {
        const size_t kv_head = query_head / query_heads_per_kv;
        Tensor scores({position + 1});
        for (size_t source = 0; source <= position; ++source) {
          float dot = 0.0f;
          for (size_t dim = 0; dim < head_dim; ++dim)
            dot += q[position * config.hidden + query_head * head_dim + dim] *
                   k[source * kv_width + kv_head * head_dim + dim];
          scores[source] = dot / std::sqrt(static_cast<float>(head_dim));
        }
        const Tensor probabilities = softmax(scores);
        for (size_t dim = 0; dim < head_dim; ++dim)
          for (size_t source = 0; source <= position; ++source)
            attention[position * config.hidden + query_head * head_dim + dim] +=
                probabilities[source] *
                v[source * kv_width + kv_head * head_dim + dim];
      }
    }
    Tensor projected = backend.linear(layer.o, attention);
    states = add(states, projected);
    normalized = rmsnorm_rows(states, layer.ffn_norm, config.eps);
    const Tensor gate = backend.linear(layer.gate, normalized);
    const Tensor up = backend.linear(layer.up, normalized);
    const Tensor down = backend.linear(layer.down, mul(silu(gate), up));
    states = add(states, down);
  }

  Tensor normalized = rmsnorm_rows(states, model.final_norm, config.eps);
  Tensor last({config.hidden});
  for (size_t hidden = 0; hidden < config.hidden; ++hidden)
    last[hidden] = normalized[(sequence_length - 1) * config.hidden + hidden];
  const WeightTensor& output = config.tie_word_embeddings ? model.embedding : model.lm_head;
  return backend.linear(output, last);
}

int greedy(const Tensor& logits) {
  if (logits.size() == 0) throw std::invalid_argument("cannot sample empty logits");
  size_t best = 0;
  for (size_t i = 1; i < logits.size(); ++i)
    if (logits[i] > logits[best]) best = i;
  return static_cast<int>(best);
}

Sampler::Sampler(SamplingOptions options)
    : options_(options), random_(options.seed) {
  if (!std::isfinite(options_.temperature) || options_.temperature < 0.0f)
    throw std::invalid_argument("temperature must be finite and non-negative");
  if (!std::isfinite(options_.top_p) || options_.top_p <= 0.0f || options_.top_p > 1.0f)
    throw std::invalid_argument("top-p must be in the interval (0, 1]");
}

int Sampler::sample(const Tensor& logits) {
  if (logits.shape().size() != 1 || logits.size() == 0)
    throw std::invalid_argument("sampling expects non-empty rank-1 logits");
  if (options_.temperature <= 0.0f) return greedy(logits);

  std::vector<size_t> candidates(logits.size());
  std::iota(candidates.begin(), candidates.end(), size_t{0});
  for (float value : logits.values())
    if (!std::isfinite(value)) throw std::runtime_error("cannot sample non-finite logits");
  std::sort(candidates.begin(), candidates.end(), [&](size_t left, size_t right) {
    if (logits[left] == logits[right]) return left < right;
    return logits[left] > logits[right];
  });
  if (options_.top_k != 0 && candidates.size() > options_.top_k)
    candidates.resize(options_.top_k);

  const float maximum = logits[candidates.front()] / options_.temperature;
  std::vector<double> weights(candidates.size());
  double total = 0.0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    weights[i] = std::exp(static_cast<double>(
        logits[candidates[i]] / options_.temperature - maximum));
    total += weights[i];
  }
  if (!std::isfinite(total) || total <= 0.0)
    throw std::runtime_error("sampling probabilities are invalid");

  if (options_.top_p < 1.0f) {
    double cumulative = 0.0;
    size_t keep = 0;
    do {
      cumulative += weights[keep] / total;
      ++keep;
    } while (keep < weights.size() && cumulative < options_.top_p);
    candidates.resize(keep);
    weights.resize(keep);
    total = std::accumulate(weights.begin(), weights.end(), 0.0);
  }

  const double draw = std::generate_canonical<double, 53>(random_) * total;
  double cumulative = 0.0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    cumulative += weights[i];
    if (draw < cumulative) return static_cast<int>(candidates[i]);
  }
  return static_cast<int>(candidates.back());
}

std::vector<int> generate(const Model& model, const Backend& backend,
                          const std::vector<int>& prompt, size_t max_tokens,
                          CacheMode mode, SamplingOptions sampling) {
  return generate_stream(model, backend, prompt, max_tokens, mode, {}, sampling);
}

std::vector<int> generate_stream(const Model& model, const Backend& backend,
                                 const std::vector<int>& prompt, size_t max_tokens,
                                 CacheMode mode,
                                 const std::function<bool(int)>& on_token,
                                 SamplingOptions sampling) {
  validate_prompt(model, prompt);
  if (max_tokens > model.config.context - prompt.size())
    throw std::invalid_argument("requested generation exceeds the model context length");
  std::vector<int> output = prompt;
  if (max_tokens == 0) return output;
  Sampler sampler(sampling);

  Tensor logits;
  std::unique_ptr<KVCache> cache;
  if (mode == CacheMode::KV) {
    const size_t capacity = prompt.size() + max_tokens;
    cache = std::make_unique<KVCache>(model.config.layers, capacity, model.config.kv_heads,
                                     model.config.hidden / model.config.heads);
    for (size_t position = 0; position < prompt.size(); ++position)
      logits = forward_token(model, backend, prompt[position], position, *cache);
  } else {
    logits = prompt_logits(model, backend, output);
  }

  for (size_t generated = 0; generated < max_tokens; ++generated) {
    const int next = sampler.sample(logits);
    output.push_back(next);
    if (on_token && !on_token(next)) break;
    if (next == model.tokenizer.eos_id()) break;
    if (generated + 1 < max_tokens) {
      if (mode == CacheMode::KV)
        logits = forward_token(model, backend, next, prompt.size() + generated, *cache);
      else
        logits = prompt_logits(model, backend, output);
    }
  }
  return output;
}
}  // namespace miniinfer
