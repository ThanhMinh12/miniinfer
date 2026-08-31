#include "miniinfer/kernels.h"
#include "miniinfer/backend.h"
#include "miniinfer/c_api.h"
#include "miniinfer/model.h"
#include "miniinfer/tokenizer.h"
#include "miniinfer/transformer.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace miniinfer;

namespace {
bool near(float actual, float expected, float tolerance = 1e-5f) {
  return std::fabs(actual - expected) <= tolerance;
}
}

int main() {
  Tensor a({2, 3});
  Tensor b({3, 2});
  for (size_t i = 0; i < 6; ++i) { a[i] = static_cast<float>(i + 1); b[i] = static_cast<float>(i + 1); }
  Tensor c = matmul(a, b);
  assert(c.shape() == std::vector<size_t>({2, 2}));
  assert(near(c[0], 22.0f));
  assert(near(c[3], 64.0f));

  Tensor linear_weight_values({3, 5});
  Tensor linear_input({5});
  for (size_t i = 0; i < linear_weight_values.size(); ++i)
    linear_weight_values[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.2f;
  for (size_t i = 0; i < linear_input.size(); ++i)
    linear_input[i] = static_cast<float>(i + 1) * 0.1f;
  WeightTensor linear_weight(linear_weight_values);
  CPUBackend scalar(CPUOptions{1, CPUKernel::Scalar});
  CPUBackend threaded(CPUOptions{3, CPUKernel::Scalar});
  Tensor expected_linear = scalar.linear(linear_weight, linear_input);
  Tensor threaded_linear = threaded.linear(linear_weight, linear_input);
  for (size_t i = 0; i < expected_linear.size(); ++i)
    assert(near(expected_linear[i], threaded_linear[i]));
  Tensor batched_input({2, 5});
  for (size_t i = 0; i < batched_input.size(); ++i) batched_input[i] = linear_input[i % 5];
  Tensor batched_output = threaded.linear(linear_weight, batched_input);
  assert(batched_output.shape() == std::vector<size_t>({2, 3}));
  for (size_t i = 0; i < 3; ++i) {
    assert(near(batched_output[i], expected_linear[i]));
    assert(near(batched_output[i + 3], expected_linear[i]));
  }
  WeightTensor quantized = WeightTensor::quantize_q8(linear_weight_values);
  Tensor reconstructed = quantized.dequantize();
  for (size_t row = 0; row < quantized.rows(); ++row)
    for (size_t column = 0; column < quantized.columns(); ++column)
      assert(std::fabs(reconstructed[row * quantized.columns() + column] -
                       linear_weight_values[row * quantized.columns() + column]) <=
             quantized.scales()[row] * 0.501f + 1e-7f);
  Tensor quantized_output = threaded.linear(quantized, linear_input);
  for (float value : quantized_output.values()) assert(std::isfinite(value));
  CPUBackend requested_avx2(CPUOptions{2, CPUKernel::AVX2});
  Tensor dispatched_output = requested_avx2.linear(linear_weight, linear_input);
  for (size_t i = 0; i < expected_linear.size(); ++i)
    assert(near(expected_linear[i], dispatched_output[i], 1e-5f));
  assert(requested_avx2.selected_kernel() == CPUKernel::Scalar ||
         requested_avx2.selected_kernel() == CPUKernel::AVX2);
#ifndef MINIINFER_ENABLE_AVX2
  assert(requested_avx2.selected_kernel() == CPUKernel::Scalar);
#endif

  Tensor probabilities = softmax(Tensor({3}, 1.0f));
  assert(near(probabilities[0] + probabilities[1] + probabilities[2], 1.0f));
  Tensor normalized = rmsnorm(Tensor({2}, 3.0f), Tensor({2}, 1.0f), 1e-5f);
  assert(near(normalized[0], 1.0f, 1e-3f));

  // Llama non-interleaved RoPE rotates dimensions i and i + head_dim / 2.
  Tensor rotated({4});
  rotated[0] = 1.0f; rotated[1] = 2.0f; rotated[2] = 3.0f; rotated[3] = 4.0f;
  rope(rotated, 1, 10000.0f);
  assert(near(rotated[0], std::cos(1.0f) - 3.0f * std::sin(1.0f)));
  assert(near(rotated[2], 3.0f * std::cos(1.0f) + std::sin(1.0f)));
  assert(near(rotated[1], 2.0f * std::cos(0.01f) - 4.0f * std::sin(0.01f)));
  assert(near(rotated[3], 4.0f * std::cos(0.01f) + 2.0f * std::sin(0.01f)));

  Tokenizer tokenizer;
  tokenizer.add_token(0, "a"); tokenizer.add_token(1, "b"); tokenizer.add_token(2, "ab");
  tokenizer.add_token(3, "1"); tokenizer.add_token(4, "2");
  tokenizer.add_special_token(5, "<special>");
  tokenizer.add_merge("a", "b");
  assert(tokenizer.encode("ab") == std::vector<int>({2}));
  assert(tokenizer.encode("12") == std::vector<int>({3, 4}));
  assert(tokenizer.encode("a<special>b") == std::vector<int>({0, 5, 1}));
  assert(tokenizer.decode({2, 5}) == "ab<special>");

  Model model;
  model.config.vocab = 2; model.config.hidden = 2; model.config.layers = 1;
  model.config.heads = 1; model.config.kv_heads = 1; model.config.intermediate = 2;
  model.config.context = 4; model.config.tie_word_embeddings = true;
  Tensor tiny_embedding({2, 2});
  tiny_embedding[0] = 0.2f; tiny_embedding[1] = 0.4f;
  tiny_embedding[2] = 1.0f; tiny_embedding[3] = -0.5f;
  model.embedding = tiny_embedding;
  model.final_norm = Tensor({2}, 1.0f);
  model.tokenizer.set_special_ids(-1, -1);
  model.tokenizer.add_token(0, "a");
  model.tokenizer.add_token(1, "b");
  model.layers.resize(1);
  auto& layer = model.layers[0];
  layer.attn_norm = Tensor({2}, 1.0f); layer.ffn_norm = Tensor({2}, 1.0f);
  Tensor tiny_matrix({2, 2});
  tiny_matrix[0] = 0.1f; tiny_matrix[1] = -0.2f;
  tiny_matrix[2] = 0.3f; tiny_matrix[3] = 0.05f;
  layer.q = tiny_matrix; layer.k = tiny_matrix; layer.v = tiny_matrix;
  layer.o = tiny_matrix; layer.gate = tiny_matrix;
  layer.up = tiny_matrix; layer.down = tiny_matrix;
  CPUBackend cpu;
  const std::vector<int> cached = generate(model, cpu, {1}, 3, CacheMode::KV);
  const std::vector<int> recomputed = generate(model, cpu, {1}, 3, CacheMode::Recompute);
  assert(cached.size() == 4);
  assert(cached == recomputed);
  KVCache decode_cache(model.config.layers, 3, model.config.kv_heads,
                       model.config.hidden / model.config.heads);
  Tensor first_logits = forward_token(model, cpu, 1, 0, decode_cache);
  const int first_token = greedy(first_logits);
  Tensor cached_logits = forward_token(model, cpu, first_token, 1, decode_cache);
  Tensor recomputed_logits = prompt_logits(model, cpu, {1, first_token});
  for (size_t i = 0; i < cached_logits.size(); ++i)
    assert(cached_logits[i] == recomputed_logits[i]);
  Tensor incremental_prefill = prompt_logits(model, cpu, {1, 0});
  Tensor batched_prefill = prompt_logits_batched(model, cpu, {1, 0});
  for (size_t i = 0; i < incremental_prefill.size(); ++i)
    assert(near(incremental_prefill[i], batched_prefill[i]));
  bool rejected_empty = false, rejected_context = false;
  try { generate(model, cpu, {}, 1); } catch (const std::invalid_argument&) { rejected_empty = true; }
  try { generate(model, cpu, {1}, 4); } catch (const std::invalid_argument&) { rejected_context = true; }
  assert(rejected_empty && rejected_context);

  const std::string c_api_model = "miniinfer-c-api-test.model";
  assert(save_model(model, c_api_model));
  MiniInferHandle* handle = miniinfer_create(c_api_model.c_str(), 2, MINIINFER_CPU_SCALAR);
  assert(handle != nullptr);
  int tokenized[2] = {-1, -1};
  assert(miniinfer_tokenize(handle, "ba", tokenized, 2) == 2);
  assert(tokenized[0] == 1 && tokenized[1] == 0);
  std::vector<int> streamed;
  auto callback = [](int token, const char*, void* user_data) {
    static_cast<std::vector<int>*>(user_data)->push_back(token);
    return 1;
  };
  assert(miniinfer_generate(handle, "b", 2, MINIINFER_CACHE_KV,
                            callback, &streamed) == 2);
  assert(streamed.size() == 2);
  miniinfer_destroy(handle);
  assert(std::remove(c_api_model.c_str()) == 0);
  return 0;
}
