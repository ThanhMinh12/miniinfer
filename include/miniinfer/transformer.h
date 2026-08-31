#pragma once
#include "miniinfer/backend.h"
#include "miniinfer/model.h"
#include <functional>
namespace miniinfer {
enum class CacheMode { Recompute, KV };

class KVCache {
 public:
  KVCache(size_t layers, size_t context, size_t kv_heads, size_t head_dim);
  void clear();
  size_t length() const { return length_; }
  void append(size_t layer, const Tensor& k, const Tensor& v);
  const Tensor& key(size_t layer) const { return keys_.at(layer); }
  const Tensor& value(size_t layer) const { return values_.at(layer); }
 private:
  size_t context_, length_=0;
  std::vector<Tensor> keys_, values_;
};
struct LayerTrace {
  Tensor attn_norm, q, k, v, q_rope, k_rope;
  Tensor attention, projected_attention, after_attention;
  Tensor ffn_norm, gate, up, down, output;
};
struct ForwardTrace {
  Tensor embedding, final_norm, logits;
  std::vector<LayerTrace> layers;
};
Tensor forward_token(const Model&, const Backend&, int token, size_t position,
                     KVCache&, ForwardTrace* trace=nullptr);
Tensor prompt_logits(const Model&, const Backend&, const std::vector<int>& prompt,
                     ForwardTrace* trace=nullptr);
// Layer-major prompt execution using matrix-matrix linear calls. It is an
// independent prefill reference and intentionally does not populate a decode cache.
Tensor prompt_logits_batched(const Model&, const Backend&,
                             const std::vector<int>& prompt);
int greedy(const Tensor& logits);
std::vector<int> generate(const Model&, const Backend&, const std::vector<int>& prompt,
                          size_t max_tokens, CacheMode mode = CacheMode::KV);
std::vector<int> generate_stream(const Model&, const Backend&,
                                 const std::vector<int>& prompt, size_t max_tokens,
                                 CacheMode mode,
                                 const std::function<bool(int)>& on_token);
}
