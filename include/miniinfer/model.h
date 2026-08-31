#pragma once
#include "miniinfer/tensor.h"
#include "miniinfer/tokenizer.h"
#include "miniinfer/weight.h"
#include <string>
#include <vector>
namespace miniinfer {
struct Config {
  size_t vocab=0, hidden=0, layers=0, heads=0, kv_heads=0;
  size_t intermediate=0, context=0;
  float eps=1e-5f, rope_theta=10000.0f;
  bool tie_word_embeddings=false;
};
struct LayerWeights {
  Tensor attn_norm, ffn_norm;
  WeightTensor q, k, v, o, gate, up, down;
};
struct Model {
  Config config;
  WeightTensor embedding, lm_head;
  Tensor final_norm;
  std::vector<LayerWeights> layers;
  Tokenizer tokenizer;
};
bool save_model(const Model&, const std::string& path);
bool load_model(const std::string& path, Model& model, std::string* error=nullptr);
}
