#include "miniinfer/model.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>

namespace miniinfer {
namespace {
constexpr uint32_t kMagic = 0x4d494e49;
constexpr uint32_t kVersion = 4;
constexpr uint32_t kLegacyVersion = 3;
constexpr uint32_t kFlagTiedEmbeddings = 1;
enum class DiskDType : uint32_t { F32 = 1, F16 = 2, BF16 = 3, Q8PerRow = 4 };

template<class T> void write_scalar(std::ofstream& file, const T& value) {
  file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
template<class T> bool read_scalar(std::ifstream& file, T& value) {
  return bool(file.read(reinterpret_cast<char*>(&value), sizeof(value)));
}

void write_string(std::ofstream& file, const std::string& value) {
  uint64_t size = value.size();
  write_scalar(file, size);
  file.write(value.data(), static_cast<std::streamsize>(size));
}
bool read_string(std::ifstream& file, std::string& value) {
  uint64_t size;
  if (!read_scalar(file, size) || size > (1ull << 30)) return false;
  value.assign(static_cast<size_t>(size), '\0');
  return bool(file.read(value.data(), static_cast<std::streamsize>(size)));
}

void write_tensor(std::ofstream& file, const Tensor& tensor) {
  write_scalar(file, static_cast<uint32_t>(DiskDType::F32));
  uint64_t rank = tensor.shape().size();
  write_scalar(file, rank);
  for (size_t dim : tensor.shape()) write_scalar(file, static_cast<uint64_t>(dim));
  write_scalar(file, static_cast<uint64_t>(tensor.size()));
  write_scalar(file, static_cast<uint64_t>(tensor.size() * sizeof(float)));
  file.write(reinterpret_cast<const char*>(tensor.data()),
             static_cast<std::streamsize>(tensor.size() * sizeof(float)));
}

void write_weight(std::ofstream& file, const WeightTensor& weight) {
  DiskDType dtype = DiskDType::F32;
  if (weight.type() == WeightType::BF16) dtype = DiskDType::BF16;
  if (weight.type() == WeightType::Q8PerRow) dtype = DiskDType::Q8PerRow;
  write_scalar(file, static_cast<uint32_t>(dtype));
  write_scalar(file, static_cast<uint64_t>(weight.shape().size()));
  for (size_t dim : weight.shape()) write_scalar(file, static_cast<uint64_t>(dim));
  write_scalar(file, static_cast<uint64_t>(weight.size()));
  uint64_t bytes = weight.size() * (dtype == DiskDType::F32 ? 4 : 2);
  if (dtype == DiskDType::Q8PerRow)
    bytes = weight.rows() * (sizeof(float) + weight.columns());
  write_scalar(file, bytes);
  if (dtype == DiskDType::F32) {
    file.write(reinterpret_cast<const char*>(weight.f32_values().data()),
               static_cast<std::streamsize>(bytes));
  } else if (dtype == DiskDType::BF16) {
    file.write(reinterpret_cast<const char*>(weight.bf16_values().data()),
               static_cast<std::streamsize>(bytes));
  } else {
    for (size_t row = 0; row < weight.rows(); ++row) {
      write_scalar(file, weight.scales()[row]);
      file.write(reinterpret_cast<const char*>(weight.q8_values().data() + row * weight.columns()),
                 static_cast<std::streamsize>(weight.columns()));
    }
  }
}

float half_to_float(uint16_t half) {
  uint32_t sign = (half & 0x8000u) << 16;
  int exponent = (half >> 10) & 0x1fu;
  uint32_t mantissa = half & 0x3ffu;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) bits = sign;
    else {
      exponent = 1;
      while ((mantissa & 0x400u) == 0) { mantissa <<= 1; --exponent; }
      mantissa &= 0x3ffu;
      bits = sign | (static_cast<uint32_t>(exponent + 112) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) bits = sign | 0x7f800000u | (mantissa << 13);
  else bits = sign | (static_cast<uint32_t>(exponent + 112) << 23) | (mantissa << 13);
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

bool read_tensor_v2(std::ifstream& file, Tensor& tensor, std::string* error) {
  uint32_t dtype_raw;
  uint64_t rank, elements, bytes;
  if (!read_scalar(file, dtype_raw) || !read_scalar(file, rank) || rank > 8) return false;
  std::vector<size_t> shape(static_cast<size_t>(rank));
  uint64_t expected_elements = 1;
  for (size_t& dim : shape) {
    uint64_t value;
    if (!read_scalar(file, value) || value > std::numeric_limits<size_t>::max()) return false;
    dim = static_cast<size_t>(value);
    if (value != 0 && expected_elements > std::numeric_limits<uint64_t>::max() / value) return false;
    expected_elements *= value;
  }
  if (!read_scalar(file, elements) || !read_scalar(file, bytes) || elements != expected_elements) return false;
  DiskDType dtype = static_cast<DiskDType>(dtype_raw);
  size_t width = dtype == DiskDType::F32 ? 4 :
                 (dtype == DiskDType::F16 || dtype == DiskDType::BF16 ? 2 : 0);
  if (width == 0 || elements > std::numeric_limits<uint64_t>::max() / width ||
      bytes != elements * width) {
    if (error) *error = "unsupported or inconsistent tensor dtype";
    return false;
  }
  tensor = Tensor(std::move(shape));
  if (dtype == DiskDType::F32)
    return bool(file.read(reinterpret_cast<char*>(tensor.data()), static_cast<std::streamsize>(bytes)));
  constexpr size_t kChunkElements = 1 << 18;
  std::vector<uint16_t> buffer(kChunkElements);
  size_t done = 0;
  while (done < tensor.size()) {
    size_t count = std::min(kChunkElements, tensor.size() - done);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(count * 2))) return false;
    for (size_t i = 0; i < count; ++i) {
      if (dtype == DiskDType::BF16) {
        uint32_t bits = static_cast<uint32_t>(buffer[i]) << 16;
        std::memcpy(&tensor[done + i], &bits, sizeof(float));
      } else tensor[done + i] = half_to_float(buffer[i]);
    }
    done += count;
  }
  return true;
}

bool read_weight(std::ifstream& file, uint32_t version, WeightTensor& weight,
                 std::string* error) {
  uint32_t dtype_raw;
  uint64_t rank, elements, bytes;
  if (!read_scalar(file, dtype_raw) || !read_scalar(file, rank) || rank > 8) return false;
  std::vector<size_t> shape(static_cast<size_t>(rank));
  uint64_t expected_elements = 1;
  for (size_t& dim : shape) {
    uint64_t value;
    if (!read_scalar(file, value) || value > std::numeric_limits<size_t>::max()) return false;
    dim = static_cast<size_t>(value);
    if (value != 0 && expected_elements > std::numeric_limits<uint64_t>::max() / value)
      return false;
    expected_elements *= value;
  }
  if (!read_scalar(file, elements) || !read_scalar(file, bytes) ||
      elements != expected_elements || elements > std::numeric_limits<size_t>::max())
    return false;
  const DiskDType dtype = static_cast<DiskDType>(dtype_raw);
  if (dtype == DiskDType::Q8PerRow) {
    const bool payload_overflows = shape.size() == 2 &&
        shape[0] > (std::numeric_limits<uint64_t>::max() - elements) / sizeof(float);
    const uint64_t expected_bytes = payload_overflows ? 0 :
        elements + shape[0] * sizeof(float);
    if (version < kVersion || shape.size() != 2 ||
        payload_overflows || bytes != expected_bytes) {
      if (error) *error = "invalid Q8-per-row tensor";
      return false;
    }
    std::vector<int8_t> values(static_cast<size_t>(elements));
    std::vector<float> scales(shape[0]);
    for (size_t row = 0; row < shape[0]; ++row) {
      if (!read_scalar(file, scales[row]) ||
          !file.read(reinterpret_cast<char*>(values.data() + row * shape[1]),
                     static_cast<std::streamsize>(shape[1])))
        return false;
    }
    try {
      weight = WeightTensor::from_q8(std::move(shape), std::move(values),
                                     std::move(scales));
    } catch (const std::exception& exception) {
      if (error) *error = exception.what();
      return false;
    }
    return true;
  }
  const size_t width = dtype == DiskDType::F32 ? 4 :
                       (dtype == DiskDType::F16 || dtype == DiskDType::BF16 ? 2 : 0);
  if (width == 0 || elements > std::numeric_limits<uint64_t>::max() / width ||
      bytes != elements * width) {
    if (error) *error = "unsupported or inconsistent weight dtype";
    return false;
  }
  if (dtype == DiskDType::BF16) {
    std::vector<uint16_t> values(static_cast<size_t>(elements));
    if (!file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes)))
      return false;
    weight = WeightTensor::from_bf16(std::move(shape), std::move(values));
    return true;
  }
  Tensor tensor(shape);
  if (dtype == DiskDType::F32) {
    if (!file.read(reinterpret_cast<char*>(tensor.data()), static_cast<std::streamsize>(bytes)))
      return false;
  } else {
    std::vector<uint16_t> values(static_cast<size_t>(elements));
    if (!file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes)))
      return false;
    for (size_t i = 0; i < tensor.size(); ++i) tensor[i] = half_to_float(values[i]);
  }
  weight = WeightTensor(std::move(tensor));
  return true;
}

template<class TensorLike>
bool validate_shape(const TensorLike& tensor, std::initializer_list<size_t> expected,
                    const char* name, std::string* error) {
  if (tensor.shape() == std::vector<size_t>(expected)) return true;
  if (error) *error = std::string("invalid shape for ") + name;
  return false;
}

bool validate_model(const Model& model, std::string* error) {
  const Config& c = model.config;
  if (!c.hidden || !c.heads || !c.kv_heads || c.hidden % c.heads || c.heads % c.kv_heads) {
    if (error) *error = "invalid attention configuration";
    return false;
  }
  const size_t kv_dim = (c.hidden / c.heads) * c.kv_heads;
  if (!validate_shape(model.embedding, {c.vocab, c.hidden}, "embedding", error) ||
      !validate_shape(model.final_norm, {c.hidden}, "final_norm", error)) return false;
  if (!c.tie_word_embeddings &&
      !validate_shape(model.lm_head, {c.vocab, c.hidden}, "lm_head", error)) return false;
  if (model.layers.size() != c.layers) return false;
  for (size_t i = 0; i < c.layers; ++i) {
    const auto& l = model.layers[i];
    if (!validate_shape(l.attn_norm, {c.hidden}, "attn_norm", error) ||
        !validate_shape(l.ffn_norm, {c.hidden}, "ffn_norm", error) ||
        !validate_shape(l.q, {c.hidden, c.hidden}, "q_proj", error) ||
        !validate_shape(l.k, {kv_dim, c.hidden}, "k_proj", error) ||
        !validate_shape(l.v, {kv_dim, c.hidden}, "v_proj", error) ||
        !validate_shape(l.o, {c.hidden, c.hidden}, "o_proj", error) ||
        !validate_shape(l.gate, {c.intermediate, c.hidden}, "gate_proj", error) ||
        !validate_shape(l.up, {c.intermediate, c.hidden}, "up_proj", error) ||
        !validate_shape(l.down, {c.hidden, c.intermediate}, "down_proj", error)) return false;
  }
  return true;
}
}  // namespace

bool save_model(const Model& model, const std::string& path) {
  std::ofstream file(path, std::ios::binary);
  if (!file) return false;
  write_scalar(file, kMagic);
  write_scalar(file, kVersion);
  const Config& c = model.config;
  for (size_t value : {c.vocab, c.hidden, c.layers, c.heads, c.kv_heads, c.intermediate, c.context})
    write_scalar(file, static_cast<uint64_t>(value));
  write_scalar(file, c.eps);
  write_scalar(file, c.rope_theta);
  write_scalar(file, c.tie_word_embeddings ? kFlagTiedEmbeddings : 0u);
  write_scalar(file, static_cast<int32_t>(model.tokenizer.bos_id()));
  write_scalar(file, static_cast<int32_t>(model.tokenizer.eos_id()));
  write_scalar(file, static_cast<uint64_t>(model.tokenizer.vocabulary().size()));
  std::vector<int> ids;
  for (const auto& item : model.tokenizer.vocabulary()) ids.push_back(item.first);
  std::sort(ids.begin(), ids.end());
  for (int id : ids) {
    write_scalar(file, static_cast<int32_t>(id));
    write_string(file, model.tokenizer.vocabulary().at(id));
  }
  write_scalar(file, static_cast<uint64_t>(model.tokenizer.merges().size()));
  for (const auto& merge : model.tokenizer.merges()) {
    write_string(file, merge.first); write_string(file, merge.second);
  }
  write_scalar(file, static_cast<uint64_t>(model.tokenizer.special_tokens().size()));
  for (const auto& special : model.tokenizer.special_tokens()) {
    write_scalar(file, static_cast<int32_t>(special.first));
    write_string(file, special.second);
  }
  write_weight(file, model.embedding);
  write_tensor(file, model.final_norm);
  if (!c.tie_word_embeddings) write_weight(file, model.lm_head);
  for (const auto& layer : model.layers) {
    write_tensor(file, layer.attn_norm);
    write_tensor(file, layer.ffn_norm);
    for (const WeightTensor* tensor : {&layer.q, &layer.k, &layer.v, &layer.o,
                                       &layer.gate, &layer.up, &layer.down})
      write_weight(file, *tensor);
  }
  return bool(file);
}

bool load_model(const std::string& path, Model& model, std::string* error) {
  std::ifstream file(path, std::ios::binary);
  uint32_t magic, version;
  if (!file || !read_scalar(file, magic) || !read_scalar(file, version) ||
      magic != kMagic || (version != kLegacyVersion && version != kVersion)) {
    if (error) *error = "invalid or unsupported MiniInfer model";
    return false;
  }
  model = Model{};
  uint64_t values[7];
  for (uint64_t& value : values) if (!read_scalar(file, value)) return false;
  model.config.vocab=values[0]; model.config.hidden=values[1]; model.config.layers=values[2];
  model.config.heads=values[3]; model.config.kv_heads=values[4];
  model.config.intermediate=values[5]; model.config.context=values[6];
  uint32_t flags; int32_t bos, eos; uint64_t vocab_size;
  if (!read_scalar(file, model.config.eps) || !read_scalar(file, model.config.rope_theta) ||
      !read_scalar(file, flags) || !read_scalar(file, bos) || !read_scalar(file, eos) ||
      !read_scalar(file, vocab_size)) return false;
  model.config.tie_word_embeddings = (flags & kFlagTiedEmbeddings) != 0;
  model.tokenizer.set_special_ids(bos, eos);
  for (uint64_t i = 0; i < vocab_size; ++i) {
    int32_t id; std::string token;
    if (!read_scalar(file, id) || !read_string(file, token)) return false;
    model.tokenizer.add_token(id, std::move(token));
  }
  uint64_t merge_count;
  if (!read_scalar(file, merge_count)) return false;
  for (uint64_t i = 0; i < merge_count; ++i) {
    std::string left, right;
    if (!read_string(file, left) || !read_string(file, right)) return false;
    model.tokenizer.add_merge(std::move(left), std::move(right));
  }
  uint64_t special_count;
  if (!read_scalar(file, special_count)) return false;
  for (uint64_t i = 0; i < special_count; ++i) {
    int32_t id; std::string token;
    if (!read_scalar(file, id) || !read_string(file, token)) return false;
    model.tokenizer.add_special_token(id, std::move(token));
  }
  if (!read_weight(file, version, model.embedding, error) ||
      !read_tensor_v2(file, model.final_norm, error)) return false;
  if (!model.config.tie_word_embeddings &&
      !read_weight(file, version, model.lm_head, error)) return false;
  model.layers.resize(model.config.layers);
  for (auto& layer : model.layers) {
    if (!read_tensor_v2(file, layer.attn_norm, error) ||
        !read_tensor_v2(file, layer.ffn_norm, error)) return false;
    for (WeightTensor* tensor : {&layer.q, &layer.k, &layer.v, &layer.o,
                                 &layer.gate, &layer.up, &layer.down})
      if (!read_weight(file, version, *tensor, error)) return false;
  }
  if (!validate_model(model, error)) return false;
  if (file.peek() != std::ifstream::traits_type::eof()) {
    if (error) *error = "unexpected trailing model data";
    return false;
  }
  return true;
}
}  // namespace miniinfer
