#pragma once

#include "miniinfer/tensor.h"

#include <cstdint>
#include <vector>

namespace miniinfer {

enum class WeightType { F32, BF16, Q8PerRow };

// Immutable model parameters. Activations deliberately remain FP32 Tensor values.
class WeightTensor {
 public:
  WeightTensor() = default;
  WeightTensor(const Tensor& tensor);
  WeightTensor(Tensor&& tensor);

  static WeightTensor from_bf16(std::vector<size_t> shape,
                                std::vector<uint16_t> values);
  static WeightTensor from_q8(std::vector<size_t> shape,
                              std::vector<int8_t> values,
                              std::vector<float> scales);
  static WeightTensor quantize_q8(const Tensor& tensor);

  WeightType type() const { return type_; }
  const std::vector<size_t>& shape() const { return shape_; }
  size_t dim(size_t index) const { return shape_.at(index); }
  size_t size() const;
  size_t rows() const { return shape_.empty() ? 0 : shape_[0]; }
  size_t columns() const { return shape_.size() == 2 ? shape_[1] : 0; }
  float value(size_t index) const;
  void copy_row(size_t row, float* destination) const;
  Tensor dequantize() const;

  const std::vector<float>& f32_values() const { return f32_; }
  const std::vector<uint16_t>& bf16_values() const { return bf16_; }
  const std::vector<int8_t>& q8_values() const { return q8_; }
  const std::vector<float>& scales() const { return scales_; }

 private:
  WeightType type_ = WeightType::F32;
  std::vector<size_t> shape_;
  std::vector<float> f32_;
  std::vector<uint16_t> bf16_;
  std::vector<int8_t> q8_;
  std::vector<float> scales_;
};

}  // namespace miniinfer
