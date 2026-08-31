#include "miniinfer/weight.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace miniinfer {
namespace {
size_t element_count(const std::vector<size_t>& shape) {
  size_t count = 1;
  for (size_t dimension : shape) {
    if (dimension != 0 && count > std::numeric_limits<size_t>::max() / dimension)
      throw std::overflow_error("weight tensor is too large");
    count *= dimension;
  }
  return count;
}

float bf16_to_float(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}
}  // namespace

WeightTensor::WeightTensor(const Tensor& tensor)
    : shape_(tensor.shape()), f32_(tensor.values()) {}

WeightTensor::WeightTensor(Tensor&& tensor)
    : shape_(tensor.shape()), f32_(std::move(tensor.values())) {}

WeightTensor WeightTensor::from_bf16(std::vector<size_t> shape,
                                     std::vector<uint16_t> values) {
  if (values.size() != element_count(shape))
    throw std::invalid_argument("BF16 weight payload does not match its shape");
  WeightTensor result;
  result.type_ = WeightType::BF16;
  result.shape_ = std::move(shape);
  result.bf16_ = std::move(values);
  return result;
}

WeightTensor WeightTensor::from_q8(std::vector<size_t> shape,
                                   std::vector<int8_t> values,
                                   std::vector<float> scales) {
  if (shape.size() != 2 || values.size() != element_count(shape) ||
      scales.size() != shape[0])
    throw std::invalid_argument("Q8 weight payload does not match its shape");
  for (float scale : scales)
    if (!std::isfinite(scale) || scale < 0.0f)
      throw std::invalid_argument("Q8 scales must be finite and non-negative");
  WeightTensor result;
  result.type_ = WeightType::Q8PerRow;
  result.shape_ = std::move(shape);
  result.q8_ = std::move(values);
  result.scales_ = std::move(scales);
  return result;
}

WeightTensor WeightTensor::quantize_q8(const Tensor& tensor) {
  if (tensor.shape().size() != 2)
    throw std::invalid_argument("only rank-2 weights can be quantized");
  std::vector<int8_t> values(tensor.size());
  std::vector<float> scales(tensor.dim(0));
  for (size_t row = 0; row < tensor.dim(0); ++row) {
    float maximum = 0.0f;
    for (size_t column = 0; column < tensor.dim(1); ++column)
      maximum = std::max(maximum, std::fabs(tensor[row * tensor.dim(1) + column]));
    const float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
    scales[row] = scale;
    for (size_t column = 0; column < tensor.dim(1); ++column) {
      const float original = tensor[row * tensor.dim(1) + column];
      const int quantized = scale == 0.0f ? 0 :
          static_cast<int>(std::lround(original / scale));
      values[row * tensor.dim(1) + column] = static_cast<int8_t>(
          std::max(-127, std::min(127, quantized)));
    }
  }
  return from_q8(tensor.shape(), std::move(values), std::move(scales));
}

size_t WeightTensor::size() const {
  if (shape_.empty()) return 0;
  return element_count(shape_);
}

float WeightTensor::value(size_t index) const {
  if (index >= size()) throw std::out_of_range("weight index");
  if (type_ == WeightType::F32) return f32_[index];
  if (type_ == WeightType::BF16) return bf16_to_float(bf16_[index]);
  return static_cast<float>(q8_[index]) * scales_[index / columns()];
}

void WeightTensor::copy_row(size_t row, float* destination) const {
  if (shape_.size() != 2 || row >= rows())
    throw std::out_of_range("weight row");
  const size_t offset = row * columns();
  for (size_t column = 0; column < columns(); ++column)
    destination[column] = value(offset + column);
}

Tensor WeightTensor::dequantize() const {
  Tensor result(shape_);
  for (size_t i = 0; i < size(); ++i) result[i] = value(i);
  return result;
}

}  // namespace miniinfer
