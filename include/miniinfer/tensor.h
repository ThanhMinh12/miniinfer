#pragma once
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace miniinfer {
class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(std::vector<size_t> shape, float value = 0.0f);
  Tensor(std::initializer_list<size_t> shape, float value = 0.0f);
  float& operator[](size_t i) { return data_[i]; }
  const float& operator[](size_t i) const { return data_[i]; }
  float& at(const std::vector<size_t>& index);
  const float& at(const std::vector<size_t>& index) const;
  const std::vector<size_t>& shape() const { return shape_; }
  const std::vector<size_t>& strides() const { return strides_; }
  size_t size() const { return data_.size(); }
  size_t dim(size_t i) const { return shape_.at(i); }
  float* data() { return data_.data(); }
  const float* data() const { return data_.data(); }
  std::vector<float>& values() { return data_; }
  const std::vector<float>& values() const { return data_; }
 private:
  std::vector<size_t> shape_, strides_;
  std::vector<float> data_;
};
}
