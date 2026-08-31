#include "miniinfer/tensor.h"
#include <numeric>
namespace miniinfer {
Tensor::Tensor(std::vector<size_t> shape, float value) : shape_(std::move(shape)) {
  strides_.resize(shape_.size()); size_t n=1;
  for (size_t i=shape_.size(); i-- > 0;) { strides_[i]=n; n*=shape_[i]; }
  data_.assign(n, value);
}
Tensor::Tensor(std::initializer_list<size_t> s, float v) : Tensor(std::vector<size_t>(s),v) {}
float& Tensor::at(const std::vector<size_t>& ix) { if(ix.size()!=shape_.size()) throw std::invalid_argument("rank"); size_t p=0; for(size_t i=0;i<ix.size();++i) p+=ix[i]*strides_[i]; if(p>=size()) throw std::out_of_range("index"); return data_[p]; }
const float& Tensor::at(const std::vector<size_t>& ix) const { return const_cast<Tensor*>(this)->at(ix); }
}
