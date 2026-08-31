#pragma once
#include "miniinfer/tensor.h"
#include <vector>
namespace miniinfer {
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor matvec(const Tensor& a, const Tensor& x);
Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps);
Tensor softmax(const Tensor& x);
Tensor silu(const Tensor& x);
void rope(Tensor& x, size_t position, float theta);
}
