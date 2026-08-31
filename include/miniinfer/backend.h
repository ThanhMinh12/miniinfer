#pragma once
#include "miniinfer/tensor.h"
#include "miniinfer/weight.h"

#include <cstddef>
#include <memory>

namespace miniinfer {

enum class CPUKernel { Auto, Scalar, AVX2 };

struct CPUOptions {
  size_t threads = 1;
  CPUKernel kernel = CPUKernel::Auto;
};

class Backend {
 public:
  virtual ~Backend() = default;
  // weight is [output, input]. Activations may be [input] or [batch, input].
  virtual Tensor linear(const WeightTensor& weight,
                        const Tensor& activations) const = 0;
  virtual const char* name() const = 0;
};

class CPUBackend final : public Backend {
 public:
  explicit CPUBackend(CPUOptions options = {});
  explicit CPUBackend(size_t threads);
  ~CPUBackend() override;
  CPUBackend(CPUBackend&&) noexcept;
  CPUBackend& operator=(CPUBackend&&) noexcept;
  CPUBackend(const CPUBackend&) = delete;
  CPUBackend& operator=(const CPUBackend&) = delete;

  Tensor linear(const WeightTensor& weight,
                const Tensor& activations) const override;
  const char* name() const override;
  const CPUOptions& options() const;
  CPUKernel selected_kernel() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

const char* cpu_kernel_name(CPUKernel kernel);
bool cpu_avx2_available();

}  // namespace miniinfer
