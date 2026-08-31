#pragma once

#include "miniinfer/backend.h"

#ifdef MINIINFER_ENABLE_WEBGPU

#include <memory>

namespace miniinfer {

// Dawn-backed synchronous linear backend. Transformer operations that are not
// linear remain on the host until their WGSL implementations are enabled.
class WebGPUBackend final : public Backend {
 public:
  WebGPUBackend();
  ~WebGPUBackend() override;
  WebGPUBackend(WebGPUBackend&&) noexcept;
  WebGPUBackend& operator=(WebGPUBackend&&) noexcept;
  WebGPUBackend(const WebGPUBackend&) = delete;
  WebGPUBackend& operator=(const WebGPUBackend&) = delete;

  Tensor linear(const WeightTensor& weight,
                const Tensor& activations) const override;
  const char* name() const override { return "webgpu-dawn"; }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace miniinfer
#endif
