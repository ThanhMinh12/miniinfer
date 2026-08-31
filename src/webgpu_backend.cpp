#include "miniinfer/webgpu_backend.h"

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace miniinfer {
namespace {
constexpr char kLinearShader[] = R"(
struct Parameters { rows: u32, columns: u32, batch: u32, padding: u32 }
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> activations: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;
@compute @workgroup_size(8, 8)
fn linear(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  let batch_index = id.y;
  if (row >= parameters.rows || batch_index >= parameters.batch) { return; }
  var sum = 0.0;
  for (var column = 0u; column < parameters.columns; column += 1u) {
    sum += weights[row * parameters.columns + column] *
           activations[batch_index * parameters.columns + column];
  }
  output[batch_index * parameters.rows + row] = sum;
})";

struct Parameters { uint32_t rows, columns, batch, padding; };

wgpu::Buffer create_buffer(wgpu::Device device, uint64_t size,
                           wgpu::BufferUsage usage) {
  wgpu::BufferDescriptor descriptor{};
  descriptor.size = size;
  descriptor.usage = usage;
  return device.CreateBuffer(&descriptor);
}
}  // namespace

class WebGPUBackend::Impl {
 public:
  Impl() {
    static constexpr auto timed_wait = wgpu::InstanceFeatureName::TimedWaitAny;
    wgpu::InstanceDescriptor instance_descriptor{};
    instance_descriptor.requiredFeatureCount = 1;
    instance_descriptor.requiredFeatures = &timed_wait;
    instance = wgpu::CreateInstance(&instance_descriptor);
    if (!instance) throw std::runtime_error("Dawn instance creation failed");

    wgpu::RequestAdapterOptions adapter_options{};
    auto adapter_callback = [](wgpu::RequestAdapterStatus status, wgpu::Adapter result,
                               wgpu::StringView, void* userdata) {
      if (status == wgpu::RequestAdapterStatus::Success)
        *static_cast<wgpu::Adapter*>(userdata) = std::move(result);
    };
    instance.WaitAny(instance.RequestAdapter(&adapter_options,
        wgpu::CallbackMode::WaitAnyOnly, adapter_callback, &adapter), UINT64_MAX);
    if (!adapter) throw std::runtime_error("Dawn could not find a WebGPU adapter");

    auto device_callback = [](wgpu::RequestDeviceStatus status, wgpu::Device result,
                              wgpu::StringView, void* userdata) {
      if (status == wgpu::RequestDeviceStatus::Success)
        *static_cast<wgpu::Device*>(userdata) = std::move(result);
    };
    instance.WaitAny(adapter.RequestDevice(nullptr, wgpu::CallbackMode::WaitAnyOnly,
                                           device_callback, &device), UINT64_MAX);
    if (!device) throw std::runtime_error("Dawn WebGPU device creation failed");
    queue = device.GetQueue();

    wgpu::ShaderSourceWGSL source{};
    source.code = kLinearShader;
    wgpu::ShaderModuleDescriptor shader_descriptor{};
    shader_descriptor.nextInChain = &source;
    wgpu::ShaderModule shader = device.CreateShaderModule(&shader_descriptor);
    wgpu::ComputePipelineDescriptor pipeline_descriptor{};
    pipeline_descriptor.compute.module = shader;
    pipeline_descriptor.compute.entryPoint = "linear";
    pipeline = device.CreateComputePipeline(&pipeline_descriptor);
    if (!pipeline) throw std::runtime_error("WebGPU linear pipeline creation failed");
  }

  wgpu::Buffer weight_buffer(const WeightTensor& weight) {
    auto found = weights.find(&weight);
    if (found != weights.end()) return found->second;
    if (weight.type() != WeightType::F32)
      throw std::invalid_argument(
          "WebGPU linear currently supports F32 weights; use the CPU backend for BF16/Q8");
    const uint64_t bytes = weight.size() * sizeof(float);
    wgpu::Buffer buffer = create_buffer(device, bytes,
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
    queue.WriteBuffer(buffer, 0, weight.f32_values().data(), bytes);
    weights.emplace(&weight, buffer);
    return buffer;
  }

  wgpu::Instance instance;
  wgpu::Adapter adapter;
  wgpu::Device device;
  wgpu::Queue queue;
  wgpu::ComputePipeline pipeline;
  std::unordered_map<const WeightTensor*, wgpu::Buffer> weights;
};

WebGPUBackend::WebGPUBackend() : impl_(std::make_unique<Impl>()) {}
WebGPUBackend::~WebGPUBackend() = default;
WebGPUBackend::WebGPUBackend(WebGPUBackend&&) noexcept = default;
WebGPUBackend& WebGPUBackend::operator=(WebGPUBackend&&) noexcept = default;

Tensor WebGPUBackend::linear(const WeightTensor& weight,
                             const Tensor& activations) const {
  if (weight.shape().size() != 2 ||
      (activations.shape().size() != 1 && activations.shape().size() != 2))
    throw std::invalid_argument("WebGPU linear shape mismatch");
  const bool vector = activations.shape().size() == 1;
  const size_t batch = vector ? 1 : activations.dim(0);
  const size_t columns = vector ? activations.dim(0) : activations.dim(1);
  if (columns != weight.columns()) throw std::invalid_argument("WebGPU linear input mismatch");
  if (weight.rows() > std::numeric_limits<uint32_t>::max() ||
      columns > std::numeric_limits<uint32_t>::max() ||
      batch > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("WebGPU linear dimensions exceed u32");

  const uint64_t input_bytes = activations.size() * sizeof(float);
  const uint64_t output_bytes = batch * weight.rows() * sizeof(float);
  const wgpu::Buffer weights = impl_->weight_buffer(weight);
  const wgpu::Buffer input = create_buffer(impl_->device, input_bytes,
      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
  const wgpu::Buffer output = create_buffer(impl_->device, output_bytes,
      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
  const wgpu::Buffer readback = create_buffer(impl_->device, output_bytes,
      wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
  const wgpu::Buffer parameters_buffer = create_buffer(impl_->device, sizeof(Parameters),
      wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  impl_->queue.WriteBuffer(input, 0, activations.data(), input_bytes);
  const Parameters parameters{static_cast<uint32_t>(weight.rows()),
      static_cast<uint32_t>(columns), static_cast<uint32_t>(batch), 0};
  impl_->queue.WriteBuffer(parameters_buffer, 0, &parameters, sizeof(parameters));

  const wgpu::BindGroupEntry entries[] = {
      {.binding = 0, .buffer = weights, .offset = 0, .size = weight.size() * sizeof(float)},
      {.binding = 1, .buffer = input, .offset = 0, .size = input_bytes},
      {.binding = 2, .buffer = output, .offset = 0, .size = output_bytes},
      {.binding = 3, .buffer = parameters_buffer, .offset = 0, .size = sizeof(Parameters)}};
  wgpu::BindGroupDescriptor bind_group_descriptor{};
  bind_group_descriptor.layout = impl_->pipeline.GetBindGroupLayout(0);
  bind_group_descriptor.entryCount = 4;
  bind_group_descriptor.entries = entries;
  const wgpu::BindGroup bind_group = impl_->device.CreateBindGroup(&bind_group_descriptor);

  wgpu::CommandEncoder encoder = impl_->device.CreateCommandEncoder();
  wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
  pass.SetPipeline(impl_->pipeline);
  pass.SetBindGroup(0, bind_group);
  pass.DispatchWorkgroups((parameters.rows + 7) / 8, (parameters.batch + 7) / 8);
  pass.End();
  encoder.CopyBufferToBuffer(output, 0, readback, 0, output_bytes);
  wgpu::CommandBuffer commands = encoder.Finish();
  impl_->queue.Submit(1, &commands);

  wgpu::MapAsyncStatus map_status = wgpu::MapAsyncStatus::Error;
  auto map_callback = [](wgpu::MapAsyncStatus status, wgpu::StringView, void* userdata) {
    *static_cast<wgpu::MapAsyncStatus*>(userdata) = status;
  };
  impl_->instance.WaitAny(readback.MapAsync(wgpu::MapMode::Read, 0, output_bytes,
      wgpu::CallbackMode::WaitAnyOnly, map_callback, &map_status), UINT64_MAX);
  if (map_status != wgpu::MapAsyncStatus::Success)
    throw std::runtime_error("WebGPU output readback failed");
  Tensor result = vector ? Tensor({weight.rows()}) : Tensor({batch, weight.rows()});
  std::memcpy(result.data(), readback.GetConstMappedRange(0, output_bytes), output_bytes);
  readback.Unmap();
  return result;
}

}  // namespace miniinfer
