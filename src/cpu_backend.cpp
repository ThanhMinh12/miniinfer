#include "miniinfer/backend.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace miniinfer {
#ifdef MINIINFER_ENABLE_AVX2
float avx2_dot_f32(const float* left, const float* right, size_t count);
float avx2_dot_bf16(const uint16_t* left, const float* right, size_t count);
float avx2_dot_q8(const int8_t* left, const float* right, size_t count);
bool runtime_has_avx2();
#else
bool runtime_has_avx2() { return false; }
#endif

namespace {

float bf16_to_float(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

class RowPool {
 public:
  explicit RowPool(size_t thread_count) {
    const size_t worker_count = thread_count > 1 ? thread_count - 1 : 0;
    workers_.reserve(worker_count);
    for (size_t index = 0; index < worker_count; ++index)
      workers_.emplace_back([this] { worker(); });
  }

  ~RowPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      ++generation_;
    }
    work_ready_.notify_all();
    for (std::thread& worker_thread : workers_) worker_thread.join();
  }

  void run(size_t rows, std::function<void(size_t)> function) {
    if (workers_.empty() || rows < 2) {
      for (size_t row = 0; row < rows; ++row) function(row);
      return;
    }
    std::unique_lock<std::mutex> invocation_lock(invocation_mutex_);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      function_ = std::move(function);
      rows_ = rows;
      next_.store(0, std::memory_order_relaxed);
      active_workers_ = workers_.size();
      ++generation_;
    }
    work_ready_.notify_all();
    consume(rows);
    std::unique_lock<std::mutex> lock(mutex_);
    work_done_.wait(lock, [this] { return active_workers_ == 0; });
    function_ = {};
  }

 private:
  void consume(size_t rows) {
    for (;;) {
      const size_t row = next_.fetch_add(1, std::memory_order_relaxed);
      if (row >= rows) break;
      function_(row);
    }
  }

  void worker() {
    size_t observed_generation = 0;
    for (;;) {
      std::unique_lock<std::mutex> lock(mutex_);
      work_ready_.wait(lock, [this, observed_generation] {
        return stopping_ || generation_ != observed_generation;
      });
      if (stopping_) return;
      observed_generation = generation_;
      const size_t rows = rows_;
      lock.unlock();
      consume(rows);
      lock.lock();
      if (--active_workers_ == 0) work_done_.notify_one();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex invocation_mutex_, mutex_;
  std::condition_variable work_ready_, work_done_;
  std::function<void(size_t)> function_;
  std::atomic<size_t> next_{0};
  size_t rows_ = 0;
  size_t active_workers_ = 0;
  size_t generation_ = 0;
  bool stopping_ = false;
};

}  // namespace

class CPUBackend::Impl {
 public:
  explicit Impl(CPUOptions requested)
      : options(requested), pool(requested.threads ? requested.threads : 1) {
    if (options.threads == 0) options.threads = 1;
    selected = options.kernel;
    if (selected == CPUKernel::Auto)
      selected = runtime_has_avx2() ? CPUKernel::AVX2 : CPUKernel::Scalar;
    if (selected == CPUKernel::AVX2 && !runtime_has_avx2())
      selected = CPUKernel::Scalar;
  }

  float dot(const WeightTensor& weight, size_t row, const float* activation) const {
    const size_t columns = weight.columns();
    if (weight.type() == WeightType::F32) {
      const float* values = weight.f32_values().data() + row * columns;
#ifdef MINIINFER_ENABLE_AVX2
      if (selected == CPUKernel::AVX2) return avx2_dot_f32(values, activation, columns);
#endif
      float sum = 0.0f;
      for (size_t column = 0; column < columns; ++column)
        sum += values[column] * activation[column];
      return sum;
    }
    if (weight.type() == WeightType::BF16) {
      const uint16_t* values = weight.bf16_values().data() + row * columns;
#ifdef MINIINFER_ENABLE_AVX2
      if (selected == CPUKernel::AVX2) return avx2_dot_bf16(values, activation, columns);
#endif
      float sum = 0.0f;
      for (size_t column = 0; column < columns; ++column)
        sum += bf16_to_float(values[column]) * activation[column];
      return sum;
    }
    const int8_t* values = weight.q8_values().data() + row * columns;
#ifdef MINIINFER_ENABLE_AVX2
    if (selected == CPUKernel::AVX2)
      return avx2_dot_q8(values, activation, columns) * weight.scales()[row];
#endif
    float sum = 0.0f;
    for (size_t column = 0; column < columns; ++column)
      sum += static_cast<float>(values[column]) * activation[column];
    return sum * weight.scales()[row];
  }

  CPUOptions options;
  CPUKernel selected = CPUKernel::Scalar;
  mutable RowPool pool;
};

CPUBackend::CPUBackend(CPUOptions options) : impl_(std::make_unique<Impl>(options)) {}
CPUBackend::CPUBackend(size_t threads) : CPUBackend(CPUOptions{threads, CPUKernel::Auto}) {}
CPUBackend::~CPUBackend() = default;
CPUBackend::CPUBackend(CPUBackend&&) noexcept = default;
CPUBackend& CPUBackend::operator=(CPUBackend&&) noexcept = default;

Tensor CPUBackend::linear(const WeightTensor& weight,
                          const Tensor& activations) const {
  if (weight.shape().size() != 2 ||
      (activations.shape().size() != 1 && activations.shape().size() != 2))
    throw std::invalid_argument(
        "linear expects [output,input] weights and rank-1/2 activations");
  const bool vector = activations.shape().size() == 1;
  const size_t batch = vector ? 1 : activations.dim(0);
  const size_t input = vector ? activations.dim(0) : activations.dim(1);
  if (weight.columns() != input) throw std::invalid_argument("linear input shape mismatch");
  Tensor output = vector ? Tensor({weight.rows()}) : Tensor({batch, weight.rows()});
  const size_t work_items = batch * weight.rows();
  impl_->pool.run(work_items, [&](size_t item) {
    const size_t batch_index = item / weight.rows();
    const size_t row = item % weight.rows();
    output[item] = impl_->dot(weight, row,
                              activations.data() + batch_index * input);
  });
  return output;
}

const char* CPUBackend::name() const {
  return impl_->selected == CPUKernel::AVX2 ? "cpu-avx2" : "cpu-scalar";
}
const CPUOptions& CPUBackend::options() const { return impl_->options; }
CPUKernel CPUBackend::selected_kernel() const { return impl_->selected; }

const char* cpu_kernel_name(CPUKernel kernel) {
  switch (kernel) {
    case CPUKernel::Auto: return "auto";
    case CPUKernel::Scalar: return "scalar";
    case CPUKernel::AVX2: return "avx2";
  }
  return "unknown";
}

bool cpu_avx2_available() { return runtime_has_avx2(); }

}  // namespace miniinfer
