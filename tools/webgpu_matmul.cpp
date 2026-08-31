#include "miniinfer/backend.h"
#include "miniinfer/webgpu_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

using namespace miniinfer;

int main() {
  try {
    Tensor matrix({257, 259});
    Tensor input({17, 259});
    for (size_t i = 0; i < matrix.size(); ++i)
      matrix[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 31.0f;
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 17.0f;
    const WeightTensor weight(matrix);
    CPUBackend cpu(CPUOptions{1, CPUKernel::Scalar});
    WebGPUBackend gpu;
    const Tensor expected = cpu.linear(weight, input);
    const auto start = std::chrono::steady_clock::now();
    const Tensor actual = gpu.linear(weight, input);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    float maximum = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i)
      maximum = std::max(maximum, std::fabs(actual[i] - expected[i]));
    std::cout << "webgpu_linear_s=" << seconds << " max_abs_error=" << maximum << '\n';
    return maximum <= 1e-4f ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
