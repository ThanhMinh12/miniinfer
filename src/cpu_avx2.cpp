#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace miniinfer {
namespace {
bool detect_avx2() {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2");
#elif defined(_MSC_VER)
  int registers[4] = {};
  __cpuidex(registers, 7, 0);
  return (registers[1] & (1 << 5)) != 0;
#else
  return false;
#endif
}
}  // namespace

bool runtime_has_avx2() {
  static const bool available = detect_avx2();
  return available;
}

float avx2_dot_f32(const float* left, const float* right, size_t count) {
  __m256 accumulator = _mm256_setzero_ps();
  size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    const __m256 a = _mm256_loadu_ps(left + index);
    const __m256 b = _mm256_loadu_ps(right + index);
    accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(a, b));
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, accumulator);
  float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                 lanes[4] + lanes[5] + lanes[6] + lanes[7];
  for (; index < count; ++index) result += left[index] * right[index];
  return result;
}

float avx2_dot_bf16(const uint16_t* left, const float* right, size_t count) {
  __m256 accumulator = _mm256_setzero_ps();
  size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    const __m128i packed = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(left + index));
    const __m256i widened = _mm256_cvtepu16_epi32(packed);
    const __m256 values = _mm256_castsi256_ps(_mm256_slli_epi32(widened, 16));
    accumulator = _mm256_add_ps(
        accumulator, _mm256_mul_ps(values, _mm256_loadu_ps(right + index)));
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, accumulator);
  float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                 lanes[4] + lanes[5] + lanes[6] + lanes[7];
  for (; index < count; ++index) {
    const uint32_t bits = static_cast<uint32_t>(left[index]) << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    result += value * right[index];
  }
  return result;
}

float avx2_dot_q8(const int8_t* left, const float* right, size_t count) {
  __m256 accumulator = _mm256_setzero_ps();
  size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    const __m128i packed = _mm_loadl_epi64(
        reinterpret_cast<const __m128i*>(left + index));
    const __m256 values = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed));
    accumulator = _mm256_add_ps(
        accumulator, _mm256_mul_ps(values, _mm256_loadu_ps(right + index)));
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, accumulator);
  float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                 lanes[4] + lanes[5] + lanes[6] + lanes[7];
  for (; index < count; ++index)
    result += static_cast<float>(left[index]) * right[index];
  return result;
}

}  // namespace miniinfer
