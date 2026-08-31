#include "miniinfer/backend.h"
#include "miniinfer/model.h"
#include "miniinfer/transformer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace miniinfer;

namespace {
using Clock = std::chrono::steady_clock;

struct Result {
  double prefill_seconds = 0.0;
  double decode_seconds = 0.0;
  std::vector<int> tokens;
  std::vector<Tensor> logits;
};

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

Result run(const Model& model, const Backend& backend, const std::vector<int>& prompt,
           size_t max_tokens, CacheMode mode) {
  Result result;
  result.tokens = prompt;
  const auto prefill_start = Clock::now();
  Tensor next_logits;
  std::unique_ptr<KVCache> cache;
  if (mode == CacheMode::KV) {
    cache = std::make_unique<KVCache>(model.config.layers, prompt.size() + max_tokens,
                                     model.config.kv_heads,
                                     model.config.hidden / model.config.heads);
    for (size_t i = 0; i < prompt.size(); ++i)
      next_logits = forward_token(model, backend, prompt[i], i, *cache);
  } else {
    next_logits = prompt_logits(model, backend, prompt);
  }
  result.prefill_seconds = elapsed(prefill_start);

  const auto decode_start = Clock::now();
  for (size_t i = 0; i < max_tokens; ++i) {
    result.logits.push_back(next_logits);
    int token = greedy(next_logits);
    result.tokens.push_back(token);
    if (token == model.tokenizer.eos_id()) break;
    if (i + 1 == max_tokens) continue;
    if (mode == CacheMode::KV)
      next_logits = forward_token(model, backend, token, prompt.size() + i, *cache);
    else
      next_logits = prompt_logits(model, backend, result.tokens);
  }
  result.decode_seconds = elapsed(decode_start);
  return result;
}

const char* name(CacheMode mode) {
  return mode == CacheMode::KV ? "kv" : "recompute";
}

void report(CacheMode mode, const Result& result, size_t prompt_size) {
  const size_t generated = result.tokens.size() - prompt_size;
  const double total = result.prefill_seconds + result.decode_seconds;
  std::cout << name(mode)
            << " prefill_s=" << result.prefill_seconds
            << " decode_s=" << result.decode_seconds
            << " total_s=" << total
            << " generated=" << generated
            << " decode_tok_s=" << (result.decode_seconds > 0 ? generated / result.decode_seconds : 0)
            << " total_tok_s=" << (total > 0 ? generated / total : 0) << '\n';
}

float max_difference(const std::vector<Tensor>& left, const std::vector<Tensor>& right) {
  if (left.size() != right.size()) return INFINITY;
  float difference = 0.0f;
  for (size_t step = 0; step < left.size(); ++step) {
    if (left[step].shape() != right[step].shape()) return INFINITY;
    for (size_t i = 0; i < left[step].size(); ++i)
      difference = std::max(difference, std::fabs(left[step][i] - right[step][i]));
  }
  return difference;
}

float max_difference(const Tensor& left, const Tensor& right) {
  if (left.shape() != right.shape()) return INFINITY;
  float difference = 0.0f;
  for (size_t i = 0; i < left.size(); ++i)
    difference = std::max(difference, std::fabs(left[i] - right[i]));
  return difference;
}

void usage() {
  std::cerr << "usage: miniinfer-bench model.miniinfer \"prompt\" "
               "[--preset quick|extended] [--cache kv|recompute|both] "
               "[--threads N] [--cpu-kernel auto|scalar|avx2]\n";
}

CPUKernel parse_kernel(const std::string& value) {
  if (value == "auto") return CPUKernel::Auto;
  if (value == "scalar") return CPUKernel::Scalar;
  if (value == "avx2") return CPUKernel::AVX2;
  throw std::invalid_argument("CPU kernel must be auto, scalar, or avx2");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) { usage(); return 2; }
    std::string preset = "quick";
    std::string cache_choice = "both";
    CPUOptions cpu_options;
    for (int i = 3; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--preset" && i + 1 < argc) preset = argv[++i];
      else if (option == "--cache" && i + 1 < argc) cache_choice = argv[++i];
      else if (option == "--threads" && i + 1 < argc)
        cpu_options.threads = std::stoull(argv[++i]);
      else if (option == "--cpu-kernel" && i + 1 < argc)
        cpu_options.kernel = parse_kernel(argv[++i]);
      else { usage(); return 2; }
    }
    if (preset != "quick" && preset != "extended")
      throw std::invalid_argument("preset must be quick or extended");
    if (cache_choice != "kv" && cache_choice != "recompute" && cache_choice != "both")
      throw std::invalid_argument("cache must be kv, recompute, or both");
    const size_t max_tokens = preset == "quick" ? 2 : 16;

    const auto load_start = Clock::now();
    Model model;
    std::string error;
    if (!load_model(argv[1], model, &error))
      throw std::runtime_error(error.empty() ? "failed to load model" : error);
    const double load_seconds = elapsed(load_start);
    const std::vector<int> prompt = model.tokenizer.encode(argv[2]);
    CPUBackend backend(cpu_options);
    std::cout << std::fixed << std::setprecision(6)
              << "preset=" << preset << " load_s=" << load_seconds
              << " prompt_tokens=" << prompt.size() << " max_tokens=" << max_tokens
              << " backend=" << backend.name()
              << " requested_kernel=" << cpu_kernel_name(cpu_options.kernel)
              << " threads=" << backend.options().threads << '\n';

    Result kv, recompute;
    if (cache_choice == "kv" || cache_choice == "both") {
      kv = run(model, backend, prompt, max_tokens, CacheMode::KV);
      report(CacheMode::KV, kv, prompt.size());
    }
    if (cache_choice == "recompute" || cache_choice == "both") {
      recompute = run(model, backend, prompt, max_tokens, CacheMode::Recompute);
      report(CacheMode::Recompute, recompute, prompt.size());
    }
    if (cache_choice == "both") {
      const float difference = max_difference(kv.logits, recompute.logits);
      const bool tokens_match = kv.tokens == recompute.tokens;
      std::cout << "equivalence tokens=" << (tokens_match ? "identical" : "DIFFERENT")
                << " max_logit_abs=" << difference << '\n';
      if (!tokens_match || difference != 0.0f) return 1;
    }
    const auto batched_start = Clock::now();
    const Tensor batched = prompt_logits_batched(model, backend, prompt);
    const double batched_seconds = elapsed(batched_start);
    Tensor incremental;
    if (!kv.logits.empty()) incremental = kv.logits.front();
    else if (!recompute.logits.empty()) incremental = recompute.logits.front();
    const float batched_difference = max_difference(batched, incremental);
    std::cout << "batched_prefill_s=" << batched_seconds
              << " batched_prompt_tok_s="
              << (batched_seconds > 0 ? prompt.size() / batched_seconds : 0)
              << " incremental_max_logit_abs=" << batched_difference << '\n';
    if (batched_difference > 1e-5f) return 1;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
