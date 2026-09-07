#include "miniinfer/backend.h"
#include "miniinfer/model.h"
#include "miniinfer/transformer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
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

struct Summary {
  Result reference;
  double median_prefill_seconds = 0.0;
  double median_decode_seconds = 0.0;
};

struct BatchedSummary {
  Tensor logits;
  double median_seconds = 0.0;
};

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

double median(std::vector<double> values) {
  if (values.empty()) throw std::invalid_argument("cannot summarize an empty benchmark");
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() % 2 ? values[middle] :
      (values[middle - 1] + values[middle]) / 2.0;
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
    const int token = greedy(next_logits);
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

Summary run_repeated(const Model& model, const Backend& backend,
                     const std::vector<int>& prompt, size_t max_tokens,
                     CacheMode mode, size_t repetitions) {
  Summary summary;
  std::vector<double> prefill_times, decode_times;
  prefill_times.reserve(repetitions);
  decode_times.reserve(repetitions);
  for (size_t repetition = 0; repetition < repetitions; ++repetition) {
    Result result = run(model, backend, prompt, max_tokens, mode);
    if (repetition == 0) summary.reference = result;
    else if (result.tokens != summary.reference.tokens)
      throw std::runtime_error("greedy benchmark tokens changed between repetitions");
    prefill_times.push_back(result.prefill_seconds);
    decode_times.push_back(result.decode_seconds);
  }
  summary.median_prefill_seconds = median(std::move(prefill_times));
  summary.median_decode_seconds = median(std::move(decode_times));
  return summary;
}

BatchedSummary run_batched_repeated(const Model& model, const Backend& backend,
                                    const std::vector<int>& prompt,
                                    size_t repetitions) {
  BatchedSummary summary;
  std::vector<double> times;
  times.reserve(repetitions);
  for (size_t repetition = 0; repetition < repetitions; ++repetition) {
    const auto start = Clock::now();
    Tensor logits = prompt_logits_batched(model, backend, prompt);
    times.push_back(elapsed(start));
    if (repetition == 0) summary.logits = std::move(logits);
  }
  summary.median_seconds = median(std::move(times));
  return summary;
}

float max_difference(const std::vector<Tensor>& left,
                     const std::vector<Tensor>& right) {
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

void report_text(CacheMode mode, const Summary& summary, size_t prompt_size) {
  const size_t generated = summary.reference.tokens.size() - prompt_size;
  const double total = summary.median_prefill_seconds + summary.median_decode_seconds;
  std::cout << (mode == CacheMode::KV ? "kv" : "recompute")
            << " median_prefill_s=" << summary.median_prefill_seconds
            << " median_decode_s=" << summary.median_decode_seconds
            << " median_total_s=" << total
            << " generated=" << generated
            << " decode_tok_s="
            << (summary.median_decode_seconds > 0 ?
                generated / summary.median_decode_seconds : 0)
            << " total_tok_s=" << (total > 0 ? generated / total : 0) << '\n';
}

void report_json_summary(const std::optional<Summary>& summary, size_t prompt_size) {
  if (!summary) { std::cout << "null"; return; }
  const size_t generated = summary->reference.tokens.size() - prompt_size;
  const double total = summary->median_prefill_seconds + summary->median_decode_seconds;
  std::cout << "{\"median_prefill_seconds\":" << summary->median_prefill_seconds
            << ",\"median_decode_seconds\":" << summary->median_decode_seconds
            << ",\"median_total_seconds\":" << total
            << ",\"generated_tokens\":" << generated
            << ",\"decode_tokens_per_second\":"
            << (summary->median_decode_seconds > 0 ?
                generated / summary->median_decode_seconds : 0)
            << ",\"total_tokens_per_second\":"
            << (total > 0 ? generated / total : 0) << '}';
}

void usage() {
  std::cerr << "usage: miniinfer-bench model.miniinfer \"prompt\" "
               "[--preset quick|extended] [--cache kv|recompute|both] "
               "[--max-tokens N] [--repetitions N] [--threads N] "
               "[--cpu-kernel auto|scalar|avx2] [--json]\n";
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
    size_t max_tokens = 0, repetitions = 0;
    bool json = false;
    for (int i = 3; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--preset" && i + 1 < argc) preset = argv[++i];
      else if (option == "--cache" && i + 1 < argc) cache_choice = argv[++i];
      else if (option == "--max-tokens" && i + 1 < argc)
        max_tokens = std::stoull(argv[++i]);
      else if (option == "--repetitions" && i + 1 < argc)
        repetitions = std::stoull(argv[++i]);
      else if (option == "--threads" && i + 1 < argc)
        cpu_options.threads = std::stoull(argv[++i]);
      else if (option == "--cpu-kernel" && i + 1 < argc)
        cpu_options.kernel = parse_kernel(argv[++i]);
      else if (option == "--json") json = true;
      else { usage(); return 2; }
    }
    if (preset != "quick" && preset != "extended")
      throw std::invalid_argument("preset must be quick or extended");
    if (cache_choice != "kv" && cache_choice != "recompute" && cache_choice != "both")
      throw std::invalid_argument("cache must be kv, recompute, or both");
    if (max_tokens == 0) max_tokens = preset == "quick" ? 2 : 16;
    if (repetitions == 0) repetitions = preset == "quick" ? 1 : 3;

    const auto load_start = Clock::now();
    Model model;
    std::string error;
    if (!load_model(argv[1], model, &error))
      throw std::runtime_error(error.empty() ? "failed to load model" : error);
    const double load_seconds = elapsed(load_start);
    const std::vector<int> prompt = model.tokenizer.encode(argv[2]);
    if (prompt.empty()) throw std::invalid_argument("prompt must encode to at least one token");
    if (prompt.size() > model.config.context ||
        max_tokens > model.config.context - prompt.size())
      throw std::invalid_argument("prompt plus generated tokens exceeds model context");
    CPUBackend backend(cpu_options);

    std::optional<Summary> kv, recompute;
    if (cache_choice == "kv" || cache_choice == "both")
      kv = run_repeated(model, backend, prompt, max_tokens, CacheMode::KV, repetitions);
    if (cache_choice == "recompute" || cache_choice == "both")
      recompute = run_repeated(model, backend, prompt, max_tokens,
                               CacheMode::Recompute, repetitions);

    bool tokens_match = true;
    float cache_difference = 0.0f;
    if (kv && recompute) {
      tokens_match = kv->reference.tokens == recompute->reference.tokens;
      cache_difference = max_difference(kv->reference.logits,
                                        recompute->reference.logits);
      if (!tokens_match || cache_difference != 0.0f)
        throw std::runtime_error("KV and recompute outputs differ");
    }

    const BatchedSummary batched = run_batched_repeated(
        model, backend, prompt, repetitions);
    const Tensor& incremental = kv ? kv->reference.logits.front() :
                                      recompute->reference.logits.front();
    const float batched_difference = max_difference(batched.logits, incremental);
    if (batched_difference > 1e-5f)
      throw std::runtime_error("batched and incremental prefill logits differ");

    std::cout << std::fixed << std::setprecision(6);
    if (json) {
      std::cout << "{\"preset\":\"" << preset
                << "\",\"repetitions\":" << repetitions
                << ",\"load_seconds\":" << load_seconds
                << ",\"prompt_tokens\":" << prompt.size()
                << ",\"max_tokens\":" << max_tokens
                << ",\"backend\":\"" << backend.name()
                << "\",\"requested_kernel\":\""
                << cpu_kernel_name(cpu_options.kernel)
                << "\",\"threads\":" << backend.options().threads
                << ",\"kv\":";
      report_json_summary(kv, prompt.size());
      std::cout << ",\"recompute\":";
      report_json_summary(recompute, prompt.size());
      std::cout << ",\"equivalence\":{\"tokens_identical\":"
                << (tokens_match ? "true" : "false")
                << ",\"max_logit_abs\":" << cache_difference
                << "},\"batched_prefill\":{\"median_seconds\":"
                << batched.median_seconds << ",\"prompt_tokens_per_second\":"
                << (batched.median_seconds > 0 ?
                    prompt.size() / batched.median_seconds : 0)
                << ",\"incremental_max_logit_abs\":" << batched_difference
                << "}}\n";
    } else {
      std::cout << "preset=" << preset << " repetitions=" << repetitions
                << " load_s=" << load_seconds
                << " prompt_tokens=" << prompt.size()
                << " max_tokens=" << max_tokens
                << " backend=" << backend.name()
                << " requested_kernel=" << cpu_kernel_name(cpu_options.kernel)
                << " threads=" << backend.options().threads << '\n';
      if (kv) report_text(CacheMode::KV, *kv, prompt.size());
      if (recompute) report_text(CacheMode::Recompute, *recompute, prompt.size());
      if (kv && recompute)
        std::cout << "equivalence tokens=identical max_logit_abs="
                  << cache_difference << '\n';
      std::cout << "batched_median_prefill_s=" << batched.median_seconds
                << " batched_prompt_tok_s="
                << (batched.median_seconds > 0 ?
                    prompt.size() / batched.median_seconds : 0)
                << " incremental_max_logit_abs=" << batched_difference << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
