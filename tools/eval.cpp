#include "miniinfer/backend.h"
#include "miniinfer/model.h"
#include "miniinfer/transformer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace miniinfer;

namespace {
using Clock = std::chrono::steady_clock;

CPUKernel parse_kernel(const std::string& value) {
  if (value == "auto") return CPUKernel::Auto;
  if (value == "scalar") return CPUKernel::Scalar;
  if (value == "avx2") return CPUKernel::AVX2;
  throw std::invalid_argument("CPU kernel must be auto, scalar, or avx2");
}

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

double token_nll(const Tensor& logits, int target) {
  if (target < 0 || static_cast<size_t>(target) >= logits.size())
    throw std::out_of_range("evaluation target is outside the vocabulary");
  const float maximum = *std::max_element(logits.values().begin(), logits.values().end());
  if (!std::isfinite(maximum)) throw std::runtime_error("model produced non-finite logits");
  double exponential_sum = 0.0;
  for (float logit : logits.values()) {
    if (!std::isfinite(logit)) throw std::runtime_error("model produced non-finite logits");
    exponential_sum += std::exp(static_cast<double>(logit - maximum));
  }
  return std::log(exponential_sum) + maximum - logits[static_cast<size_t>(target)];
}

void usage() {
  std::cerr << "usage: miniinfer-eval model.miniinfer corpus.txt "
               "[--max-tokens N] [--threads N] "
               "[--cpu-kernel auto|scalar|avx2] [--json]\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) { usage(); return 2; }
    CPUOptions cpu_options;
    size_t max_tokens = std::numeric_limits<size_t>::max();
    bool json = false;
    for (int index = 3; index < argc; ++index) {
      const std::string option = argv[index];
      if (option == "--max-tokens" && index + 1 < argc)
        max_tokens = std::stoull(argv[++index]);
      else if (option == "--threads" && index + 1 < argc)
        cpu_options.threads = std::stoull(argv[++index]);
      else if (option == "--cpu-kernel" && index + 1 < argc)
        cpu_options.kernel = parse_kernel(argv[++index]);
      else if (option == "--json") json = true;
      else { usage(); return 2; }
    }
    if (max_tokens < 2) throw std::invalid_argument("evaluation requires at least two tokens");

    const auto load_start = Clock::now();
    Model model;
    std::string error;
    if (!load_model(argv[1], model, &error))
      throw std::runtime_error(error.empty() ? "failed to load model" : error);
    const double load_seconds = seconds_since(load_start);

    std::ifstream corpus_file(argv[2], std::ios::binary);
    if (!corpus_file) throw std::runtime_error("cannot open evaluation corpus");
    const std::string corpus((std::istreambuf_iterator<char>(corpus_file)),
                             std::istreambuf_iterator<char>());
    std::vector<int> tokens = model.tokenizer.encode(corpus);
    if (tokens.size() > max_tokens) tokens.resize(max_tokens);
    if (tokens.size() < 2)
      throw std::invalid_argument("evaluation corpus must encode to at least two tokens");
    if (tokens.size() > model.config.context)
      throw std::invalid_argument(
          "evaluation token count exceeds context; use --max-tokens to select a prefix");

    CPUBackend backend(cpu_options);
    KVCache cache(model.config.layers, tokens.size() - 1, model.config.kv_heads,
                  model.config.hidden / model.config.heads);
    const auto evaluation_start = Clock::now();
    double negative_log_likelihood = 0.0;
    for (size_t position = 0; position + 1 < tokens.size(); ++position) {
      const Tensor logits = forward_token(model, backend, tokens[position], position, cache);
      negative_log_likelihood += token_nll(logits, tokens[position + 1]);
    }
    const double evaluation_seconds = seconds_since(evaluation_start);
    const size_t predictions = tokens.size() - 1;
    const double cross_entropy = negative_log_likelihood / predictions;
    const double perplexity = std::exp(cross_entropy);

    std::cout << std::setprecision(10);
    if (json) {
      std::cout << "{\"tokens\":" << tokens.size()
                << ",\"predictions\":" << predictions
                << ",\"negative_log_likelihood\":" << negative_log_likelihood
                << ",\"cross_entropy\":" << cross_entropy
                << ",\"perplexity\":" << perplexity
                << ",\"load_seconds\":" << load_seconds
                << ",\"evaluation_seconds\":" << evaluation_seconds
                << ",\"tokens_per_second\":"
                << (evaluation_seconds > 0.0 ? predictions / evaluation_seconds : 0.0)
                << ",\"backend\":\"" << backend.name() << "\"}\n";
    } else {
      std::cout << "tokens=" << tokens.size() << " predictions=" << predictions
                << " nll=" << negative_log_likelihood
                << " cross_entropy=" << cross_entropy
                << " perplexity=" << perplexity << '\n'
                << "load_s=" << load_seconds << " evaluation_s=" << evaluation_seconds
                << " tokens_s="
                << (evaluation_seconds > 0.0 ? predictions / evaluation_seconds : 0.0)
                << " backend=" << backend.name()
                << " threads=" << backend.options().threads << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
