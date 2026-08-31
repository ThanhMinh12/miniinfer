#include "miniinfer/backend.h"
#include "miniinfer/model.h"
#include "miniinfer/transformer.h"
#ifdef MINIINFER_ENABLE_WEBGPU
#include "miniinfer/webgpu_backend.h"
#endif

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace miniinfer;

namespace {
void usage() {
  std::cerr << "usage: infer model.miniinfer \"prompt\" [--max-tokens N] "
               "[--cache kv|recompute] [--threads N] "
               "[--cpu-kernel auto|scalar|avx2] [--backend cpu|webgpu]\n"
            << "       infer --inspect model.miniinfer\n"
            << "       infer --tokenize model.miniinfer \"text\"\n"
            << "       infer --logits model.miniinfer \"prompt\"\n"
            << "       infer --trace model.miniinfer \"prompt\"\n"
            << "       infer --generate-ids model.miniinfer \"prompt\" N "
               "[--cache kv|recompute] [--threads N] "
               "[--cpu-kernel auto|scalar|avx2] [--backend cpu|webgpu]\n";
}

CPUKernel parse_kernel(const std::string& value) {
  if (value == "auto") return CPUKernel::Auto;
  if (value == "scalar") return CPUKernel::Scalar;
  if (value == "avx2") return CPUKernel::AVX2;
  throw std::invalid_argument("CPU kernel must be auto, scalar, or avx2");
}

CacheMode parse_cache(const std::string& value) {
  if (value == "kv") return CacheMode::KV;
  if (value == "recompute") return CacheMode::Recompute;
  throw std::invalid_argument("cache mode must be kv or recompute");
}

const char* weight_type_name(WeightType type) {
  if (type == WeightType::BF16) return "bf16";
  if (type == WeightType::Q8PerRow) return "q8-per-row";
  return "f32";
}

struct RuntimeOptions {
  CPUOptions cpu;
  CacheMode cache = CacheMode::KV;
  std::string backend = "cpu";
};

void parse_runtime_option(int& index, int argc, char** argv, RuntimeOptions& options) {
  const std::string option = argv[index];
  if (option == "--cache" && index + 1 < argc)
    options.cache = parse_cache(argv[++index]);
  else if (option == "--threads" && index + 1 < argc)
    options.cpu.threads = std::stoull(argv[++index]);
  else if (option == "--cpu-kernel" && index + 1 < argc)
    options.cpu.kernel = parse_kernel(argv[++index]);
  else if (option == "--backend" && index + 1 < argc) {
    options.backend = argv[++index];
    if (options.backend != "cpu" && options.backend != "webgpu")
      throw std::invalid_argument("backend must be cpu or webgpu");
  }
  else
    throw std::invalid_argument("unknown or incomplete runtime option: " + option);
}

std::unique_ptr<Backend> make_backend(const RuntimeOptions& options) {
  if (options.backend == "cpu") return std::make_unique<CPUBackend>(options.cpu);
#ifdef MINIINFER_ENABLE_WEBGPU
  return std::make_unique<WebGPUBackend>();
#else
  throw std::runtime_error("WebGPU support was not enabled in this build");
#endif
}

Model load(const std::string& path) {
  Model model;
  std::string error;
  if (!load_model(path, model, &error))
    throw std::runtime_error(error.empty() ? "failed to load model" : error);
  return model;
}

template<class T>
void print_array(const std::vector<T>& values) {
  std::cout << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << values[i];
  }
  std::cout << ']';
}

void print_tensor(const Tensor& tensor) { print_array(tensor.values()); }

void print_named_tensor(const char* name, const Tensor& tensor, bool& first) {
  if (!first) std::cout << ',';
  first = false;
  std::cout << '\"' << name << "\":";
  print_tensor(tensor);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--inspect") {
      Model model = load(argv[2]);
      const Config& c = model.config;
      std::cout << "vocab=" << c.vocab << " hidden=" << c.hidden
                << " layers=" << c.layers << " heads=" << c.heads
                << " kv_heads=" << c.kv_heads << " intermediate=" << c.intermediate
                << " context=" << c.context
                << " tied_embeddings=" << (c.tie_word_embeddings ? "yes" : "no")
                << " matrix_storage=" << weight_type_name(model.embedding.type()) << '\n';
      return 0;
    }

    if (argc == 4 && std::string(argv[1]) == "--tokenize") {
      Model model = load(argv[2]);
      print_array(model.tokenizer.encode(argv[3]));
      std::cout << '\n';
      return 0;
    }

    if (argc == 4 && std::string(argv[1]) == "--logits") {
      Model model = load(argv[2]);
      std::vector<int> tokens = model.tokenizer.encode(argv[3]);
      CPUBackend cpu;
      Tensor logits = prompt_logits(model, cpu, tokens);
      std::cout << "{\"tokens\":";
      print_array(tokens);
      std::cout << ",\"logits\":[" << std::setprecision(9);
      for (size_t i = 0; i < logits.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << logits[i];
      }
      std::cout << "]}\n";
      return 0;
    }

    if (argc == 4 && std::string(argv[1]) == "--trace") {
      Model model = load(argv[2]);
      std::vector<int> tokens = model.tokenizer.encode(argv[3]);
      CPUBackend cpu;
      ForwardTrace trace;
      prompt_logits(model, cpu, tokens, &trace);
      std::cout << std::setprecision(9) << "{\"tokens\":";
      print_array(tokens);
      std::cout << ",\"embedding\":"; print_tensor(trace.embedding);
      std::cout << ",\"layers\":[";
      for (size_t i = 0; i < trace.layers.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << '{'; bool first = true;
        const LayerTrace& layer = trace.layers[i];
        print_named_tensor("attn_norm", layer.attn_norm, first);
        print_named_tensor("q", layer.q, first); print_named_tensor("k", layer.k, first);
        print_named_tensor("v", layer.v, first); print_named_tensor("q_rope", layer.q_rope, first);
        print_named_tensor("k_rope", layer.k_rope, first);
        print_named_tensor("attention", layer.attention, first);
        print_named_tensor("projected_attention", layer.projected_attention, first);
        print_named_tensor("after_attention", layer.after_attention, first);
        print_named_tensor("ffn_norm", layer.ffn_norm, first);
        print_named_tensor("gate", layer.gate, first); print_named_tensor("up", layer.up, first);
        print_named_tensor("down", layer.down, first); print_named_tensor("output", layer.output, first);
        std::cout << '}';
      }
      std::cout << "],\"final_norm\":"; print_tensor(trace.final_norm);
      std::cout << ",\"logits\":"; print_tensor(trace.logits);
      std::cout << "}\n";
      return 0;
    }

    if (argc >= 5 && std::string(argv[1]) == "--generate-ids") {
      Model model = load(argv[2]);
      std::vector<int> prompt = model.tokenizer.encode(argv[3]);
      RuntimeOptions options;
      for (int index = 5; index < argc; ++index)
        parse_runtime_option(index, argc, argv, options);
      std::unique_ptr<Backend> backend = make_backend(options);
      std::vector<int> tokens = generate(model, *backend, prompt, std::stoull(argv[4]),
                                         options.cache);
      std::cout << "{\"prompt_tokens\":"; print_array(prompt);
      std::cout << ",\"tokens\":"; print_array(tokens);
      std::cout << ",\"generated\":";
      print_array(std::vector<int>(tokens.begin() + prompt.size(), tokens.end()));
      std::cout << "}\n";
      return 0;
    }

    if (argc < 3) { usage(); return 2; }
    size_t max_tokens = 32;
    RuntimeOptions options;
    for (int i = 3; i < argc; ++i) {
      if (std::string(argv[i]) == "--max-tokens" && i + 1 < argc)
        max_tokens = std::stoull(argv[++i]);
      else
        parse_runtime_option(i, argc, argv, options);
    }
    Model model = load(argv[1]);
    std::vector<int> prompt = model.tokenizer.encode(argv[2]);
    std::unique_ptr<Backend> backend = make_backend(options);
    const auto start = std::chrono::steady_clock::now();
    std::vector<int> tokens = generate(model, *backend, prompt, max_tokens, options.cache);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    const size_t generated = tokens.size() - prompt.size();
    std::cout << model.tokenizer.decode(tokens) << '\n'
              << "generated " << generated << " tokens in " << seconds << " s ("
              << (seconds > 0.0 ? generated / seconds : 0.0) << " tok/s), backend="
              << backend->name();
    if (options.backend == "cpu")
      std::cout << ", threads=" << options.cpu.threads;
    std::cout
              << ", cache=" << (options.cache == CacheMode::KV ? "kv" : "recompute")
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
