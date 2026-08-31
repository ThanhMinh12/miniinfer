#include "miniinfer/c_api.h"

#include "miniinfer/backend.h"
#include "miniinfer/model.h"
#include "miniinfer/transformer.h"

#include <atomic>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using miniinfer::CPUBackend;
using miniinfer::CPUKernel;
using miniinfer::CPUOptions;
using miniinfer::CacheMode;
using miniinfer::Model;

struct MiniInferHandle {
  explicit MiniInferHandle(CPUOptions options) : backend(options) {}
  Model model;
  CPUBackend backend;
  std::atomic<bool> canceled{false};
  std::string last_error;
  std::string callback_piece;
};

namespace {
thread_local std::string global_error;

CPUKernel kernel_from_c(int kernel) {
  if (kernel == MINIINFER_CPU_AUTO) return CPUKernel::Auto;
  if (kernel == MINIINFER_CPU_SCALAR) return CPUKernel::Scalar;
  if (kernel == MINIINFER_CPU_AVX2) return CPUKernel::AVX2;
  throw std::invalid_argument("invalid CPU kernel");
}

CacheMode cache_from_c(int mode) {
  if (mode == MINIINFER_CACHE_RECOMPUTE) return CacheMode::Recompute;
  if (mode == MINIINFER_CACHE_KV) return CacheMode::KV;
  throw std::invalid_argument("invalid cache mode");
}

template<class Function>
int protect(MiniInferHandle* handle, Function function) {
  try {
    if (!handle) throw std::invalid_argument("MiniInfer handle is null");
    handle->last_error.clear();
    return function();
  } catch (const std::exception& error) {
    if (handle) handle->last_error = error.what();
    else global_error = error.what();
    return -1;
  }
}
}  // namespace

extern "C" {

MiniInferHandle* miniinfer_create(const char* model_path, size_t threads,
                                  int cpu_kernel) {
  try {
    if (!model_path) throw std::invalid_argument("model path is null");
    auto handle = std::make_unique<MiniInferHandle>(
        CPUOptions{threads ? threads : 1, kernel_from_c(cpu_kernel)});
    if (!miniinfer::load_model(model_path, handle->model, &handle->last_error)) {
      global_error = handle->last_error.empty() ? "failed to load model" : handle->last_error;
      return nullptr;
    }
    global_error.clear();
    return handle.release();
  } catch (const std::exception& error) {
    global_error = error.what();
    return nullptr;
  }
}

void miniinfer_destroy(MiniInferHandle* handle) { delete handle; }

int miniinfer_generate(MiniInferHandle* handle, const char* prompt,
                       size_t max_tokens, int cache_mode,
                       MiniInferTokenCallback callback, void* user_data) {
  return protect(handle, [&] {
    if (!prompt) throw std::invalid_argument("prompt is null");
    handle->canceled.store(false, std::memory_order_relaxed);
    const std::vector<int> prompt_tokens = handle->model.tokenizer.encode(prompt);
    const std::vector<int> result = miniinfer::generate_stream(
        handle->model, handle->backend, prompt_tokens, max_tokens,
        cache_from_c(cache_mode), [&](int token) {
          if (handle->canceled.load(std::memory_order_relaxed)) return false;
          if (!callback) return true;
          handle->callback_piece = handle->model.tokenizer.decode({token});
          return callback(token, handle->callback_piece.c_str(), user_data) != 0;
        });
    return static_cast<int>(result.size() - prompt_tokens.size());
  });
}

void miniinfer_cancel(MiniInferHandle* handle) {
  if (handle) handle->canceled.store(true, std::memory_order_relaxed);
}

int miniinfer_tokenize(MiniInferHandle* handle, const char* text,
                       int* output, size_t capacity) {
  return protect(handle, [&] {
    if (!text) throw std::invalid_argument("text is null");
    const std::vector<int> tokens = handle->model.tokenizer.encode(text);
    if (output && capacity >= tokens.size())
      std::memcpy(output, tokens.data(), tokens.size() * sizeof(int));
    return static_cast<int>(tokens.size());
  });
}

int miniinfer_decode(MiniInferHandle* handle, const int* token_ids, size_t count,
                     char* output, size_t capacity) {
  return protect(handle, [&] {
    if (count && !token_ids) throw std::invalid_argument("token IDs are null");
    std::vector<int> tokens;
    if (count) tokens.assign(token_ids, token_ids + count);
    const std::string decoded = handle->model.tokenizer.decode(tokens);
    const size_t required = decoded.size() + 1;
    if (output && capacity >= required)
      std::memcpy(output, decoded.c_str(), required);
    return static_cast<int>(required);
  });
}

const char* miniinfer_last_error(const MiniInferHandle* handle) {
  return handle ? handle->last_error.c_str() : global_error.c_str();
}

}  // extern "C"
