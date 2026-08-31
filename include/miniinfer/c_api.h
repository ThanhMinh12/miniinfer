#pragma once

#include <stddef.h>

#if defined(_WIN32) && defined(MINIINFER_SHARED)
#define MINIINFER_API __declspec(dllexport)
#else
#define MINIINFER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MiniInferHandle MiniInferHandle;
typedef int (*MiniInferTokenCallback)(int token_id, const char* token_piece,
                                      void* user_data);

enum MiniInferCPUKernel {
  MINIINFER_CPU_AUTO = 0,
  MINIINFER_CPU_SCALAR = 1,
  MINIINFER_CPU_AVX2 = 2
};

enum MiniInferCacheMode {
  MINIINFER_CACHE_RECOMPUTE = 0,
  MINIINFER_CACHE_KV = 1
};

MINIINFER_API MiniInferHandle* miniinfer_create(const char* model_path,
                                                size_t threads,
                                                int cpu_kernel);
MINIINFER_API void miniinfer_destroy(MiniInferHandle* handle);

// Returns the number of generated tokens, or -1 on error. The callback runs
// synchronously after each token and may return zero to stop generation.
MINIINFER_API int miniinfer_generate(MiniInferHandle* handle, const char* prompt,
                                    size_t max_tokens, int cache_mode,
                                    MiniInferTokenCallback callback,
                                    void* user_data);
MINIINFER_API void miniinfer_cancel(MiniInferHandle* handle);

// If output is null or too small, tokenize/decode return the required size.
MINIINFER_API int miniinfer_tokenize(MiniInferHandle* handle, const char* text,
                                    int* output, size_t capacity);
MINIINFER_API int miniinfer_decode(MiniInferHandle* handle, const int* token_ids,
                                  size_t count, char* output, size_t capacity);
MINIINFER_API const char* miniinfer_last_error(const MiniInferHandle* handle);

#ifdef __cplusplus
}
#endif
