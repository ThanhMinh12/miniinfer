# MiniInfer

MiniInfer is a dependency-light C++ inference engine built to answer one
question: **what actually has to happen between a language-model checkpoint and
the next generated token?**

The project runs SmolLM2-135M without PyTorch, Transformers, llama.cpp, or
Python at inference time. It started as a deliberately slow scalar reference,
then added one optimization at a time—KV caching, a real backend boundary,
persistent threading, AVX2, compact weight storage, and int8 quantization—while
keeping each stage comparable with the original implementation.

This is an educational engine and benchmark project, not a production serving
runtime. The CPU path is complete and validated end to end. Native WebGPU and
WASM are optional, earlier-stage backends with their current boundaries
documented below.

## What is implemented

- A contiguous row-major FP32 activation `Tensor`.
- SmolLM2/Llama inference with RMSNorm, split-half RoPE, grouped-query
  attention, SwiGLU, residual connections, final normalization, and tied LM
  head.
- SmolLM2's 9 query heads to 3 KV heads mapping.
- Exact byte-level BPE tokenization from `tokenizer.json`, including added
  tokens, Unicode pre-tokenization, merges, and decoding.
- Greedy and seeded temperature/top-k/top-p autoregressive generation with EOS
  and context-length checks.
- Incremental KV caching and a deliberately slow recompute reference mode.
- Scalar, persistent multithreaded, and runtime-dispatched AVX2 linear kernels.
- F32, BF16, and symmetric Q8-per-output-row model weights with FP32
  accumulation.
- Batched layer-major prefill for exercising matrix-matrix linear operations.
- An offline, streaming Hugging Face safetensors converter.
- Development trace/export tools for comparison with Hugging Face.
- Cross-entropy and perplexity evaluation over a local tokenized corpus.
- An optional Dawn WebGPU F32 linear backend and numerical matmul benchmark.
- A C API and CPU-WASM browser demo with token callbacks.

The primary model has 135M parameters, 30 transformer layers, hidden size 576,
9 query heads, 3 KV heads, and intermediate size 1536. Architecture values are
always read from `config.json`; they are not hardcoded into execution.

## How the engine evolved

### 1. Begin with a slow implementation that can be trusted

The first goal was not speed. It was making every tensor transition visible and
testable:

```text
prompt
  -> ByteLevel BPE token IDs
  -> token embedding
  -> 30 x [RMSNorm -> Q/K/V -> RoPE -> GQA -> residual
           -> RMSNorm -> SwiGLU MLP -> residual]
  -> final RMSNorm
  -> tied embedding / LM head
  -> greedy or sampled next token
```

All activations and arithmetic used scalar FP32 loops. Each projection was a
plain matrix-vector operation. This was intentionally easy to inspect: the CLI
can export embeddings, normalized states, Q/K/V values, rotated vectors,
attention outputs, MLP outputs, and final logits for the last prompt token.

The scalar path was compared layer by layer with Hugging Face. For the prompt
`Hello world`, the final maximum absolute logit difference was
`4.48e-05`, and three generated greedy tokens were identical. That scalar path
remains the correctness reference for every later kernel.

Greedy decoding is the default because it makes token-for-token comparisons
unambiguous. Once that path matched, seeded temperature, top-k, and nucleus
sampling were added as a policy above logits; sampling does not alter model
execution or cache behavior.

### 2. Identify what was bad about the baseline

Correctness exposed several structural problems:

- Transformer projections bypassed the nominal backend and called a free
  scalar `matvec` function directly.
- The `--threads` value existed conceptually but did not affect computation.
- Enabling AVX2 only changed global compiler flags; there was no separately
  dispatched AVX2 kernel or unsupported-CPU fallback.
- BF16 checkpoint tensors were expanded to FP32 during loading, taking roughly
  540 MiB before inference started.
- Prompt execution was token-major, so every projection behaved like
  matrix-vector multiplication even when an entire prompt was available.
- A KV cache existed, but there was no deliberately uncached path or benchmark
  proving how much work it avoided.

For a 135M-parameter decoder, the linear projections dominate the arithmetic:
Q/K/V/O, three MLP matrices, and the vocabulary projection all read large
weight matrices for every token. Optimizing small elementwise operations first
would not address that data movement.

### 3. Make the KV-cache benefit measurable

MiniInfer now exposes two execution modes:

- `kv`: prefill once, append only the new token's K/V values, and attend over
  the bounded cache.
- `recompute`: discard sequence state and replay the complete sequence before
  every generated token.

They share the same transformer logic and must produce bit-identical logits and
greedy tokens. On the development machine, four-thread scalar decoding for the
quick `Hello` benchmark improved from `3.34` tokens/sec in recompute mode to
`5.47` tokens/sec with KV caching—a **64% decode-throughput improvement**. The
gap grows with sequence length because recompute repeatedly evaluates previous
tokens while KV mode evaluates one new token.

The cache is allocated for `prompt length + requested generation`, not the full
8192-token model context, and context overflow is rejected before allocation.

### 4. Replace the decorative backend with a real linear interface

The backend API became:

```cpp
Tensor Backend::linear(const WeightTensor& weight,
                       const Tensor& activations);
```

Weights are `[output, input]`. Activations may be `[input]` for decode or
`[batch, input]` for prefill. Every Q/K/V/O, MLP, and LM-head projection now
passes through this method. Transformer code no longer decides whether a row is
computed by scalar C++, several workers, AVX2, or WebGPU.

A second layer-major prefill path submits full prompt matrices to the backend.
Its final logits are checked against incremental token-major prefill. This
separates matrix-matrix prompt work from decode-time matrix-vector work in the
benchmark without weakening the original reference path.

### 5. Add threading without paying thread-creation cost per projection

Linear output rows are independent, so MiniInfer partitions
`batch * output_rows` across a persistent worker pool. Workers are created once
with the CPU backend and reused by every transformer layer. The calling thread
also consumes rows instead of waiting idle.

This matters because one generated token launches hundreds of projections. A
new set of threads for every matrix would spend too much time in thread startup
and teardown, particularly on small machines.

### 6. Add AVX2 while preserving a portable scalar binary

AVX2 lives in an isolated translation unit. The rest of the library is compiled
without `-mavx2`, and runtime feature detection selects the optimized code only
on supported CPUs. `--cpu-kernel avx2` safely falls back to scalar when the
kernel was not compiled or the CPU does not support it.

The AVX2 implementation handles F32, BF16, and Q8 rows directly, including odd
column counts and scalar tails. BF16 is widened in registers; Q8 values are
converted and multiplied by their per-row scale after accumulation. No complete
weight matrix is dequantized before a projection.

Combining four persistent workers with AVX2 increased quick-benchmark total
throughput from `1.00` token/sec for one-thread scalar BF16 to `7.29`
tokens/sec—about **7.2x end-to-end** on the development machine.

### 7. Stop expanding weights, then quantize them

`WeightTensor` is separate from activation `Tensor`. Activations remain FP32,
while model parameters retain one of three storage types:

| Storage | Representation | Accumulation |
|---|---|---|
| F32 | One FP32 value per weight | FP32 |
| BF16 | Original checkpoint words | Converted per dot-product chunk |
| Q8-per-row | Signed int8 values + one FP32 scale per output row | FP32 |

Merely retaining BF16 during loading reduced peak memory from roughly 540 MiB
to 275 MiB. Offline Q8 conversion reduced the checkpoint from 258.6 MiB to
131.1 MiB and measured peak RSS to about 148 MiB.

Q8 uses symmetric row quantization:

```text
scale[row] = max(abs(weight[row])) / 127
q[row, column] = clamp(round(weight / scale), -127, 127)
weight ~= q * scale
```

This simple scheme prioritizes implementation clarity and size, not perfect
quality. For `Hello`, Q8 preserved the first three greedy tokens, but its full
logits differed from Hugging Face by max/mean absolute values `10.85/1.94`
after 30 quantized layers. Perplexity has not been measured, so the project does
not claim accuracy-neutral quantization.

### 8. Establish, but do not overstate, the GPU/browser path

The first native WebGPU milestone is implemented through Dawn: an F32 WGSL
linear kernel, cached GPU weight buffers, explicit upload/readback, and an
odd-sized CPU numerical comparison. This proves the backend boundary, but it is
not yet a faster end-to-end GPU engine. BF16/Q8 GPU kernels and device-resident
RMSNorm, RoPE, attention, and complete transformer layers remain future work.

The same inference core also has a C API for loading, tokenizing, generation,
cancellation, decoding, and per-token callbacks. Emscripten can compile that
API into a CPU-WASM module used by the minimal page in `web/`. Browser WebGPU is
intentionally deferred until the native GPU path is stable.

## Build

The default build has no Dawn, PyTorch, Transformers, or Python runtime
dependency:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Build the isolated AVX2 kernels with:

```sh
cmake -S . -B build-avx2 -DMINIINFER_ENABLE_AVX2=ON
cmake --build build-avx2
ctest --test-dir build-avx2 --output-on-failure
```

The project uses C++17 for CPU builds and provides scalar fallback on all
platforms supported by CMake and the standard library.

## Convert SmolLM2

Obtain a local `HuggingFaceTB/SmolLM2-135M` directory containing at least
`config.json`, `tokenizer.json`, and `model.safetensors` (or its shard index).
Then run:

```sh
./build/miniinfer-convert /path/to/SmolLM2-135M smollm2-135m.miniinfer
./build/miniinfer-convert /path/to/SmolLM2-135M smollm2-135m-q8.miniinfer \
  --quantize int8
./build/infer --inspect smollm2-135m.miniinfer
```

The converter uses only Python's standard library. It validates architecture
parameters, tokenizer behavior, every expected tensor name, shape, dtype, and
byte count. Safetensors payloads are streamed, and output is atomically renamed
only after successful conversion.

The documented [MiniInfer version 4 format](docs/model-format.md) remains
backward-compatible with version 3 and stores tied embeddings only once.

## Generate text

```sh
./build-avx2/infer smollm2-135m.miniinfer \
  "The meaning of life is" \
  --max-tokens 32 \
  --cache kv \
  --threads 4 \
  --cpu-kernel auto
```

For reproducible stochastic sampling:

```sh
./build-avx2/infer smollm2-135m.miniinfer "Once upon a time" \
  --max-tokens 64 --temperature 0.8 --top-k 40 --top-p 0.95 --seed 42
```

Useful runtime options:

| Option | Values | Purpose |
|---|---|---|
| `--max-tokens` | integer | Maximum new greedy tokens |
| `--cache` | `kv`, `recompute` | Incremental decode or slow reference replay |
| `--threads` | integer | Persistent CPU worker count |
| `--cpu-kernel` | `auto`, `scalar`, `avx2` | Kernel dispatch policy |
| `--backend` | `cpu`, `webgpu` | Execution backend when WebGPU is compiled |
| `--temperature` | float >= 0 | Zero selects greedy; positive values sample |
| `--top-k` | integer | Keep the highest K logits; zero keeps all |
| `--top-p` | float in `(0,1]` | Keep the smallest nucleus reaching P mass |
| `--seed` | integer | Reproducible sampling RNG seed |

Generation stops on EOS and rejects empty prompts, invalid token IDs, and
requests exceeding the configured context.

## Benchmark and measured progression

Use `quick` on low-power machines and `extended` for longer measurements:

```sh
./build/miniinfer-bench smollm2-135m.miniinfer "Hello" \
  --preset quick --cache both --threads 4 --cpu-kernel scalar

./build-avx2/miniinfer-bench smollm2-135m.miniinfer "Hello" \
  --preset extended --cache kv --threads 4 --cpu-kernel avx2
```

`quick` generates two tokens; `extended` generates 16. The executable reports
model load, incremental prefill, decode, total throughput, and layer-major
batched prefill. `--cache both` exits with failure unless cache and recompute
logits and tokens are identical.

Measurements below use the one-token prompt `Hello` and quick preset. They are
development-machine observations, not portable performance promises:

| Weights / kernel | Workers | Load | KV prefill | Decode, 2 tokens | Total tok/s | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| BF16 scalar | 1 | 4.23 s | 1.00 s | 0.99 s | 1.00 | ~275 MiB |
| BF16 scalar | 4 | 1.40 s | 0.32 s | 0.37 s | 2.90 | ~275 MiB |
| BF16 AVX2 | 4 | 0.96 s | 0.14 s | 0.14 s | 7.29 | ~275 MiB |
| Q8 AVX2 | 4 | 0.47 s | 0.13 s | 0.15 s | 7.00 | ~148 MiB |

The useful story is not that every optimization always wins. Q8 substantially
reduced size and loading cost, but this straightforward dequantizing kernel did
not outperform BF16 AVX2 end to end in the short benchmark. The benchmark keeps
those tradeoffs visible instead of assuming a predetermined speedup.

## Measure cross-entropy and perplexity

Use the same text and token prefix to compare floating-point and quantized
models:

```sh
./build-avx2/miniinfer-eval smollm2-135m.miniinfer corpus.txt \
  --max-tokens 128 --threads 4 --cpu-kernel avx2 --json

./build-avx2/miniinfer-eval smollm2-135m-q8.miniinfer corpus.txt \
  --max-tokens 128 --threads 4 --cpu-kernel avx2 --json
```

For tokens `t[0..N-1]`, the evaluator performs `N-1` next-token predictions and
reports total negative log-likelihood, mean cross-entropy, perplexity, load
time, evaluation time, and tokens/sec. It uses the runtime tokenizer and KV
cache rather than a Python reference. Inputs longer than the model context are
rejected unless `--max-tokens` selects a valid prefix, making comparisons
explicit and repeatable.

## Correctness and diagnostics

Deterministic CLI modes expose intermediate state:

```sh
./build/infer --tokenize model.miniinfer "Hello world"
./build/infer --logits model.miniinfer "Hello world"
./build/infer --trace model.miniinfer "Hello"
./build/infer --generate-ids model.miniinfer "Hello" 3
```

With development-only PyTorch and Transformers installed, compare against a
local Hugging Face checkpoint:

```sh
python3 tools/reference_hf.py compare \
  --hf-model /path/to/SmolLM2-135M \
  --miniinfer-model smollm2-135m.miniinfer \
  --infer ./build/infer \
  --prompt "Hello world" \
  --trace \
  --generate-tokens 3
```

PyTorch and Transformers are used only by this offline validation utility.
They are never imported or linked by normal MiniInfer conversion or inference.

Tests cover tensor operations, split-half RoPE, tokenizer behavior, GQA and
multi-token attention, cached/recomputed equivalence, batched/incremental
prefill, odd-sized threaded and AVX2 linears, Q8 reconstruction, malformed
files, conversion, deterministic generation, and the C API.

## Optional native WebGPU

Install Dawn externally and point CMake to its installation:

```sh
cmake -S . -B build-webgpu \
  -DMINIINFER_ENABLE_WEBGPU=ON \
  -DCMAKE_PREFIX_PATH=/path/to/dawn/install/Release
cmake --build build-webgpu
./build-webgpu/miniinfer-webgpu-matmul
```

CPU-only builds never search for or link Dawn. The current WebGPU backend
supports F32 vector and batched linear calls. Unsupported BF16/Q8 GPU weights
fail explicitly instead of silently producing CPU results.

## Optional CPU-WASM demo

```sh
emcmake cmake -S . -B build-wasm \
  -DMINIINFER_BUILD_WASM=ON \
  -DMINIINFER_BUILD_TESTS=OFF
cmake --build build-wasm --target miniinfer-wasm
```

Copy `build-wasm/miniinfer_wasm.{js,wasm}` beside `web/index.html`, serve the
directory over HTTP, and select a local `.miniinfer` model. The page calls the
API in `include/miniinfer/c_api.h` and does not use Transformers.js or
llama.cpp.

## Repository layout

```text
include/miniinfer/   Public C++ and C interfaces
src/                 Tensor, tokenizer, model, transformer, and backend code
tools/               CLI, converter, benchmark, and Hugging Face comparison
tests/               C++ unit tests and end-to-end conversion/evaluation test
docs/                Binary model-format specification
shaders/             WGSL compute kernels
web/                 Minimal browser demo
```

## Current limitations and next work

- Sampling uses a fixed seed for reproducibility but does not yet expose
  repetition, frequency, or presence penalties.
- Q8 is a simple per-row scheme. The evaluator can measure it on local text,
  but no representative benchmark corpus has been published yet.
- AVX2 is the only optimized CPU ISA; ARM NEON and other SIMD paths are absent.
- WebGPU currently accelerates only F32 linear operations and synchronizes
  output to the host after each call.
- RMSNorm, RoPE, attention, KV state, and complete layers still need to remain
  device-resident for meaningful end-to-end GPU speedups.
- WASM currently uses CPU inference; browser WebGPU is a later milestone.

The project intentionally keeps the scalar path permanent. Every future
optimization must first match it numerically, then earn its complexity with a
measured benchmark.
