# MiniInfer model format

All integers and floating-point values are little-endian. Version 4 readers
must also accept version 3 files.

## Header and tokenizer

The file starts with two `u32` values: magic `0x4d494e49` (`MINI`) and version
`4`. They are followed by seven `u64` configuration values in this order:
vocabulary size, hidden size, layer count, query-head count, KV-head count,
intermediate size, and context length. Next are FP32 RMSNorm epsilon, FP32 RoPE
theta, `u32` flags, and signed `i32` BOS and EOS IDs. Flag bit zero means token
embeddings and the LM head are tied.

The vocabulary is a `u64` count followed by `(i32 id, string token)` entries.
A string is a `u64` UTF-8 byte count followed by those bytes. BPE merges are a
`u64` count followed by two strings per merge. Added special tokens are a
`u64` count followed by `(i32 id, string token)` entries.

## Tensor record

Each tensor has this header:

| Field | Type | Meaning |
|---|---:|---|
| dtype | `u32` | 1=F32, 2=F16, 3=BF16, 4=Q8-per-row |
| rank | `u64` | Number of dimensions |
| shape | `rank * u64` | Row-major dimensions |
| elements | `u64` | Product of dimensions |
| bytes | `u64` | Payload byte count |

F32, F16, and BF16 payloads are contiguous row-major values. Q8-per-row is
valid only for rank-two matrices. Every output row stores one FP32 scale and
then `columns` signed int8 values. Reconstruction is `float(q) * scale`.
Quantization is symmetric, uses `max(abs(row)) / 127`, and never emits -128.

## Tensor order

Records appear as token embeddings, final RMSNorm, and (only when untied) the
LM head. Each transformer layer then stores input RMSNorm, post-attention
RMSNorm, Q/K/V/O projections, gate/up projections, and down projection.

Version 3 has the same tokenizer and tensor order but does not permit dtype 4.
