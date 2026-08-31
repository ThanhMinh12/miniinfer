#!/usr/bin/env python3
"""Convert a local Hugging Face SmolLM2 directory to MiniInfer.

Only Python's standard library is used. Safetensors payloads are streamed into
the output, so conversion does not materialize the model in memory.
"""

from __future__ import annotations

import argparse
from array import array
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import BinaryIO

MAGIC = 0x4D494E49
VERSION = 4
FLAG_TIED_EMBEDDINGS = 1
DTYPE_CODE = {"F32": 1, "F16": 2, "BF16": 3}
DTYPE_WIDTH = {"F32": 4, "F16": 2, "BF16": 2}
Q8_PER_ROW = 4


class ConversionError(RuntimeError):
    pass


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read {path}: {exc}") from exc


def require_int(config: dict, name: str) -> int:
    value = config.get(name)
    if not isinstance(value, int) or value <= 0:
        raise ConversionError(f"config field {name!r} must be a positive integer")
    return value


def validate_config(config: dict) -> dict:
    if config.get("model_type") != "llama":
        raise ConversionError("only LlamaForCausalLM/SmolLM2 models are supported")
    hidden = require_int(config, "hidden_size")
    heads = require_int(config, "num_attention_heads")
    kv_heads = int(config.get("num_key_value_heads", heads))
    if hidden % heads or heads % kv_heads:
        raise ConversionError("hidden_size/attention head configuration is inconsistent")
    if config.get("rope_interleaved", False):
        raise ConversionError("interleaved RoPE is not supported")
    if config.get("rope_scaling") is not None:
        raise ConversionError("RoPE scaling is not supported")
    if config.get("hidden_act", "silu") != "silu":
        raise ConversionError("only the SiLU Llama MLP activation is supported")
    if config.get("attention_bias", False) or config.get("mlp_bias", False):
        raise ConversionError("attention/MLP bias tensors are not supported")
    return {
        "vocab": require_int(config, "vocab_size"),
        "hidden": hidden,
        "layers": require_int(config, "num_hidden_layers"),
        "heads": heads,
        "kv_heads": kv_heads,
        "intermediate": require_int(config, "intermediate_size"),
        "context": require_int(config, "max_position_embeddings"),
        "eps": float(config.get("rms_norm_eps", 1e-5)),
        "rope_theta": float(config.get("rope_theta", 10000.0)),
        "tied": bool(config.get("tie_word_embeddings", False)),
        "bos": int(config.get("bos_token_id", -1) if config.get("bos_token_id") is not None else -1),
        "eos": int(config.get("eos_token_id", -1) if config.get("eos_token_id") is not None else -1),
    }


def load_tokenizer(path: Path) -> tuple[
        list[tuple[int, str]], list[tuple[str, str]], list[tuple[int, str]]]:
    data = load_json(path)
    model = data.get("model", {})
    if model.get("type") != "BPE" or not isinstance(model.get("vocab"), dict):
        raise ConversionError("tokenizer.json must contain a BPE vocabulary")
    pretokenizer = data.get("pre_tokenizer")
    expected_pretokenizers = (pretokenizer or {}).get("pretokenizers", [])
    valid_preprocessing = (
        data.get("normalizer") is None
        and (pretokenizer or {}).get("type") == "Sequence"
        and len(expected_pretokenizers) == 2
        and expected_pretokenizers[0].get("type") == "Digits"
        and expected_pretokenizers[0].get("individual_digits") is True
        and expected_pretokenizers[1].get("type") == "ByteLevel"
        and expected_pretokenizers[1].get("add_prefix_space") is False
        and expected_pretokenizers[1].get("use_regex") is True
        and (data.get("decoder") or {}).get("type") == "ByteLevel"
        and model.get("dropout") is None
        and model.get("unk_token") is None
        and model.get("byte_fallback", False) is False
    )
    if not valid_preprocessing:
        raise ConversionError("tokenizer preprocessing does not match the supported SmolLM2 ByteLevel BPE")
    by_id: dict[int, str] = {int(token_id): token for token, token_id in model["vocab"].items()}
    specials: list[tuple[int, str]] = []
    for added in data.get("added_tokens", []):
        if isinstance(added, dict) and isinstance(added.get("id"), int):
            content = str(added.get("content", ""))
            by_id[added["id"]] = content
            if added.get("special", False):
                if (added.get("lstrip", False) or added.get("rstrip", False)
                        or added.get("single_word", False) or added.get("normalized", False)):
                    raise ConversionError("special tokens with normalization/strip/single-word behavior are unsupported")
                specials.append((added["id"], content))
    vocab = sorted(by_id.items())
    merges: list[tuple[str, str]] = []
    for merge in model.get("merges", []):
        if isinstance(merge, str):
            parts = merge.split(" ")
        elif isinstance(merge, list):
            parts = merge
        else:
            raise ConversionError("unsupported tokenizer merge entry")
        if len(parts) != 2:
            raise ConversionError(f"invalid tokenizer merge: {merge!r}")
        merges.append((str(parts[0]), str(parts[1])))
    return vocab, merges, specials


class SafeTensorFile:
    def __init__(self, path: Path):
        self.path = path
        try:
            with path.open("rb") as handle:
                raw = handle.read(8)
                if len(raw) != 8:
                    raise ConversionError(f"truncated safetensors file: {path}")
                header_size = struct.unpack("<Q", raw)[0]
                if header_size > 100_000_000:
                    raise ConversionError(f"unreasonable safetensors header size: {path}")
                header = json.loads(handle.read(header_size))
                self.data_start = 8 + header_size
        except (OSError, json.JSONDecodeError) as exc:
            raise ConversionError(f"cannot read safetensors header {path}: {exc}") from exc
        self.tensors = {name: meta for name, meta in header.items() if name != "__metadata__"}
        self.file_size = path.stat().st_size

    def metadata(self, name: str) -> dict:
        if name not in self.tensors:
            raise ConversionError(f"missing tensor {name!r} in {self.path.name}")
        meta = self.tensors[name]
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        offsets = meta.get("data_offsets")
        if dtype not in DTYPE_CODE or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ConversionError(f"unsupported metadata for tensor {name!r}")
        elements = 1
        for dim in shape:
            if not isinstance(dim, int) or dim < 0:
                raise ConversionError(f"invalid shape for tensor {name!r}")
            elements *= dim
        byte_count = offsets[1] - offsets[0]
        if byte_count != elements * DTYPE_WIDTH[dtype]:
            raise ConversionError(f"byte count does not match shape for tensor {name!r}")
        if offsets[0] < 0 or self.data_start + offsets[1] > self.file_size:
            raise ConversionError(f"out-of-range data offset for tensor {name!r}")
        return {"dtype": dtype, "shape": shape, "offset": offsets[0],
                "bytes": byte_count, "elements": elements}

    def copy_payload(self, name: str, destination: BinaryIO, meta: dict) -> None:
        with self.path.open("rb") as source:
            source.seek(self.data_start + meta["offset"])
            remaining = meta["bytes"]
            while remaining:
                chunk = source.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise ConversionError(f"truncated tensor payload for {name!r}")
                destination.write(chunk)
                remaining -= len(chunk)

    def quantize_payload(self, name: str, destination: BinaryIO, meta: dict) -> None:
        if len(meta["shape"]) != 2:
            raise ConversionError(f"only rank-2 tensor {name!r} can use Q8-per-row")
        rows, columns = meta["shape"]
        width = DTYPE_WIDTH[meta["dtype"]]
        with self.path.open("rb") as source:
            source.seek(self.data_start + meta["offset"])
            for _ in range(rows):
                raw = source.read(columns * width)
                if len(raw) != columns * width:
                    raise ConversionError(f"truncated tensor payload for {name!r}")
                values = decode_values(raw, meta["dtype"])
                maximum = max((abs(value) for value in values), default=0.0)
                scale = maximum / 127.0 if maximum else 0.0
                if scale:
                    quantized = array("b", (max(-127, min(127, round(value / scale)))
                                             for value in values))
                else:
                    quantized = array("b", [0]) * columns
                destination.write(struct.pack("<f", scale))
                destination.write(quantized.tobytes())


class SafeTensorSet:
    def __init__(self, directory: Path):
        index_path = directory / "model.safetensors.index.json"
        single_path = directory / "model.safetensors"
        self.files: dict[Path, SafeTensorFile] = {}
        if index_path.exists():
            index = load_json(index_path)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ConversionError("invalid model.safetensors.index.json")
            self.weight_map = {name: directory / filename for name, filename in weight_map.items()}
        elif single_path.exists():
            file = SafeTensorFile(single_path)
            self.files[single_path] = file
            self.weight_map = {name: single_path for name in file.tensors}
        else:
            raise ConversionError("model.safetensors or model.safetensors.index.json was not found")

    def get(self, name: str) -> tuple[SafeTensorFile, dict]:
        path = self.weight_map.get(name)
        if path is None:
            raise ConversionError(f"required tensor {name!r} is missing")
        if path not in self.files:
            self.files[path] = SafeTensorFile(path)
        file = self.files[path]
        return file, file.metadata(name)


def write_u32(handle: BinaryIO, value: int) -> None:
    handle.write(struct.pack("<I", value))


def write_u64(handle: BinaryIO, value: int) -> None:
    handle.write(struct.pack("<Q", value))


def write_string(handle: BinaryIO, value: str) -> None:
    encoded = value.encode("utf-8")
    write_u64(handle, len(encoded))
    handle.write(encoded)


def decode_values(raw: bytes, dtype: str) -> list[float] | array:
    if dtype == "F32":
        values = array("f")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        return values
    words = array("H")
    words.frombytes(raw)
    if sys.byteorder != "little":
        words.byteswap()
    if dtype == "BF16":
        result = []
        for word in words:
            sign = -1.0 if word & 0x8000 else 1.0
            exponent = (word >> 7) & 0xFF
            mantissa = word & 0x7F
            if exponent == 0:
                result.append(sign * math.ldexp(mantissa / 128.0, -126))
            elif exponent == 0xFF:
                result.append(sign * (math.inf if mantissa == 0 else math.nan))
            else:
                result.append(sign * math.ldexp(1.0 + mantissa / 128.0, exponent - 127))
        return result
    # Python's struct module provides a correctly rounded IEEE binary16 decoder.
    return [struct.unpack("<e", raw[index:index + 2])[0]
            for index in range(0, len(raw), 2)]


def expected_tensors(config: dict) -> list[tuple[str, tuple[int, ...]]]:
    h, qh, kvh = config["hidden"], config["heads"], config["kv_heads"]
    kv_dim = (h // qh) * kvh
    result: list[tuple[str, tuple[int, ...]]] = [
        ("model.embed_tokens.weight", (config["vocab"], h)),
        ("model.norm.weight", (h,)),
    ]
    if not config["tied"]:
        result.append(("lm_head.weight", (config["vocab"], h)))
    for layer in range(config["layers"]):
        prefix = f"model.layers.{layer}."
        result.extend([
            (prefix + "input_layernorm.weight", (h,)),
            (prefix + "post_attention_layernorm.weight", (h,)),
            (prefix + "self_attn.q_proj.weight", (h, h)),
            (prefix + "self_attn.k_proj.weight", (kv_dim, h)),
            (prefix + "self_attn.v_proj.weight", (kv_dim, h)),
            (prefix + "self_attn.o_proj.weight", (h, h)),
            (prefix + "mlp.gate_proj.weight", (config["intermediate"], h)),
            (prefix + "mlp.up_proj.weight", (config["intermediate"], h)),
            (prefix + "mlp.down_proj.weight", (h, config["intermediate"])),
        ])
    return result


def write_tensor(handle: BinaryIO, tensors: SafeTensorSet, name: str,
                 expected_shape: tuple[int, ...], quantize: str) -> None:
    source, meta = tensors.get(name)
    if tuple(meta["shape"]) != expected_shape:
        raise ConversionError(
            f"tensor {name!r} has shape {meta['shape']}, expected {list(expected_shape)}")
    use_q8 = quantize == "int8" and len(meta["shape"]) == 2
    write_u32(handle, Q8_PER_ROW if use_q8 else DTYPE_CODE[meta["dtype"]])
    write_u64(handle, len(meta["shape"]))
    for dim in meta["shape"]:
        write_u64(handle, dim)
    write_u64(handle, meta["elements"])
    if use_q8:
        rows, columns = meta["shape"]
        write_u64(handle, rows * (4 + columns))
        source.quantize_payload(name, handle, meta)
    else:
        write_u64(handle, meta["bytes"])
        source.copy_payload(name, handle, meta)


def convert(source_dir: Path, output: Path, quantize: str = "none") -> None:
    config = validate_config(load_json(source_dir / "config.json"))
    vocab, merges, specials = load_tokenizer(source_dir / "tokenizer.json")
    if len(vocab) != config["vocab"]:
        raise ConversionError(
            f"tokenizer has {len(vocab)} IDs but config vocab_size is {config['vocab']}")
    if [token_id for token_id, _ in vocab] != list(range(config["vocab"])):
        raise ConversionError("tokenizer IDs must be contiguous from zero to vocab_size - 1")
    if config["bos"] not in range(-1, config["vocab"]) or config["eos"] not in range(-1, config["vocab"]):
        raise ConversionError("BOS/EOS token IDs are outside the vocabulary")
    tensors = SafeTensorSet(source_dir)
    tensor_list = expected_tensors(config)
    for name, shape in tensor_list:
        _, meta = tensors.get(name)
        if tuple(meta["shape"]) != shape:
            raise ConversionError(f"tensor {name!r} has shape {meta['shape']}, expected {list(shape)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(struct.pack("<II", MAGIC, VERSION))
            for key in ("vocab", "hidden", "layers", "heads", "kv_heads", "intermediate", "context"):
                write_u64(handle, config[key])
            handle.write(struct.pack("<ffIii", config["eps"], config["rope_theta"],
                                     FLAG_TIED_EMBEDDINGS if config["tied"] else 0,
                                     config["bos"], config["eos"]))
            write_u64(handle, len(vocab))
            for token_id, token in vocab:
                handle.write(struct.pack("<i", token_id))
                write_string(handle, token)
            write_u64(handle, len(merges))
            for left, right in merges:
                write_string(handle, left)
                write_string(handle, right)
            write_u64(handle, len(specials))
            for token_id, token in specials:
                handle.write(struct.pack("<i", token_id))
                write_string(handle, token)
            for index, (name, shape) in enumerate(tensor_list, 1):
                print(f"[{index}/{len(tensor_list)}] {name}", file=sys.stderr)
                write_tensor(handle, tensors, name, shape, quantize)
        os.replace(temporary, output)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    print(f"wrote {output} ({output.stat().st_size / (1024 * 1024):.1f} MiB)", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_directory", type=Path, help="local SmolLM2 Hugging Face directory")
    parser.add_argument("output", type=Path, help="output .miniinfer file")
    parser.add_argument("--quantize", choices=("none", "int8"), default="none",
                        help="weight-only matrix quantization (default: none)")
    args = parser.parse_args()
    try:
        convert(args.hf_directory.resolve(), args.output.resolve(), args.quantize)
    except ConversionError as exc:
        print(f"conversion failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
