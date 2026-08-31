#!/usr/bin/env python3
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def tensor_shapes(config):
    hidden = config["hidden_size"]
    head_dim = hidden // config["num_attention_heads"]
    kv_dim = head_dim * config["num_key_value_heads"]
    tensors = {
        "model.embed_tokens.weight": [config["vocab_size"], hidden],
        "model.norm.weight": [hidden],
    }
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        tensors.update({
            prefix + "input_layernorm.weight": [hidden],
            prefix + "post_attention_layernorm.weight": [hidden],
            prefix + "self_attn.q_proj.weight": [hidden, hidden],
            prefix + "self_attn.k_proj.weight": [kv_dim, hidden],
            prefix + "self_attn.v_proj.weight": [kv_dim, hidden],
            prefix + "self_attn.o_proj.weight": [hidden, hidden],
            prefix + "mlp.gate_proj.weight": [config["intermediate_size"], hidden],
            prefix + "mlp.up_proj.weight": [config["intermediate_size"], hidden],
            prefix + "mlp.down_proj.weight": [hidden, config["intermediate_size"]],
        })
    return tensors


def element_count(shape):
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def deterministic_weights(shapes):
    result = {}
    for tensor_number, (name, shape) in enumerate(shapes.items()):
        count = element_count(shape)
        if "layernorm" in name or name == "model.norm.weight":
            values = [1.0 + 0.01 * ((i + tensor_number) % 3 - 1) for i in range(count)]
        else:
            values = [0.025 * (((i * 3 + tensor_number) % 11) - 5) for i in range(count)]
        result[name] = values
    return result


def write_safetensors(path, shapes, dtype="BF16", weights=None):
    width = 4 if dtype == "F32" else 2
    header, payload, offset = {}, bytearray(), 0
    for name, shape in shapes.items():
        count = element_count(shape)
        size = count * width
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        if dtype == "F32":
            payload.extend(struct.pack(f"<{count}f", *(weights[name] if weights else [0.0] * count)))
        else:
            payload.extend(b"\0" * size)
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(payload)


def matvec(matrix, rows, columns, vector):
    return [sum(matrix[row * columns + col] * vector[col] for col in range(columns))
            for row in range(rows)]


def rmsnorm(vector, weight, epsilon):
    scale = 1.0 / math.sqrt(sum(value * value for value in vector) / len(vector) + epsilon)
    return [value * scale * weight[i] for i, value in enumerate(vector)]


def rope(vector, position, theta):
    half = len(vector) // 2
    result = vector[:]
    for i in range(half):
        angle = position * theta ** (-i / half)
        cosine, sine = math.cos(angle), math.sin(angle)
        result[i] = vector[i] * cosine - vector[i + half] * sine
        result[i + half] = vector[i + half] * cosine + vector[i] * sine
    return result


def reference_logits(config, weights, tokens):
    hidden = config["hidden_size"]
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    head_dim, groups = hidden // heads, heads // kv_heads
    intermediate = config["intermediate_size"]
    caches = [[] for _ in range(config["num_hidden_layers"])]
    embedding = weights["model.embed_tokens.weight"]
    logits = None
    for position, token in enumerate(tokens):
        x = embedding[token * hidden:(token + 1) * hidden]
        for layer in range(config["num_hidden_layers"]):
            prefix = f"model.layers.{layer}."
            normalized = rmsnorm(x, weights[prefix + "input_layernorm.weight"], config["rms_norm_eps"])
            q = matvec(weights[prefix + "self_attn.q_proj.weight"], hidden, hidden, normalized)
            kv_dim = head_dim * kv_heads
            k = matvec(weights[prefix + "self_attn.k_proj.weight"], kv_dim, hidden, normalized)
            v = matvec(weights[prefix + "self_attn.v_proj.weight"], kv_dim, hidden, normalized)
            q_heads = [rope(q[h * head_dim:(h + 1) * head_dim], position,
                            config["rope_theta"]) for h in range(heads)]
            k_heads = [rope(k[h * head_dim:(h + 1) * head_dim], position,
                            config["rope_theta"]) for h in range(kv_heads)]
            v_heads = [v[h * head_dim:(h + 1) * head_dim] for h in range(kv_heads)]
            caches[layer].append((k_heads, v_heads))
            attention = []
            for query_head in range(heads):
                kv_head = query_head // groups
                scores = [sum(q_heads[query_head][d] * cached_k[kv_head][d]
                              for d in range(head_dim)) / math.sqrt(head_dim)
                          for cached_k, _ in caches[layer]]
                maximum = max(scores)
                exponentials = [math.exp(score - maximum) for score in scores]
                total = sum(exponentials)
                probabilities = [value / total for value in exponentials]
                attention.extend([
                    sum(probabilities[t] * caches[layer][t][1][kv_head][d]
                        for t in range(len(caches[layer])))
                    for d in range(head_dim)
                ])
            projected = matvec(weights[prefix + "self_attn.o_proj.weight"], hidden, hidden, attention)
            x = [x[i] + projected[i] for i in range(hidden)]
            normalized = rmsnorm(x, weights[prefix + "post_attention_layernorm.weight"],
                                 config["rms_norm_eps"])
            gate = matvec(weights[prefix + "mlp.gate_proj.weight"], intermediate, hidden, normalized)
            up = matvec(weights[prefix + "mlp.up_proj.weight"], intermediate, hidden, normalized)
            gated = [(gate[i] / (1.0 + math.exp(-gate[i]))) * up[i]
                     for i in range(intermediate)]
            down = matvec(weights[prefix + "mlp.down_proj.weight"], hidden, intermediate, gated)
            x = [x[i] + down[i] for i in range(hidden)]
        normalized = rmsnorm(x, weights["model.norm.weight"], config["rms_norm_eps"])
        logits = matvec(embedding, config["vocab_size"], hidden, normalized)
    return logits


def main():
    converter, infer = sys.argv[1:]
    config = {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "vocab_size": 8, "hidden_size": 4, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "intermediate_size": 8, "max_position_embeddings": 16,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
        "rope_interleaved": False, "tie_word_embeddings": True,
        "bos_token_id": 0, "eos_token_id": 1,
    }
    tokenizer = {
        "model": {"type": "BPE", "vocab": {
            "<bos>": 0, "<eos>": 1, "a": 2, "b": 3,
            "c": 4, "Ġ": 5, "ab": 6, "Ġa": 7,
        }, "dropout": None, "unk_token": None, "byte_fallback": False,
           "merges": ["a b", "Ġ a"]},
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Digits", "individual_digits": True},
            {"type": "ByteLevel", "add_prefix_space": False,
             "trim_offsets": True, "use_regex": True},
        ]},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True,
                    "trim_offsets": True, "use_regex": True},
        "added_tokens": [
            {"id": 0, "content": "<bos>", "single_word": False,
             "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 1, "content": "<eos>", "single_word": False,
             "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        ],
    }
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
        shapes = tensor_shapes(config)

        # BF16 zero tensors exercise compact checkpoint loading and inspection.
        write_safetensors(root / "model.safetensors", shapes)
        output = root / "tiny-bf16.miniinfer"
        subprocess.run([converter, str(root), str(output)], check=True)
        inspected = subprocess.run([infer, "--inspect", str(output)], check=True,
                                   text=True, capture_output=True).stdout
        assert "vocab=8 hidden=4 layers=1 heads=2 kv_heads=1" in inspected
        assert "tied_embeddings=yes" in inspected

        # Nonzero F32 weights compare the complete two-token GQA/RoPE/KV path.
        weights = deterministic_weights(shapes)
        write_safetensors(root / "model.safetensors", shapes, "F32", weights)
        output = root / "tiny-f32.miniinfer"
        subprocess.run([converter, str(root), str(output)], check=True)
        actual = json.loads(subprocess.run(
            [infer, "--logits", str(output), "ab a"], check=True,
            text=True, capture_output=True).stdout)
        assert actual["tokens"] == [6, 7]
        expected = reference_logits(config, weights, actual["tokens"])
        assert max(abs(a - b) for a, b in zip(actual["logits"], expected)) < 2e-5
        trace = json.loads(subprocess.run(
            [infer, "--trace", str(output), "ab a"], check=True,
            text=True, capture_output=True).stdout)
        assert len(trace["layers"]) == 1
        assert max(abs(a - b) for a, b in zip(trace["logits"], expected)) < 2e-5

        # Version 4 Q8-per-row weights stay compressed and produce finite,
        # deterministic logits without expanding the model at load time.
        quantized_output = root / "tiny-q8.miniinfer"
        subprocess.run([converter, str(root), str(quantized_output),
                        "--quantize", "int8"], check=True)
        assert quantized_output.stat().st_size < output.stat().st_size
        quantized_inspect = subprocess.run(
            [infer, "--inspect", str(quantized_output)], check=True,
            text=True, capture_output=True).stdout
        assert "matrix_storage=q8-per-row" in quantized_inspect
        quantized = json.loads(subprocess.run(
            [infer, "--logits", str(quantized_output), "ab a"], check=True,
            text=True, capture_output=True).stdout)
        assert all(math.isfinite(value) for value in quantized["logits"])
        assert max(abs(a - b) for a, b in zip(quantized["logits"], expected)) < 0.05
        first_generation = subprocess.run(
            [infer, "--generate-ids", str(quantized_output), "ab a", "2"],
            check=True, text=True, capture_output=True).stdout
        second_generation = subprocess.run(
            [infer, "--generate-ids", str(quantized_output), "ab a", "2"],
            check=True, text=True, capture_output=True).stdout
        assert first_generation == second_generation

        truncated_q8 = root / "truncated-q8.miniinfer"
        truncated_q8.write_bytes(quantized_output.read_bytes()[:-1])
        rejected_q8 = subprocess.run(
            [infer, "--inspect", str(truncated_q8)], text=True, capture_output=True)
        assert rejected_q8.returncode != 0

        # A missing tensor is rejected and no partial output is published.
        broken_shapes = dict(shapes)
        broken_shapes.pop("model.layers.0.mlp.down_proj.weight")
        write_safetensors(root / "model.safetensors", broken_shapes)
        broken_output = root / "broken.miniinfer"
        failed = subprocess.run([converter, str(root), str(broken_output)],
                                text=True, capture_output=True)
        assert failed.returncode != 0
        assert not broken_output.exists()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
