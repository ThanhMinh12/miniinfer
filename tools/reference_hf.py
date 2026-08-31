#!/usr/bin/env python3
"""Export or compare MiniInfer logits with a Hugging Face Llama model.

This is an offline development utility. Transformers and PyTorch are never
used by the MiniInfer runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def load_dependencies():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "reference tooling requires torch and transformers; install them in a development environment") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def load_hf(model_path: str, allow_download: bool):
    torch, model_class, tokenizer_class = load_dependencies()
    options = {"local_files_only": not allow_download}
    tokenizer = tokenizer_class.from_pretrained(model_path, **options)
    model = model_class.from_pretrained(
        model_path, torch_dtype=torch.float32, **options)
    model.eval()
    return torch, tokenizer, model


def capture_reference(model_path: str, prompt: str, top_k: int,
                      full_logits: bool, include_trace: bool,
                      allow_download: bool, generate_tokens: int = 0) -> dict:
    torch, tokenizer, model = load_hf(model_path, allow_download)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    traces: dict[str, list[float]] = {}
    hooks = []
    if include_trace:
        def save_embedding(_module, _inputs, output):
            traces["embedding"] = output[0, -1].detach().float().cpu().tolist()

        def save_final_norm(_module, _inputs, output):
            traces["final_norm"] = output[0, -1].detach().float().cpu().tolist()

        hooks.append(model.model.embed_tokens.register_forward_hook(save_embedding))
        hooks.append(model.model.norm.register_forward_hook(save_final_norm))
        for layer_index, layer in enumerate(model.model.layers):
            modules = {
                "attn_norm": layer.input_layernorm,
                "q": layer.self_attn.q_proj,
                "k": layer.self_attn.k_proj,
                "v": layer.self_attn.v_proj,
                "o": layer.self_attn.o_proj,
                "ffn_norm": layer.post_attention_layernorm,
                "gate": layer.mlp.gate_proj,
                "up": layer.mlp.up_proj,
                "down": layer.mlp.down_proj,
                "output": layer,
            }
            for name, module in modules.items():
                key = f"layer.{layer_index}.{name}"

                def save_output(_module, _inputs, output, trace_key=key):
                    value = output[0] if isinstance(output, tuple) else output
                    traces[trace_key] = value[0, -1].detach().float().cpu().tolist()

                hooks.append(module.register_forward_hook(save_output))

            def save_after_attention(_module, inputs, trace_key=f"layer.{layer_index}.after_attention"):
                traces[trace_key] = inputs[0][0, -1].detach().float().cpu().tolist()

            hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(save_after_attention))
    try:
        with torch.no_grad():
            output = model(**inputs, use_cache=False, output_hidden_states=include_trace)
    finally:
        for hook in hooks:
            hook.remove()
    logits = output.logits[0, -1].detach().float().cpu()
    count = min(top_k, logits.numel())
    values, indices = torch.topk(logits, count)
    result = {
        "prompt": prompt,
        "token_ids": inputs["input_ids"][0].tolist(),
        "top_token_ids": indices.tolist(),
        "top_logits": values.tolist(),
    }
    if full_logits:
        result["logits"] = logits.tolist()
    if include_trace:
        result["hidden_states"] = [
            hidden[0, -1].detach().float().cpu().tolist()
            for hidden in output.hidden_states
        ]
        result["module_outputs"] = traces
    if generate_tokens:
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=generate_tokens,
                                       do_sample=False, use_cache=True)
        result["generated_ids"] = generated[0].tolist()
    return result


def miniinfer_logits(infer: Path, model: Path, prompt: str) -> dict:
    completed = subprocess.run(
        [str(infer), "--logits", str(model), prompt], check=True,
        text=True, capture_output=True)
    return json.loads(completed.stdout)


def miniinfer_trace(infer: Path, model: Path, prompt: str) -> dict:
    completed = subprocess.run(
        [str(infer), "--trace", str(model), prompt], check=True,
        text=True, capture_output=True)
    return json.loads(completed.stdout)


def miniinfer_generate(infer: Path, model: Path, prompt: str, count: int) -> dict:
    completed = subprocess.run(
        [str(infer), "--generate-ids", str(model), prompt, str(count)], check=True,
        text=True, capture_output=True)
    return json.loads(completed.stdout)


def command_export(args) -> int:
    reference = capture_reference(args.hf_model, args.prompt, args.top_k,
                                  args.full_logits, args.trace, args.allow_download)
    args.output.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def command_compare(args) -> int:
    # MiniInfer runs first and exits before PyTorch is loaded, limiting peak RAM.
    actual = (miniinfer_trace(args.infer, args.miniinfer_model, args.prompt)
              if args.trace else miniinfer_logits(
                  args.infer, args.miniinfer_model, args.prompt))
    actual_generation = (miniinfer_generate(args.infer, args.miniinfer_model,
                                            args.prompt, args.generate_tokens)
                         if args.generate_tokens else None)
    reference = capture_reference(args.hf_model, args.prompt, args.top_k,
                                  True, args.trace, args.allow_download,
                                  args.generate_tokens)
    if actual["tokens"] != reference["token_ids"]:
        print(f"token mismatch\nMiniInfer: {actual['tokens']}\nHugging Face: {reference['token_ids']}",
              file=sys.stderr)
        return 1
    if len(actual["logits"]) != len(reference["logits"]):
        print("logit vector length mismatch", file=sys.stderr)
        return 1
    differences = [abs(a - b) for a, b in zip(actual["logits"], reference["logits"])]
    max_abs = max(differences)
    mean_abs = sum(differences) / len(differences)
    rms = math.sqrt(sum(value * value for value in differences) / len(differences))
    mini_top = max(range(len(actual["logits"])), key=actual["logits"].__getitem__)
    hf_top = reference["top_token_ids"][0]
    print(f"tokens: identical ({len(actual['tokens'])})")
    print(f"logits: max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} rms={rms:.8g}")
    print(f"greedy: MiniInfer={mini_top} HuggingFace={hf_top}")
    generation_failed = False
    if args.generate_tokens:
        generation_failed = actual_generation["tokens"] != reference["generated_ids"]
        print("generation:", "identical" if not generation_failed else "mismatch")
        if generation_failed:
            print(f"MiniInfer: {actual_generation['tokens']}")
            print(f"Hugging Face: {reference['generated_ids']}")
    trace_failed = False
    if args.trace:
        reference_trace = reference["module_outputs"]
        comparisons = [("embedding", actual["embedding"], reference_trace["embedding"]),
                       ("final_norm", actual["final_norm"], reference_trace["final_norm"])]
        fields = {
            "attn_norm": "attn_norm", "q": "q", "k": "k", "v": "v",
            "projected_attention": "o", "after_attention": "after_attention",
            "ffn_norm": "ffn_norm", "gate": "gate", "up": "up",
            "down": "down", "output": "output",
        }
        for layer_index, layer in enumerate(actual["layers"]):
            for mini_name, hf_name in fields.items():
                comparisons.append((f"layer.{layer_index}.{mini_name}", layer[mini_name],
                                    reference_trace[f"layer.{layer_index}.{hf_name}"]))
        for name, mini_values, hf_values in comparisons:
            difference = max(abs(a - b) for a, b in zip(mini_values, hf_values))
            scale = max(1.0, max(abs(value) for value in hf_values))
            relative = difference / scale
            print(f"trace {name}: max_abs={difference:.8g} scaled={relative:.8g}")
            if difference > args.trace_max_abs and relative > args.trace_max_relative:
                trace_failed = True
                break
    if max_abs > args.max_abs or mini_top != hf_top or trace_failed or generation_failed:
        print("comparison failed", file=sys.stderr)
        return 1
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--hf-model", required=True,
                        help="local Hugging Face directory or model ID")
    common.add_argument("--prompt", required=True)
    common.add_argument("--top-k", type=int, default=10)
    common.add_argument("--allow-download", action="store_true",
                        help="allow Transformers to access the network")

    export = subparsers.add_parser("export", parents=[common])
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--full-logits", action="store_true")
    export.add_argument("--trace", action="store_true",
                        help="include final-position hidden states and module outputs")
    export.set_defaults(function=command_export)

    compare = subparsers.add_parser("compare", parents=[common])
    compare.add_argument("--miniinfer-model", type=Path, required=True)
    compare.add_argument("--infer", type=Path, default=Path("build/infer"))
    compare.add_argument("--max-abs", type=float, default=2e-3)
    compare.add_argument("--trace", action="store_true",
                         help="also compare layer-level module outputs")
    compare.add_argument("--trace-max-abs", type=float, default=2e-3)
    compare.add_argument("--trace-max-relative", type=float, default=2e-3)
    compare.add_argument("--generate-tokens", type=int, default=0,
                         help="also require an identical greedy token sequence")
    compare.set_defaults(function=command_compare)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return args.function(args)
    except (RuntimeError, subprocess.CalledProcessError, OSError, json.JSONDecodeError) as exc:
        print(f"reference operation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
