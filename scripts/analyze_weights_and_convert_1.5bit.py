#!/usr/bin/env python3
"""
Analyze BitSkip/BitNet checkpoints and convert to 1.5-bit (ternary) representation.

- Loads a state dict (e.g. from model.safetensors).
- Identifies BitLinear/HBitLinear weight keys (q/k/v/o_proj, up/down_proj).
- Computes insights: ternary distribution (-1/0/+1), scale stats, sparsity.
- Optionally writes a 1.5-bit state dict (ternary weights + per-tensor scale).

Usage:
  python scripts/analyze_weights_and_convert_1.5bit.py --checkpoint path/to/model.safetensors
  python scripts/analyze_weights_and_convert_1.5bit.py --checkpoint model.safetensors --output-1.5bit model_1.5bit.safetensors --json insights.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

# Keys that hold quantizable (BitLinear/HBitLinear) weights
QUANTIZABLE_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)


def is_quantizable_key(key: str) -> bool:
    return key.endswith(QUANTIZABLE_SUFFIXES) and "weight_scale" not in key


def ternary_quantize(w: torch.Tensor) -> tuple[torch.Tensor, float, dict[str, float]]:
    """Apply ternary quantization (scale = mean abs). Returns (w_ternary, scale, stats)."""
    scale = w.abs().mean().item()
    scale = max(scale, 1e-8)
    w_q = torch.zeros_like(w)
    w_q[w > 0.5 * scale] = 1.0
    w_q[w < -0.5 * scale] = -1.0
    n = w.numel()
    n_neg = (w < -0.5 * scale).sum().item()
    n_zero = (w_q == 0).sum().item()
    n_pos = (w > 0.5 * scale).sum().item()
    stats = {
        "scale": scale,
        "frac_neg": n_neg / n,
        "frac_zero": n_zero / n,
        "frac_pos": n_pos / n,
        "sparsity": n_zero / n,
    }
    return w_q * scale, scale, stats


def analyze_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Compute weight insights for all quantizable keys."""
    insights: dict[str, list[dict[str, Any]]] = {"layers": [], "summary": {}}
    all_stats: list[dict[str, float]] = []
    total_params = 0

    for key in sorted(state_dict.keys()):
        if not is_quantizable_key(key):
            continue
        w = state_dict[key]
        if w.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            continue
        w = w.float()
        _, scale, stats = ternary_quantize(w)
        n = w.numel()
        total_params += n
        entry = {
            "key": key,
            "shape": list(w.shape),
            "numel": n,
            **stats,
        }
        insights["layers"].append(entry)
        all_stats.append(stats)

    if not all_stats:
        insights["summary"] = {"quantizable_keys": 0, "total_quantizable_params": 0}
        return insights

    insights["summary"] = {
        "quantizable_keys": len(insights["layers"]),
        "total_quantizable_params": total_params,
        "mean_scale": sum(s["scale"] for s in all_stats) / len(all_stats),
        "mean_sparsity": sum(s["sparsity"] for s in all_stats) / len(all_stats),
        "mean_frac_pos": sum(s["frac_pos"] for s in all_stats) / len(all_stats),
        "mean_frac_neg": sum(s["frac_neg"] for s in all_stats) / len(all_stats),
    }
    return insights


def convert_to_1_5bit(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Build a new state dict with ternary weights and updated weight_scale buffers."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not is_quantizable_key(key):
            out[key] = value.detach().clone()
            continue
        w = value.float()
        w_q, scale, _ = ternary_quantize(w)
        out[key] = w_q.to(value.dtype if value.dtype != torch.float16 else torch.float32)
        scale_key = key.replace(".weight", ".weight_scale")
        if scale_key in state_dict:
            out[scale_key] = torch.tensor(scale, dtype=state_dict[scale_key].dtype, device=value.device)
        else:
            out[scale_key] = torch.tensor(scale, dtype=torch.float32)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze weights and convert to 1.5-bit")
    parser.add_argument("--checkpoint", required=True, help="Path to model.safetensors")
    parser.add_argument("--output-1.5bit", default=None, help="Path to save 1.5-bit state dict")
    parser.add_argument("--json", default=None, help="Path to save insights JSON")
    parser.add_argument("--no-print", action="store_true", help="Do not print insights to stdout")
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.exists():
        print(f"Error: checkpoint not found: {path}", file=sys.stderr)
        return 1

    state_dict = load_file(str(path))
    insights = analyze_weights(state_dict)

    if not args.no_print:
        print("Weight insights (quantizable BitLinear/HBitLinear layers)")
        print("=" * 60)
        for layer in insights["layers"][:20]:
            print(f"  {layer['key']}: scale={layer['scale']:.6f} sparsity={layer['sparsity']:.2%} +1={layer['frac_pos']:.2%} -1={layer['frac_neg']:.2%}")
        if len(insights["layers"]) > 20:
            print(f"  ... and {len(insights['layers']) - 20} more")
        print("Summary:", insights["summary"])

    if args.json:
        with open(args.json, "w") as f:
            json.dump(insights, f, indent=2)
        print(f"Wrote insights to {args.json}")

    if args.output_1_5bit:
        out_dict = convert_to_1_5bit(state_dict)
        save_file(out_dict, args.output_1_5bit)
        print(f"Wrote 1.5-bit state dict to {args.output_1_5bit}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
