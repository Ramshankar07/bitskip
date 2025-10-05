#!/usr/bin/env python3
"""
LM Head analysis utility for BitNet/H-BitLinear models.

For each target model, this script loads the model via BitNetInferenceEngine
and prints LM head statistics (shape, mean, std, min, max).

Supported targets (default):
- Ram07/bitnet-1b
- Ram07/bitnet-2b
- Ram07/hbitlinear-1b
- Ram07/hbitlinear-2b

Usage examples:
  python scripts/lm_head_analysis.py                      # analyze all defaults
  python scripts/lm_head_analysis.py --device cpu         # force CPU
  python scripts/lm_head_analysis.py --models Ram07/bitnet-1b d:\\BitSkip\\models\\local_model_dir
"""

import os
import sys
import argparse
import traceback

import torch

# Ensure env is set to avoid Xet optimized storage requirements on Windows
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

try:
    from bitnet.inference.engine import BitNetInferenceEngine
except Exception as e:
    print("ERROR: Failed to import BitNetInferenceEngine:", e)
    traceback.print_exc()
    sys.exit(1)


DEFAULT_MODELS = [
    "Ram07/bitnet-1b",
    "Ram07/bitnet-2b",
    "Ram07/hbitlinear-1b",
    "Ram07/hbitlinear-2b",
]


def analyze_lm_head_for_model(model_path: str, device: str = "cpu", tokenizer_name: str = None) -> None:
    print("\n" + "=" * 80)
    print(f"Analyzing model: {model_path}")
    print("=" * 80)

    try:
        engine = BitNetInferenceEngine(
            model_path=model_path,
            tokenizer_path=(tokenizer_name or os.getenv("TOKENIZER_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")),
            device=device,
            model_type="auto",
        )

        # Access LM head tensor
        lm_head_weight = getattr(getattr(engine, "model", None), "lm_head", None)
        if lm_head_weight is None or not hasattr(lm_head_weight, "weight"):
            print("WARNING: LM head not found on loaded model")
            return

        weight = lm_head_weight.weight
        # Ensure on CPU for stats to avoid device transfer noise
        w_cpu = weight.detach().to("cpu")

        print("\nLM Head Analysis:")
        print(f"Weight shape: {tuple(w_cpu.shape)}")
        try:
            print(f"Weight mean: {w_cpu.mean().item():.6f}")
            print(f"Weight std: {w_cpu.std(unbiased=False).item():.6f}")
            print(f"Weight min: {w_cpu.min().item():.6f}")
            print(f"Weight max: {w_cpu.max().item():.6f}")
        except Exception as stat_err:
            print(f"ERROR computing stats: {stat_err}")

    except Exception as e:
        print("ERROR: Failed to load/analyze model:")
        print(str(e))
        print(traceback.format_exc())


def parse_args():
    parser = argparse.ArgumentParser(description="LM Head analysis for BitNet/H-BitLinear models")
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="List of model repo IDs or local paths to analyze",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for loading weights (stats computed on CPU)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer name/path to use (defaults to Meta Llama 3 8B Instruct)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA is not available; falling back to CPU")
        args.device = "cpu"

    for model in args.models:
        analyze_lm_head_for_model(model, device=args.device, tokenizer_name=args.tokenizer)


if __name__ == "__main__":
    main()


