"""
BitSkip v2 - Modal Training Script

Run a single training experiment on Modal with B200 GPU.

Usage:
    # Deploy and run
    modal run experiments/modal_train.py --model-id my_experiment --dataset wikitext2

    # With v2 losses
    modal run experiments/modal_train.py --model-id v2_routing --lambda-r 0.05 --lambda-q 0.1

    # With wandb logging
    modal run experiments/modal_train.py --model-id tracked_run --wandb
"""

import modal
from pathlib import Path

# Paths relative to this script so they work from repo root or experiments/
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_BITNET_DIR = _REPO_ROOT / "bitnet"
_WIKITEXT2_DIR = _SCRIPT_DIR / "bitskip_wikitext2"

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "safetensors",
        "tqdm",
        "numpy",
        "einops>=0.7.0",
        "wandb>=0.15.0",
        "huggingface_hub>=0.19.0",
    )
    .add_local_dir(str(_BITNET_DIR), "/root/bitnet")
    .add_local_dir(str(_WIKITEXT2_DIR), "/root/experiments/bitskip_wikitext2")
)

vol = modal.Volume.from_name("bitskip-data", create_if_missing=True)

app = modal.App("bitskip-training", image=image)

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------


@app.function(
    gpu="B200",
    timeout=86400,  # 24h max
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(
    model_id: str = "bitskip_modal_run",
    model_size: str = "125M",
    dataset: str = "wikitext2",
    precision: str = "int8",
    use_hadamard: bool = False,
    # Early exit
    early_exit_lambda: float = 0.3,
    p_max: float = 0.5,
    dropout_schedule: str = "quadratic",
    no_early_exit: bool = False,
    # V2 losses
    lambda_q: float = 0.0,
    lambda_r: float = 0.0,
    # Training
    batch_size: int = 128,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 6e-4,
    num_steps: int = 500,
    eval_every_steps: int = 250,
    seed: int = 42,
    # Compilation
    compile: bool = True,
    compile_mode: str = "default",
    # Mixed precision
    use_amp: bool = True,
    # Diagnostics
    disable_quantization: bool = False,
    disable_hadamard: bool = False,
    h_init_scale: float = 0.1,
    run_fwht_test: bool = False,
    # Logging
    wandb_enabled: bool = False,
    wandb_project: str = "bitskip-v2",
):
    """Train BitSkip model on Modal B200 GPU."""
    import os
    import sys
    import math
    import random

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    from datasets import load_dataset
    from safetensors.torch import save_file
    from tqdm import tqdm

    # Setup paths
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/experiments/bitskip_wikitext2")

    from bitnet.utils.wandb_logger import WandbLogger
    from config import ExperimentConfig
    from model_factory import create_model

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    config = ExperimentConfig(model_size=model_size)
    config.dataset = dataset

    if precision == "fp16":
        config.weight_bits = 16
    elif precision == "int8":
        config.weight_bits = 8
    elif precision == "int4":
        config.weight_bits = 4

    config.use_hadamard = use_hadamard

    if no_early_exit:
        config.use_early_exit = False
        config.early_exit_loss_weight = 0.0
        config.dropout_probability_max = 0.0
    else:
        config.use_early_exit = True
        config.early_exit_loss_weight = early_exit_lambda
        config.dropout_probability_max = p_max
        config.dropout_schedule = dropout_schedule

    config.lambda_q = lambda_q
    config.lambda_r = lambda_r
    config.disable_quantization = disable_quantization
    config.disable_hadamard = disable_hadamard
    config.h_init_scale = h_init_scale
    config.batch_size = batch_size
    config.gradient_accumulation_steps = gradient_accumulation_steps
    config.learning_rate = learning_rate
    config.seed = seed
    config.num_steps = num_steps
    config.eval_every_steps = eval_every_steps

    # Output
    run_dir = f"/data/results/{model_id}"
    os.makedirs(run_dir, exist_ok=True)
    cache_dir = "/data/datasets"
    os.makedirs(cache_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # wandb
    # -----------------------------------------------------------------------
    wb = WandbLogger(
        project=wandb_project,
        run_name=model_id,
        config=vars(config),
        enabled=wandb_enabled,
        tags=[dataset, precision, f"lr{learning_rate}"],
    )

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print(f"Loading tokenizer and {dataset} dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    seq_length = config.max_position_embeddings

    def load_split(split_name):
        if dataset == "wikitext2":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split_name, cache_dir=cache_dir)
        elif dataset == "wikitext103":
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split_name, cache_dir=cache_dir)
        elif dataset == "ptb":
            ds = load_dataset("ptb_text_only", split=split_name, cache_dir=cache_dir)
            ds = ds.map(lambda ex: {"text": ex["sentence"]}, remove_columns=["sentence"])
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        tok = ds.map(
            lambda ex: tokenizer(ex["text"], return_special_tokens_mask=True),
            batched=True,
            num_proc=4,
            remove_columns=["text"] if "text" in ds.column_names else ds.column_names,
        )

        def group(examples):
            concat = {k: sum(examples[k], []) for k in examples.keys()}
            total = len(concat[list(examples.keys())[0]])
            total = (total // seq_length) * seq_length
            result = {k: [t[i : i + seq_length] for i in range(0, total, seq_length)] for k, t in concat.items()}
            result["labels"] = result["input_ids"].copy()
            return result

        return tok.map(group, batched=True, num_proc=4)

    train_ds = load_split("train")
    val_ds = load_split("validation")

    def collate_fn(batch):
        ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        mask = [[1] * len(i) for i in ids]
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask),
            "labels": torch.tensor(labels),
        }

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, pin_memory=True, num_workers=4,
        prefetch_factor=2, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, pin_memory=True, num_workers=2,
    )

    # -----------------------------------------------------------------------
    # FWHT unit test (runs on GPU before training if requested)
    # -----------------------------------------------------------------------
    if run_fwht_test:
        device = torch.device("cuda")
        from bitnet.modeling.kernels import fwht as fwht_fn
        print("\n=== FWHT Unit Test ===")
        all_passed = True
        for n in [64, 128, 256, 512, 1024, 2048]:
            x = torch.randn(4, 32, n, device=device)
            y = fwht_fn(fwht_fn(x))
            max_err = (y - x).abs().max().item()
            rel_err = max_err / x.abs().max().item()
            status = "PASS" if rel_err < 1e-4 else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(f"  n={n:5d}: H(H(x))==x  max_err={max_err:.2e}  rel_err={rel_err:.2e}  [{status}]")

        x32 = torch.randn(4, 32, 512, device=device, dtype=torch.float32)
        x16 = x32.half()
        y32 = fwht_fn(fwht_fn(x32))
        y16 = fwht_fn(fwht_fn(x16))
        err32 = (y32 - x32).abs().max().item()
        err16 = (y16.float() - x32).abs().max().item()
        print(f"  fp32 roundtrip err: {err32:.2e}")
        print(f"  fp16 roundtrip err: {err16:.2e}")
        print(f"  fp16/fp32 ratio:    {err16/max(err32, 1e-12):.1f}x")

        x_stats = torch.randn(2, 16, 512, device=device)
        h_x = fwht_fn(x_stats)
        print(f"\n  Input  stats: mean={x_stats.mean():.4f} std={x_stats.std():.4f} max={x_stats.abs().max():.4f}")
        print(f"  H(x)   stats: mean={h_x.mean():.4f} std={h_x.std():.4f} max={h_x.abs().max():.4f}")
        print(f"  L2 preserved: input={x_stats.norm():.4f} output={h_x.norm():.4f} ratio={h_x.norm()/x_stats.norm():.6f}")
        print(f"=== FWHT Test {'PASSED' if all_passed else 'FAILED'} ===\n")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    if disable_quantization:
        print("*** DIAGNOSTIC: quantization DISABLED in HBitLinear (FWHT-only mode) ***")
    if disable_hadamard:
        print("*** DIAGNOSTIC: ALL Hadamard transforms DISABLED (pure LayerNorm → Linear) ***")
    if h_init_scale != 0.1:
        print(f"*** DIAGNOSTIC: HBitLinear init_scale={h_init_scale} (default 0.1) ***")
    print(f"Creating model (hadamard={use_hadamard}, precision={precision})...")
    model = create_model(config)
    device = torch.device("cuda")
    model.to(device)

    # Compile for B200 performance
    if compile and hasattr(torch, "compile"):
        import torch._inductor.config
        torch._inductor.config.fx_graph_cache = True
        # Persist cache on Modal Volume so it survives across containers
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data/.torch_cache/inductor")
        os.makedirs("/data/.torch_cache/inductor", exist_ok=True)
        print(f"Compiling model with torch.compile(mode={compile_mode!r}, cache=/data/.torch_cache)...")
        model = torch.compile(model, mode=compile_mode)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count / 1e6:.1f}M")

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(100, num_steps // 10), num_training_steps=num_steps
    )
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'DISABLED (full float32)'}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"Starting training: {num_steps} steps, batch_size={batch_size}, "
          f"grad_accum={gradient_accumulation_steps}")

    model.train()
    global_step = 0
    best_val_ppl = float("inf")
    patience_counter = 0
    best_model_state = None
    early_stopped = False

    pbar = tqdm(total=num_steps, desc=f"Training {model_id}", unit="step")

    while global_step < num_steps:
        for batch in train_loader:
            if global_step >= num_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    training_step=global_step,
                )

            loss = outputs.loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

            if (global_step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                current_loss = loss.item() * gradient_accumulation_steps
                pbar.set_postfix({"loss": f"{current_loss:.4f}", "ppl": f"{best_val_ppl:.1f}"})
                wb.log({"train/loss": current_loss, "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

            global_step += 1
            pbar.update(1)

            # Validation
            if global_step % eval_every_steps == 0:
                model.eval()
                val_loss, val_steps = 0.0, 0
                max_val_batches = 50

                with torch.no_grad():
                    for vb in val_loader:
                        if val_steps >= max_val_batches:
                            break
                        vo = model(
                            input_ids=vb["input_ids"].to(device),
                            attention_mask=vb["attention_mask"].to(device),
                            labels=vb["labels"].to(device),
                        )
                        val_loss += vo.loss.item()
                        val_steps += 1

                avg_val_loss = val_loss / max(val_steps, 1)
                val_ppl = math.exp(min(avg_val_loss, 20))  # cap to avoid overflow
                print(f"\n  Step {global_step}: val_ppl={val_ppl:.2f}  best={best_val_ppl:.2f}")
                wb.log({"val/loss": avg_val_loss, "val/perplexity": val_ppl}, step=global_step)

                if val_ppl < best_val_ppl - 0.01:
                    best_val_ppl = val_ppl
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    save_file(best_model_state, os.path.join(run_dir, "model.safetensors"))
                    vol.commit()
                else:
                    patience_counter += 1
                    if patience_counter >= 5:
                        print(f"Early stopping at step {global_step}")
                        early_stopped = True
                        break

                model.train()

            if early_stopped:
                break
        if early_stopped:
            break

    pbar.close()

    # -----------------------------------------------------------------------
    # Final evaluation
    # -----------------------------------------------------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    total_eval_loss, eval_steps = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            total_eval_loss += outputs.loss.item()
            eval_steps += 1

    final_ppl = math.exp(total_eval_loss / max(eval_steps, 1))
    print(f"\nFinal Validation Perplexity: {final_ppl:.2f}")

    # Save results
    results_file = os.path.join(run_dir, "results.txt")
    with open(results_file, "w") as f:
        f.write(f"Validation Perplexity: {final_ppl:.2f}\n")
        f.write(f"Best Validation PPL: {best_val_ppl:.2f}\n")
        f.write(f"Final Step: {global_step}\n")
        f.write(f"Early Stopped: {early_stopped}\n")
    vol.commit()

    wb.summary("final_ppl", final_ppl)
    wb.summary("best_val_ppl", best_val_ppl)
    wb.finish()

    return {"model_id": model_id, "final_ppl": final_ppl, "best_val_ppl": best_val_ppl, "steps": global_step}


# ---------------------------------------------------------------------------
# Weight analysis + 1.5-bit conversion (inline for Modal, no script dependency)
# ---------------------------------------------------------------------------

_QUANTIZABLE_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)


def _is_quantizable_key(key: str) -> bool:
    return key.endswith(_QUANTIZABLE_SUFFIXES) and "weight_scale" not in key


def _ternary_quantize(w):
    import torch
    scale = float(w.abs().mean().clamp(min=1e-8))
    w_q = torch.zeros_like(w)
    w_q[w > 0.5 * scale] = 1.0
    w_q[w < -0.5 * scale] = -1.0
    n = w.numel()
    return w_q * scale, scale, {
        "scale": scale,
        "sparsity": (w_q == 0).sum().item() / n,
        "frac_pos": (w > 0.5 * scale).sum().item() / n,
        "frac_neg": (w < -0.5 * scale).sum().item() / n,
    }


def _analyze_weights(state_dict):
    import torch
    layers, all_stats = [], []
    for key in sorted(state_dict.keys()):
        if not _is_quantizable_key(key):
            continue
        w = state_dict[key].float()
        _, scale, stats = _ternary_quantize(w)
        layers.append({"key": key, "numel": w.numel(), **stats})
        all_stats.append(stats)
    summary = {}
    if all_stats:
        scales = [s["scale"] for s in all_stats]
        sparsities = [s["sparsity"] for s in all_stats]
        n = len(all_stats)
        summary = {
            "quantizable_keys": len(layers),
            "total_quantizable_params": sum(e["numel"] for e in layers),
            "mean_scale": sum(scales) / n,
            "min_scale": min(scales),
            "max_scale": max(scales),
            "mean_sparsity": sum(sparsities) / n,
            "max_sparsity": max(sparsities),
            "layers_high_sparsity": sum(1 for s in sparsities if s >= 0.5),
            "mean_frac_pos": sum(s["frac_pos"] for s in all_stats) / n,
            "mean_frac_neg": sum(s["frac_neg"] for s in all_stats) / n,
        }
    return {"layers": layers, "summary": summary}


def _convert_to_1_5bit(state_dict):
    import torch
    out = {}
    for key, value in state_dict.items():
        if not _is_quantizable_key(key):
            out[key] = value.detach().clone()
            continue
        w = value.float()
        w_q, scale, _ = _ternary_quantize(w)
        out[key] = w_q.to(value.dtype if value.dtype != torch.float16 else torch.float32)
        scale_key = key.replace(".weight", ".weight_scale")
        out[scale_key] = torch.tensor(scale, dtype=torch.float32)
    return out


# ---------------------------------------------------------------------------
# Push to Hugging Face (analysis + 1.5-bit + upload)
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def push_to_huggingface(
    model_id: str,
    repo_id: str,
    private: bool = False,
    upload_1_5bit: bool = True,
):
    """
    Load best checkpoint from /data/results/{model_id}/, run weight insights,
    optionally build 1.5-bit state dict, and upload to Hugging Face.

    Requires Modal secret "huggingface-secret" with HF_TOKEN (or HUGGINGFACE_TOKEN).
    """
    import os
    import json
    import tempfile
    from pathlib import Path
    from safetensors.torch import load_file, save_file
    from huggingface_hub import HfApi, create_repo, upload_folder

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return {"ok": False, "error": "HF_TOKEN or HUGGINGFACE_TOKEN not set"}

    run_dir = Path(f"/data/results/{model_id}")
    ckpt_path = run_dir / "model.safetensors"
    results_path = run_dir / "results.txt"

    if not ckpt_path.exists():
        return {"ok": False, "error": f"Checkpoint not found: {ckpt_path}"}

    state_dict = load_file(str(ckpt_path))
    insights = _analyze_weights(state_dict)
    insights_json = json.dumps(insights, indent=2)

    results_text = ""
    if results_path.exists():
        results_text = results_path.read_text()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_file(state_dict, str(tmp / "model.safetensors"))
        (tmp / "weight_insights.json").write_text(insights_json)

        if upload_1_5bit:
            state_1_5bit = _convert_to_1_5bit(state_dict)
            save_file(state_1_5bit, str(tmp / "model_1.5bit.safetensors"))

        embed_key = next((k for k in state_dict if "embed_tokens.weight" in k), None)
        hidden_size = int(state_dict[embed_key].shape[1]) if embed_key else 512
        config = {
            "model_type": "bitskip",
            "model_id": model_id,
            "vocab_size": 50257,
            "hidden_size": hidden_size,
        }
        (tmp / "config.json").write_text(json.dumps(config, indent=2))

        readme = f"""---
license: apache-2.0
tags:
- bitnet
- bitskip
- quantization
- causal-lm
---

# BitSkip – {model_id}

Trained with BitSkip (WikiText-2). Best checkpoint uploaded.

## Files

- `model.safetensors` – full-precision checkpoint
- `model_1.5bit.safetensors` – ternary (1.5-bit) weights + scales
- `weight_insights.json` – ternary distribution and scale stats
- `config.json` – model config

## Results

```
{results_text}
```

## Weight insights (summary)

```json
{json.dumps(insights["summary"], indent=2)}
```
"""
        (tmp / "README.md").write_text(readme)

        api = HfApi(token=token)
        create_repo(repo_id=repo_id, private=private, exist_ok=True)
        upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )

    return {"ok": True, "repo_id": repo_id, "summary": insights["summary"]}


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    model_id: str = "bitskip_modal_run",
    model_size: str = "125M",
    dataset: str = "wikitext2",
    precision: str = "int8",
    use_hadamard: bool = False,
    no_early_exit: bool = False,
    early_exit_lambda: float = 0.3,
    p_max: float = 0.5,
    dropout_schedule: str = "quadratic",
    lambda_q: float = 0.0,
    lambda_r: float = 0.0,
    batch_size: int = 128,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 6e-4,
    num_steps: int = 500,
    eval_every_steps: int = 250,
    seed: int = 42,
    compile: bool = True,
    compile_mode: str = "default",
    use_amp: bool = True,
    disable_quantization: bool = False,
    disable_hadamard: bool = False,
    h_init_scale: float = 0.1,
    run_fwht_test: bool = False,
    wandb: bool = False,
    wandb_project: str = "bitskip-v2",
):
    result = train.remote(
        model_id=model_id,
        model_size=model_size,
        dataset=dataset,
        precision=precision,
        use_hadamard=use_hadamard,
        early_exit_lambda=early_exit_lambda,
        p_max=p_max,
        dropout_schedule=dropout_schedule,
        no_early_exit=no_early_exit,
        lambda_q=lambda_q,
        lambda_r=lambda_r,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_steps=num_steps,
        eval_every_steps=eval_every_steps,
        seed=seed,
        compile=compile,
        compile_mode=compile_mode,
        use_amp=use_amp,
        disable_quantization=disable_quantization,
        disable_hadamard=disable_hadamard,
        h_init_scale=h_init_scale,
        run_fwht_test=run_fwht_test,
        wandb_enabled=wandb,
        wandb_project=wandb_project,
    )
    print(f"\nResults: {result}")


# Default Hugging Face username for push_to_hf from composition
HF_USERNAME = "Ram07"


@app.local_entrypoint(name="composition")
def run_composition(
    model_size: str = "85M_H",
    num_steps: int = 500,
    eval_every_steps: int = 50,
    batch_size: int = 16,
    learning_rate: float = 6e-4,
    seed: int = 42,
    wandb: bool = False,
    wandb_project: str = "bitskip-v2",
    push_to_hf: bool = True,
):
    """
    Run the 4-way composition ablation: Baseline, H-only, EE-only, BitSkip (H+EE).
    Then, if push_to_hf is True, push each model to Hugging Face (Ram07/{model_id}, private).

    Uses best hyperparameters from benchmarks:
      lambda_r=0.2, lambda_q=0.05, p_max=0.7, early_exit_lambda=0.3

    Usage:
        modal run modal_train.py::app.composition
        modal run modal_train.py::app.composition --num-steps 1000
        modal run modal_train.py::app.composition --no-push-to-hf   # skip HF upload
    """
    BEST = dict(
        lambda_r=0.2,
        lambda_q=0.05,
        early_exit_lambda=0.3,
        p_max=0.7,
        dropout_schedule="quadratic",
    )

    experiments = [
        {
            "model_id": f"comp_baseline_s{seed}",
            "use_hadamard": False,
            "no_early_exit": True,
            **{k: 0.0 for k in ("lambda_r", "lambda_q", "early_exit_lambda", "p_max")},
            "dropout_schedule": "quadratic",
            "label": "Baseline (no H, no EE)",
        },
        {
            "model_id": f"comp_H_only_s{seed}",
            "use_hadamard": True,
            "no_early_exit": True,
            **{k: 0.0 for k in ("lambda_r", "lambda_q", "early_exit_lambda", "p_max")},
            "dropout_schedule": "quadratic",
            "label": "H-only (Hadamard, no EE)",
        },
        {
            "model_id": f"comp_EE_only_s{seed}",
            "use_hadamard": False,
            "no_early_exit": False,
            **BEST,
            "label": "EE-only (no Hadamard, early exit)",
        },
        {
            "model_id": f"comp_BitSkip_s{seed}",
            "use_hadamard": True,
            "no_early_exit": False,
            **BEST,
            "label": "BitSkip (H + EE)",
        },
    ]

    shared = dict(
        model_size=model_size,
        dataset="wikitext2",
        precision="int8",
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        num_steps=num_steps,
        eval_every_steps=eval_every_steps,
        seed=seed,
        compile=True,
        compile_mode="default",
        use_amp=True,
        disable_quantization=False,
        disable_hadamard=False,
        h_init_scale=0.1,
        run_fwht_test=False,
        wandb_enabled=wandb,
        wandb_project=wandb_project,
    )

    print("=" * 70)
    print("BitSkip Composition Ablation")
    print(f"Model: {model_size} | Steps: {num_steps} | Seed: {seed}")
    print("=" * 70)

    results = []
    for exp in experiments:
        label = exp.pop("label")
        params = {**shared, **exp}
        print(f"\n>>> Running: {label} ({exp['model_id']})")
        result = train.remote(**params)
        result["label"] = label
        results.append(result)
        print(f"    Result: PPL={result['best_val_ppl']:.2f} (steps={result['steps']})")

    print("\n" + "=" * 70)
    print("COMPOSITION ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Config':<35} {'Best Val PPL':>12} {'Steps':>6}")
    print("-" * 55)
    for r in results:
        print(f"{r['label']:<35} {r['best_val_ppl']:>12.2f} {r['steps']:>6}")
    print("=" * 70)

    if push_to_hf:
        print("\n" + "=" * 70)
        print("Pushing to Hugging Face (private repos)")
        print("=" * 70)
        for r in results:
            mid = r["model_id"]
            repo_id = f"{HF_USERNAME}/{mid}"
            print(f">>> Pushing {mid} -> {repo_id} ...")
            out = push_to_huggingface.remote(
                model_id=mid,
                repo_id=repo_id,
                private=True,
                upload_1_5bit=True,
            )
            if out.get("ok"):
                ppl = r.get("best_val_ppl")
                print(f"    OK: {repo_id}  (best val PPL: {ppl:.2f})")
                summary = out.get("summary") or {}
                if summary:
                    print(f"    Weight insights: keys={summary.get('quantizable_keys', '—')} "
                          f"params={summary.get('total_quantizable_params', '—')} "
                          f"scale=[{summary.get('min_scale', 0):.4f}, {summary.get('max_scale', 0):.4f}] "
                          f"mean_scale={summary.get('mean_scale', 0):.6f}")
                    print(f"    Sparsity: mean={summary.get('mean_sparsity', 0):.2%} "
                          f"max={summary.get('max_sparsity', 0):.2%} "
                          f"layers_high(≥50%)={summary.get('layers_high_sparsity', 0)}")
                    print(f"    Ternary: frac +1={summary.get('mean_frac_pos', 0):.2%} "
                          f"frac -1={summary.get('mean_frac_neg', 0):.2%}")
            else:
                print(f"    Failed: {out.get('error', out)}")
        print("=" * 70)


@app.local_entrypoint(name="push_to_hf")
def run_push_to_hf(
    model_id: str,
    repo_id: str,
    private: bool = False,
    upload_1_5bit: bool = True,
):
    """
    Run weight analysis, build 1.5-bit checkpoint, and push to Hugging Face.

    Requires Modal secret "huggingface-secret" with HF_TOKEN.

    Usage:
        modal run modal_train.py::app.push_to_hf --model-id comp_baseline_s42 --repo-id username/bitskip-85M
    """
    result = push_to_huggingface.remote(
        model_id=model_id,
        repo_id=repo_id,
        private=private,
        upload_1_5bit=upload_1_5bit,
    )
    print("Result:", result)
