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
    # Model
    # -----------------------------------------------------------------------
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
    scaler = torch.amp.GradScaler("cuda")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(100, num_steps // 10), num_training_steps=num_steps
    )

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

            with torch.amp.autocast("cuda"):
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
        wandb_enabled=wandb,
        wandb_project=wandb_project,
    )
    print(f"\nResults: {result}")
