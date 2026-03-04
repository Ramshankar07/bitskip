"""
BitSkip v2 - Modal Batch Experiment Runner

Runs multiple experiments concurrently on Modal using B200 GPUs.

Usage:
    # Run all v2 ablation stages
    modal run experiments/modal_run_experiments.py::app.run_experiments

    # Run specific stage
    modal run experiments/modal_run_experiments.py::app.run_experiments --stage 1

    # Dry run to see what would be executed
    modal run experiments/modal_run_experiments.py::app.run_experiments --dry-run

    # With wandb
    modal run experiments/modal_run_experiments.py::app.run_experiments --wandb
"""

import modal

from modal_train import app, train, image, vol

# ---------------------------------------------------------------------------
# Ablation parameter space
# ---------------------------------------------------------------------------

LAMBDA_R_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]
LAMBDA_Q_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]

# Paper best config (baseline for v2 experiments)
PAPER_BEST = {
    "early_exit_lambda": 0.3,
    "p_max": 0.5,
    "dropout_schedule": "quadratic",
}


def build_experiments(stage: int, best_lambda_r: float = 0.05, best_lambda_q: float = 0.0):
    """Build experiment configs for a given stage."""
    experiments = []

    if stage == 1:
        # Stage V2-1: Lambda_r sweep
        for lr_val in LAMBDA_R_VALUES:
            for had in [False, True]:
                had_tag = "H" if had else "noH"
                experiments.append({
                    "model_id": f"V2_LambdaR_{lr_val}_int8_{had_tag}_s42",
                    "precision": "int8",
                    "use_hadamard": had,
                    "lambda_q": 0.0,
                    "lambda_r": lr_val,
                    "seed": 42,
                    **PAPER_BEST,
                })

    elif stage == 2:
        # Stage V2-2: Lambda_q sweep (using best lambda_r from stage 1)
        for lq_val in LAMBDA_Q_VALUES:
            for had in [False, True]:
                had_tag = "H" if had else "noH"
                experiments.append({
                    "model_id": f"V2_LambdaQ_{lq_val}_int8_{had_tag}_s42",
                    "precision": "int8",
                    "use_hadamard": had,
                    "lambda_q": lq_val,
                    "lambda_r": best_lambda_r,
                    "seed": 42,
                    **PAPER_BEST,
                })

    elif stage == 3:
        # Stage V2-3: Routing on/off at different p_max
        for p_max in [0.2, 0.5, 0.7]:
            for routing_on in [True, False]:
                for had in [False, True]:
                    had_tag = "H" if had else "noH"
                    route_tag = "routeON" if routing_on else "routeOFF"
                    experiments.append({
                        "model_id": f"V2_Routing_{route_tag}_p{p_max}_int8_{had_tag}_s42",
                        "precision": "int8",
                        "use_hadamard": had,
                        "lambda_q": best_lambda_q if routing_on else 0.0,
                        "lambda_r": best_lambda_r if routing_on else 0.0,
                        "seed": 42,
                        "early_exit_lambda": PAPER_BEST["early_exit_lambda"],
                        "p_max": p_max,
                        "dropout_schedule": PAPER_BEST["dropout_schedule"],
                    })

    elif stage == 4:
        # Stage V2-4: Golden config with 3 seeds
        for seed in [42, 123, 456]:
            for had in [True]:
                had_tag = "H" if had else "noH"
                experiments.append({
                    "model_id": f"V2_Golden_int8_{had_tag}_s{seed}",
                    "precision": "int8",
                    "use_hadamard": had,
                    "lambda_q": best_lambda_q,
                    "lambda_r": best_lambda_r,
                    "seed": seed,
                    **PAPER_BEST,
                })

    return experiments


@app.local_entrypoint(name="run_experiments")
def main(
    stage: int = 0,
    best_lambda_r: float = 0.05,
    best_lambda_q: float = 0.0,
    dataset: str = "wikitext2",
    batch_size: int = 128,
    num_steps: int = 5000,
    eval_every_steps: int = 250,
    learning_rate: float = 6e-4,
    wandb: bool = False,
    wandb_project: str = "bitskip-v2",
    dry_run: bool = False,
):
    stages = [stage] if stage > 0 else [1, 2, 3, 4]

    for s in stages:
        experiments = build_experiments(s, best_lambda_r, best_lambda_q)
        print(f"\n{'#'*60}")
        print(f"  STAGE V2-{s}: {len(experiments)} experiments")
        print(f"{'#'*60}")

        if dry_run:
            for exp in experiments:
                print(f"  {exp['model_id']}: lr={exp['lambda_r']} lq={exp['lambda_q']} "
                      f"had={exp['use_hadamard']}")
            continue

        # Launch all experiments in this stage concurrently
        handles = []
        for exp in experiments:
            h = train.spawn(
                model_id=exp["model_id"],
                dataset=dataset,
                precision=exp["precision"],
                use_hadamard=exp["use_hadamard"],
                early_exit_lambda=exp["early_exit_lambda"],
                p_max=exp["p_max"],
                dropout_schedule=exp["dropout_schedule"],
                lambda_q=exp["lambda_q"],
                lambda_r=exp["lambda_r"],
                batch_size=batch_size,
                gradient_accumulation_steps=1,
                learning_rate=learning_rate,
                num_steps=num_steps,
                eval_every_steps=eval_every_steps,
                seed=exp["seed"],
                wandb_enabled=wandb,
                wandb_project=wandb_project,
            )
            handles.append((exp["model_id"], h))
            print(f"  Spawned: {exp['model_id']}")

        # Collect results
        print(f"\nWaiting for {len(handles)} experiments to complete...")
        results = []
        for name, h in handles:
            try:
                result = h.get()
                print(f"  Done: {name} -> PPL={result.get('final_ppl', '?'):.2f}")
                results.append(result)
            except Exception as e:
                print(f"  FAILED: {name} -> {e}")
                results.append({"model_id": name, "error": str(e)})

        # Find best for next stage
        valid = [r for r in results if "final_ppl" in r]
        if valid:
            best = min(valid, key=lambda r: r["final_ppl"])
            print(f"\n  Best in stage {s}: {best['model_id']} (PPL={best['final_ppl']:.2f})")

    print("\nAll stages complete!")
