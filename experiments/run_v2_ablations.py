"""
BitSkip v2 Ablation Study Runner

Runs the v2-specific ablation experiments in 4 stages:
  Stage V2-1: Routing loss (lambda_r) sweep
  Stage V2-2: Quantization loss (lambda_q) sweep
  Stage V2-3: Learnable routing on/off
  Stage V2-4: V2 golden configuration (best of all, 3 seeds)

Usage:
    # Run locally
    python experiments/run_v2_ablations.py --output_dir ./results/v2 --dataset wikitext2

    # Run specific stage
    python experiments/run_v2_ablations.py --stage 1 --output_dir ./results/v2

    # Run on Modal
    python experiments/run_v2_ablations.py --modal --wandb
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# V2 ablation parameter space
# ---------------------------------------------------------------------------

LAMBDA_R_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]
LAMBDA_Q_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]
PRECISIONS = ["int8"]
HADAMARD_OPTIONS = [False, True]

# Paper best config (baseline for v2 experiments)
PAPER_BEST = {
    "early_exit_lambda": 0.3,
    "p_max": 0.5,
    "dropout_schedule": "quadratic",
}


@dataclass
class Experiment:
    model_id: str
    stage: int
    precision: str
    use_hadamard: bool
    early_exit_lambda: float
    p_max: float
    dropout_schedule: str
    lambda_q: float
    lambda_r: float
    seed: int = 42
    model_size: str = "85M_H"


def build_stage_v2_1(best_lambda_q: float = 0.0) -> List[Experiment]:
    """Stage V2-1: Routing loss (lambda_r) ablation."""
    experiments = []
    for lr_val in LAMBDA_R_VALUES:
        for prec in PRECISIONS:
            for had in HADAMARD_OPTIONS:
                had_tag = "H" if had else "noH"
                exp = Experiment(
                    model_id=f"V2_LambdaR_{lr_val}_{prec}_{had_tag}_s42",
                    stage=1,
                    precision=prec,
                    use_hadamard=had,
                    lambda_q=best_lambda_q,
                    lambda_r=lr_val,
                    seed=42,
                    model_size="85M_H",
                    **PAPER_BEST,
                )
                experiments.append(exp)
    return experiments


def build_stage_v2_2(best_lambda_r: float = 0.05) -> List[Experiment]:
    """Stage V2-2: Quantization loss (lambda_q) ablation."""
    experiments = []
    for lq_val in LAMBDA_Q_VALUES:
        for prec in PRECISIONS:
            for had in HADAMARD_OPTIONS:
                had_tag = "H" if had else "noH"
                exp = Experiment(
                    model_id=f"V2_LambdaQ_{lq_val}_{prec}_{had_tag}_s42",
                    stage=2,
                    precision=prec,
                    use_hadamard=had,
                    lambda_q=lq_val,
                    lambda_r=best_lambda_r,
                    seed=42,
                    model_size="85M_H",
                    **PAPER_BEST,
                )
                experiments.append(exp)
    return experiments


def build_stage_v2_3(
    best_lambda_r: float = 0.05, best_lambda_q: float = 0.05
) -> List[Experiment]:
    """Stage V2-3: Routing on (with loss) vs off (lambda_r=0)."""
    experiments = []
    p_max_values = [0.2, 0.5, 0.7]
    for p_max in p_max_values:
        for routing_on in [True, False]:
            for prec in PRECISIONS:
                for had in HADAMARD_OPTIONS:
                    had_tag = "H" if had else "noH"
                    route_tag = "routeON" if routing_on else "routeOFF"
                    lr = best_lambda_r if routing_on else 0.0
                    lq = best_lambda_q if routing_on else 0.0
                    exp = Experiment(
                        model_id=f"V2_Routing_{route_tag}_p{p_max}_{prec}_{had_tag}_s42",
                        stage=3,
                        precision=prec,
                        use_hadamard=had,
                        lambda_q=lq,
                        lambda_r=lr,
                        seed=42,
                        model_size="85M_H",
                        early_exit_lambda=PAPER_BEST["early_exit_lambda"],
                        p_max=p_max,
                        dropout_schedule=PAPER_BEST["dropout_schedule"],
                    )
                    experiments.append(exp)
    return experiments


def build_stage_v2_4(
    best_lambda_r: float = 0.05,
    best_lambda_q: float = 0.05,
    best_p_max: float = 0.5,
    best_had: bool = True,
) -> List[Experiment]:
    """Stage V2-4: V2 golden configuration with 3 seeds."""
    experiments = []
    for seed in [42, 123, 456]:
        exp = Experiment(
            model_id=f"V2_Golden_int8_{'H' if best_had else 'noH'}_s{seed}",
            stage=4,
            precision="int8",
            use_hadamard=best_had,
            lambda_q=best_lambda_q,
            lambda_r=best_lambda_r,
            seed=seed,
            model_size="85M_H",
            early_exit_lambda=PAPER_BEST["early_exit_lambda"],
            p_max=best_p_max,
            dropout_schedule=PAPER_BEST["dropout_schedule"],
        )
        experiments.append(exp)
    return experiments


def run_experiment_local(exp: Experiment, output_dir: str, args) -> dict:
    """Run a single experiment locally via subprocess."""
    train_script = os.path.join(
        os.path.dirname(__file__), "bitskip_wikitext2", "train.py"
    )

    cmd = [
        sys.executable, train_script,
        "--model_id", exp.model_id,
        "--model_size", exp.model_size,
        "--output_dir", output_dir,
        "--dataset", args.dataset,
        "--precision", exp.precision,
        "--early_exit_lambda", str(exp.early_exit_lambda),
        "--p_max", str(exp.p_max),
        "--dropout_schedule", exp.dropout_schedule,
        "--lambda_q", str(exp.lambda_q),
        "--lambda_r", str(exp.lambda_r),
        "--seed", str(exp.seed),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--num_steps", str(args.num_steps),
        "--eval_every_steps", str(args.eval_every_steps),
    ]

    if exp.use_hadamard:
        cmd.append("--use_hadamard")
    if args.wandb:
        cmd.extend(["--wandb", "--wandb_project", args.wandb_project])
    if args.compile:
        cmd.append("--compile")

    print(f"\n{'='*60}")
    print(f"Running: {exp.model_id}")
    print(f"  lambda_r={exp.lambda_r}, lambda_q={exp.lambda_q}, "
          f"hadamard={exp.use_hadamard}, seed={exp.seed}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)

    # Read results
    results_file = os.path.join(output_dir, exp.model_id, "results.txt")
    ppl = None
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                if "Validation Perplexity" in line:
                    ppl = float(line.split(":")[-1].strip())
                    break

    return {"model_id": exp.model_id, "ppl": ppl, "returncode": result.returncode}


def run_experiment_modal(exp: Experiment, args) -> dict:
    """Run a single experiment on Modal."""
    cmd = [
        "modal", "run", os.path.join(os.path.dirname(__file__), "modal_train.py"),
        "--model-id", exp.model_id,
        "--model-size", exp.model_size,
        "--dataset", args.dataset,
        "--precision", exp.precision,
        "--early-exit-lambda", str(exp.early_exit_lambda),
        "--p-max", str(exp.p_max),
        "--dropout-schedule", exp.dropout_schedule,
        "--lambda-q", str(exp.lambda_q),
        "--lambda-r", str(exp.lambda_r),
        "--seed", str(exp.seed),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--num-steps", str(args.num_steps),
        "--eval-every-steps", str(args.eval_every_steps),
    ]

    if exp.use_hadamard:
        cmd.append("--use-hadamard")
    if args.compile:
        cmd.extend(["--compile", "--compile-mode", args.compile_mode])
    else:
        cmd.append("--no-compile")
    if args.wandb:
        cmd.extend(["--wandb", "--wandb-project", args.wandb_project])

    print(f"\n{'='*60}")
    print(f"[Modal] Running: {exp.model_id}")
    print(f"  lambda_r={exp.lambda_r}, lambda_q={exp.lambda_q}, "
          f"hadamard={exp.use_hadamard}, seed={exp.seed}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    return {"model_id": exp.model_id, "returncode": result.returncode}


def find_best(results: List[dict], key: str = "ppl") -> Optional[dict]:
    """Find experiment with best (lowest) perplexity."""
    valid = [r for r in results if r.get(key) is not None]
    if not valid:
        return None
    return min(valid, key=lambda r: r[key])


def parse_args():
    parser = argparse.ArgumentParser(description="BitSkip v2 Ablation Runner")
    parser.add_argument("--stage", type=int, default=None,
                        help="Run specific stage (1-4). Default: all stages sequentially")
    parser.add_argument("--output_dir", type=str, default="./results/v2")
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "wikitext103", "ptb"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--eval_every_steps", type=int, default=50)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="bitskip-v2")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (default: 'default')")
    parser.add_argument("--modal", action="store_true", help="Run experiments on Modal")
    parser.add_argument("--dry_run", action="store_true", help="Print experiments without running")

    # Override best values from previous stages
    parser.add_argument("--best_lambda_r", type=float, default=None)
    parser.add_argument("--best_lambda_q", type=float, default=None)
    parser.add_argument("--best_p_max", type=float, default=None)
    parser.add_argument("--best_hadamard", type=bool, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_fn = run_experiment_modal if args.modal else run_experiment_local

    all_results = {}
    stages_to_run = [args.stage] if args.stage else [1, 2, 3, 4]

    # Track best values across stages
    best_lambda_r = args.best_lambda_r if args.best_lambda_r is not None else 0.05
    best_lambda_q = args.best_lambda_q if args.best_lambda_q is not None else 0.0
    best_p_max = args.best_p_max if args.best_p_max is not None else 0.5
    best_had = args.best_hadamard if args.best_hadamard is not None else True

    for stage in stages_to_run:
        print(f"\n{'#'*60}")
        print(f"  STAGE V2-{stage}")
        print(f"{'#'*60}")

        if stage == 1:
            experiments = build_stage_v2_1()
        elif stage == 2:
            experiments = build_stage_v2_2(best_lambda_r=best_lambda_r)
        elif stage == 3:
            experiments = build_stage_v2_3(best_lambda_r=best_lambda_r, best_lambda_q=best_lambda_q)
        elif stage == 4:
            experiments = build_stage_v2_4(
                best_lambda_r=best_lambda_r,
                best_lambda_q=best_lambda_q,
                best_p_max=best_p_max,
                best_had=best_had,
            )
        else:
            print(f"Unknown stage {stage}, skipping")
            continue

        print(f"  {len(experiments)} experiments to run")

        if args.dry_run:
            for exp in experiments:
                print(f"    {exp.model_id}: lr={exp.lambda_r} lq={exp.lambda_q} "
                      f"had={exp.use_hadamard} seed={exp.seed}")
            continue

        stage_results = []
        for exp in experiments:
            if args.modal:
                result = run_fn(exp, args)
            else:
                result = run_fn(exp, args.output_dir, args)
            stage_results.append(result)

        all_results[f"stage_{stage}"] = stage_results

        # Update best values for next stage (local only)
        if not args.modal:
            best = find_best(stage_results)
            if best:
                print(f"\n  Best in stage {stage}: {best['model_id']} (PPL={best.get('ppl', '?')})")
                # Parse best values from model_id
                mid = best["model_id"]
                if stage == 1:
                    for lr in LAMBDA_R_VALUES:
                        if f"LambdaR_{lr}_" in mid:
                            best_lambda_r = lr
                            break
                elif stage == 2:
                    for lq in LAMBDA_Q_VALUES:
                        if f"LambdaQ_{lq}_" in mid:
                            best_lambda_q = lq
                            break

        # Save intermediate results
        results_path = os.path.join(args.output_dir, "v2_ablation_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to {results_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("V2 ABLATION COMPLETE")
    print(f"  Best lambda_r: {best_lambda_r}")
    print(f"  Best lambda_q: {best_lambda_q}")
    print(f"  Best p_max:    {best_p_max}")
    print(f"  Best hadamard: {best_had}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
