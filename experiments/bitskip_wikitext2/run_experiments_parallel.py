#!/usr/bin/env python3
"""
Parallel experiment runner for Bitskip experiments.
Runs experiments concurrently within stages and saves results for later stages.
"""

import os
import subprocess
import json
import time
import re
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, ALL_COMPLETED
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Define base paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
BASE_CMD = f"{sys.executable} {os.path.join(SCRIPT_DIR, 'train.py')}"
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULTS_CACHE_FILE = os.path.join(OUTPUT_BASE_DIR, "results_cache.json")

# Ensure output directories exist
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def load_results_cache() -> Dict:
    """Load results cache from JSON file."""
    if os.path.exists(RESULTS_CACHE_FILE):
        try:
            with open(RESULTS_CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load results cache: {e}")
    return {}


def save_results_cache(results: Dict):
    """Save results cache to JSON file."""
    try:
        with open(RESULTS_CACHE_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save results cache: {e}")


def parse_val_perplexity(model_id: str) -> float:
    """Parse validation perplexity from results.txt or cache."""
    # Check cache first
    cache = load_results_cache()
    if model_id in cache:
        return cache[model_id].get('perplexity', float('inf'))
    
    # Check results file
    try:
        results_file = os.path.join(OUTPUT_BASE_DIR, model_id, "results.txt")
        if not os.path.exists(results_file):
            return float('inf')
            
        with open(results_file, 'r') as f:
            content = f.read()
            match = re.search(r"Validation Perplexity: ([\d.]+)", content)
            if match:
                ppl = float(match.group(1))
                # Update cache
                cache[model_id] = {'perplexity': ppl, 'timestamp': time.time()}
                save_results_cache(cache)
                return ppl
            else:
                return float('inf')
    except Exception as e:
        print(f"Error parsing results for {model_id}: {e}")
        return float('inf')


def run_single_experiment(args_tuple: Tuple[Dict, bool]) -> Tuple[str, float, bool]:
    """
    Run a single experiment and return results.
    Wrapped to work with ProcessPoolExecutor on Windows.
    
    Args:
        args_tuple: (exp_config, compile_flag)
    
    Returns:
        (model_id, perplexity, success)
    """
    exp_config, compile_flag = args_tuple
    """
    Run a single experiment and return results.
    
    Returns:
        (model_id, perplexity, success)
    """
    model_id = exp_config["id"]
    log_file = os.path.join(LOGS_DIR, f"{model_id}.log")
    
    # Check if already completed
    existing_ppl = parse_val_perplexity(model_id)
    if existing_ppl != float('inf'):
        print(f"✓ {model_id} already completed (PPL: {existing_ppl:.2f})")
        return (model_id, existing_ppl, True)
    
    # Build command
    cmd_parts = [BASE_CMD]
    cmd_parts.append(f"--model_id {model_id}")
    cmd_parts.append(f"--precision {exp_config['precision']}")
    
    if compile_flag:
        cmd_parts.append("--compile")
        
    if exp_config["hadamard"]:
        cmd_parts.append("--use_hadamard")
        
    if not exp_config["early_exit"]:
        cmd_parts.append("--no_early_exit")
    else:
        cmd_parts.append(f"--early_exit_lambda {exp_config.get('lambda', 0.0)}")
        cmd_parts.append(f"--p_max {exp_config.get('p_max', 0.0)}")
        cmd_parts.append(f"--dropout_schedule {exp_config.get('schedule', 'quadratic')}")
        
    if "seed" in exp_config:
        cmd_parts.append(f"--seed {exp_config['seed']}")
    
    cmd = " ".join(cmd_parts)
    
    # Run experiment
    print(f"▶ Starting {model_id}...")
    start_time = time.time()
    
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=f, 
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT
            )
            process.wait()
        
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            ppl = parse_val_perplexity(model_id)
            if ppl != float('inf'):
                print(f"✓ {model_id} completed in {elapsed:.1f}s (PPL: {ppl:.2f})")
                return (model_id, ppl, True)
            else:
                print(f"✗ {model_id} completed but no results found")
                return (model_id, float('inf'), False)
        else:
            print(f"✗ {model_id} failed (check {log_file})")
            return (model_id, float('inf'), False)
            
    except Exception as e:
        print(f"✗ {model_id} raised exception: {e}")
        return (model_id, float('inf'), False)


def run_stage_parallel(
    stage_name: str, 
    experiments: List[Dict], 
    max_workers: int = 4,
    compile_flag: bool = False
) -> Dict[str, float]:
    """
    Run a stage's experiments in parallel and wait for all to complete.
    
    Returns:
        Dictionary mapping model_id to perplexity
    """
    print(f"\n{'='*60}")
    print(f"Stage: {stage_name}")
    print(f"Experiments: {len(experiments)}")
    print(f"Max Workers: {max_workers}")
    print(f"{'='*60}")
    
    results = {}
    stage_start_time = time.time()
    
    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments (wrap args for Windows compatibility)
        future_to_exp = {
            executor.submit(run_single_experiment, (exp, compile_flag)): exp 
            for exp in experiments
        }
        
        # Collect results as they complete - as_completed() blocks until ALL futures are done
        completed = 0
        pending = len(experiments)
        
        print(f"Waiting for all {pending} experiments to complete...")
        print("(This may take a while for long-running experiments)\n")
        
        # Iterate through completed futures - this blocks until all are done
        # as_completed() yields futures as they finish, but waits for ALL to complete
        for future in as_completed(future_to_exp):
            try:
                model_id, ppl, success = future.result()
                results[model_id] = ppl
                completed += 1
                pending -= 1
                
                status = "✓" if success else "✗"
                elapsed = time.time() - stage_start_time
                ppl_str = f"{ppl:.2f}" if ppl != float('inf') else "N/A"
                print(f"{status} [{completed}/{len(experiments)}] {model_id} completed (PPL: {ppl_str}, {pending} remaining, {elapsed/60:.1f}m elapsed)")
            except Exception as e:
                exp_config = future_to_exp[future]
                model_id = exp_config.get("id", "unknown")
                print(f"✗ [{completed}/{len(experiments)}] {model_id} raised exception: {e}")
                results[model_id] = float('inf')
                completed += 1
                pending -= 1
        
        # Explicitly verify all futures completed (as_completed should have handled this, but double-check)
        all_futures = list(future_to_exp.keys())
        done_futures, not_done_futures = wait(all_futures, timeout=0, return_when=ALL_COMPLETED)
        
        if not_done_futures:
            print(f"WARNING: {len(not_done_futures)} futures still pending, waiting for completion...")
            for future in not_done_futures:
                exp_config = future_to_exp[future]
                model_id = exp_config.get("id", "unknown")
                try:
                    _, ppl, success = future.result(timeout=3600)  # 1 hour timeout per experiment
                    results[model_id] = ppl
                except Exception as e:
                    print(f"✗ {model_id} failed with exception: {e}")
                    results[model_id] = float('inf')
        
        print(f"\n✓ All {len(experiments)} experiments have completed.")
    
    # Verify all experiments completed
    if len(results) != len(experiments):
        print(f"WARNING: Expected {len(experiments)} results, got {len(results)}")
        missing = set(exp["id"] for exp in experiments) - set(results.keys())
        if missing:
            print(f"Missing results for: {missing}")
            for exp_id in missing:
                results[exp_id] = float('inf')
    
    # Save results to cache
    cache = load_results_cache()
    for model_id, ppl in results.items():
        cache[model_id] = {
            'perplexity': ppl,
            'timestamp': time.time(),
            'stage': stage_name
        }
    save_results_cache(cache)
    
    # Print summary
    stage_elapsed = time.time() - stage_start_time
    successful = sum(1 for ppl in results.values() if ppl != float('inf'))
    print(f"\n{'='*60}")
    print(f"Stage Complete: {successful}/{len(experiments)} successful")
    print(f"Stage Duration: {stage_elapsed/60:.1f} minutes")
    if successful > 0:
        best_ppl = min(ppl for ppl in results.values() if ppl != float('inf'))
        print(f"Best PPL: {best_ppl:.2f}")
    print(f"{'='*60}\n")
    
    return results


def get_best_param_from_cache(param_name: str, search_values: List, prefix_template: str) -> float:
    """
    Find best parameter value from cached results.
    Averages PPL across precisions (FP16/INT8).
    """
    cache = load_results_cache()
    avg_ppls = {}
    
    for val in search_values:
        ppls = []
        for precision in ["FP16", "INT8"]:
            val_str = f"{val}"
            run_id = f"{prefix_template.format(val=val_str)}_{precision}"
            
            if run_id in cache:
                ppl = cache[run_id].get('perplexity', float('inf'))
                if ppl != float('inf'):
                    ppls.append(ppl)
            else:
                # Fallback to file system
                ppl = parse_val_perplexity(run_id)
                if ppl != float('inf'):
                    ppls.append(ppl)
        
        if ppls:
            avg_ppls[val] = sum(ppls) / len(ppls)
        else:
            print(f"Warning: No valid results found for {param_name}={val}")
            avg_ppls[val] = float('inf')
    
    # Find best
    if all(v == float('inf') for v in avg_ppls.values()):
        print(f"Critical Warning: Could not find best {param_name}. Using default.")
        return search_values[0] if search_values else 0.0
    
    best_val = min(avg_ppls, key=avg_ppls.get)
    print(f"✓ Best {param_name}: {best_val} (Avg PPL: {avg_ppls[best_val]:.2f})")
    return best_val


def get_best_schedule_from_cache(schedules: List[str]) -> str:
    """Find best schedule from cached results."""
    cache = load_results_cache()
    avg_ppls = {}
    
    for sched in schedules:
        ppls = []
        for prec in ["FP16", "INT8"]:
            run_id = f"Exp4_S{sched}_{prec}"
            
            if run_id in cache:
                ppl = cache[run_id].get('perplexity', float('inf'))
                if ppl != float('inf'):
                    ppls.append(ppl)
            else:
                ppl = parse_val_perplexity(run_id)
                if ppl != float('inf'):
                    ppls.append(ppl)
        
        if ppls:
            avg_ppls[sched] = sum(ppls) / len(ppls)
        else:
            avg_ppls[sched] = float('inf')
    
    if any(v != float('inf') for v in avg_ppls.values()):
        best_sched = min(avg_ppls, key=avg_ppls.get)
        print(f"✓ Best Schedule: {best_sched} (Avg PPL: {avg_ppls[best_sched]:.2f})")
        return best_sched
    else:
        print("Warning: Could not determine best schedule. Using default.")
        return schedules[0] if schedules else "quadratic"


def main():
    parser = argparse.ArgumentParser(description="BitSkip Parallel Experiments Runner")
    parser.add_argument("--stage", type=int, default=0, help="Run specific stage (1-6). 0 runs all.")
    parser.add_argument("--max_workers", type=int, default=4, help="Max parallel workers per stage")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for all experiments")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = parser.parse_args()
    
    # Define Search Spaces
    LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    P_MAXS = [0.0, 0.3, 0.5, 0.7]
    SCHEDULES = ["quadratic", "linear", "uniform"]
    
    print(f"\n{'='*60}")
    print("BitSkip Parallel Experiment Runner")
    print(f"{'='*60}")
    print(f"Results Cache: {RESULTS_CACHE_FILE}")
    print(f"Max Workers: {args.max_workers}")
    print(f"Compile: {args.compile}")
    print(f"{'='*60}\n")
    
    if args.dry_run:
        print("DRY RUN MODE - Showing what would be executed\n")
        # Still show the structure but don't execute
        print("Would run experiments in parallel with the following configuration:")
        print(f"  Max Workers: {args.max_workers}")
        print(f"  Compile: {args.compile}")
        print(f"  Stages: {args.stage if args.stage > 0 else 'All (1-6)'}")
        return
    
    start_time = time.time()
    
    # --- Stage 1: Baselines ---
    if args.stage == 0 or args.stage == 1:
        exp1_configs = []
        for prec in ["fp16", "int8", "int4"]:
            exp1_configs.append({
                "id": f"B_Base_{prec.upper()}",
                "precision": prec,
                "hadamard": False,
                "early_exit": False
            })
        for prec in ["int8", "int4"]:
            exp1_configs.append({
                "id": f"B_H_Base_{prec.upper()}",
                "precision": prec,
                "hadamard": True,
                "early_exit": False
            })
        
        run_stage_parallel("Exp 1: Baselines", exp1_configs, args.max_workers, args.compile)
    
    # --- Stage 2: Lambda Ablation ---
    if args.stage == 0 or args.stage == 2:
        exp2_configs = []
        for lam in LAMBDAS:
            for prec in ["fp16", "int8"]:
                exp2_configs.append({
                    "id": f"Exp2_L{lam}_{prec.upper()}",
                    "precision": prec,
                    "hadamard": False,
                    "early_exit": True,
                    "lambda": lam,
                    "p_max": 0.5,
                    "schedule": "quadratic"
                })
        run_stage_parallel("Exp 2: Lambda Ablation", exp2_configs, args.max_workers, args.compile)
    
    # Determine Best Lambda (Needed for Stage 3, 4, 5)
    best_lambda = LAMBDAS[3]  # Default 0.3
    if args.stage == 0 or args.stage >= 3:
        print("\n" + "="*60)
        print("Determining Best Lambda from Stage 2 results...")
        print("="*60)
        best_lambda = get_best_param_from_cache("Lambda", LAMBDAS, "Exp2_L{val}")
    
    # --- Stage 3: P_max Ablation ---
    if args.stage == 0 or args.stage == 3:
        exp3_configs = []
        for p in P_MAXS:
            for prec in ["fp16", "int8"]:
                exp3_configs.append({
                    "id": f"Exp3_P{p}_{prec.upper()}",
                    "precision": prec,
                    "hadamard": False,
                    "early_exit": True,
                    "lambda": best_lambda,
                    "p_max": p,
                    "schedule": "quadratic"
                })
        run_stage_parallel(
            f"Exp 3: P_max Ablation (Lambda={best_lambda})",
            exp3_configs,
            args.max_workers,
            args.compile
        )
    
    # Determine Best P_max (Needed for Stage 4, 5)
    best_p_max = P_MAXS[2]  # Default 0.5
    if args.stage == 0 or args.stage >= 4:
        print("\n" + "="*60)
        print("Determining Best P_max from Stage 3 results...")
        print("="*60)
        best_p_max = get_best_param_from_cache("P_max", P_MAXS, "Exp3_P{val}")
    
    # --- Stage 4: Schedule Ablation ---
    if args.stage == 0 or args.stage == 4:
        exp4_configs = []
        for sched in SCHEDULES:
            for prec in ["fp16", "int8"]:
                exp4_configs.append({
                    "id": f"Exp4_S{sched}_{prec.upper()}",
                    "precision": prec,
                    "hadamard": False,
                    "early_exit": True,
                    "lambda": best_lambda,
                    "p_max": best_p_max,
                    "schedule": sched
                })
        run_stage_parallel(
            f"Exp 4: Schedule Ablation (L={best_lambda}, P={best_p_max})",
            exp4_configs,
            args.max_workers,
            args.compile
        )
    
    # Determine Best Schedule
    best_sched = SCHEDULES[0]  # Default quadratic
    if args.stage == 0 or args.stage >= 5:
        print("\n" + "="*60)
        print("Determining Best Schedule from Stage 4 results...")
        print("="*60)
        best_sched = get_best_schedule_from_cache(SCHEDULES)
    
    # --- Stage 5: Full Comparison ---
    if args.stage == 0 or args.stage == 5:
        full_models_configs = []
        seeds = [42, 123, 456]
        base_models = [
            {"name": "F1", "precision": "fp16", "hadamard": False},
            {"name": "F2", "precision": "int8", "hadamard": False},
            {"name": "F3", "precision": "int4", "hadamard": False},
            {"name": "F4", "precision": "int8", "hadamard": True},
            {"name": "F5", "precision": "int4", "hadamard": True},
        ]
        
        for fm in base_models:
            for seed in seeds:
                full_models_configs.append({
                    "id": f"{fm['name']}_S{seed}",
                    "precision": fm['precision'],
                    "hadamard": fm['hadamard'],
                    "early_exit": True,
                    "lambda": best_lambda,
                    "p_max": best_p_max,
                    "schedule": best_sched,
                    "seed": seed
                })
        run_stage_parallel(
            f"Exp 5: Full Comparison (L={best_lambda}, P={best_p_max}, S={best_sched})",
            full_models_configs,
            args.max_workers,
            args.compile
        )
    
    # --- Stage 6: Hadamard Analysis ---
    if args.stage == 0 or args.stage == 6:
        print("\n" + "="*60)
        print("Stage: Exp 6: Hadamard Analysis")
        print("="*60)
        print("Analysis task reusing models from Exp 5.")
        print("Please run custom analysis scripts on the generated checkpoints.")
    
    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("All Stages Completed!")
    print(f"Total Time: {elapsed/60:.1f} minutes")
    print(f"Results Cache: {RESULTS_CACHE_FILE}")
    print(f"{'='*60}\n")
    
    # Print final best parameters
    print("Final Best Parameters:")
    print(f"  Lambda: {best_lambda}")
    print(f"  P_max: {best_p_max}")
    print(f"  Schedule: {best_sched}")


if __name__ == "__main__":
    main()

