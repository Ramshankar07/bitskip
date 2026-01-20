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
import signal
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, ALL_COMPLETED
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Define base paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
BASE_CMD = f"{sys.executable} {os.path.join(SCRIPT_DIR, 'train.py')} --num_steps 50 --eval_every_steps 10"
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULTS_CACHE_FILE = os.path.join(OUTPUT_BASE_DIR, "results_cache.json")

# Ensure output directories exist
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Precision-aware batch sizes for optimal GPU utilization
# Higher batch sizes for lower precision (less memory per param)
# H200 (80GB): Can use much larger batches
# RTX 3090 (24GB): More conservative
PRECISION_BATCH_SIZES = {
    # For H200 or high-VRAM GPUs (80GB+)
    "h200": {
        "fp16": 64,
        "int8": 96,
        "int4": 128,
    },
    # For RTX 3090 or similar (24GB)
    "rtx3090": {
        "fp16": 16,
        "int8": 24,
        "int4": 32,
    },
    # Default / conservative
    "default": {
        "fp16": 16,
        "int8": 24,
        "int4": 32,
    }
}

# Corresponding gradient accumulation to maintain effective batch size
PRECISION_GRAD_ACCUM = {
    "h200": {
        "fp16": 1,
        "int8": 1,
        "int4": 1,
    },
    "rtx3090": {
        "fp16": 4,
        "int8": 3,
        "int4": 2,
    },
    "default": {
        "fp16": 4,
        "int8": 3,
        "int4": 2,
    }
}

# Global GPU profile (set via --gpu_profile argument)
_gpu_profile = "default"

# Global variables for cleanup
_executor = None
_cleanup_called = False


def cleanup_processes():
    """Cleanup all active processes on exit."""
    global _executor, _cleanup_called
    
    # Prevent double cleanup
    if _cleanup_called:
        return
    _cleanup_called = True
    
    print("\nCleaning up processes...")
    
    # Kill all Python processes running train.py (except current process)
    current_pid = os.getpid()
    killed_count = 0
    
    try:
        # Try using pkill first (more reliable)
        result = subprocess.run(
            ["pkill", "-f", "train.py"],
            timeout=5,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL
        )
        if result.returncode == 0:
            print("  Sent termination signal to training processes")
            time.sleep(1)  # Give processes time to terminate
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Fallback: find and kill processes manually using pgrep
        try:
            result = subprocess.run(
                ["pgrep", "-f", "train.py"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for pid_str in result.stdout.strip().split('\n'):
                    if pid_str.strip():
                        try:
                            pid = int(pid_str.strip())
                            if pid != current_pid:
                                # Try graceful termination first
                                try:
                                    os.kill(pid, signal.SIGTERM)
                                    time.sleep(0.5)
                                    # Check if still running, force kill if needed
                                    try:
                                        os.kill(pid, 0)  # Check if process exists
                                        os.kill(pid, signal.SIGKILL)  # Force kill
                                        killed_count += 1
                                    except ProcessLookupError:
                                        killed_count += 1  # Already terminated
                                except ProcessLookupError:
                                    pass  # Process already dead
                                except PermissionError:
                                    print(f"  Warning: No permission to kill process {pid}")
                        except ValueError:
                            pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    if killed_count > 0:
        print(f"  Killed {killed_count} training process(es)")
    
    # Shutdown executor
    if _executor:
        try:
            print("  Shutting down executor...")
            _executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            print(f"  Error shutting down executor: {e}")
    
    # Clear CUDA cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("  Cleared CUDA cache")
    except Exception:
        pass
    
    print("Cleanup complete.")


def signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C)."""
    print("\n\nInterrupt received. Cleaning up...")
    cleanup_processes()
    sys.exit(1)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_processes)


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


def get_available_gpus() -> List[int]:
    """Get list of available GPU IDs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpu_count = len(result.stdout.strip().split('\n'))
            return list(range(gpu_count))
    except Exception:
        pass
    return [0]  # Default to GPU 0 if nvidia-smi fails


def run_single_experiment(args_tuple: Tuple[Dict, int, bool]) -> Tuple[str, float, bool]:
    """
    Run a single experiment and return results.
    Wrapped to work with ProcessPoolExecutor on Windows.
    
    Args:
        args_tuple: (exp_config, gpu_id, compile_flag)
    
    Returns:
        (model_id, perplexity, success)
    """
    exp_config, gpu_id, compile_flag = args_tuple
    model_id = exp_config["id"]
    log_file = os.path.join(LOGS_DIR, f"{model_id}.log")
    
    # Check if already completed
    existing_ppl = parse_val_perplexity(model_id)
    if existing_ppl != float('inf'):
        print(f"[OK] {model_id} already completed (PPL: {existing_ppl:.2f})")
        return (model_id, existing_ppl, True)
    
    # Build command with GPU assignment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Fix OpenMP library conflict on macOS (common with multiprocessing)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Get precision-aware batch size and gradient accumulation
    precision = exp_config['precision'].lower()
    batch_sizes = PRECISION_BATCH_SIZES.get(_gpu_profile, PRECISION_BATCH_SIZES["default"])
    grad_accums = PRECISION_GRAD_ACCUM.get(_gpu_profile, PRECISION_GRAD_ACCUM["default"])
    batch_size = batch_sizes.get(precision, 16)
    grad_accum = grad_accums.get(precision, 4)
    
    cmd_parts = [BASE_CMD]
    cmd_parts.append(f"--model_id {model_id}")
    cmd_parts.append(f"--precision {exp_config['precision']}")
    cmd_parts.append(f"--batch_size {batch_size}")
    cmd_parts.append(f"--gradient_accumulation_steps {grad_accum}")
    cmd_parts.append(f"--output_dir {OUTPUT_BASE_DIR}")
    
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
    print(f">> Starting {model_id} on GPU {gpu_id} (batch={batch_size}, grad_accum={grad_accum})...")
    start_time = time.time()
    
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=f, 
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
                env=env
            )
            process.wait()
        
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            ppl = parse_val_perplexity(model_id)
            if ppl != float('inf'):
                print(f"[OK] {model_id} completed in {elapsed:.1f}s (PPL: {ppl:.2f})")
                return (model_id, ppl, True)
            else:
                print(f"[FAIL] {model_id} completed but no results found")
                return (model_id, float('inf'), False)
        else:
            # Check log file for OOM errors
            oom_detected = False
            try:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        if "CUDA out of memory" in log_content or "OutOfMemoryError" in log_content:
                            oom_detected = True
            except Exception:
                pass
            
            if oom_detected:
                print(f"[FAIL] {model_id} failed due to CUDA OOM (check {log_file})")
                print(f"  Tip: Try reducing --max_workers or --max_gpu_workers, or reduce batch size in config")
            else:
                print(f"[FAIL] {model_id} failed (check {log_file})")
            return (model_id, float('inf'), False)
            
    except Exception as e:
        print(f"[FAIL] {model_id} raised exception: {e}")
        return (model_id, float('inf'), False)


def run_stage_parallel(
    stage_name: str, 
    experiments: List[Dict], 
    max_workers: int = 4,
    compile_flag: bool = False,
    max_gpu_workers: Optional[int] = None
) -> Dict[str, float]:
    """
    Run a stage's experiments in parallel and wait for all to complete.
    
    Args:
        stage_name: Name of the stage
        experiments: List of experiment configurations
        max_workers: Maximum number of parallel workers
        compile_flag: Whether to compile models
        max_gpu_workers: Maximum number of GPU workers (if None, uses max_workers)
    
    Returns:
        Dictionary mapping model_id to perplexity
    """
    global _executor
    
    print(f"\n{'='*60}")
    print(f"Stage: {stage_name}")
    print(f"Experiments: {len(experiments)}")
    print(f"Max Workers: {max_workers}")
    
    # Get available GPUs and determine GPU worker limit
    available_gpus = get_available_gpus()
    if max_gpu_workers is None:
        max_gpu_workers = min(max_workers, len(available_gpus))
    else:
        max_gpu_workers = min(max_gpu_workers, len(available_gpus))
    
    # Limit GPU assignments to max_gpu_workers (cycle through available GPUs)
    gpus_to_use = available_gpus[:max_gpu_workers]
    
    print(f"Available GPUs: {len(available_gpus)}")
    print(f"Using GPUs: {gpus_to_use}")
    print(f"Max GPU Workers: {max_gpu_workers}")
    
    # Warn if max_workers exceeds GPU capacity
    if max_workers > len(available_gpus) and max_gpu_workers < max_workers:
        print(f"WARNING: max_workers ({max_workers}) > available GPUs ({len(available_gpus)})")
        print(f"  Limiting concurrent GPU workers to {max_gpu_workers} to prevent OOM errors")
        print(f"  Consider setting --max_gpu_workers={max_gpu_workers} or reducing --max_workers")
    
    print(f"{'='*60}")
    
    results = {}
    stage_start_time = time.time()
    
    # Create GPU assignment for each experiment (round-robin through limited GPUs)
    gpu_assignments = [gpus_to_use[i % len(gpus_to_use)] for i in range(len(experiments))]
    
    # Run experiments in parallel
    _executor = ProcessPoolExecutor(max_workers=max_workers)
    try:
        # Submit all experiments with GPU assignments
        future_to_exp = {}
        for exp, gpu_id in zip(experiments, gpu_assignments):
            future = _executor.submit(run_single_experiment, (exp, gpu_id, compile_flag))
            future_to_exp[future] = exp
        
        # Initialize progress bar
        pbar = tqdm(total=len(experiments), desc=f"Stage: {stage_name}", unit="exp")
        pbar.set_postfix({"completed": 0, "successful": 0, "failed": 0, "best_ppl": "N/A"})
        
        # Track experiment times for ETA calculation
        experiment_times = []
        
        # Collect results as they complete - as_completed() blocks until ALL futures are done
        completed = 0
        successful = 0
        failed = 0
        
        # Iterate through completed futures - this blocks until all are done
        # as_completed() yields futures as they finish, but waits for ALL to complete
        for future in as_completed(future_to_exp):
            exp_start_time = time.time()
            try:
                model_id, ppl, success = future.result()
                results[model_id] = ppl
                completed += 1
                
                if success:
                    successful += 1
                    status_str = "OK"
                else:
                    failed += 1
                    status_str = "FAIL"
                
                elapsed = time.time() - stage_start_time
                exp_time = time.time() - exp_start_time
                experiment_times.append(exp_time)
                
                # Calculate ETA
                if completed > 0 and len(experiment_times) > 0:
                    avg_time = sum(experiment_times) / len(experiment_times)
                    remaining = len(experiments) - completed
                    eta_seconds = avg_time * remaining
                    eta_str = f"{eta_seconds/60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
                else:
                    eta_str = "N/A"
                
                # Update progress bar
                best_ppl = min([ppl for ppl in results.values() if ppl != float('inf')], default=float('inf'))
                best_ppl_str = f"{best_ppl:.2f}" if best_ppl != float('inf') else "N/A"
                pbar.set_postfix({
                    "completed": completed,
                    "successful": successful,
                    "failed": failed,
                    "best_ppl": best_ppl_str,
                    "eta": eta_str
                })
                pbar.set_description(f"Stage: {stage_name} | {model_id} [{status_str}]")
                pbar.update(1)
                
            except Exception as e:
                exp_config = future_to_exp[future]
                model_id = exp_config.get("id", "unknown")
                results[model_id] = float('inf')
                completed += 1
                failed += 1
                
                elapsed = time.time() - stage_start_time
                pbar.set_postfix({
                    "completed": completed,
                    "successful": successful,
                    "failed": failed,
                    "best_ppl": "N/A"
                })
                pbar.set_description(f"Stage: {stage_name} | {model_id} [ERROR]")
                pbar.update(1)
        
        pbar.close()
        
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
                    print(f"[FAIL] {model_id} failed with exception: {e}")
                    results[model_id] = float('inf')
        
        print(f"\nAll {len(experiments)} experiments have completed.")
    finally:
        # Cleanup executor
        _executor.shutdown(wait=True)
        _executor = None
    
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
    successful_count = sum(1 for ppl in results.values() if ppl != float('inf'))
    print(f"\n{'='*60}")
    print(f"Stage Complete: {successful_count}/{len(experiments)} successful")
    print(f"Stage Duration: {stage_elapsed/60:.1f} minutes")
    if successful_count > 0:
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
    print(f"[OK] Best {param_name}: {best_val} (Avg PPL: {avg_ppls[best_val]:.2f})")
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
        print(f"[OK] Best Schedule: {best_sched} (Avg PPL: {avg_ppls[best_sched]:.2f})")
        return best_sched
    else:
        print("Warning: Could not determine best schedule. Using default.")
        return schedules[0] if schedules else "quadratic"


def main():
    global _gpu_profile
    
    parser = argparse.ArgumentParser(description="BitSkip Minimal Parallel Evaluation Runner")
    parser.add_argument("--max_workers", type=int, default=4, help="Max parallel workers per stage")
    parser.add_argument("--max_gpu_workers", type=int, default=None, help="Max concurrent GPU workers (default: min(max_workers, num_gpus))")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for all experiments")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--gpu_profile", type=str, default="default", 
                        choices=["h200", "rtx3090", "default"],
                        help="GPU profile for precision-aware batch sizes (h200, rtx3090, default)")
    args = parser.parse_args()
    
    # Set global GPU profile for precision-aware batch sizes
    _gpu_profile = args.gpu_profile
    
    # Optimal parameters identified in research
    GOLDEN_LAMBDA = 0.3
    GOLDEN_P_MAX = 0.5
    GOLDEN_SCHEDULE = "quadratic"

    print(f"\n{'='*60}")
    print("BitSkip Minimal Parallel Evaluation Suite")
    print(f"{'='*60}")
    print(f"Results Cache: {RESULTS_CACHE_FILE}")
    print(f"Max Workers: {args.max_workers}")
    print(f"Compile: {args.compile}")
    print(f"GPU Profile: {args.gpu_profile}")
    
    # Show precision-aware batch sizes
    batch_sizes = PRECISION_BATCH_SIZES.get(args.gpu_profile, PRECISION_BATCH_SIZES["default"])
    grad_accums = PRECISION_GRAD_ACCUM.get(args.gpu_profile, PRECISION_GRAD_ACCUM["default"])
    print(f"Batch Sizes: FP16={batch_sizes['fp16']}, INT8={batch_sizes['int8']}, INT4={batch_sizes['int4']}")
    print(f"Grad Accum:  FP16={grad_accums['fp16']}, INT8={grad_accums['int8']}, INT4={grad_accums['int4']}")
    print(f"{'='*60}\n")
    
    if args.dry_run:
        print("DRY RUN MODE - Showing what would be executed\n")
        return
    
    start_time = time.time()
    
    # Define minimal set of experiments
    minimal_configs = [
        # --- Baselines ---
        {
            "id": "Baseline_FP16",
            "precision": "fp16",
            "hadamard": False,
            "early_exit": False
        },
        {
            "id": "Baseline_INT8",
            "precision": "int8",
            "hadamard": False,
            "early_exit": False
        },
        # --- The "Best" BitSkip Setup ---
        {
            "id": "BitSkip_Golden_INT8_H",
            "precision": "int8",
            "hadamard": True,
            "early_exit": True,
            "lambda": GOLDEN_LAMBDA,
            "p_max": GOLDEN_P_MAX,
            "schedule": GOLDEN_SCHEDULE
        }
    ]
    
    run_stage_parallel("Minimal Evaluation", minimal_configs, args.max_workers, args.compile, args.max_gpu_workers)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("Evaluation Suite Finished")
    print(f"Total Time: {elapsed/60:.1f} minutes")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

