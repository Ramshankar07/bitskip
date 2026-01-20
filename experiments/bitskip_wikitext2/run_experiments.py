
import os
import subprocess
import time
import re
import math
import argparse
import sys

# Define base paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
# Use sys.executable to ensure child processes use the same environment
BASE_CMD = f"{sys.executable} {os.path.join(SCRIPT_DIR, 'train.py')} --num_steps 50 --eval_every_steps 10"
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Ensure output directories exist
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def run_command(cmd, log_file):
    print(f"Running: {cmd}")
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        process.wait()
    return process.returncode

def parse_val_perplexity(model_id):
    """Parses validation perplexity from results.txt"""
    try:
        results_file = os.path.join(OUTPUT_BASE_DIR, model_id, "results.txt")
        if not os.path.exists(results_file):
            # Check log file for fallback (sometimes results.txt isn't written if crash, but we want to know)
            return float('inf')
            
        with open(results_file, 'r') as f:
            content = f.read()
            match = re.search(r"Validation Perplexity: ([\d.]+)", content)
            if match:
                return float(match.group(1))
            else:
                return float('inf')
    except Exception as e:
        print(f"Error parsing results for {model_id}: {e}")
        return float('inf')

def run_experiment_stage(stage_name, experiments, dry_run=False, compile=False):
    print(f"\n--- Starting Stage: {stage_name} ---")
    results = {}
    
    for exp in experiments:
        model_id = exp["id"]
        log_file = os.path.join(LOGS_DIR, f"{model_id}.log")
        
        # Skip if already done successfully
        if os.path.exists(log_file) and parse_val_perplexity(model_id) != float('inf'):
            print(f"Skipping {model_id}, already completed.")
            results[model_id] = parse_val_perplexity(model_id)
            continue
            
        cmd_parts = [BASE_CMD]
        cmd_parts.append(f"--model_id {model_id}")
        cmd_parts.append(f"--precision {exp['precision']}")
        
        if compile:
            cmd_parts.append("--compile")
            
        if exp["hadamard"]:
            cmd_parts.append("--use_hadamard")
            
        if not exp["early_exit"]:
            cmd_parts.append("--no_early_exit")
        else:
            cmd_parts.append(f"--early_exit_lambda {exp.get('lambda', 0.0)}")
            cmd_parts.append(f"--p_max {exp.get('p_max', 0.0)}")
            cmd_parts.append(f"--dropout_schedule {exp.get('schedule', 'quadratic')}")
            
        if "seed" in exp:
            cmd_parts.append(f"--seed {exp['seed']}")
            
        cmd = " ".join(cmd_parts)
        
        if not dry_run:
            ret = run_command(cmd, log_file)
            if ret != 0:
                print(f"Experiment {model_id} failed. Check {log_file}")
                results[model_id] = float('inf')
            else:
                ppl = parse_val_perplexity(model_id)
                print(f"Experiment {model_id} completed. PPL: {ppl}")
                results[model_id] = ppl
        else:
            print(f"[DRY RUN] Would run: {cmd}")
            results[model_id] = 0.0 # Placeholder
            
    return results

def get_best_param(results, param_name, search_values, prefix_template):
    """
    Finds best parameter value by averaging PPL across precisions (FP16/INT8).
    """
    avg_ppls = {}
    for val in search_values:
        ppls = []
        for precision in ["FP16", "INT8"]:
            val_str = f"{val}"
            try:
                run_id = f"{prefix_template.format(val=val_str)}_{precision}"
                # We need to potentially re-scan directory if results dict is partial
                # But here we assume results dict contains what we need or we scan
                ppl = parse_val_perplexity(run_id)
                if ppl != float('inf'):
                    ppls.append(ppl)
            except KeyError:
                pass
        
        if ppls:
            avg_ppls[val] = sum(ppls) / len(ppls)
        else:
            print(f"Warning: No valid results found for {param_name}={val}")
            avg_ppls[val] = float('inf')
            
    # Find min
    if all(v == float('inf') for v in avg_ppls.values()):
        print(f"Critical Warning: Could not find best {param_name}. Using default.")
        return search_values[0] # Return first as fallback
        
    best_val = min(avg_ppls, key=avg_ppls.get)
    print(f"Best {param_name}: {best_val} (Avg PPL: {avg_ppls[best_val]:.2f})")
    return best_val

def generate_slurm_scripts():
    slurm_header_template = """#!/bin/bash
#SBATCH --job-name=BitSkip_Stage_{}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bhuvaneshwaran.r@northeastern.edu
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=01:00:00

# Load Anaconda and activate environment
module load anaconda3/2024.06
conda activate env_pytorch

echo "Job started at $(date)"
# Use SLURM_SUBMIT_DIR to find the project root relative to where sbatch was called
cd "$SLURM_SUBMIT_DIR/../../"
echo "Current directory: $(pwd)"
"""

    scripts = [
        ("run_stage_1.slurm", 1),
        ("run_stage_2.slurm", 2),
        ("run_stage_3.slurm", 3),
        ("run_stage_4.slurm", 4),
        ("run_stage_5.slurm", 5),
        ("run_stage_6.slurm", 6),
    ]
    
    for filename, stage in scripts:
        filepath = os.path.join(SCRIPT_DIR, filename)
        with open(filepath, "w") as f:
            f.write(slurm_header_template.format(stage))
            f.write(f"\n# Run Stage {stage}\n")
            f.write(f"python experiments/bitskip_wikitext2/run_experiments.py --stage {stage}\n")
            f.write("\necho \"Job finished at $(date)\"\n")
        
        print(f"Generated {filepath}")
        # Make executable
        os.chmod(filepath, 0o755)

def main():
    parser = argparse.ArgumentParser(description="BitSkip Minimal Evaluation Runner")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for all experiments")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("BitSkip Minimal Evaluation Suite")
    print("="*60)
    print(f"Output Directory: {os.path.abspath(OUTPUT_BASE_DIR)}")
    print(f"Logs Directory:   {os.path.abspath(LOGS_DIR)}")
    print("="*60 + "\n")

    # Optimal parameters identified in research
    GOLDEN_LAMBDA = 0.3
    GOLDEN_P_MAX = 0.5
    GOLDEN_SCHEDULE = "quadratic"

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
    
    run_experiment_stage("Minimal Evaluation", minimal_configs, dry_run=args.dry_run, compile=args.compile)
    
    print("\n" + "="*60)
    print("Evaluation Suite Finished")
    print("="*60)

if __name__ == "__main__":
    main()
