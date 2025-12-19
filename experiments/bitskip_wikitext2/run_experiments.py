
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
BASE_CMD = f"python3 {os.path.join(SCRIPT_DIR, 'train.py')}"
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "results")

# Ensure output directory exists
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

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

def run_experiment_stage(stage_name, experiments, dry_run=False):
    print(f"\n--- Starting Stage: {stage_name} ---")
    results = {}
    
    for exp in experiments:
        model_id = exp["id"]
        log_file = os.path.join(OUTPUT_BASE_DIR, f"{model_id}.log")
        
        # Skip if already done successfully
        if os.path.exists(log_file) and parse_val_perplexity(model_id) != float('inf'):
            print(f"Skipping {model_id}, already completed.")
            results[model_id] = parse_val_perplexity(model_id)
            continue
            
        cmd_parts = [BASE_CMD]
        cmd_parts.append(f"--model_id {model_id}")
        cmd_parts.append(f"--precision {exp['precision']}")
        
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
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=04:00:00

# Activate environment if needed
# source ~/.bashrc
# conda activate bitskip

echo "Job started at $(date)"
cd {}
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
            f.write(slurm_header_template.format(stage, PROJECT_ROOT))
            f.write(f"\n# Run Stage {stage}\n")
            f.write(f"python3 {os.path.abspath(__file__)} --stage {stage}\n")
            f.write("\necho \"Job finished at $(date)\"\n")
        
        print(f"Generated {filepath}")
        # Make executable
        os.chmod(filepath, 0o755)

def main():
    parser = argparse.ArgumentParser(description="BitSkip Experiments Runner")
    parser.add_argument("--stage", type=int, default=0, help="Run specific stage (1-6). 0 runs all.")
    parser.add_argument("--generate_slurm", action="store_true", help="Generate Slurm batch scripts")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = parser.parse_args()
    
    if args.generate_slurm:
        generate_slurm_scripts()
        return

    # Define Search Spaces
    LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    P_MAXS = [0.0, 0.3, 0.5, 0.7]
    SCHEDULES = ["quadratic", "linear", "uniform"]
    
    # --- Experiment 1: Baselines ---
    if args.stage == 0 or args.stage == 1:
        exp1_configs = []
        for prec in ["fp16", "int8", "int4"]:
             exp1_configs.append({"id": f"B_Base_{prec.upper()}", "precision": prec, "hadamard": False, "early_exit": False})
        for prec in ["int8", "int4"]:
             exp1_configs.append({"id": f"B_H_Base_{prec.upper()}", "precision": prec, "hadamard": True, "early_exit": False})
             
        run_experiment_stage("Exp 1: Baselines", exp1_configs, dry_run=args.dry_run)
    
    # --- Experiment 2: Lambda Ablation ---
    if args.stage == 0 or args.stage == 2:
        exp2_configs = []
        for lam in LAMBDAS:
            for prec in ["fp16", "int8"]:
                exp2_configs.append({
                    "id": f"Exp2_L{lam}_{prec.upper()}", 
                    "precision": prec, "hadamard": False, "early_exit": True,
                    "lambda": lam, "p_max": 0.5, "schedule": "quadratic"
                })
        run_experiment_stage("Exp 2: Lambda Ablation", exp2_configs, dry_run=args.dry_run)
    
    # Determine Best Lambda (Needed for Stage 3, 4, 5)
    # We always need to calculate this if we are running later stages
    best_lambda = LAMBDAS[3] # Default 0.3
    if args.stage == 0 or args.stage >= 3:
        if not args.dry_run:
            best_lambda = get_best_param({}, "Lambda", LAMBDAS, "Exp2_L{val}")
            
    # --- Experiment 3: P_max Ablation ---
    if args.stage == 0 or args.stage == 3:
        exp3_configs = []
        for p in P_MAXS:
            for prec in ["fp16", "int8"]:
                 exp3_configs.append({
                    "id": f"Exp3_P{p}_{prec.upper()}", 
                    "precision": prec, "hadamard": False, "early_exit": True,
                    "lambda": best_lambda, "p_max": p, "schedule": "quadratic"
                })
        run_experiment_stage(f"Exp 3: P_max Ablation (Lambda={best_lambda})", exp3_configs, dry_run=args.dry_run)
        
    # Determine Best P_max (Needed for Stage 4, 5)
    best_p_max = P_MAXS[2] # Default 0.5
    if args.stage == 0 or args.stage >= 4:
        if not args.dry_run:
            best_p_max = get_best_param({}, "P_max", P_MAXS, "Exp3_P{val}")

    # --- Experiment 4: Schedule Ablation ---
    if args.stage == 0 or args.stage == 4:
        exp4_configs = []
        for sched in SCHEDULES:
             for prec in ["fp16", "int8"]:
                exp4_configs.append({
                    "id": f"Exp4_S{sched}_{prec.upper()}", 
                    "precision": prec, "hadamard": False, "early_exit": True,
                    "lambda": best_lambda, "p_max": best_p_max, "schedule": sched
                })
        run_experiment_stage(f"Exp 4: Schedule Ablation (L={best_lambda}, P={best_p_max})", exp4_configs, dry_run=args.dry_run)

    # Determine Best Schedule
    best_sched = SCHEDULES[0] # Default quadratic
    if args.stage == 0 or args.stage >= 5:
        if not args.dry_run:
             # Custom polling for schedules
            avg_ppls = {}
            for sched in SCHEDULES:
                ppls = []
                for prec in ["FP16", "INT8"]:
                     run_id = f"Exp4_S{sched}_{prec}"
                     ppl = parse_val_perplexity(run_id)
                     if ppl != float('inf'): ppls.append(ppl)
                if ppls: avg_ppls[sched] = sum(ppls) / len(ppls)
                else: avg_ppls[sched] = float('inf')
            
            if any(v != float('inf') for v in avg_ppls.values()):
                best_sched = min(avg_ppls, key=avg_ppls.get)
                print(f"Best Schedule: {best_sched} (Avg PPL: {avg_ppls[best_sched]:.2f})")
            else:
                 print("Warning: Could not determine best schedule. Using default.")

    # --- Experiment 5: Full Comparison ---
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
        run_experiment_stage(f"Exp 5: Full Comparison (L={best_lambda}, P={best_p_max}, S={best_sched})", full_models_configs, dry_run=args.dry_run)
        
    # --- Experiment 6: Hadamard Analysis ---
    if args.stage == 0 or args.stage == 6:
        print("\n--- Starting Stage: Exp 6: Hadamard Analysis ---")
        print("Analysis task reusing models from Exp 5.")
        print("Please run custom analysis scripts on the generated checkpoints.")
        # This stage might just be a placeholder or run specific analysis scripts
        # For now, we'll leave it as a placeholder or you can add analysis calls here

if __name__ == "__main__":
    main()
