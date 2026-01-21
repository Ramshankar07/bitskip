#!/usr/bin/env python3
"""
Utility script to view and manage experiment results cache.
Enhanced with seed aggregation (mean ± std) and publication-ready tables.
"""

import os
import json
import sys
import re
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
RESULTS_CACHE_FILE = os.path.join(PROJECT_ROOT, "results", "results_cache.json")


def load_cache():
    """Load results cache."""
    if not os.path.exists(RESULTS_CACHE_FILE):
        print(f"Cache file not found: {RESULTS_CACHE_FILE}")
        return {}
    
    try:
        with open(RESULTS_CACHE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading cache: {e}")
        return {}


def aggregate_by_config(cache):
    """
    Aggregate results by configuration (removing seed suffix).
    Returns dict: config_name -> list of perplexities
    """
    aggregated = defaultdict(list)
    
    for model_id, data in cache.items():
        ppl = data.get('perplexity', float('inf'))
        if ppl == float('inf'):
            continue
        
        # Remove seed suffix (e.g., _s42, _s123, _s456)
        config_name = re.sub(r'_s\d+$', '', model_id)
        aggregated[config_name].append(ppl)
    
    return aggregated


def compute_stats(ppls):
    """Compute mean and std from a list of perplexities."""
    if not ppls:
        return None, None
    
    import math
    mean = sum(ppls) / len(ppls)
    if len(ppls) > 1:
        variance = sum((x - mean) ** 2 for x in ppls) / (len(ppls) - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return mean, std


def view_all_results():
    """View all results in the cache."""
    cache = load_cache()
    
    if not cache:
        print("No results found in cache.")
        return
    
    print(f"\n{'='*80}")
    print(f"Experiment Results ({len(cache)} experiments)")
    print(f"{'='*80}\n")
    
    # Group by stage
    by_stage = {}
    for model_id, data in cache.items():
        stage = data.get('stage', 'Unknown')
        if stage not in by_stage:
            by_stage[stage] = []
        by_stage[stage].append((model_id, data))
    
    # Print by stage
    for stage, experiments in sorted(by_stage.items()):
        print(f"\n{stage}:")
        print("-" * 80)
        
        # Sort by perplexity
        experiments.sort(key=lambda x: x[1].get('perplexity', float('inf')))
        
        for model_id, data in experiments:
            ppl = data.get('perplexity', float('inf'))
            if ppl != float('inf'):
                print(f"  {model_id:40s} PPL: {ppl:6.2f}")
            else:
                print(f"  {model_id:40s} PPL: Failed")


def view_aggregated_results():
    """View results aggregated by config (mean ± std across seeds)."""
    cache = load_cache()
    
    if not cache:
        print("No results found in cache.")
        return
    
    aggregated = aggregate_by_config(cache)
    
    print(f"\n{'='*80}")
    print("Aggregated Results (mean ± std across seeds)")
    print(f"{'='*80}\n")
    
    # Sort by mean PPL
    results = []
    for config_name, ppls in aggregated.items():
        mean, std = compute_stats(ppls)
        if mean is not None:
            results.append((config_name, mean, std, len(ppls)))
    
    results.sort(key=lambda x: x[1])
    
    print(f"{'Config':<45} {'PPL (mean ± std)':<20} {'Seeds':>6}")
    print("-" * 75)
    
    for config_name, mean, std, n_seeds in results:
        print(f"{config_name:<45} {mean:6.2f} ± {std:5.2f}     {n_seeds:>6}")


def view_ablation_table(ablation_type: str):
    """Generate publication-ready table for a specific ablation."""
    cache = load_cache()
    aggregated = aggregate_by_config(cache)
    
    print(f"\n{'='*80}")
    print(f"Ablation Table: {ablation_type}")
    print(f"{'='*80}\n")
    
    # Filter configs for this ablation
    if ablation_type == "lambda":
        pattern = r"Lambda_([0-9.]+)_(FP16|INT8)"
        header = "| λ | FP16 PPL | INT8 PPL |"
    elif ablation_type == "p_max":
        pattern = r"PMax_([0-9.]+)_(FP16|INT8)"
        header = "| p_max | FP16 PPL | INT8 PPL |"
    elif ablation_type == "schedule":
        pattern = r"Schedule_(\w+)_(FP16|INT8)"
        header = "| Schedule | FP16 PPL | INT8 PPL |"
    elif ablation_type == "hadamard":
        pattern = r"Hadamard_(H|NoH)_(FP16|INT8)"
        header = "| Hadamard | FP16 PPL | INT8 PPL |"
    else:
        print(f"Unknown ablation type: {ablation_type}")
        return
    
    # Parse and organize data
    data = {}
    for config_name, ppls in aggregated.items():
        match = re.match(pattern, config_name)
        if match:
            param_val = match.group(1)
            precision = match.group(2)
            mean, std = compute_stats(ppls)
            if param_val not in data:
                data[param_val] = {}
            data[param_val][precision] = (mean, std, len(ppls))
    
    if not data:
        print(f"No data found for ablation type: {ablation_type}")
        return
    
    # Print Markdown table
    print(header)
    print("|" + "|".join(["-" * 12] * 3) + "|")
    
    for param_val in sorted(data.keys()):
        fp16 = data[param_val].get("FP16", (None, None, 0))
        int8 = data[param_val].get("INT8", (None, None, 0))
        
        fp16_str = f"{fp16[0]:.2f} ± {fp16[1]:.2f}" if fp16[0] else "N/A"
        int8_str = f"{int8[0]:.2f} ± {int8[1]:.2f}" if int8[0] else "N/A"
        
        print(f"| {param_val:<10} | {fp16_str:<10} | {int8_str:<10} |")


def view_best_results():
    """View best results for each stage."""
    cache = load_cache()
    aggregated = aggregate_by_config(cache)
    
    if not aggregated:
        print("No results found in cache.")
        return
    
    print(f"\n{'='*80}")
    print("Best Results (Aggregated)")
    print(f"{'='*80}\n")
    
    # Find best by mean PPL
    results = []
    for config_name, ppls in aggregated.items():
        mean, std = compute_stats(ppls)
        if mean is not None:
            results.append((config_name, mean, std, len(ppls)))
    
    results.sort(key=lambda x: x[1])
    
    print("Top 10 configurations:")
    print(f"{'Config':<45} {'PPL (mean ± std)':<20} {'Seeds':>6}")
    print("-" * 75)
    
    for config_name, mean, std, n_seeds in results[:10]:
        print(f"{config_name:<45} {mean:6.2f} ± {std:5.2f}     {n_seeds:>6}")


def clear_cache():
    """Clear the results cache."""
    if os.path.exists(RESULTS_CACHE_FILE):
        os.remove(RESULTS_CACHE_FILE)
        print(f"Cache cleared: {RESULTS_CACHE_FILE}")
    else:
        print("Cache file does not exist.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="View experiment results with statistical aggregation")
    parser.add_argument("--all", action="store_true", help="View all raw results")
    parser.add_argument("--aggregated", action="store_true", help="View aggregated results (mean ± std)")
    parser.add_argument("--best", action="store_true", help="View best results")
    parser.add_argument("--table", type=str, choices=["lambda", "p_max", "schedule", "hadamard"],
                        help="Generate publication table for ablation")
    parser.add_argument("--clear", action="store_true", help="Clear the cache")
    args = parser.parse_args()
    
    if args.clear:
        clear_cache()
    elif args.table:
        view_ablation_table(args.table)
    elif args.aggregated:
        view_aggregated_results()
    elif args.best:
        view_best_results()
    else:
        view_all_results()


if __name__ == "__main__":
    main()
