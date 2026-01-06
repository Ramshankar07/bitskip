#!/usr/bin/env python3
"""
Utility script to view and manage experiment results cache.
"""

import os
import json
import sys
from pathlib import Path

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


def view_best_results():
    """View best results for each stage."""
    cache = load_cache()
    
    if not cache:
        print("No results found in cache.")
        return
    
    print(f"\n{'='*80}")
    print("Best Results by Stage")
    print(f"{'='*80}\n")
    
    # Group by stage
    by_stage = {}
    for model_id, data in cache.items():
        stage = data.get('stage', 'Unknown')
        ppl = data.get('perplexity', float('inf'))
        if ppl != float('inf'):
            if stage not in by_stage:
                by_stage[stage] = []
            by_stage[stage].append((model_id, ppl))
    
    # Find best for each stage
    for stage in sorted(by_stage.keys()):
        experiments = by_stage[stage]
        best_model, best_ppl = min(experiments, key=lambda x: x[1])
        print(f"{stage}:")
        print(f"  Best: {best_model} (PPL: {best_ppl:.2f})")
        print()


def view_stage_summary():
    """View summary statistics for each stage."""
    cache = load_cache()
    
    if not cache:
        print("No results found in cache.")
        return
    
    print(f"\n{'='*80}")
    print("Stage Summary Statistics")
    print(f"{'='*80}\n")
    
    # Group by stage
    by_stage = {}
    for model_id, data in cache.items():
        stage = data.get('stage', 'Unknown')
        ppl = data.get('perplexity', float('inf'))
        if stage not in by_stage:
            by_stage[stage] = []
        if ppl != float('inf'):
            by_stage[stage].append(ppl)
    
    # Print statistics
    for stage in sorted(by_stage.keys()):
        ppls = by_stage[stage]
        if ppls:
            print(f"{stage}:")
            print(f"  Count: {len(ppls)}")
            print(f"  Best:  {min(ppls):.2f}")
            print(f"  Worst: {max(ppls):.2f}")
            print(f"  Mean:  {sum(ppls)/len(ppls):.2f}")
            print()


def clear_cache():
    """Clear the results cache."""
    if os.path.exists(RESULTS_CACHE_FILE):
        os.remove(RESULTS_CACHE_FILE)
        print(f"Cache cleared: {RESULTS_CACHE_FILE}")
    else:
        print("Cache file does not exist.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="View experiment results cache")
    parser.add_argument("--all", action="store_true", help="View all results")
    parser.add_argument("--best", action="store_true", help="View best results per stage")
    parser.add_argument("--summary", action="store_true", help="View summary statistics")
    parser.add_argument("--clear", action="store_true", help="Clear the cache")
    args = parser.parse_args()
    
    if args.clear:
        clear_cache()
    elif args.best:
        view_best_results()
    elif args.summary:
        view_stage_summary()
    else:
        view_all_results()


if __name__ == "__main__":
    main()

