#!/usr/bin/env python3
"""
Simple BitNet Benchmark Runner
Uses already downloaded models to avoid hanging issues
"""

import os
import sys
import json
from pathlib import Path

# Set up environment
os.environ["PYTHONPATH"] = "."

def run_simple_benchmark():
    """Run a simple benchmark using the downloaded BitNet 1B model."""
    
    print("Running BitNet 1B benchmark with local model...")
    
    # Use the already downloaded BitNet 1B model
    model_path = "models/models--Ram07--bitnet-1b/snapshots/43d33885fa0c4bea30986172e63ee6bc3e8c18ae"
    
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        return False
    
    try:
        # Import the benchmark
        sys.path.insert(0, '.')
        from evals.inference_benchmark_1b_improved import InferenceBenchmark
        
        benchmark = InferenceBenchmark(
            model_path=model_path,
            device="cpu"
        )
        
        print("Loading model from local path...")
        if benchmark.load_model():
            print("Model loaded successfully!")
            
            print("Running quick benchmark...")
            results = benchmark.run_benchmark(
                max_new_tokens=20,  # Smaller for quick test
                num_runs=1,
                early_exit_thresholds=[0.0, 0.5]  # Fewer thresholds for speed
            )
            
            # Save results
            output_file = "results/bitnet1b_quick_test.json"
            os.makedirs("results", exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Quick benchmark completed! Results saved to {output_file}")
            
            # Show summary
            if "prompt_results" in results:
                for prompt_key, prompt_data in results["prompt_results"].items():
                    if "results" in prompt_data:
                        for threshold_key, threshold_data in prompt_data["results"].items():
                            tokens_per_sec = threshold_data.get("tokens_per_second_avg", 0)
                            ttft = threshold_data.get("ttft_avg", 0)
                            print(f"Threshold {threshold_key}: {tokens_per_sec:.3f} tok/s, TTFT: {ttft:.3f}s")
            
            return True
        else:
            print("Failed to load model")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_simple_benchmark()
    if success:
        print("\nSUCCESS: Quick BitNet benchmark completed!")
    else:
        print("\nFAILED: Quick BitNet benchmark failed!")
        print("You can analyze your existing results in results/bitnet1bcpu.json")

