#!/usr/bin/env python3
"""
Simple runner script for HuggingFace LayerSkip Benchmark
"""

import subprocess
import sys
import os

def main():
    """Run the HuggingFace LayerSkip benchmark"""
    print("🚀 Starting HuggingFace LayerSkip Benchmark")
    print("=" * 50)
    
    # Check if required packages are installed
    try:
        import torch
        from llama_cpp import Llama
        print("✅ Required packages are installed")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("💡 Install with: pip install llama-cpp-python")
        return 1
    
    # Check if model exists
    model_path = "./models/models--RichardErkhov--facebook_-_layerskip-llama3.2-1B-gguf"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("💡 Make sure the model is downloaded to the models folder")
        return 1
    
    print(f"✅ Model found at: {model_path}")
    
    # Run the benchmark
    cmd = [
        sys.executable, 
        "evals/huggingface_layerskip_benchmark.py",
        "--model-name", "RichardErkhov/facebook_-_layerskip-llama3.2-1B-gguf",
        "--model-file", "layerskip-llama3.2-1B.Q2_K.gguf",
        "--output-file", "huggingface_layerskip_benchmark_results.json"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Benchmark completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Benchmark failed with exit code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())