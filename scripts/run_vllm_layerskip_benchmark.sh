#!/bin/bash
# vLLM LayerSkip Benchmark Runner
# Runs the vLLM benchmark for LayerSkip Llama3.2-1B model

set -e

echo "🚀 Starting vLLM LayerSkip Benchmark"
echo "=================================="

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "❌ vLLM is not installed. Installing..."
    pip install vllm
fi

# Check if required packages are installed
python -c "import requests, psutil, GPUtil" 2>/dev/null || {
    echo "📦 Installing required packages..."
    pip install requests psutil GPUtil
}

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_MODELSCOPE=False

# Create results directory
mkdir -p results/vllm_benchmarks
cd results/vllm_benchmarks

echo "📊 Running vLLM LayerSkip Benchmark..."
echo "Model: RichardErkhov/facebook_-_layerskip-llama3.2-1B-gguf"
echo "File: layerskip-llama3.2-1B.Q2_K.gguf"
echo ""

# Run the benchmark
python ../../evals/vllm_layerskip_benchmark.py \
    --model-name "RichardErkhov/facebook_-_layerskip-llama3.2-1B-gguf" \
    --model-file "layerskip-llama3.2-1B.Q2_K.gguf" \
    --server-host "localhost" \
    --server-port 8000 \
    --output-file "vllm_layerskip_benchmark_results.json" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9

echo ""
echo "✅ Benchmark completed!"
echo "📊 Results saved to: results/vllm_benchmarks/vllm_layerskip_benchmark_results.json"
echo ""
echo "🔍 To view results:"
echo "   cat results/vllm_benchmarks/vllm_layerskip_benchmark_results.json | jq ."
echo ""
echo "📈 To compare with other benchmarks:"
echo "   python ../../evals/compare_benchmarks.py"
