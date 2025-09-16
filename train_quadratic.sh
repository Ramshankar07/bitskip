#!/bin/bash

# BitSkip Quadratic Schedule Training Script
# This script trains the BitNet model with quadratic dropout schedule
# Model: Native BitNet + Layer Dropout + Early Exit + Quadratic Schedule

echo "=========================================="
echo "BitSkip Quadratic Schedule Training"
echo "=========================================="
echo "Model: Native BitNet + Quadratic Schedule"
echo "Steps: 1000"
echo "Output: ./output-quadratic"
echo "=========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZER_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
# Require HUGGINGFACE_TOKEN to be set in the environment; do not hardcode secrets
: "${HUGGINGFACE_TOKEN:?Set HUGGINGFACE_TOKEN in your environment before running this script}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Create output directory
mkdir -p ./output-quadratic

# Run training with 1000 steps
python train_quadratic.py \
    --num_steps 1000 \
    --output_dir ./output-quadratic \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 50 \
    --lambda_q 0.1 \
    --lambda_r 0.05 \
    --early_exit_threshold 0.95 \
    --quadratic_constant 0.3

echo "=========================================="
echo "Quadratic Schedule Training Completed!"
echo "Check output in: ./output-quadratic/"
echo "=========================================="
