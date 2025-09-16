#!/bin/bash

# BitSkip Quadratic Schedule H-BitLinear Training Script
# This script trains the H-BitLinear model with quadratic dropout schedule
# Model: Enhanced BitNet (model2) + H-BitLinear + Layer Dropout + Early Exit + Quadratic Schedule

echo "=========================================="
echo "BitSkip Quadratic Schedule H-BitLinear Training"
echo "=========================================="
echo "Model: Enhanced BitNet (model2) + H-BitLinear + Quadratic Schedule"
echo "Steps: 1000"
echo "Output: ./output-quadratic-hbitlinear"
echo "=========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZER_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
# Require HUGGINGFACE_TOKEN to be set in the environment; do not hardcode secrets
: "${HUGGINGFACE_TOKEN:?Set HUGGINGFACE_TOKEN in your environment before running this script}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Create output directory
mkdir -p ./output-quadratic-hbitlinear

# Run training with 1000 steps
python train_quadratic_hbitlinear.py \
    --num_steps 1000 \
    --output_dir ./output-quadratic-hbitlinear \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 50 \
    --lambda_q 0.1 \
    --lambda_r 0.05 \
    --early_exit_threshold 0.95 \
    --quadratic_constant 0.3

echo "=========================================="
echo "Quadratic Schedule H-BitLinear Training Completed!"
echo "Check output in: ./output-quadratic-hbitlinear/"
echo "=========================================="
