#!/bin/bash

# =============================================================================
# ABLATION STUDY SHELL SCRIPT
# =============================================================================
# This script runs all 5 training files for the BitNet ablation study:
# 1. train_no_dropout.py     - Control Group (No Layer Dropout)
# 2. train_layer_dropout.py  - Baseline (Layer Dropout)
# 3. train_early_exit.py     - Early Exit (Layer Dropout + Early Exit)
# 4. train_quadratic.py      - Quadratic Schedule (All Features)
# 5. train_quadratic_hbitlinear.py - H-BitLinear with Quadratic Schedule
# =============================================================================

echo "================================================================"
echo "STARTING BITNET ABLATION STUDY TRAINING"
echo "================================================================"
echo "Time: $(date)"
echo "================================================================"

# Activate virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZER_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
# Require HUGGINGFACE_TOKEN to be set in the environment; do not hardcode secrets
: "${HUGGINGFACE_TOKEN:?Set HUGGINGFACE_TOKEN in your environment before running this script}"

# Create output directories
mkdir -p ./output-no-dropout
mkdir -p ./output-baseline
mkdir -p ./output-early-exit
mkdir -p ./output-quadratic
mkdir -p ./output-quadratic-hbitlinear

# =============================================================================
# SCRIPT 1: CONTROL GROUP - NO DROPOUT
# =============================================================================
echo ""
echo "================================================================"
echo "RUNNING SCRIPT 1/4: CONTROL GROUP (NO DROPOUT)"
echo "================================================================"
echo "Features: Native BitNet + BitLinear + NO Layer Dropout"
echo "Purpose: Baseline control group for ablation study"
echo "Output: ./output-no-dropout"
echo "================================================================"

python train_no_dropout.py \
    --output_dir ./output-no-dropout

CONTROL_EXIT_CODE=$?
if [ $CONTROL_EXIT_CODE -eq 0 ]; then
    echo "✓ Control group training completed successfully!"
else
    echo "✗ Control group training failed with exit code $CONTROL_EXIT_CODE"
fi

# =============================================================================
# SCRIPT 2: BASELINE - LAYER DROPOUT
# =============================================================================
echo ""
echo "================================================================"
echo "RUNNING SCRIPT 2/4: BASELINE (LAYER DROPOUT)"
echo "================================================================"
echo "Features: Native BitNet + Layer Dropout + BitLinear"
echo "Purpose: Baseline with basic layer dropout"
echo "Output: ./output-baseline"
echo "================================================================"

python train_layer_dropout.py \
    --output_dir ./output-baseline

BASELINE_EXIT_CODE=$?
if [ $BASELINE_EXIT_CODE -eq 0 ]; then
    echo "✓ Baseline training completed successfully!"
else
    echo "✗ Baseline training failed with exit code $BASELINE_EXIT_CODE"
fi

# =============================================================================
# SCRIPT 3: EARLY EXIT - LAYER DROPOUT + EARLY EXIT
# =============================================================================
echo ""
echo "================================================================"
echo "RUNNING SCRIPT 3/4: EARLY EXIT (LAYER DROPOUT + EARLY EXIT)"
echo "================================================================"
echo "Features: Native BitNet + Layer Dropout + Early Exit + BitLinear"
echo "Purpose: Baseline + Early Exit functionality"
echo "Output: ./output-early-exit"
echo "================================================================"

python train_early_exit.py \
    --output_dir ./output-early-exit

EARLY_EXIT_EXIT_CODE=$?
if [ $EARLY_EXIT_EXIT_CODE -eq 0 ]; then
    echo "✓ Early Exit training completed successfully!"
else
    echo "✗ Early Exit training failed with exit code $EARLY_EXIT_EXIT_CODE"
fi

# =============================================================================
# SCRIPT 4: QUADRATIC SCHEDULE - ALL FEATURES
# =============================================================================
echo ""
echo "================================================================"
echo "RUNNING SCRIPT 4/5: QUADRATIC SCHEDULE (ALL FEATURES)"
echo "================================================================"
echo "Features: Native BitNet + Layer Dropout + Early Exit + Quadratic Schedule + BitLinear"
echo "Purpose: All features enabled with quadratic dropout schedule"
echo "Output: ./output-quadratic"
echo "================================================================"

python train_quadratic.py \
    --output_dir ./output-quadratic

QUADRATIC_EXIT_CODE=$?
if [ $QUADRATIC_EXIT_CODE -eq 0 ]; then
    echo "✓ Quadratic Schedule training completed successfully!"
else
    echo "✗ Quadratic Schedule training failed with exit code $QUADRATIC_EXIT_CODE"
fi

# =============================================================================
# SCRIPT 5: H-BITLINEAR QUADRATIC SCHEDULE - ALL FEATURES
# =============================================================================
echo ""
echo "================================================================"
echo "RUNNING SCRIPT 5/5: H-BITLINEAR QUADRATIC SCHEDULE (ALL FEATURES)"
echo "================================================================"
echo "Features: H-BitLinear + Layer Dropout + Early Exit + Quadratic Schedule"
echo "Purpose: All features enabled with H-BitLinear and quadratic dropout schedule"
echo "Output: ./output-quadratic-hbitlinear"
echo "================================================================"

python train_quadratic_hbitlinear.py \
    --output_dir ./output-quadratic-hbitlinear

H_BITLINEAR_EXIT_CODE=$?
if [ $H_BITLINEAR_EXIT_CODE -eq 0 ]; then
    echo "✓ H-BitLinear Quadratic Schedule training completed successfully!"
else
    echo "✗ H-BitLinear Quadratic Schedule training failed with exit code $H_BITLINEAR_EXIT_CODE"
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo "================================================================"
echo "ABLATION STUDY TRAINING COMPLETED"
echo "================================================================"
echo "Final Status Summary:"
echo "  - Control Group (No Dropout):     $([ $CONTROL_EXIT_CODE -eq 0 ] && echo "✓ SUCCESS" || echo "✗ FAILED")"
echo "  - Baseline (Layer Dropout):       $([ $BASELINE_EXIT_CODE -eq 0 ] && echo "✓ SUCCESS" || echo "✗ FAILED")"
echo "  - Early Exit:                     $([ $EARLY_EXIT_EXIT_CODE -eq 0 ] && echo "✓ SUCCESS" || echo "✗ FAILED")"
echo "  - Quadratic Schedule:             $([ $QUADRATIC_EXIT_CODE -eq 0 ] && echo "✓ SUCCESS" || echo "✗ FAILED")"
echo "  - H-BitLinear Quadratic:          $([ $H_BITLINEAR_EXIT_CODE -eq 0 ] && echo "✓ SUCCESS" || echo "✗ FAILED")"
echo ""
echo "Output Directories:"
echo "  - Control:     ./output-no-dropout"
echo "  - Baseline:    ./output-baseline"
echo "  - Early Exit:  ./output-early-exit"
echo "  - Quadratic:   ./output-quadratic"
echo "  - H-BitLinear: ./output-quadratic-hbitlinear"
echo ""
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "================================================================"

# Check if all scripts succeeded
if [ $CONTROL_EXIT_CODE -eq 0 ] && [ $BASELINE_EXIT_CODE -eq 0 ] && [ $EARLY_EXIT_EXIT_CODE -eq 0 ] && [ $QUADRATIC_EXIT_CODE -eq 0 ] && [ $H_BITLINEAR_EXIT_CODE -eq 0 ]; then
    echo "🎉 ALL TRAINING SCRIPTS COMPLETED SUCCESSFULLY!"
    exit 0
else
    echo "⚠️  SOME TRAINING SCRIPTS FAILED. Check the logs above for details."
    exit 1
fi
