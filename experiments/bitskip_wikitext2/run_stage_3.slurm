#!/bin/bash
#SBATCH --job-name=BitSkip_Stage_3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bhuvaneshwaran.r@northeastern.edu
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=01:00:00

# Activate environment
source ~/.bashrc
conda activate /home/bhuvaneshwaran.r/.conda/envs/env_pytorch

echo "Job started at $(date)"
# Use SLURM_SUBMIT_DIR to find the project root relative to where sbatch was called
cd "$SLURM_SUBMIT_DIR/../../"
echo "Current directory: $(pwd)"

# Run Stage 3
python3 experiments/bitskip_wikitext2/run_experiments.py --stage 3

echo "Job finished at $(date)"
