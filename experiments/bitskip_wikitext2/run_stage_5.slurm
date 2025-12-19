#!/bin/bash
#SBATCH --job-name=BitSkip_Stage_5
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
cd /Users/ram/Public/Study/BitSkipv2/bitskip

# Run Stage 5
python3 /Users/ram/Public/Study/BitSkipv2/bitskip/experiments/bitskip_wikitext2/run_experiments.py --stage 5

echo "Job finished at $(date)"
