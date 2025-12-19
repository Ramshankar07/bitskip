#!/bin/bash
#SBATCH --job-name=BitSkip_Stage_6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bhuvaneshwaran.r@northeastern.edu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=04:00:00

# Activate environment
source ~/.bashrc
conda activate /home/bhuvaneshwaran.r/.conda/envs/env_pytorch

echo "Job started at $(date)"
cd /Users/ram/Public/Study/BitSkipv2/bitskip

# Run Stage 6
python3 /Users/ram/Public/Study/BitSkipv2/bitskip/experiments/bitskip_wikitext2/run_experiments.py --stage 6

echo "Job finished at $(date)"
