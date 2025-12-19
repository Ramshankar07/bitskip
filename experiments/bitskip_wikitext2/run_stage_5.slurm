#!/bin/bash
#SBATCH --job-name=BitSkip_Stage_5
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
# Go to project root (2 levels up from experiments/bitskip_wikitext2/)
cd "$(dirname "$0")/../../"
echo "Current directory: $(pwd)"

# Run Stage 5
python3 experiments/bitskip_wikitext2/run_experiments.py --stage 5

echo "Job finished at $(date)"
