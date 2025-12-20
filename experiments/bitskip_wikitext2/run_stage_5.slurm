#!/bin/bash
#SBATCH --job-name=BitSkip_Stage_1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bhuvaneshwaran.r@northeastern.edu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=04:00:00

# Load Anaconda and activate environment
module load anaconda3/2024.06
conda activate env_pytorch

echo "Job started at $(date)"
# Use SLURM_SUBMIT_DIR to find the project root relative to where sbatch was called
cd "$SLURM_SUBMIT_DIR/../../"
echo "Current directory: $(pwd)"

# Run Stage 5
python experiments/bitskip_wikitext2/run_experiments.py --stage 5

echo "Job finished at $(date)"
