#!/bin/bash
#SBATCH --job-name=test1_Orthogonal
#SBATCH --time=0-02:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=logs/test1_Orthogonal%j.out  # Optional: Save output logs
#SBATCH --error=logs/test1_Orthogonal%j.err   # Optional: Save error logs

# Load environment
module load anaconda/3
module load cuda/11.8
conda activate mila_project

cd /home/mila/b/benj/scratch/mila_project/direction_aware_fusion

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Run Test 1 (Orthogonal Deltas)
python experiments/test1_orthogonal_deltas.py --wandb
