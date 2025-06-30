#!/bin/bash
#SBATCH --job-name=test0_soft_soup
#SBATCH --time=0-02:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=logs/test0_%j.out  # Optional: Save output logs
#SBATCH --error=logs/test0_%j.err   # Optional: Save error logs

# Load environment
module load anaconda/3
module load cuda/11.8
conda activate mila_project

# Move to project directory (edit path if needed)
cd /home/mila/b/benj/scratch/mila_project/direction_aware_fusion

# Add current directory to Python path so imports work
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run Test 0 (SoftSoup)
python experiments/test0_soft_soup.py --wandb
