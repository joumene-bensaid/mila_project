#!/bin/bash
# Script to run all tests
# Ensure the script is executable: chmod +x run_all.sh

module load anaconda/3
module load cuda/11.8
conda activate mila_project

cd /home/mila/b/benj/scratch/mila_project/direction_aware_fusion

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run Test 1 (Orthogonal Deltas)
sbatch scripts/run_test1.sh 
# Run Test 1b (Orthogonal Normalization)
sbatch scripts/run_test1b.sh
# Run Test 0 (SoftSoup)
sbatch scripts/run_test0.sh
