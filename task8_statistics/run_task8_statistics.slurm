#!/bin/bash
#SBATCH --job-name=t8_stats
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/cv_data/users/%u/logs/%x-%j.out
#SBATCH --error=/mnt/cv_data/users/%u/logs/%x-%j.err

# CPU-only analysis notebook (no training): reads the Task 6 outputs, so run it after the
# three task6_* jobs have finished.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load miniconda3/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/ml"

jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=ml \
    statistical_analysis.ipynb
