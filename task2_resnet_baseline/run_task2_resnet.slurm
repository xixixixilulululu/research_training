#!/bin/bash
#SBATCH --job-name=t2_resnet50
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/cv_data/users/%u/logs/%x-%j.out
#SBATCH --error=/mnt/cv_data/users/%u/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load miniconda3/latest
module load cuda/13.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/ml"

nvidia-smi

# Executes the notebook in place via nbconvert, so all printed output and figures
# are written back into the same .ipynb file - no need to babysit the terminal.
# --ExecutePreprocessor.timeout=-1 means no per-cell timeout (full training can take a while).
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=ml \
    BUSI_Classification_ResNet50_Baseline.ipynb
