#!/bin/bash
#SBATCH --job-name=t5_gradcam
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/cv_data/users/%u/logs/%x-%j.out
#SBATCH --error=/mnt/cv_data/users/%u/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load miniconda3/latest
module load cuda/13.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/ml"

nvidia-smi

# Inference + Grad-CAM only (loads the three Task 2 checkpoints, no training).
# Output figures/CSVs land in task5_cam/{resnet50,vgg16,densenet121}/, and all printed
# results are written back into the notebook itself.
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=ml \
    GradCAM_comparison.ipynb
