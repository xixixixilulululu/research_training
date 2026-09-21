#!/bin/bash
#SBATCH --job-name=busi_task1
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

# 用 nbconvert 在后台完整执行 notebook，并把输出（打印结果、图表）原地写回同一个 .ipynb 文件，
# 这样训练结束后可以直接打开 BUSI_Classification.ipynb 看到所有结果，不需要人一直守在屏幕前。
# --ExecutePreprocessor.timeout=-1 表示不设执行超时（训练可能要跑很久）。
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=ml \
    BUSI_Classification.ipynb
