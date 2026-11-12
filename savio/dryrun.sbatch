#!/bin/bash
# Smoke test: 3 optimisation steps + 3-face validation on a GPU. Submit from the repo root on Savio:
#   sbatch savio/dryrun.sbatch
#SBATCH --job-name=linemodel_dry
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_ica
#SBATCH --gres=gpu:A5000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=savio/logs/%x_%j.out
set -eo pipefail
source savio/savio_env.sh; source activate "$CONDA_ENV_DIR"
nvidia-smi
python -m line_segmentation.linemodel.train --dry-run --heidelberg --heights --out weights/linemodel/savio_dryrun
