#!/bin/bash
# Run 4: the run3_full recipe (40 epochs, eBL + Old Assyrian + Heidelberg faces) with the height channels.
#   sbatch savio/train.sbatch                      # default run name
#   RUN=run4_s1 SEED=1 sbatch savio/train.sbatch   # another seed / name
# GPU choice: A5000 (savio4_gpu, 24 GB, 4 CPUs per GPU) is the easiest to get.
# For an A40 (48 GB) use: --partition=savio3_gpu --qos=a40_gpu3_ica --gres=gpu:A40:1 --cpus-per-task=8
#SBATCH --job-name=linemodel_train
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_ica
#SBATCH --gres=gpu:A5000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=savio/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edwardxuming.lin@gmail.com
set -eo pipefail
source savio/savio_env.sh; source activate "$CONDA_ENV_DIR"
RUN="${RUN:-run4_savio_s0}"; SEED="${SEED:-0}"; EPOCHS="${EPOCHS:-40}"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m line_segmentation.linemodel.train --epochs "$EPOCHS" --seed "$SEED" --heidelberg --heights \
    --workers "$SLURM_CPUS_PER_TASK" --plot-every 5 --out "weights/linemodel/$RUN"
echo "weights in weights/linemodel/$RUN (model.pt = best checkpoint)"
