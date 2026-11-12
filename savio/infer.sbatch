#!/bin/bash
# Score a run 4 model on the eBL test faces: lines, then crops with the predicted heights against the best fixed height.
#   RUN=run4_savio_s0 sbatch savio/infer.sbatch
#   RUN=run4_savio_s0 sbatch --dependency=afterok:<train job id> savio/infer.sbatch   # run after training finishes
# Inference is light: a GTX2080TI (2 CPUs per GPU) is enough and queues quickly.
# Crops are scored from crops.csv alone (--no-images), about 20,000 fewer files on shared scratch.
#SBATCH --job-name=linemodel_infer
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_ica
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=savio/logs/%x_%j.out
set -eo pipefail
source savio/savio_env.sh; source activate "$CONDA_ENV_DIR"
RUN="${RUN:-run4_savio_s0}"
WEIGHTS="weights/linemodel/$RUN/model.pt"
[ -f "$WEIGHTS" ] || { echo "no weights at $WEIGHTS (training not finished? ls weights/linemodel/)"; exit 1; }
FACES=tablet_data/ebl_tablets/faces/faces.csv
LM=tablet_data/ebl_tablets/linemodel
PRED="$LM/compare/$RUN"; CROPS="$LM/compare_crops/$RUN"
python -m line_segmentation.linemodel.infer --weights "$WEIGHTS" --faces "$FACES" --out "$PRED" --method unet --split test
python -m line_segmentation.eval.evaluate_lines --faces "$FACES" --pred "$PRED"
crops() { name=$1; shift; python -m line_segmentation.crops.make_crops --faces "$FACES" --pred "$PRED" --method unet --split test --no-images --out "$CROPS/$name" "$@"; }
crops fixed --bounds segments --extend 0.2 --shift 0.06
crops heights --bounds heights
crops heights_tight --bounds heights --height-scale 0.9
python -m line_segmentation.eval.box_containment --sets fixed="$CROPS/fixed/crops.csv" heights="$CROPS/heights/crops.csv" \
    heights_tight="$CROPS/heights_tight/crops.csv" --out "$CROPS/box_containment"
echo "lines: $PRED/evaluation.csv  crops: $CROPS/box_containment/summary.json"
