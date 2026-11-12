#!/bin/bash
# One-time environment setup. Run ON A SAVIO LOGIN NODE after sync_to_savio.sh:
#   ssh edwardlin@hpc.brc.berkeley.edu
#   bash /global/scratch/users/edwardlin/ProtoSnap/savio/setup_env.sh
set -eo pipefail
source "$(dirname "$0")/savio_env.sh"

echo "== Slurm associations (pick --account / --qos from these) =="
sacctmgr -p show associations user=$USER format=account,partition,qos | sort -u

mkdir -p "$SAVIO_SCRATCH/envs" "$TORCH_HOME"
# keep conda's package cache off the (small) home directory
mkdir -p "$SAVIO_SCRATCH/.conda/pkgs"; conda config --add pkgs_dirs "$SAVIO_SCRATCH/.conda/pkgs"

if [ ! -x "$CONDA_ENV_DIR/bin/python" ]; then
  conda create -y -p "$CONDA_ENV_DIR" python=3.11
fi
source activate "$CONDA_ENV_DIR"
pip install --no-cache-dir -r "$PROTOSNAP_DIR/savio/requirements-linemodel.txt"

# cache the ImageNet encoder weights now: compute nodes may not reach the internet
python -c "import torchvision; torchvision.models.resnet34(weights='IMAGENET1K_V1'); print('resnet34 weights cached in', __import__('os').environ['TORCH_HOME'])"
python -c "import torch, cv2; print('torch', torch.__version__, 'cv2', cv2.__version__, 'cuda build', torch.version.cuda)"
cd "$PROTOSNAP_DIR" && python -m line_segmentation.paths
echo "== setup done. Submit with: sbatch savio/dryrun.sbatch =="
