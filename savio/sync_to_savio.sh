#!/bin/bash
# Run LOCALLY (WSL) from the repo root: copies the code and only the data the line model needs (~3.6 GB) to Savio scratch.
#   bash savio/sync_to_savio.sh            # user defaults to edwardlin; pass another as $1
# Uses the data-transfer node. One SSH connection is opened first (PIN+OTP once) and every rsync reuses it
# (ControlMaster), so a fresh one-time code is never needed mid-run. Safe to rerun: rsync only sends what is missing.
set -euo pipefail
U="${1:-edwardlin}"
command -v rsync >/dev/null || { echo "rsync is not installed in this WSL: sudo apt install rsync"; exit 1; }
HOST="dtn.brc.berkeley.edu"
DST="$U@$HOST:/global/scratch/users/$U/ProtoSnap"
SOCK="$HOME/.ssh/ctl-savio-%r@%h:%p"; mkdir -p ~/.ssh; chmod 700 ~/.ssh
SSH="ssh -o ControlMaster=auto -o ControlPath=$SOCK -o ControlPersist=2h"
$SSH -fN "$U@$HOST"                       # master connection: enter PIN+OTP here, once
trap '$SSH -O exit "$U@$HOST" 2>/dev/null || true' EXIT
R() { rsync -avhm --progress --relative -e "$SSH" "$@"; }

# code
R --exclude '__pycache__' --exclude '.venv' --exclude '*.pyc' \
   ./line_cropping ./savio ./requirements.txt ./README.md "$DST/"
# eBL faces (with line_heights.csv), ground truth, split, manifest; fragment.json for make_crops' ATF labels, annotations.json for box_containment
R ./tablet_data/ebl_tablets/faces ./tablet_data/ebl_tablets/ground_truth \
   ./tablet_data/ebl_tablets/split.csv ./tablet_data/ebl_tablets/manifest.csv "$DST/"
R --include '*/' --include 'fragment.json' --include 'annotations.json' --exclude '*' \
   ./tablet_data/ebl_tablets/NA ./tablet_data/ebl_tablets/NB ./tablet_data/ebl_tablets/OA ./tablet_data/ebl_tablets/OB "$DST/"
# Old Assyrian prepared faces + hand labels, Heidelberg faces + annotations
R ./tablet_data/fat-cross_processed/prepared ./tablet_data/fat-cross_processed/ground_truth \
   ./tablet_data/heidelberg/faces ./data/heidelberg/*.csv "$DST/"
# the images of the hand-labelled Old Assyrian faces (88 files; the rest of that 8 GB folder is not used)
LIST="$(mktemp)"; trap 'rm -f "$LIST"; $SSH -O exit "$U@$HOST" 2>/dev/null || true' EXIT
OA=tablet_data/fat-cross_processed
awk -F, 'NR==FNR { if (FNR==1) { for (i=1;i<=NF;i++) { if ($i=="face_id") c=i; if ($i=="status") t=i } } else if ($t=="done") ok[$c]=1; next }
         FNR==1 { for (i=1;i<=NF;i++) { if ($i=="face") a=i; if ($i=="file") b=i } next } ($a in ok) { print $b }' \
    "$OA/ground_truth/labels.csv" "$OA/prepared/faces_accepted.csv" > "$LIST"
R --files-from="$LIST" ./ "$DST/"
# existing trained weights, for inference without retraining
R ./weights/linemodel/run3_full_s0/model.pt ./weights/linemodel/run3_full_s0/config.json "$DST/"
echo "synced to $DST"
