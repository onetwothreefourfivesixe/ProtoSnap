# Running the line model on Savio

These scripts train and run `line_cropping/linemodel` (the U-Net tablet line segmenter) on Berkeley's Savio cluster.
The Savio docs are at https://datadisco.cdss.berkeley.edu/ (jobs and software) and
https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/ (GPU partitions and QoS).

| file | what |
|---|---|
| `savio_env.sh` | paths + modules, sourced by everything |
| `sync_to_savio.sh` | run locally: rsync code + the ~3.6 GB of data the model needs to scratch |
| `setup_env.sh` | run once on a login node: conda env on scratch, pip packages, cache ResNet-34 weights |
| `requirements-linemodel.txt` | the subset of `requirements.txt` the line model needs (headless OpenCV) |
| `check_data.py` | run on Savio before submitting: every file training and scoring need is there |
| `dryrun.sbatch` | 15-minute GPU smoke test (`train.py --dry-run --heights`) |
| `train.sbatch` | run 4: the `run3_full` recipe with the height channels; `RUN=` / `SEED=` / `EPOCHS=` env overrides |
| `infer.sbatch` | test-split inference, line scores, and crops with predicted heights against the best fixed height |

## Run 4: steps

Run 4 adds two output channels that predict how far each line's signs reach above and below it, so that crop height is
set per line instead of by one fixed extension (`docs/line_model/line_model.md`, Appendix A.10). It uses two seeds, as in run 3.

```bash
# 0. local, from the repo root: copy everything over (PIN+OTP once; needs rsync: sudo apt install rsync). Also rerun after any code change.
bash savio/sync_to_savio.sh

# 1. on Savio (login node)
ssh edwardlin@hpc.brc.berkeley.edu
cd /global/scratch/users/edwardlin/ProtoSnap
bash savio/setup_env.sh            # first time only; prints your account/partition/QoS table: see 'QoS' below
source savio/savio_env.sh; source activate "$CONDA_ENV_DIR"
python -m savio.check_data         # must end with 'data OK' (556 training faces)
sacctmgr show qos a5k_gpu4_ica format=Name,MaxWall   # must be at least the 2 h train.sbatch asks for

# 2. submit, always from the repo root (logs land in savio/logs/)
sbatch savio/dryrun.sbatch         # wait for it to succeed; its log shows "height" in the loss line
T0=$(RUN=run4_savio_s0 SEED=0 sbatch --parsable savio/train.sbatch)
T1=$(RUN=run4_savio_s1 SEED=1 sbatch --parsable savio/train.sbatch)
RUN=run4_savio_s0 sbatch --dependency=afterok:$T0 savio/infer.sbatch
RUN=run4_savio_s1 sbatch --dependency=afterok:$T1 savio/infer.sbatch
squeue -u $USER                    # watch; sacct -j <id> --format=JobID,Elapsed,MaxRSS,State afterwards

# 3. copy results back (local); then scratch can be cleaned
for R in run4_savio_s0 run4_savio_s1; do
  rsync -avh edwardlin@dtn.brc.berkeley.edu:/global/scratch/users/edwardlin/ProtoSnap/weights/linemodel/$R weights/linemodel/
  rsync -avh edwardlin@dtn.brc.berkeley.edu:/global/scratch/users/edwardlin/ProtoSnap/tablet_data/ebl_tablets/linemodel/compare/$R tablet_data/ebl_tablets/linemodel/compare/
  rsync -avh edwardlin@dtn.brc.berkeley.edu:/global/scratch/users/edwardlin/ProtoSnap/tablet_data/ebl_tablets/linemodel/compare_crops/$R tablet_data/ebl_tablets/linemodel/compare_crops/
done
```

Each training job uses 4 cores on `savio4_gpu` at 4.67 SU per core-hour, so at most about 37 SU for its 2 h limit.

## QoS (ic_cdss170fall)

`sacctmgr -p show associations user=$USER format=account,partition,qos` shows that the account's GPU QoS names end in
`_ica` (the FCA ones end in `_normal`). They are `a5k_gpu4_ica` on `savio4_gpu`, and `a40_gpu3_ica`, `gtx2080_gpu3_ica`
and `v100_gpu3_ica` on `savio3_gpu`. `savio_lowprio` and `savio_debug` are not available on the GPU partitions.

## GPU choice

| partition | `--gres` | `--qos` | `--cpus-per-task` per GPU | notes |
|---|---|---|---|---|
| savio4_gpu | `gpu:A5000:1` | `a5k_gpu4_ica` | 4 | 24 GB: default here |
| savio3_gpu | `gpu:A40:1` | `a40_gpu3_ica` | 8 | 48 GB |
| savio3_gpu | `gpu:GTX2080TI:1` | `gtx2080_gpu3_ica` | 2 | 11 GB, fine for inference |
| savio3_gpu | `gpu:V100:1` | `v100_gpu3_ica` | 4 or 8 (by node) | 32 GB, 4 GPUs total |

TITAN, L40 and H200 GPUs need `savio_lowprio`, which this account does not have. A job that asks
for fewer CPUs per GPU than the ratio above is rejected by the scheduler. The `_ica` GPU QoS cap wall time per job below the
general 72 h. `sacctmgr show qos a5k_gpu4_ica format=Name,MaxWall` prints the cap, and a longer `--time` leaves the job
pending with reason QOSMaxWallDurationPerJobLimit. The 40-epoch run took 54 min locally, so `train.sbatch` asks for 2 h.

## Interactive debugging

```bash
srun --pty -A ic_cdss170fall -p savio4_gpu -q a5k_gpu4_ica --gres=gpu:A5000:1 --cpus-per-task=4 -t 01:00:00 bash -i
source savio/savio_env.sh; source activate "$CONDA_ENV_DIR"
python -m line_cropping.linemodel.train --dry-run --heidelberg --heights --out weights/linemodel/tmp
```

Scratch is not backed up, and Research IT asks users to delete intermediate files when a job ends. A file-count overload made scratch unavailable from 23 Jan to 10 Feb 2026, and a purge policy is planned. Copy weights back, then delete what you no longer need. Do not set `CUDA_VISIBLE_DEVICES`
yourself. The ProtoSnap stages (`main.py`, diffusers) are not covered by this environment. Install the full
`requirements.txt` into the same env if you need them.
