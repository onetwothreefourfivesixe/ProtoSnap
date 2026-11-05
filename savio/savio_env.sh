# Sourced by every job script and by setup_env.sh.
# Login: ssh edwardlin@hpc.brc.berkeley.edu. The Slurm --account (ic_cdss170fall, not the username) is set in the
# three *.sbatch files; find it with: sacctmgr -p show associations user=$USER format=account,partition,qos
export SAVIO_USER="edwardlin"
export SAVIO_SCRATCH="/global/scratch/users/$USER"
export PROTOSNAP_DIR="$SAVIO_SCRATCH/ProtoSnap"        # repo + data live on scratch (home dir quota is small)
export CONDA_ENV_DIR="$SAVIO_SCRATCH/envs/protosnap"   # conda env on scratch, not in ~/.conda
export TORCH_HOME="$SAVIO_SCRATCH/torch_cache"         # ImageNet ResNet-34 weights cached here (downloaded once in setup_env.sh)
export MPLBACKEND=Agg

module purge
module load anaconda3/2024.02-1-11.4   # the version the Savio docs name for conda environments; 'module avail anaconda3' lists others
