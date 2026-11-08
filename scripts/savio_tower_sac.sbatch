#!/bin/bash
# End-to-end tower policy, SAC with the script's episodes in the replay buffer (scripts/train_tower_sac.py), on Savio.
# From the repo root on Savio:  sbatch scripts/savio_tower_sac.sbatch
# Continue from a checkpoint, at a given level:
#   sbatch scripts/savio_tower_sac.sbatch tower_sac_<job>/checkpoints/sac_5000000_steps.zip 4
# Output: /global/scratch/users/$USER/tower_sac_<job id>
#SBATCH --job-name=tower_sac
#SBATCH --account=ic_piml
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=14:00:00
#SBATCH --output=tower_sac_%j.log

set -e
cd "$SLURM_SUBMIT_DIR"
module load python 2>/dev/null || true
VENV=/global/scratch/users/$USER/pensig-venv
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q -e . tensorboard
fi
[ -d assets/mujoco_menagerie ] || "$VENV/bin/python" scripts/fetch_assets.py

RUNS=/global/scratch/users/$USER
RESUME=()
[ -n "$1" ] && RESUME=(--resume "$RUNS/$1")
[ -n "$2" ] && RESUME+=(--level "$2")
"$VENV/bin/python" -u scripts/train_tower_sac.py --out "$RUNS/tower_sac_$SLURM_JOB_ID" "${RESUME[@]}" \
    --timesteps 30000000 --max-minutes 780 --n-envs 32 --threads 8 --eval-towers 100
