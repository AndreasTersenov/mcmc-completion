#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=1
#SBATCH --time=02:59:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH -o jobout/train_%x_%j.out
# Usage: sbatch -J <arm> scripts/slurm/train_arm.sh <data_subdir> [--nograd]
# arm name doubles as ckpt/status suffix. Resumable: safe to chain with
# --dependency=afterany; exits fast if status says complete.
module load python/3.11.5 gcc cuda/12.6
source "$HOME/ics-env/bin/activate"
cd "$HOME/software/mcmc-completion"
ARM="${SLURM_JOB_NAME}"
DATA="$SCRATCH/ics-zoo/$1"
STATUS="$SCRATCH/ics-zoo/status_${ARM}.json"
if [ -f "$STATUS" ] && python -c "import json,sys;sys.exit(0 if json.load(open('$STATUS'))['complete'] else 1)"; then
  echo "already complete"; exit 0
fi
shift
python -u scripts/train_full.py --data "$DATA" \
  --ckpt "$SCRATCH/ics-zoo/ckpt_${ARM}.pkl" --status "$STATUS" \
  --steps 1600000 --time-budget-sec 9000 "$@"
