#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=1
#SBATCH --time=02:59:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH -o jobout/evalcurve_%j.out
module load python/3.11.5 gcc cuda/12.6
source "$HOME/ics-env/bin/activate"
cd "$HOME/software/mcmc-completion"
STATUS="$SCRATCH/ics-zoo/status_2m.json"
if ! python -c "import json,sys;sys.exit(0 if json.load(open('$STATUS'))['complete'] else 1)" 2>/dev/null; then
  echo "!! 2M CHAIN NOT COMPLETE ($STATUS) — refusing to run a partial curve; resubmit after the chain finishes"
  exit 1
fi
python -u scripts/eval_curve.py
