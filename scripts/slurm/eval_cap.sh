#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH -o jobout/evalcap_%j.out
module load python/3.11.5 gcc cuda/12.6
source "$HOME/ics-env/bin/activate"
cd "$HOME/software/mcmc-completion"
STATUS="$SCRATCH/ics-zoo/status_cap.json"
if ! python -c "import json,sys;sys.exit(0 if json.load(open('$STATUS'))['complete'] else 1)" 2>/dev/null; then
  echo "!! capacity chain not complete — resubmit after it finishes"; exit 1
fi
python -u scripts/eval_curve.py --wide --out eval_cap.json --refs-from eval_curve.json \
  --ckpts "cap0.25M=$SCRATCH/ics-zoo/ckpt_cap_step250000.pkl,cap0.5M=$SCRATCH/ics-zoo/ckpt_cap_step500000.pkl,cap1M=$SCRATCH/ics-zoo/ckpt_cap_step1000000.pkl"
