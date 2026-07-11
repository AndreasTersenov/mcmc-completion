#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=1
#SBATCH --time=02:59:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH -o jobout/tcap_%j.out
module load python/3.11.5 gcc cuda/12.6
source "$HOME/ics-env/bin/activate"
cd "$HOME/software/mcmc-completion"
STATUS="$SCRATCH/ics-zoo/status_cap.json"
if [ -f "$STATUS" ] && python -c "import json,sys;sys.exit(0 if json.load(open('$STATUS'))['complete'] else 1)"; then
  echo "already complete"; exit 0
fi
python -u scripts/train128_cap.py --ckpt "$SCRATCH/ics-zoo/ckpt_cap.pkl" \
  --status "$STATUS" --time-budget-sec 9000
