#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH -o jobout/eval_untrained_%j.out
module load python/3.11.5 gcc cuda/12.6
source "$HOME/ics-env/bin/activate"
cd "$HOME/software/mcmc-completion"
python -u scripts/eval_full.py --ckpt "$SCRATCH/ics-zoo/ckpt_untrained.pkl" \
  --evalset "$SCRATCH/ics-zoo/eval/eval_set.pkl" --out results/eval_untrained.json
