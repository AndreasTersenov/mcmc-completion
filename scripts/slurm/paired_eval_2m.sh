#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH -o jobout/paired2m_%j.out
module load python/3.11.5 gcc cuda/12.6
source "$HOME/ics-env/bin/activate"
cd "$HOME/software/mcmc-completion"
python -u scripts/paired_eval.py --z128-ckpt "$SCRATCH/ics-zoo/ckpt_2m.pkl" \
  --out paired_eval_2m.json
