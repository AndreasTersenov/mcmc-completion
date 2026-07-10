#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=1
#SBATCH --time=02:59:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH -o jobout/eval_%x_%j.out
# Usage: sbatch -J eval_<arm> scripts/slurm/eval_arm.sh <arm> [--nograd]
module load python/3.11.5 gcc cuda/12.6
source "$HOME/ics-env/bin/activate"
cd "$HOME/software/mcmc-completion"
ARM="$1"; shift
python -u scripts/eval_full.py --ckpt "$SCRATCH/ics-zoo/ckpt_${ARM}.pkl" \
  --evalset "$SCRATCH/ics-zoo/eval/eval_set.pkl" \
  --out "results/eval_${ARM}.json" "$@"
