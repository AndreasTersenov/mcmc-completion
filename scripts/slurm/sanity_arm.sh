#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH -o jobout/sanity_%x_%j.out
module load python/3.11.5 gcc cuda/12.6
source "$HOME/ics-env/bin/activate"
cd "$HOME/software/mcmc-completion"
ARM="$1"; DATA="$2"; shift 2
python -u scripts/instrument_sanity.py --ckpt "$SCRATCH/ics-zoo/ckpt_${ARM}.pkl" \
  --targets "$SCRATCH/ics-zoo/${DATA}/targets.pkl" --out "results/sanity_${ARM}.json" "$@"
