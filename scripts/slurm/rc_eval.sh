#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH -o jobout/rceval_%j.out
# full H100: Readout-C cost accounting uses the same convention as readout_b
module load python/3.11.5 gcc cuda/12.6
source "$HOME/ics-env/bin/activate"
cd "$HOME/software/mcmc-completion"
CKPT="$1"; TAG="$2"
if [ ! -f "$CKPT" ]; then echo "!! checkpoint $CKPT missing"; exit 1; fi
python -u scripts/readout_c.py --eval --ckpt "$CKPT" --tag "$TAG"
