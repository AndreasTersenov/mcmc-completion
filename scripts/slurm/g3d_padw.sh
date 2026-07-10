#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH -o jobout/g3d_padw_%j.out
module load python/3.11.5 gcc cuda/12.6
source "$HOME/ics-env/bin/activate"
cd "$HOME/software/mcmc-completion"
python -u scripts/gate3_minizoo.py --attn --aux --pad-weight 0.1 --criteria p1 --out gate3_padw.json
