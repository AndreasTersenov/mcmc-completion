#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH -o jobout/g3d_1a_reeval_%j.out
module load python/3.11.5 gcc cuda/12.6
source "$HOME/ics-env/bin/activate"
cd "$HOME/software/mcmc-completion"
python -u scripts/gate3_minizoo.py --attn --aux --donehot --criteria p1 \
  --eval-only --ckpt-in results/gate3_d1_params.pkl --out gate3_d1_fixed.json
