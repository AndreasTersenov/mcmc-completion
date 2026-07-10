#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -o jobout/gate3_%j.out
module load python/3.11.5 gcc cuda/12.6
source "$HOME/ics-env/bin/activate"
cd "$HOME/software/mcmc-completion"
python -u scripts/gate3_minizoo.py --attn --aux --shortk
