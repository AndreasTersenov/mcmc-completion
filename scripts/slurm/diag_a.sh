#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=00:45:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH -o jobout/diag_a_%j.out
module load python/3.11.5 gcc cuda/12.6
source "$HOME/ics-env/bin/activate"
cd "$HOME/software/mcmc-completion"
python -u scripts/gate2_context_pair.py --attn 2 --aux --out gate2_newarch.json
