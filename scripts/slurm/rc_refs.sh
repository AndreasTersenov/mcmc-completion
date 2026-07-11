#!/bin/bash
#SBATCH --account=def-lplevass
#SBATCH --time=02:59:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH -o jobout/rcrefs_%j.out
# CPU job: fit+validate the WL surrogate, then build all NUTS references
# (closed-form Gaussian validation gate runs first inside readout_c.py).
module load python/3.11.5 gcc
source "$HOME/ics-env/bin/activate"
cd "$HOME/software/mcmc-completion"
export JAX_PLATFORMS=cpu
python -u scripts/wl_surrogate.py
python -u scripts/readout_c.py --build-refs
