#!/bin/bash
#SBATCH --account=def-lplevass
#SBATCH --time=02:59:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH -o jobout/wlgrid_%j.out
# CPU job; runs with eft-sbi's OWN .venv (camb lives there); that repo is
# read-only for us — the CAMB cache goes to $SCRATCH/ics-wl.
module load python/3.11.5 gcc
cd "$HOME/software/mcmc-completion"
"$HOME/software/eft-sbi/.venv/bin/python" -u scripts/wl_grid.py
