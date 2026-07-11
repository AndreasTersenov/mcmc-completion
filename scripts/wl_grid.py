"""Readout C target (c): CAMB band-power grid for the WL surrogate.

Runs with the eft-sbi repo's OWN .venv (camb lives there); that repo is used
READ-ONLY — the ClEngine disk cache is redirected to our scratch. Restarting is
cheap: every CAMB P(k,z) call is disk-cached, so a rerun skips finished points.

Box (pre-registered, log/2026-07-11-phase1b.md): center (0.27, 0.83, 0.94),
halfwidths (0.12, 0.15, 0.08), 9 points per axis = 729 CAMB calls.
Writes $SCRATCH/ics-wl/wl_grid.npz (thetas, cls (729,120), cl_fid, cov,
center, hw, theta_fid).
"""

import os
import sys
import time

import numpy as np

EFT = os.path.expanduser("~/software/eft-sbi")
sys.path.insert(0, EFT)

from phase05.cls import FIDUCIAL, ClEngine, flatten_band_major
from phase05.survey import Survey, knox_covariance, shape_noise

CENTER = np.array([0.27, 0.83, 0.94])
HW = np.array([0.12, 0.15, 0.08])
NPTS = 9

out_dir = os.path.join(os.environ["SCRATCH"], "ics-wl")
os.makedirs(out_dir, exist_ok=True)

sv = Survey()
eng = ClEngine(sv, cache_dir=os.path.join(out_dir, "cls_cache"))

axes = [np.linspace(c - h, c + h, NPTS) for c, h in zip(CENTER, HW)]
thetas = np.array([[om, s8, ns] for om in axes[0] for s8 in axes[1]
                   for ns in axes[2]])

t0 = time.time()
cl_fid_pairs = eng.cl_bands(FIDUCIAL)
cov = knox_covariance(cl_fid_pairs, shape_noise(sv), sv)
cl_fid = flatten_band_major(cl_fid_pairs)
print(f"fiducial + Knox cov done [{time.time()-t0:.0f}s]", flush=True)

cls = np.empty((len(thetas), cl_fid.size))
for i, (om, s8, ns) in enumerate(thetas):
    cls[i] = flatten_band_major(eng.cl_bands({"Om": om, "s8": s8, "ns": ns}))
    if (i + 1) % 27 == 0:
        print(f"[{i+1}/{len(thetas)}] {time.time()-t0:.0f}s", flush=True)

np.savez(os.path.join(out_dir, "wl_grid.npz"), thetas=thetas, cls=cls,
         cl_fid=cl_fid, cov=cov, center=CENTER, hw=HW,
         theta_fid=np.array([FIDUCIAL["Om"], FIDUCIAL["s8"], FIDUCIAL["ns"]]))
print(f"WL-GRID-DONE [{time.time()-t0:.0f}s]", flush=True)
