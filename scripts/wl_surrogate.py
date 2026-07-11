"""Readout C target (c): fit + validate the differentiable WL band-power
surrogate from the CAMB grid (pre-registration: log/2026-07-11-phase1b.md).

Polynomial in scaled coords t = (theta - center)/hw, total degree escalated
4 -> 8 until the held-out gate passes: max over 64 held-out grid points and all
120 band powers of |pred - CAMB| / Knox sigma < 0.01. Final coefficients are
refit on ALL 729 points at the selected degree (validation number reported from
the train/held-out split). Data vector d_obs = C(theta_fid) + Knox noise draw,
seed zlib.crc32(b"wl-bandpower-data"). Writes results/wl_surrogate.npz.
"""

import itertools
import os
import sys
import zlib

import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")


def design(tsc, powers):
    return np.prod(tsc[:, None, :] ** powers[None, :, :], axis=2)  # (n, ncoef)


def main():
    z = np.load(os.path.join(os.environ["SCRATCH"], "ics-wl", "wl_grid.npz"))
    thetas, cls, cov = z["thetas"], z["cls"], z["cov"]
    center, hw = z["center"], z["hw"]
    tsc = (thetas - center) / hw
    sigma = np.sqrt(np.diag(cov))

    rng = np.random.default_rng(zlib.crc32(b"wl-surrogate-split"))
    held = rng.choice(len(thetas), size=64, replace=False)
    mask = np.zeros(len(thetas), bool)
    mask[held] = True

    chosen = None
    for deg in range(4, 9):
        powers = np.array([p for p in itertools.product(range(deg + 1), repeat=3)
                           if sum(p) <= deg])
        X = design(tsc, powers)
        coef, *_ = np.linalg.lstsq(X[~mask], cls[~mask], rcond=None)
        err = np.abs(X[mask] @ coef - cls[mask]) / sigma[None, :]
        max_rel = float(err.max())
        print(f"degree {deg}: ncoef={len(powers)} held-out max |dC|/sigma = "
              f"{max_rel:.2e}", flush=True)
        if max_rel < 0.01:
            chosen = (deg, powers, max_rel)
            break
    if chosen is None:
        print("SURROGATE-GATE-FAIL: no degree <= 8 meets the 1% gate")
        sys.exit(1)

    deg, powers, max_rel = chosen
    X = design(tsc, powers)
    coef, *_ = np.linalg.lstsq(X, cls, rcond=None)  # refit on all points

    L = np.linalg.cholesky(cov)
    eps = np.random.default_rng(zlib.crc32(b"wl-bandpower-data")).standard_normal(
        cov.shape[0])
    d_obs = z["cl_fid"] + L @ eps
    from scipy.linalg import solve_triangular
    chol_w = solve_triangular(L, np.eye(cov.shape[0]), lower=True)

    np.savez(os.path.join(R, "wl_surrogate.npz"), coef=coef, powers=powers,
             center=center, hw=hw, d_obs=d_obs, chol_w=chol_w,
             theta_fid=z["theta_fid"], heldout_max_rel_err=max_rel, degree=deg,
             cov=cov)
    print(f"WL-SURROGATE-DONE degree={deg} heldout_max_rel_err={max_rel:.2e}",
          flush=True)


if __name__ == "__main__":
    main()
