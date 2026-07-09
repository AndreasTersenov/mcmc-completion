"""M1 — the certificate envelope (PLAN.md FROZEN CORE).

For target p* in {Gaussian, well-separated GMM, funnel} and controlled
proposals q with known mismatch, measure ESS/N and log-Z bias/variance on the
(eps, d) grid, d in {2,4,8,16,32,64}, N = 1e6 per point.

Per grid point:
  - ess_cf   : closed/semi-closed form ESS/N where available (= exp(-D2))
  - ess_emp  : empirical ESS/N at N=1e6 (streaming)
  - N-doubling stability (gate 4): 5e5 vs 1e6, rel tol 0.25 (ESS),
    abs tol 0.05 (logZ); a failed check marks the number as non-reportable
  - relsd_analytic: analytic relative sd of the ESS estimator itself
    (4th-moment criterion; inf when E[w^4] diverges)
  - logz_emp at N=1e6; logz bias/sd across 100 independent chunks at N=1e4
    (true logZ = 0 everywhere by construction: targets are normalized)

Output: results/m1_grid.csv (one row per point). Runtime ~minutes on CPU
(multiprocessing over grid points; keep OMP_NUM_THREADS=1).
"""

import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stage0.is_estimators import (
    is_summary,
    gaussian_scale_ess_frac,
    gaussian_shift_ess_frac,
)
from stage0.stability import n_doubling
from stage0.targets import GMM, Funnel, Gaussian

DS = [2, 4, 8, 16, 32, 64]
N_MAIN = 1_000_000
N_CHUNK, CHUNKS = 10_000, 100
GMM_SEP = 8.0  # mode separation along e1; "well-separated" per PLAN

EPS_SHIFT = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
EPS_SCALE = [-0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.5]
GMM_DELTA = [0.1, 0.2, 0.3, 0.4, 0.45]
EPS_FUNNEL_COND = [-0.3, -0.15, -0.05, 0.05, 0.15, 0.3, 0.5]
EPS_FUNNEL_V = [-0.3, -0.15, 0.15, 0.3, 0.5]


def _relsd_shift(eps, d, n):
    ratio_log = 4.0 * d * eps**2  # log of E w^4 / (E w^2)^2
    if ratio_log > 60:
        return float("inf")
    return float(np.sqrt((np.exp(ratio_log) - 1.0) / n))


def _relsd_scale(eps, d, n):
    s2 = 1.0 + eps
    if s2 <= 0.75:
        return float("inf")
    ew4 = s2**1.5 / np.sqrt(4.0 - 3.0 / s2)
    rho = s2 / np.sqrt(2.0 * s2 - 1.0)
    ratio_log = d * np.log(ew4 / rho**2)
    if ratio_log > 60:
        return float("inf")
    return float(np.sqrt((np.exp(ratio_log) - 1.0) / n))


def _gmm_pair(d, weights_q=None, drop=False):
    mu = np.zeros(d)
    mu[0] = GMM_SEP / 2.0
    means = np.stack([mu, -mu])
    target = GMM(means, np.array([0.5, 0.5]))
    if drop:
        proposal = Gaussian(mean=mu, var=1.0)
    else:
        proposal = GMM(means, np.asarray(weights_q))
    return target, proposal


def _gmm_moment_ratio(p_w, q_w, k):
    # well-separated approximation: E_q[w^k] ~ sum_j p_j^k / q_j^(k-1)
    p_w, q_w = np.asarray(p_w), np.asarray(q_w)
    return float((p_w**k / q_w ** (k - 1)).sum())


def build_cases():
    cases = []
    for d in DS:
        for eps in EPS_SHIFT:
            cases.append(dict(family="gaussian", kind="shift", d=d, eps=eps))
        for eps in EPS_SCALE:
            cases.append(dict(family="gaussian", kind="scale", d=d, eps=eps))
        for delta in GMM_DELTA:
            cases.append(dict(family="gmm", kind="weight_distort", d=d, eps=delta))
        cases.append(dict(family="gmm", kind="mode_drop", d=d, eps=np.nan))
        for eps in EPS_FUNNEL_COND:
            cases.append(dict(family="funnel", kind="cond_scale", d=d, eps=eps))
        for eps in EPS_FUNNEL_V:
            cases.append(dict(family="funnel", kind="v_scale", d=d, eps=eps))
        cases.append(dict(family="funnel", kind="gauss_proposal", d=d, eps=np.nan))
    return cases


def make_point(case):
    """Returns (sample_fn, logw_fn, ess_cf, relsd_fn, note)."""
    d, eps, kind = case["d"], case["eps"], case["kind"]
    if case["family"] == "gaussian":
        target = Gaussian(mean=np.zeros(d), var=1.0)
        if kind == "shift":
            proposal = Gaussian(mean=np.full(d, eps), var=1.0)
            ess_cf = gaussian_shift_ess_frac(eps, d)
            relsd = lambda n: _relsd_shift(eps, d, n)
        else:
            proposal = Gaussian(mean=np.zeros(d), var=1.0 + eps)
            ess_cf = gaussian_scale_ess_frac(eps, d)
            relsd = lambda n: _relsd_scale(eps, d, n)
        note = ""
    elif case["family"] == "gmm":
        if kind == "weight_distort":
            qw = np.array([0.5 + eps, 0.5 - eps])
            target, proposal = _gmm_pair(d, weights_q=qw)
            ew2 = _gmm_moment_ratio([0.5, 0.5], qw, 2)
            ess_cf = 1.0 / ew2
            ratio = _gmm_moment_ratio([0.5, 0.5], qw, 4) / ew2**2
            relsd = lambda n: float(np.sqrt(max(ratio - 1.0, 0.0) / n))
            note = "semi-cf (well-separated approx)"
        else:  # mode_drop
            target, proposal = _gmm_pair(d, drop=True)
            # exact (any d): w = (1 + e^{-2 x.mu})/2 under q=N(mu,I), |mu|=sep/2
            # E w = 1 (unbiased!), E w^2 = (3 + e^{sep^2})/4  => ESS/N ~ 4 e^{-sep^2}
            ess_cf = 4.0 / (3.0 + np.exp(min(GMM_SEP**2, 700.0)))
            relsd = lambda n: float("inf")  # E w^4 ~ e^{6 sep^2}: estimator pure noise
            note = ("BLIND SPOT: E[w]=1 so Zhat unbiased only at astronomical N; "
                    "at practical N logZ locks onto -ln2 with tiny variance; "
                    "empirical ESS is seed-lottery (const-weight plateau vs tail spike)")
    else:  # funnel
        target = Funnel(d, sigma_v=3.0, cond_scale=1.0)
        if kind == "cond_scale":
            proposal = Funnel(d, sigma_v=3.0, cond_scale=float(np.sqrt(1.0 + eps)))
            ess_cf = gaussian_scale_ess_frac(eps, d - 1)
            relsd = lambda n: _relsd_scale(eps, d - 1, n)
            note = "cf exact: v-part matches, mismatch only in d-1 conditionals"
        elif kind == "v_scale":
            proposal = Funnel(d, sigma_v=3.0 * (1.0 + eps), cond_scale=1.0)
            e1 = (1.0 + eps) ** 2 - 1.0
            ess_cf = gaussian_scale_ess_frac(e1, 1)
            relsd = lambda n: _relsd_scale(e1, 1, n)
            note = "1-dim mismatch (v only): d-independent by construction"
        else:  # gauss_proposal (moment-matched diagonal Gaussian)
            var = np.full(d, np.exp(4.5))
            var[0] = 9.0
            proposal = Gaussian(mean=np.zeros(d), var=var)
            ess_cf = np.nan
            relsd = lambda n: float("inf")
            note = "E[w^2]=inf for ANY Gaussian proposal vs funnel: certificate undefined"

    def sample_fn(rng, m):
        return proposal.sample(rng, m)

    def logw_fn(x):
        return target.logpdf_u(x) - proposal.logpdf(x)

    return sample_fn, logw_fn, ess_cf, relsd, note


def run_point(args):
    idx, case = args
    seed = 100_000 + 137 * idx
    sample_fn, logw_fn, ess_cf, relsd, note = make_point(case)
    t0 = time.time()

    cache = {}

    def summary(n, s):
        if (n, s) not in cache:
            cache[(n, s)] = is_summary(sample_fn, logw_fn, n=n, batch=100_000, seed=s)
        return cache[(n, s)]

    main = summary(N_MAIN, seed)
    dbl_ess = n_doubling(lambda n, s: summary(n, s)["ess_frac"],
                         n=N_MAIN // 2, tol=0.25, rel=True, seed=seed + 1)
    dbl_logz = n_doubling(lambda n, s: summary(n, s)["logz"],
                          n=N_MAIN // 2, tol=0.05, rel=False, seed=seed + 1)
    chunk_logz = np.array(
        [summary(N_CHUNK, seed + 1000 + i)["logz"] for i in range(CHUNKS)]
    )
    d2_cf = -np.log(ess_cf) if np.isfinite(ess_cf) and ess_cf > 0 else np.nan
    row = dict(
        family=case["family"], kind=case["kind"], d=case["d"], eps=case["eps"],
        ess_cf=ess_cf, d2_cf=d2_cf,
        ess_emp=main["ess_frac"], ess_emp_half=dbl_ess["v_n"],
        ess_dbl_pass=dbl_ess["passed"],
        relsd_analytic=relsd(N_MAIN),
        logz_emp=main["logz"], logz_dbl_pass=dbl_logz["passed"],
        logz_bias_1e4=float(chunk_logz.mean()), logz_sd_1e4=float(chunk_logz.std(ddof=1)),
        seconds=round(time.time() - t0, 2), note=note,
    )
    return row


def main():
    cases = build_cases()
    print(f"M1: {len(cases)} grid points, N={N_MAIN} per point")
    rows = []
    with Pool(12) as pool:
        for i, row in enumerate(pool.imap(run_point, list(enumerate(cases)))):
            rows.append(row)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(cases)} done", flush=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "m1_grid.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
