"""M2 — the in-context oracle (PLAN.md FROZEN CORE).

Zoo v0 families (stage0.m2_families) with explicit uniform box priors.
Contexts: short MALA chains (4 chains, K/4 steps each) from overdispersed
inits (N(0, 5^2 I)), K in {8,32,128,512}, temperature T in {1,2,5}; the
recorded (E, gradE) are the EXACT untempered values at visited points.
Posterior p(theta|C): dense two-stage grid (theta-dim 2) or adaptive SMC
(gmm8, theta-dim 8), through the centered-energy pseudo-likelihood
(sigma_E = sigma_G = 0.05; the contraction FLOOR is set by this choice — the
with/without-gradient comparison is the pre-registered target, not the floor).

Measured:
 (a) posterior contraction vs K, WITH and WITHOUT gradient values
     -> results/m2_contraction.csv   (grid: res 121 primary, 241 doubling;
        SMC: 1024 particles primary, 2048 doubling)
 (b) sliced-W2^2(q, p*) vs K (T=1, grads on), q = posterior predictive
     -> results/m2_sw2.csv           (n=4096 primary, 8192 doubling)
 (c) family mismatch (context family A, zoo family B) + in-family controls:
     overconfidence, SW2(q, p*_true), and the true-energy SNIS certificate
     -> results/m2_mismatch.csv      (n=16384 primary, 32768 doubling)
"""

import csv
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stage0.is_estimators import ess_frac_from_logw, logz_from_logw
from stage0.m2_families import FAMILIES, centered_energy_loglik
from stage0.mala import mala_chains
from stage0.posterior import grid_posterior, smc_posterior
from stage0.sliced_w2 import sliced_w2_squared
from stage0.utils import logsumexp

KS = [8, 32, 128, 512]
TS = [1, 2, 5]
REPS = 6
SIG_E = SIG_G = 0.05
MALA_STEP = {"gmm2": 0.3, "funnel2": 0.05, "dwell2": 0.05, "gmm8": 0.3}
INIT_SCALE = {"gmm2": 5.0, "funnel2": 5.0, "dwell2": 2.0, "gmm8": 5.0}


def dseed(*parts):
    """Deterministic (unsalted) seed from the case labels."""
    return zlib.crc32("|".join(map(str, parts)).encode())
GRID_FAMS = ["gmm2", "funnel2", "dwell2"]
RES, RES_DBL = 121, 241
NPART, NPART_DBL = 1024, 2048
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def chunked(fn, block=8192):
    def g(theta):
        return np.concatenate([fn(theta[i:i + block]) for i in range(0, len(theta), block)])
    return g


def gen_context(fam, theta_star, K, T, rng):
    n_chains = 4
    steps = K // n_chains
    target = fam.target(theta_star)
    x0 = rng.standard_normal((n_chains, fam.d)) * INIT_SCALE[fam.name]
    xs, acc = mala_chains(target, x0, steps, MALA_STEP[fam.name] * np.sqrt(T),
                          temperature=T, rng=rng)
    pts = xs.reshape(-1, fam.d)
    E, G = fam.energy(theta_star[None, :], pts)
    return pts, E[0], G[0], acc


def make_loglik(fam, pts, E, G, use_grads):
    # RELATIVE readout precision: 5% of the context's own energy/gradient
    # spread (+ floor for degenerate contexts). An absolute sigma is broken
    # for heavy-scale families (funnel neck energies are O(10^2-10^3): the
    # likelihood becomes a spike thinner than any grid cell and the argmax
    # aliases onto wrong theta) — found and fixed 2026-07-09, see log.
    sig_e = SIG_E * float(np.std(E)) + 1e-2
    sig_g = SIG_G * float(np.std(G)) + 1e-2

    def ll(theta):
        return centered_energy_loglik(fam, theta, pts, E, sig_e,
                                      grads=G if use_grads else None, sigma_g=sig_g)
    return chunked(ll)


def two_stage_grid(fam, loglik_fn, res):
    """Coarse grid over the prior box, then a fine same-res grid over the
    logpost > max-20 bounding box (+2 cells margin) — keeps contraction
    measurements resolution-safe when the posterior is much narrower than
    the prior box."""
    axes = [np.linspace(lo, hi, res) for lo, hi in zip(fam.prior_lo, fam.prior_hi)]
    coarse = grid_posterior(axes, loglik_fn)
    lp = coarse["logpost"]
    mask = lp > lp.max() - 20.0
    fine_axes = []
    for i in range(fam.theta_dim):
        other = tuple(j for j in range(fam.theta_dim) if j != i)
        idx = np.where(mask.any(axis=other) if other else mask)[0]
        lo = axes[i][max(idx.min() - 2, 0)]
        hi = axes[i][min(idx.max() + 2, res - 1)]
        fine_axes.append(np.linspace(lo, hi, res))
    return grid_posterior(fine_axes, loglik_fn)


def contraction_metrics(fam, post, theta_star):
    ratio = float(np.mean(post["std"] / fam.prior_std))
    err = float(np.linalg.norm(post["mean"] - theta_star))
    err_norm = float(np.linalg.norm((post["mean"] - theta_star) / fam.prior_std)
                     / np.sqrt(fam.theta_dim))
    return ratio, err, err_norm


def predictive_sample(fam, post, n, rng, n_atoms=1024):
    idx = rng.choice(len(post["theta"]), size=n_atoms, p=post["w"])
    atoms = post["theta"][idx]
    counts = np.bincount(rng.choice(n_atoms, size=n), minlength=n_atoms)
    xs = [fam.target(atoms[j]).sample(rng, counts[j])
          for j in range(n_atoms) if counts[j] > 0]
    return np.vstack(xs), atoms


def predictive_logpdf(fam, atoms, x, block=2048):
    lz = fam.log_z(atoms)
    out = []
    for i in range(0, len(x), block):
        E, _ = fam.energy(atoms, x[i:i + block])
        out.append(logsumexp(-E - lz[:, None], axis=0) - np.log(len(atoms)))
    return np.concatenate(out)


# ---------------- part A/C worker: grid families ----------------

def run_grid_case(args):
    fam_name, K, T, rep = args
    fam = FAMILIES[fam_name]
    seed = dseed("grid", fam_name, K, T, rep)
    rng = np.random.default_rng(seed)
    theta_star = fam.prior_sample(rng, 1)[0]
    pts, E, G, acc = gen_context(fam, theta_star, K, T, rng)
    con_rows, sw2_row = [], None
    posts = {}
    for use_grads in (False, True):
        ll = make_loglik(fam, pts, E, G, use_grads)
        p1 = two_stage_grid(fam, ll, RES)
        p2 = two_stage_grid(fam, ll, RES_DBL)
        posts[use_grads] = p1
        r1, err, err_n = contraction_metrics(fam, p1, theta_star)
        r2, _, _ = contraction_metrics(fam, p2, theta_star)
        con_rows.append(dict(
            family=fam_name, engine="grid", K=K, T=T, grads=use_grads, rep=rep,
            contract_ratio=r1, contract_ratio_dbl=r2,
            dbl_pass=bool(abs(r2 - r1) <= 0.05),
            theta_err=err, theta_err_norm=err_n, accept=round(acc, 3),
        ))
    if T == 1:
        sw2_rng = np.random.default_rng(seed + 1)
        target = fam.target(theta_star)

        def sw2_at(n, s):
            r = np.random.default_rng(s)
            qx, _ = predictive_sample(fam, posts[True], n, r)
            px = target.sample(r, n)
            return sliced_w2_squared(qx, px, n_proj=128, rng=r)

        v1, v2 = sw2_at(4096, seed + 2), sw2_at(8192, seed + 3)
        base = sliced_w2_squared(target.sample(sw2_rng, 4096),
                                 target.sample(sw2_rng, 4096),
                                 n_proj=128, rng=sw2_rng)
        sw2_row = dict(family=fam_name, K=K, T=T, rep=rep,
                       sw2=v1, sw2_dbl=v2,
                       dbl_pass=bool(abs(v2 - v1) <= max(0.02, 0.15 * v1)),
                       sw2_baseline=base)
    return con_rows, sw2_row


# ---------------- part B worker: gmm8 via SMC ----------------

def run_gmm8_case(args):
    K, rep, use_grads = args
    fam = FAMILIES["gmm8"]
    seed = dseed("gmm8", K, rep)  # context shared across grads settings
    rng = np.random.default_rng(seed)
    theta_star = fam.prior_sample(rng, 1)[0]
    pts, E, G, acc = gen_context(fam, theta_star, K, 1, rng)
    ll = make_loglik(fam, pts, E, G, use_grads)
    rows = []
    out = {}
    for npart in (NPART, NPART_DBL):
        res = smc_posterior(fam.prior_sample, fam.logprior, ll, n_particles=npart,
                            rng=np.random.default_rng(seed + npart), n_mcmc=8)
        out[npart] = contraction_metrics(fam, res, theta_star)
    r1, err, err_n = out[NPART]
    r2, _, _ = out[NPART_DBL]
    rows.append(dict(
        family="gmm8", engine="smc", K=K, T=1, grads=use_grads, rep=rep,
        contract_ratio=r1, contract_ratio_dbl=r2,
        dbl_pass=bool(abs(r2 - r1) <= 0.1),
        theta_err=err, theta_err_norm=err_n, accept=round(acc, 3),
    ))
    return rows, None


# ---------------- part D: family mismatch ----------------

MISMATCH_PAIRS = [
    ("funnel2", "gmm2"), ("dwell2", "gmm2"), ("gmm2", "funnel2"),
    ("gmm2", "gmm2"), ("funnel2", "funnel2"),  # in-family controls
]


def run_mismatch_case(args):
    ctx_name, zoo_name, rep = args
    K, T = 128, 1
    ctx_fam, zoo_fam = FAMILIES[ctx_name], FAMILIES[zoo_name]
    seed = dseed("mismatch", ctx_name, zoo_name, rep)
    rng = np.random.default_rng(seed)
    theta_star = ctx_fam.prior_sample(rng, 1)[0]
    ctx_target = ctx_fam.target(theta_star)
    pts, E, G, acc = gen_context(ctx_fam, theta_star, K, T, rng)
    ll = make_loglik(zoo_fam, pts, E, G, True)
    post = two_stage_grid(zoo_fam, ll, RES)
    ratio = float(np.mean(post["std"] / zoo_fam.prior_std))

    def snis(n, s):
        r = np.random.default_rng(s)
        qx, atoms = predictive_sample(zoo_fam, post, n, r)
        logw = ctx_target.logpdf_u(qx) - predictive_logpdf(zoo_fam, atoms, qx)
        return ess_frac_from_logw(logw), logz_from_logw(logw), qx

    ess1, logz1, qx = snis(16384, seed + 1)
    ess2, logz2, _ = snis(32768, seed + 2)
    r2 = np.random.default_rng(seed + 3)
    sw2 = sliced_w2_squared(qx, ctx_target.sample(r2, 16384), n_proj=128, rng=r2)
    return dict(
        ctx_family=ctx_name, zoo_family=zoo_name, K=K, T=T, rep=rep,
        contract_ratio=ratio, map_1=post["map"][0], map_2=post["map"][1],
        sw2_q_true=sw2,
        ess_frac=ess1, ess_frac_2n=ess2,
        # a stable ESS/N is N-invariant; >1.6x movement on doubling flags
        # divergent weights (E[w^2]=inf regime) — the "catches it" signal
        ess_dbl_pass=bool(abs(np.log10(max(ess2, 1e-300)) - np.log10(max(ess1, 1e-300))) < 0.2),
        logz_err=logz1 - ctx_target.log_Z, logz_err_2n=logz2 - ctx_target.log_Z,
        accept=round(acc, 3),
    )


def main():
    t0 = time.time()
    grid_cases = [(f, K, T, r) for f in GRID_FAMS for K in KS for T in TS
                  for r in range(REPS)]
    gmm8_cases = [(K, r, g) for K in KS for r in range(REPS) for g in (False, True)]
    mis_cases = [(a, b, r) for (a, b) in MISMATCH_PAIRS for r in range(4)]
    print(f"M2: {len(grid_cases)} grid contexts, {len(gmm8_cases)} SMC runs, "
          f"{len(mis_cases)} mismatch rows")

    con_rows, sw2_rows, mis_rows = [], [], []
    with Pool(12) as pool:
        for i, (cr, sr) in enumerate(pool.imap(run_grid_case, grid_cases)):
            con_rows += cr
            if sr:
                sw2_rows.append(sr)
            if (i + 1) % 24 == 0:
                print(f"  grid {i + 1}/{len(grid_cases)}", flush=True)
        for i, (cr, _) in enumerate(pool.imap(run_gmm8_case, gmm8_cases)):
            con_rows += cr
            if (i + 1) % 12 == 0:
                print(f"  gmm8 {i + 1}/{len(gmm8_cases)}", flush=True)
        for row in pool.imap(run_mismatch_case, mis_cases):
            mis_rows.append(row)
    print(f"  mismatch done; total {time.time() - t0:.0f}s")

    for name, rows in [("m2_contraction.csv", con_rows), ("m2_sw2.csv", sw2_rows),
                       ("m2_mismatch.csv", mis_rows)]:
        with open(os.path.join(RESULTS, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote results/{name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
