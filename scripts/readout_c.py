"""Phase 1b Readout C: the usefulness barometer (NON-GATING; pre-registration
and accounting: log/2026-07-11-phase1b.md).

--build-refs (CPU): NUTS references for the three real targets.
  Validation gates FIRST (CLAUDE.md rule): (1) the whole NUTS+R-hat pipeline on
  a closed-form 3-d correlated Gaussian (moments < 2% of sd, R-hat < 1.01);
  (2) the banana reference is cross-checked against the target's EXACT
  transform sampler. Every reference requires split-R-hat < 1.01 on all dims.
  Saves $SCRATCH/ics-refs/<name>.npz (draws, rhat, seconds).

--eval --ckpt PATH --tag TAG (GPU): zero-shot evaluation of a checkpoint on the
  three targets, standard context protocol (4 MALA chains, K=128, T in {1,5},
  aux tokens), certificate + SW2 vs reference + all-in cost accounting (same
  convention as readout_b). Writes results/readout_c_<tag>.json.

ICS_SMOKE=1 shrinks draw counts for a CPU path test only.
"""

import argparse
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import blackjax

from ics.context import generate_context
from ics.models import ICSModel
from ics.real import (WLBandpower, eight_schools_logpdf, gym_banana_logpdf,
                      gym_banana_sample, ics_evaluate_fn, split_rhat)
from ics.train import load_checkpoint
from stage0.sliced_w2 import sliced_w2_squared

R = os.path.join(os.path.dirname(__file__), "..", "results")
SMOKE = os.environ.get("ICS_SMOKE") == "1"
REFS = (os.path.join(os.environ.get("TMPDIR", "/tmp"), "ics-refs-smoke") if SMOKE
        else os.path.join(os.environ.get("SCRATCH", "/tmp"), "ics-refs"))
N_WARM = 200 if SMOKE else 2000
# per-target draws AFTER the 2026-07-11 post-mortem: 4x20k under-converged the
# banana (IACT O(100+) along the arch; variance swung -15%/+13% across draw
# counts). Banana's SW2 reference is now its EXACT transform sampler; NUTS is
# kept as a machinery cross-check at 1M draws with MC-error-calibrated gates.
N_DRAWS = ({"gauss": 400, "eight_schools": 400, "gym_banana": 400,
            "wl_bandpower": 400} if SMOKE else
           {"gauss": 20_000, "eight_schools": 100_000, "gym_banana": 1_000_000,
            "wl_bandpower": 50_000})
N_EVAL, N_ODE = (64, 8) if SMOKE else (4096, 100)
RHAT_GATE = 1.2 if SMOKE else 1.01  # smoke tests the PATH, not the statistics
VAR_STAB_TOL = 0.5 if SMOKE else 0.03  # half-vs-full reference variance move


def nuts_chains(key, logpdf_fn, d, init_scales, n_draws, target_accept=0.85):
    """4 NUTS chains via window adaptation; returns (draws (4,N,d), seconds)."""
    ld = lambda xi: logpdf_fn(xi[None, :])[0]
    t0 = time.time()
    chains = []
    for c in range(4):
        kw, kd, ki = jr.split(jr.fold_in(key, c), 3)
        x0 = jnp.asarray(init_scales) * jr.normal(ki, (d,), jnp.float64)
        wa = blackjax.window_adaptation(blackjax.nuts, ld,
                                        target_acceptance_rate=target_accept)
        (state, parameters), _ = wa.run(kw, x0, num_steps=N_WARM)
        alg = blackjax.nuts(ld, **parameters)

        def step(s, k):
            s, _ = alg.step(k, s)
            return s, s.position

        _, pos = jax.lax.scan(step, state, jr.split(kd, n_draws))
        chains.append(np.asarray(pos))
    return np.stack(chains), time.time() - t0


def build_reference(name, key, logpdf_fn, d, init_scales, **kw):
    draws, secs = nuts_chains(key, logpdf_fn, d, init_scales, N_DRAWS[name], **kw)
    rhat = split_rhat(draws)
    # repo-standard N-doubling stability: pooled variance, first half vs full
    pooled = draws.reshape(-1, d)
    half = draws[:, : draws.shape[1] // 2].reshape(-1, d)
    var_move = np.abs(half.var(0) / pooled.var(0) - 1.0)
    print(f"{name}: {draws.shape} in {secs:.0f}s, max R-hat = {rhat.max():.4f}, "
          f"max var half-vs-full move = {var_move.max():.4f}", flush=True)
    if rhat.max() >= RHAT_GATE or var_move.max() >= VAR_STAB_TOL:
        print(f"REF-GATE-FAIL on {name}: rhat={rhat}, var_move={var_move}")
        sys.exit(1)
    os.makedirs(REFS, exist_ok=True)
    np.savez(os.path.join(REFS, f"{name}.npz"), draws=draws.astype(np.float32),
             rhat=rhat, seconds=secs, var_move=var_move)
    return draws


def build_refs():
    # ---- gate 1: closed-form 3-d correlated Gaussian through the full pipeline
    A = np.array([[1.0, 0.5, 0.2], [0.5, 2.0, 0.3], [0.2, 0.3, 0.5]])
    mean = np.array([1.0, -1.0, 0.5])
    Ainv = jnp.asarray(np.linalg.inv(A))
    mj = jnp.asarray(mean)

    def gauss_lp(x):
        r = x - mj
        return -0.5 * jnp.einsum("ni,ij,nj->n", r, Ainv, r)

    draws, _ = nuts_chains(jr.key(101), gauss_lp, 3, [1.0, 1.4, 0.7],
                           N_DRAWS["gauss"])
    pooled = draws.reshape(-1, 3)
    rhat = split_rhat(draws)
    sd = np.sqrt(np.diag(A))
    mean_err = np.abs(pooled.mean(0) - mean) / sd
    var_err = np.abs(pooled.var(0) / np.diag(A) - 1.0)
    print(f"gaussian gate: mean_err/sd {mean_err.round(4)}, var rel err "
          f"{var_err.round(4)}, max R-hat {rhat.max():.4f}", flush=True)
    if not (rhat.max() < RHAT_GATE and mean_err.max() < (0.15 if SMOKE else 0.02)
            and var_err.max() < (0.3 if SMOKE else 0.02)):
        print("GAUSSIAN-VALIDATION-FAIL")
        sys.exit(1)

    # ---- (a) eight schools (funnel-ish: higher target acceptance)
    build_reference("eight_schools", jr.key(102), eight_schools_logpdf, 10,
                    [1.0] * 10, target_accept=0.9)

    # ---- (b) gym banana: EXACT transform sampler is THE reference (superior
    # to any chain; post-mortem 2026-07-11 in log/2026-07-11-phase1b.md).
    # NUTS is retained as a machinery cross-check at 1M draws/chain; the var
    # gate 0.10 is ~2 sigma of the MC error at the measured IACT.
    exact = np.asarray(gym_banana_sample(jr.key(104), 400_000))
    os.makedirs(REFS, exist_ok=True)
    np.savez(os.path.join(REFS, "gym_banana.npz"),
             draws=exact.reshape(4, -1, 2).astype(np.float32),
             rhat=np.ones(2), seconds=0.0, var_move=np.zeros(2), exact=True)
    draws, secs = nuts_chains(jr.key(103), gym_banana_logpdf, 2, [10.0, 1.0],
                              N_DRAWS["gym_banana"])
    pooled = draws.reshape(-1, 2)
    m_err = np.abs(pooled.mean(0) - exact.mean(0)) / exact.std(0)
    v_err = np.abs(pooled.var(0) / exact.var(0) - 1.0)
    rhat = split_rhat(draws)
    print(f"banana NUTS machinery cross-check ({secs:.0f}s): mean_err/sd "
          f"{m_err.round(4)}, var rel err {v_err.round(4)}, "
          f"max R-hat {rhat.max():.4f}", flush=True)
    if not SMOKE and not (m_err.max() < 0.03 and v_err.max() < 0.10
                          and rhat.max() < RHAT_GATE):
        print("BANANA-CROSSCHECK-FAIL")
        sys.exit(1)

    # ---- (c) WL band-power (u-space; surrogate must exist)
    wl_path = os.path.join(R, "wl_surrogate.npz")
    if os.path.exists(wl_path):
        wl = WLBandpower(wl_path)
        build_reference("wl_bandpower", jr.key(105), wl.logpdf, 3, [1.0] * 3)
    else:
        print("!! wl_surrogate.npz missing — WL reference SKIPPED (grid job "
              "pending); rerun --build-refs after it lands", flush=True)
    print("REFS-DONE", flush=True)


TARGETS = [
    ("eight_schools", eight_schools_logpdf, 10),
    ("gym_banana", gym_banana_logpdf, 2),
    ("wl_bandpower", None, 3),  # logpdf bound at runtime from the surrogate
]


def evaluate(ckpt, tag):
    model = ICSModel(n_attn=2)
    params = jax.tree_util.tree_map(jnp.asarray, load_checkpoint(ckpt)["params"])
    out = {"tag": tag, "ckpt": ckpt, "targets": {}}
    wl = None
    for idx, (name, fn, d) in enumerate(TARGETS):
        ref_path = os.path.join(REFS, f"{name}.npz")
        if not os.path.exists(ref_path):
            print(f"!! reference missing for {name} — SKIPPED", flush=True)
            out["targets"][name] = {"skipped": "no reference"}
            continue
        if name == "wl_bandpower":
            wl = wl or WLBandpower(os.path.join(R, "wl_surrogate.npz"))
            fn = wl.logpdf
        ref = np.load(ref_path)
        pooled = ref["draws"].reshape(-1, d).astype(np.float64)
        ref_thin = pooled[np.random.default_rng(9).choice(
            len(pooled), 2 * N_EVAL, replace=False)]
        rec = {"ref_rhat_max": float(ref["rhat"].max()),
               "ref_seconds": float(ref["seconds"])}
        for col, temp in (("T1", 1.0), ("T5", 5.0)):
            ctx = generate_context(jr.fold_in(jr.key(777_000 + idx), int(temp)),
                                   fn, d, K=128, temperature=temp,
                                   aux_tokens=True)
            t0 = time.time()
            cert, x_gen = ics_evaluate_fn(model, params, fn, d, ctx,
                                          jr.key(778_000 + 100 * idx + int(temp)),
                                          n_eval=N_EVAL, n_ode=N_ODE)
            allin = time.time() - t0
            sw2 = float(sliced_w2_squared(
                np.asarray(x_gen, np.float64), ref_thin, n_proj=128,
                rng=np.random.default_rng(idx)))
            rec[col] = dict(sw2_vs_ref=sw2, allin_seconds=allin,
                            cert_eff_per_s=cert["ess_frac_2n"] * 2 * N_EVAL / allin,
                            **cert)
            if name == "wl_bandpower":
                th_gen = np.asarray(wl.theta_of_u(jnp.asarray(x_gen)))
                th_ref = np.asarray(wl.theta_of_u(jnp.asarray(ref_thin)))
                rec[col]["theta_mean_gen"] = th_gen.mean(0).tolist()
                rec[col]["theta_mean_ref"] = th_ref.mean(0).tolist()
                rec[col]["theta_sd_gen"] = th_gen.std(0).tolist()
                rec[col]["theta_sd_ref"] = th_ref.std(0).tolist()
            print(f"{name} {col}: ESS(2N)={cert['ess_frac_2n']:.4f} "
                  f"stable={cert['stable']} sw2={sw2:.4f} allin={allin:.1f}s",
                  flush=True)
        out["targets"][name] = rec
        json.dump(out, open(os.path.join(R, f"readout_c_{tag}.json"), "w"),
                  indent=1)
    print(f"READOUT-C[{tag}] DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-refs", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--tag")
    a = ap.parse_args()
    if a.build_refs:
        build_refs()
    if a.eval:
        assert a.ckpt and a.tag
        evaluate(a.ckpt, a.tag)
