"""Phase 1b Readout B: the proposal-engine cost-crossover (pre-registration and
frozen accounting: log/2026-07-11-phase1b.md).

12-target subset + T5 contexts verbatim from $SCRATCH/ics-zoo/eval/eval_set.pkl
(the same objects baselines.py scored B2-B4 on). Per target:
- ICS all-in wall-clock = forward sampling (2N=8192, n_ode=100) + weights (CNF
  log-density) + certificate incl. the N-vs-2N doubling check. Jit warm-up is
  excluded via a discarded first pass on target 0 (compile_seconds reported
  separately — same convention as B4's adapt/compile split).
- Crossover(j): sw2_ICS(j) <= sw2_B4(j) AND allin_ICS(j) < adapt+sample(j)
  (B4 numbers read from results/baselines.json — frozen references, not rerun).
- Certified eff. samples/s (ICS) = ess_frac_2n * 8192 / allin; MCLMC comparator
  8192/(adapt+sample) is an UPPER bound on its effective rate (generous to MCLMC).
Writes results/readout_b_<tag>.json. ICS_SMOKE=1 = CPU path test only.
"""

import argparse
import json
import os
import pickle
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ics.certificate import snis_certificate
from ics.cfm import cnf_logpdf, cond_cfm_sample, make_velocity_fn
from ics.context import Context, whiten_invert
from ics.eval import augmented_true_logpdf, mode_recovery
from ics.models import ICSModel
from ics.train import load_checkpoint
from ics.zoo import sample_x
from stage0.sliced_w2 import sliced_w2_squared

N_EVAL = 4096
N_ODE = 100
SMOKE = os.environ.get("ICS_SMOKE") == "1"
if SMOKE:
    N_EVAL, N_ODE = 64, 8
SUBSET = [(f, d, 0) for f in ("gmm", "dwell", "funnel", "warp", "banana", "funnelmix")
          for d in (2, 4)]


def ics_allin(model, params, target, ctx, seed):
    """Timed exactly as frozen: forward + weights + doubling-checked certificate."""
    tokens = ctx.tokens.astype(jnp.float64)
    params64 = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64), params)
    t0 = time.time()
    x_full = cond_cfm_sample(model, params64, tokens, jr.key(seed),
                             n=2 * N_EVAL, n_steps=N_ODE)
    velocity_fn = make_velocity_fn(model, params64, tokens)
    logq = cnf_logpdf(velocity_fn, x_full, n_steps=N_ODE)
    logp = augmented_true_logpdf(target, ctx, x_full)
    cert = snis_certificate(np.asarray(logp), np.asarray(logq))
    allin = time.time() - t0
    x_raw = np.asarray(whiten_invert(x_full[:, : target.d], ctx.mu, ctx.sigma))
    return cert, x_raw, allin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True, help="e.g. 200k or 2M")
    args = ap.parse_args()
    t_start = time.time()

    print(f"readout_b starting: devices={jax.devices()}", flush=True)
    base = os.path.join(os.path.dirname(__file__), "..", "results")
    bl = json.load(open(os.path.join(base, "baselines.json")))
    b4_by_key = {(r["family"], r["d"], r["idx"]): r["b4"] for r in bl["rows"]}
    es = pickle.load(open(os.path.join(os.environ["SCRATCH"], "ics-zoo", "eval",
                                       "eval_set.pkl"), "rb"))
    rows_by_key = {(r["family"], r["d"], r["idx"]): r for r in es}

    model = ICSModel(n_attn=2)
    params = jax.tree_util.tree_map(jnp.asarray,
                                    load_checkpoint(args.ckpt)["params"])

    subset = SUBSET[:2] if SMOKE else SUBSET
    # discarded warm-up pass on the first target: compile time, not inference
    r0 = rows_by_key[subset[0][:3]]
    ctx0 = Context(**{k: jnp.asarray(v) for k, v in r0["context_t5"]._asdict().items()})
    print("eval set loaded; starting jit warm-up", flush=True)
    t0 = time.time()
    _ = ics_allin(model, params, r0["target"], ctx0, seed=1)
    compile_seconds = time.time() - t0
    print(f"warm-up done in {compile_seconds:.1f}s", flush=True)

    out = {"tag": args.tag, "ckpt": args.ckpt, "compile_seconds": compile_seconds,
           "rows": []}
    for n, (f, d, i) in enumerate(subset):
        row = rows_by_key[(f, d, i)]
        t = row["target"]
        ctx = Context(**{k: jnp.asarray(v)
                         for k, v in row["context_t5"]._asdict().items()})
        t_row = time.time()
        cert, x_gen, allin = ics_allin(model, params, t, ctx, seed=80_000 + 100 * n)
        t_a = time.time()
        fresh = np.asarray(sample_x(jr.key(81_000 + 100 * n), t, 2 * N_EVAL),
                           np.float64)
        t_f = time.time()
        sw2 = float(sliced_w2_squared(np.asarray(x_gen, np.float64), fresh,
                                      n_proj=128, rng=np.random.default_rng(n)))
        t_s = time.time()
        print(f"  row timing: allin {t_a-t_row:.1f}s fresh {t_f-t_a:.1f}s "
              f"sw2 {t_s-t_f:.1f}s", flush=True)
        b4 = b4_by_key.get((f, d, i))
        mclmc_allin = (b4["adapt_seconds"] + b4["sample_seconds"]) if b4 else None
        rec = dict(
            family=f, d=d, idx=i, sw2=sw2, allin_seconds=allin,
            mode_recovery=mode_recovery(t, x_gen),
            cert_eff_per_s=cert["ess_frac_2n"] * 2 * N_EVAL / allin,
            b4_sw2=(b4 or {}).get("sw2"), b4_allin_seconds=mclmc_allin,
            b4_rate_upper=(2 * N_EVAL / mclmc_allin) if mclmc_allin else None,
            crossover=bool(b4 and sw2 <= b4["sw2"] and allin < mclmc_allin),
            **cert)
        out["rows"].append(rec)
        json.dump(out, open(os.path.join(base, f"readout_b_{args.tag}.json"), "w"),
                  indent=1)
        print(f"[{n+1}/{len(subset)}] {f}-d{d}: sw2={sw2:.3f} vs b4 "
              f"{rec['b4_sw2']:.3f} | allin {allin:.1f}s vs {mclmc_allin:.1f}s | "
              f"crossover={rec['crossover']} [total {time.time()-t_start:.0f}s]",
              flush=True)

    out["crossover_count"] = sum(r["crossover"] for r in out["rows"])
    out["n"] = len(out["rows"])
    out["seconds"] = round(time.time() - t_start, 1)
    json.dump(out, open(os.path.join(base, f"readout_b_{args.tag}.json"), "w"),
              indent=1)
    print(f"READOUT-B[{args.tag}]: crossover {out['crossover_count']}/{out['n']} "
          f"(frozen train4 baseline: 3/12)", flush=True)


if __name__ == "__main__":
    main()
