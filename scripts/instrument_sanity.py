"""Embedded instrument (a): trained-target sanity floor (reconvene ruling,
log/2026-07-10-reconvene-paired.md). Pre-registered subsample: targets with
in-cell indices {0,1} per (family,d) cell for train4/train4ng (24 of 1024),
indices {0,1,2} for train2 (24). Fresh contexts, T=5 scoring column, funnels
K=512, in-job bespoke refs. FLOOR (pre-registered): >= 6/24 composite AND
median ESS >= 1%. Role: catch broken arms, not re-litigate sharpness."""
import argparse, json, os, pickle, sys, time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ics.context import generate_context_for_target
from ics.models import ICSModel
from ics.train import load_checkpoint
from paired_eval import bespoke_ref_sw2, eval_one  # shared, gate-tested paths

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--targets", required=True)
ap.add_argument("--per-cell", type=int, default=2)
ap.add_argument("--out", required=True)
ap.add_argument("--nograd", action="store_true")
args = ap.parse_args()

t0 = time.time()
rows_in = pickle.load(open(args.targets, "rb"))  # [(family, d, target_tree)]
model = ICSModel(n_attn=2)
params = jax.tree_util.tree_map(jnp.asarray, load_checkpoint(args.ckpt)["params"])

seen, picked = {}, []
for j, (f, d, t) in enumerate(rows_in):
    if seen.get((f, d), 0) < args.per_cell:
        picked.append((j, f, d, t))
        seen[(f, d)] = seen.get((f, d), 0) + 1
picked = picked[:24]

from ics.zoo import DMAX
out = []
for n, (j, f, d, t) in enumerate(picked):
    k_eval = 512 if f == "funnel" else 128
    ctx = generate_context_for_target(jr.fold_in(jr.key(717171), j), t, K=k_eval,
                                      temperature=5.0, aux_tokens=True)
    if args.nograd:
        toks = np.array(ctx.tokens)  # copy: read-only source
        toks[..., DMAX + 1: 2 * DMAX + 1] = 0.0
        toks[..., 2 * DMAX + 2] = 0.0
        ctx = ctx._replace(tokens=jnp.asarray(toks))
    ref = bespoke_ref_sw2(t, ctx, 730_000 + 100 * n)
    r = eval_one(model, params, t, ctx, 740_000 + 100 * n, ref)
    r.update(family=f, d=d, sw2_ref=ref)
    out.append(r)
    print(f"[{n+1}/{len(picked)}] {f}-d{d} pass={r['passed']} ess={r['ess_frac_2n']:.4f} [{time.time()-t0:.0f}s]", flush=True)

n_pass = sum(r["passed"] for r in out)
med_ess = float(np.median([r["ess_frac_2n"] for r in out]))
floor_ok = n_pass >= 6 and med_ess >= 0.01
json.dump(dict(rows=out, n_pass=n_pass, median_ess=med_ess, floor_ok=floor_ok),
          open(args.out, "w"), indent=2)
print(f"n_pass={n_pass}/24 median_ess={med_ess:.4f}")
print("SANITY-FLOOR-PASS" if floor_ok else "SANITY-FLOOR-FAIL")
