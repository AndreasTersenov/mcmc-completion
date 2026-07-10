"""Gate (iv) frozen evaluation: run the certificate + sliced-W2 + mode
recovery over the pregenerated eval set for a trained arm checkpoint.
Writes results/eval_<arm>.json with one row per eval target.

Metrics (frozen): ESS at N/2N + stability flag, D2-hat = -ln(ESS/N),
|logZ-hat| error, sliced-W2^2 vs exact samples, mode-recovery rate for
families with declared mode structure (nearest-center assignment; mode j
recovered if its sample share >= w_j / 4).
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

from ics.context import Context
from ics.eval import ics_evaluate
from ics.models import ICSModel
from ics.train import load_checkpoint
from ics.zoo import DMAX, mode_centers, sample_x
from stage0.sliced_w2 import sliced_w2_squared


def mode_recovery(target, x_gen):
    mc = mode_centers(target)
    if mc is None:
        return None
    mc = np.asarray(mc)
    if hasattr(target, "log_weights"):
        w = np.exp(np.asarray(target.log_weights))
        w = w[np.isfinite(np.asarray(target.log_weights))] if w.ndim else w
        w = w / w.sum()
    else:
        w = np.full(len(mc), 1.0 / len(mc))
    if len(w) != len(mc):  # dwell: equal-weight well combinations
        w = np.full(len(mc), 1.0 / len(mc))
    d2 = ((x_gen[:, None, :] - mc[None, :, :]) ** 2).sum(-1)
    share = np.bincount(d2.argmin(1), minlength=len(mc)) / len(x_gen)
    recovered = share >= w / 4.0
    return float(recovered.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nograd", action="store_true")
    ap.add_argument("--n-eval", type=int, default=4096)
    args = ap.parse_args()

    with open(args.evalset, "rb") as f:
        rows = pickle.load(f)
    ck = load_checkpoint(args.ckpt)
    params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    model = ICSModel()

    out = []
    t0 = time.time()
    for i, row in enumerate(rows):
        target, ctx = row["target"], Context(**{k: jnp.asarray(v) for k, v in
                                                row["context"]._asdict().items()})
        if args.nograd:
            toks = np.asarray(ctx.tokens)
            toks[..., DMAX + 1 : 2 * DMAX + 1] = 0.0
            toks[..., 2 * DMAX + 2] = 0.0
            ctx = ctx._replace(tokens=jnp.asarray(toks))
        cert, x_gen = ics_evaluate(model, params, target, ctx,
                                   jr.key(880_000 + i), n_eval=args.n_eval, n_ode=100)
        fresh = np.asarray(sample_x(jr.key(881_000 + i), target, 2 * args.n_eval),
                           np.float64)
        fresh2 = np.asarray(sample_x(jr.key(882_000 + i), target, 2 * args.n_eval),
                            np.float64)
        sw2 = sliced_w2_squared(x_gen, fresh, n_proj=128,
                                rng=np.random.default_rng(i))
        floor = sliced_w2_squared(fresh2, fresh, n_proj=128,
                                  rng=np.random.default_rng(10_000 + i))
        out.append(dict(
            family=row["family"], d=row["d"], idx=row["idx"],
            heldout=row["heldout"], sw2=float(sw2), sw2_floor=float(floor),
            mode_recovery=mode_recovery(target, x_gen), **cert,
        ))
        if (i + 1) % 24 == 0:
            print(f"{i+1}/{len(rows)} [{time.time()-t0:.0f}s]", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"EVAL-DONE {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
