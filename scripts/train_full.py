"""Gate (iv) training: one arm of the full zoo, resumable (3h-b1-chain
pattern per the rorqual-jobs skill). Loads the pregenerated dataset from
$SCRATCH, checkpoints to $SCRATCH, stops cleanly at --time-budget-sec and
records completeness in --status (JSON) for the sbatch chain.

--nograd zeroes the gradient-token block and the log-gradient-scale feature
(the P11 ablation arm; identical shapes and compute).
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
import optax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ics.models import ICSModel
from ics.train import ZooData, load_checkpoint, make_train_step, save_checkpoint
from ics.zoo import DMAX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--steps", type=int, default=1_600_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--time-budget-sec", type=int, default=9000)
    ap.add_argument("--nograd", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    raw = np.load(os.path.join(args.data, "data.npz"))
    data = ZooData(**{k: jnp.asarray(raw[k]) for k in ZooData._fields})
    if args.nograd:
        toks = np.asarray(data.tokens)
        toks[..., DMAX + 1 : 2 * DMAX + 1] = 0.0  # grad block
        toks[..., 2 * DMAX + 2] = 0.0             # log g_scale feature
        data = data._replace(tokens=jnp.asarray(toks))
    n_targets, n_ctx = data.tokens.shape[0], data.tokens.shape[1]
    n_pool = data.pool.shape[1]
    print(f"data: {n_targets} targets x {n_ctx} ctx, pool {n_pool}", flush=True)

    model = ICSModel(n_attn=2)
    params = model.init(
        jr.key(52), jnp.ones((2, DMAX), jnp.float32), jnp.ones((2,), jnp.float32),
        data.tokens[:2, 0],
    )["params"]
    tx = optax.adam(optax.cosine_decay_schedule(args.lr, args.steps))
    opt_state = tx.init(params)
    start = 0
    if os.path.exists(args.ckpt):
        ck = load_checkpoint(args.ckpt)
        params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
        opt_state = jax.tree_util.tree_map(
            lambda a: jnp.asarray(a) if hasattr(a, "shape") else a, ck["opt_state"]
        )
        start = ck["step"]
        print(f"resumed from step {start}", flush=True)

    step = make_train_step(model, tx, args.batch, n_targets, n_ctx, n_pool)
    i = start
    while i < args.steps:
        params, opt_state, loss = step(params, opt_state, jr.fold_in(jr.key(53), i), data)
        i += 1
        if i % 10_000 == 0:
            print(f"step {i}: loss {float(loss):.4f} [{time.time()-t0:.0f}s]", flush=True)
        if i % 5_000 == 0 and time.time() - t0 > args.time_budget_sec:
            break

    save_checkpoint(args.ckpt, params, opt_state, i)
    complete = i >= args.steps
    with open(args.status, "w") as f:
        json.dump({"complete": complete, "step": i}, f)
    print(f"saved at step {i}; complete={complete}")
    print("TRAIN-COMPLETE" if complete else "TRAIN-INCOMPLETE")


if __name__ == "__main__":
    main()
