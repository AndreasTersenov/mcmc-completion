"""PAIRED-A branch: 2M-step training on the exact gate3e 128-target dataset
(deterministic rebuild in-job: jr.key(3131), T-mix [1,1,2,2,5,5], aux).
Resumable (time budget + checkpoint + status json) for the 3h-b1 chain.
Pre-registration: log/2026-07-10-paired-eval.md (branch A)."""

import argparse, json, os, sys, time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import optax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ics.models import ICSModel
from ics.train import build_zoo_dataset, load_checkpoint, make_train_step, save_checkpoint
from ics.zoo import DMAX

FAMS, DS = ("gmm", "dwell", "funnel", "warp"), (2, 4, 8)
PER_CELL = {c: (11 if i < 8 else 10) for i, c in enumerate(
    [(f, d) for f in FAMS for d in DS])}
T_MIX = [1.0, 1.0, 2.0, 2.0, 5.0, 5.0]
STEPS_TOTAL, BATCH, LR = 2_000_000, 512, 1e-3

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--status", required=True)
ap.add_argument("--time-budget-sec", type=int, default=9000)
args = ap.parse_args()

t0 = time.time()
specs = [(f, d, i) for (f, d), n in PER_CELL.items() for i in range(n)]
targets, ctxs, data = build_zoo_dataset(jr.key(3131), specs, 6, 128, 50_000,
                                        temperature=T_MIX, aux_tokens=True)
print(f"dataset rebuilt in {time.time()-t0:.0f}s", flush=True)

model = ICSModel(n_attn=2)
params = model.init(jr.key(32), jnp.ones((2, DMAX), jnp.float32),
                    jnp.ones((2,), jnp.float32), data.tokens[:2, 0])["params"]
tx = optax.adam(optax.cosine_decay_schedule(LR, STEPS_TOTAL))
opt_state = tx.init(params)
start = 0
if os.path.exists(args.ckpt):
    ck = load_checkpoint(args.ckpt)
    params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    opt_state = jax.tree_util.tree_map(
        lambda a: jnp.asarray(a) if hasattr(a, "shape") else a, ck["opt_state"])
    start = ck["step"]
    print(f"resumed from step {start}", flush=True)

step = make_train_step(model, tx, BATCH, len(specs), 6, 50_000)
i = start
while i < STEPS_TOTAL:
    params, opt_state, loss = step(params, opt_state, jr.fold_in(jr.key(33), i), data)
    i += 1
    if i % 50_000 == 0:
        print(f"step {i}: loss {float(loss):.4f} [{time.time()-t0:.0f}s]", flush=True)
    if i % 10_000 == 0 and time.time() - t0 > args.time_budget_sec:
        break
save_checkpoint(args.ckpt, params, opt_state, i)
complete = i >= STEPS_TOTAL
json.dump({"complete": complete, "step": i}, open(args.status, "w"))
print("TRAIN-COMPLETE" if complete else "TRAIN-INCOMPLETE")
