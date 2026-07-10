"""Karpathy gate (iii): 10-target mini-zoo, shared model.

10 targets across the 4 train families, d in {2,4,8}; 8 frozen-protocol
contexts per target; one shared ICSModel trained on all pairs. Eval per
target uses a FRESH context (unseen chains) — context-level generalization,
the precondition for gate (iv). Two held-out-theta probes are REPORTED (P1
preview) but do not gate.

PASS (pre-registered; SW2 bar amended after attempt 1, see log): >= 8/10
targets meet ESS/N(2N) >= 5% AND stable AND |logZ-hat| <= 0.1 AND
SW2^2 <= max(3x same-p floor, 0.1), on fresh contexts. Attempt 2: 200k steps.
Writes results/gate3.json + results/gate3_params.pkl.
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

from ics.context import generate_context
from ics.eval import ics_evaluate, mode_recovery
from ics.models import ICSModel
from ics.train import build_zoo_dataset, make_train_step, save_checkpoint
from ics.zoo import DMAX, logpdf, sample_target, sample_x
from stage0.sliced_w2 import sliced_w2_squared

SPECS = [
    ("gmm", 2), ("gmm", 4), ("gmm", 8),
    ("dwell", 2), ("dwell", 4),
    ("funnel", 4), ("funnel", 8),
    ("warp", 2), ("warp", 4), ("warp", 8),
]
N_CTX, K, N_POOL = 8, 128, 50_000
STEPS, BATCH, LR = 200_000, 512, 1e-3
N_EVAL = 4096


def eval_on(model, params, target, ctx, seed):
    cert, x_gen = ics_evaluate(model, params, target, ctx, jr.key(seed),
                               n_eval=N_EVAL, n_ode=100)
    fresh = np.asarray(sample_x(jr.key(seed + 1), target, 2 * N_EVAL), np.float64)
    fresh2 = np.asarray(sample_x(jr.key(seed + 2), target, 2 * N_EVAL), np.float64)
    sw2 = sliced_w2_squared(x_gen, fresh, n_proj=128, rng=np.random.default_rng(seed))
    floor = sliced_w2_squared(fresh2, fresh, n_proj=128,
                              rng=np.random.default_rng(seed + 1))
    ok = (cert["ess_frac_2n"] >= 0.05 and cert["stable"]
          and abs(cert["logz"]) <= 0.1 and sw2 <= max(3.0 * floor, 0.1))
    return dict(passed=bool(ok), sw2=float(sw2), sw2_floor=float(floor),
                mode_recovery=mode_recovery(target, x_gen), **cert)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", action="store_true", help="2 self-attention blocks")
    ap.add_argument("--aux", action="store_true", help="chain-summary tokens")
    ap.add_argument("--shortk", action="store_true", help="short-context augmentation")
    ap.add_argument("--wide", action="store_true",
                    help="attempt-4 encoder scale: enc_dim 256, hidden 512, 4 attn blocks")
    args = ap.parse_args()
    t0 = time.time()
    print(f"building dataset... (attn={args.attn} aux={args.aux} shortk={args.shortk})",
          flush=True)
    targets, ctxs, data = build_zoo_dataset(jr.key(31), SPECS, N_CTX, K, N_POOL,
                                            aux_tokens=args.aux)
    print(f"dataset built in {time.time()-t0:.0f}s", flush=True)

    if args.wide:
        model = ICSModel(enc_dim=256, enc_hidden=512, n_attn=4)
    else:
        model = ICSModel(n_attn=2 if args.attn else 0)
    params = model.init(
        jr.key(32), jnp.ones((2, DMAX), jnp.float32), jnp.ones((2,), jnp.float32),
        data.tokens[:2, 0],
    )["params"]
    tx = optax.adam(optax.cosine_decay_schedule(LR, STEPS))
    opt_state = tx.init(params)
    step = make_train_step(model, tx, BATCH, len(SPECS), N_CTX, N_POOL,
                           shortk=args.shortk, K=K, n_aux=4 if args.aux else 0)

    keys = jr.split(jr.key(33), STEPS)
    for i in range(STEPS):
        params, opt_state, loss = step(params, opt_state, keys[i], data)
        if (i + 1) % 10_000 == 0:
            print(f"step {i+1}: loss {float(loss):.4f} [{time.time()-t0:.0f}s]",
                  flush=True)

    out = {"gate": "iii", "targets": [], "heldout_theta_probes": []}
    n_pass = 0
    for j, ((family, d), t) in enumerate(zip(SPECS, targets)):
        fn = lambda x, _t=t: logpdf(_t, x)
        fresh_ctx = generate_context(jr.key(9000 + j), fn, d, K=K,
                                     aux_tokens=args.aux)
        r = eval_on(model, params, t, fresh_ctx, 10_000 + 10 * j)
        r.update(family=family, d=d)
        n_pass += r["passed"]
        out["targets"].append(r)
        print(json.dumps(r), flush=True)

    # held-out-theta probes (reported, non-gating): fresh targets, fresh ctx
    for j, (family, d) in enumerate([("gmm", 4), ("funnel", 4)]):
        t = sample_target(jr.key(7000 + j), family, d)
        fn = lambda x, _t=t: logpdf(_t, x)
        fresh_ctx = generate_context(jr.key(7100 + j), fn, d, K=K,
                                     aux_tokens=args.aux)
        r = eval_on(model, params, t, fresh_ctx, 7200 + 10 * j)
        r.update(family=family, d=d)
        out["heldout_theta_probes"].append(r)
        print("held-out-theta:", json.dumps(r), flush=True)

    out["n_pass"] = n_pass
    out["passed"] = bool(n_pass >= 8)
    out["seconds"] = round(time.time() - t0, 1)
    base = os.path.join(os.path.dirname(__file__), "..", "results")
    with open(os.path.join(base, "gate3.json"), "w") as f:
        json.dump(out, f, indent=2)
    save_checkpoint(os.path.join(base, "gate3_params.pkl"), params, opt_state, STEPS)
    print("GATE3-PASS" if out["passed"] else "GATE3-FAIL")


if __name__ == "__main__":
    main()
