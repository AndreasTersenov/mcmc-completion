"""Karpathy gate (ii): overfit (context, target) PAIRS through the encoder.

Two d=4 GMM targets, one fixed frozen-protocol context each (K=128). The
conditional model trains on both pairs simultaneously — with a single pair it
could ignore the context entirely, so two pairs is the minimal setup where
passing PROVES the conditioning pathway routes information.

PASS (pre-registered): both pairs meet the gate-(i) thresholds
(ESS/N >= 5%, stable, |logZ-hat| <= 0.1, SW2^2 <= 3x floor) AND the
swapped-context control degrades SW2^2 by >= 3x on both pairs.
Writes results/gate2.json.
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

from ics.cfm import cond_cfm_loss, cond_cfm_sample
from ics.context import generate_context, pad_to_dmax, whiten_apply, whiten_invert
from ics.eval import ics_evaluate
from ics.models import ICSModel
from ics.zoo import logpdf, sample_target, sample_x
from stage0.sliced_w2 import sliced_w2_squared

D = 4
K = 128
N_TRAIN = 100_000
STEPS = 20_000
BATCH = 512
LR = 1e-3
N_EVAL = 8192


AUX = False  # set from args in main()


def build_pair(seed_target, seed_ctx, seed_data):
    target = sample_target(jr.key(seed_target), "gmm", D)
    fn = lambda x: logpdf(target, x)
    ctx = generate_context(jr.key(seed_ctx), fn, D, K=K, aux_tokens=AUX)
    x_raw = sample_x(jr.key(seed_data), target, N_TRAIN)
    x_white = whiten_apply(x_raw, ctx.mu, ctx.sigma)
    x1 = pad_to_dmax(jr.key(seed_data + 1), x_white).astype(jnp.float32)
    return target, ctx, x1


def main():
    global AUX
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", type=int, default=0)
    ap.add_argument("--aux", action="store_true")
    ap.add_argument("--out", default="gate2.json")
    args = ap.parse_args()
    AUX = args.aux
    t0 = time.time()
    pairs = [build_pair(1042, 2042, 3042), build_pair(1142, 2142, 3142)]
    tokens = jnp.stack([p[1].tokens for p in pairs]).astype(jnp.float32)  # (2, K, F)

    model = ICSModel(n_attn=args.attn)
    params = model.init(
        jr.key(7),
        jnp.ones((2, 16), jnp.float32),
        jnp.ones((2,), jnp.float32),
        tokens,
    )["params"]
    tx = optax.adam(optax.cosine_decay_schedule(LR, STEPS))
    opt_state = tx.init(params)
    x1_all = jnp.stack([p[2] for p in pairs])  # (2, N, DMAX)

    @jax.jit
    def step(params, opt_state, key):
        kb, kp, kl = jr.split(key, 3)
        idx = jr.randint(kb, (BATCH,), 0, N_TRAIN)
        which = jr.randint(kp, (BATCH,), 0, 2)
        x1 = x1_all[which, idx]
        toks = tokens[which]
        loss, grads = jax.value_and_grad(cond_cfm_loss)(params, model, x1, toks, kl)
        updates, opt_state = tx.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    keys = jr.split(jr.key(8), STEPS)
    for i in range(STEPS):
        params, opt_state, loss = step(params, opt_state, keys[i])
        if (i + 1) % 4000 == 0:
            print(f"step {i+1}: loss {float(loss):.4f}", flush=True)
    print(f"trained in {time.time()-t0:.0f}s", flush=True)

    out = {"gate": "ii", "pairs": [], "seconds_train": round(time.time() - t0, 1)}
    all_pass = True
    for j, (target, ctx, _) in enumerate(pairs):
        cert, x_gen = ics_evaluate(model, params, target, ctx, jr.key(100 + j),
                                   n_eval=N_EVAL)
        fresh = np.asarray(sample_x(jr.key(200 + j), target, 2 * N_EVAL), np.float64)
        fresh2 = np.asarray(sample_x(jr.key(300 + j), target, 2 * N_EVAL), np.float64)
        sw2 = sliced_w2_squared(x_gen, fresh, n_proj=128, rng=np.random.default_rng(j))
        floor = sliced_w2_squared(fresh2, fresh, n_proj=128,
                                  rng=np.random.default_rng(10 + j))
        # swapped-context control: sample with the OTHER pair's context,
        # de-whiten with THAT context's transform, compare to THIS target
        other_ctx = pairs[1 - j][1]
        x_wrong_full = cond_cfm_sample(
            model, params, other_ctx.tokens.astype(jnp.float32),
            jr.key(400 + j), n=2 * N_EVAL, n_steps=200,
        )
        x_wrong = np.asarray(
            whiten_invert(x_wrong_full[:, :D], other_ctx.mu, other_ctx.sigma), np.float64
        )
        sw2_wrong = sliced_w2_squared(x_wrong, fresh, n_proj=128,
                                      rng=np.random.default_rng(20 + j))
        pair_pass = (
            cert["ess_frac_2n"] >= 0.05 and cert["stable"]
            and abs(cert["logz"]) <= 0.1 and sw2 <= 3.0 * floor
            and sw2_wrong >= 3.0 * sw2
        )
        all_pass &= pair_pass
        out["pairs"].append(dict(
            pair=j, passed=bool(pair_pass), sw2=float(sw2), sw2_floor=float(floor),
            sw2_wrong_context=float(sw2_wrong), **cert,
        ))
        print(json.dumps(out["pairs"][-1], indent=2), flush=True)

    out["passed"] = bool(all_pass)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", args.out), "w") as f:
        json.dump(out, f, indent=2)
    print("GATE2-PASS" if all_pass else "GATE2-FAIL")


if __name__ == "__main__":
    main()
