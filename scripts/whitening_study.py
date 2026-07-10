"""Whitening scale study (pre-registered: log/2026-07-10-reconvene-gate3.md).

Arms (scale computed FROM THE CONTEXT CHAIN ONLY, K=128 frozen protocol):
  A sigma_std      per-dim chain std (status quo)
  B sigma_std_x3   err-wide arm (stage-0 asymmetry: overdispersion is cheap)
  C sigma_range    per-dim (max - min)/2 over the chain
  D sigma_range_x3 err-wide arm

Testbed: 3 funnels (mild/mid/hard) + gmm d4 + warp d4 controls; single-target
UNCONDITIONAL FM (256,256), 4k steps, cosine 2e-3. Metrics: SW2^2 ratio over
same-p floor (all targets) + certificate (funnels). Decision rule: best
worst-case sw2/floor across the testbed; ties -> more overdispersed arm.
Writes results/whitening_study.json.
"""

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

from jax_flows import TimeConditionedMLP, cfm_loss, cfm_sample

from ics.cfm import cnf_logpdf
from ics.certificate import snis_certificate
from ics.context import generate_context_for_target
from ics.zoo import FunnelTarget, logpdf, sample_target, sample_x
from ics.warps import random_rotation
from stage0.sliced_w2 import sliced_w2_squared

STEPS, BATCH = 4000, 512


def make_funnel(key, d, sigma_v, c):
    return FunnelTarget(Q=random_rotation(key, d), sigma_v=jnp.asarray(sigma_v),
                        c=jnp.asarray(c))


TARGETS = [
    ("funnel_mild", make_funnel(jr.key(61), 4, 1.5, 1.0)),
    ("funnel_mid", make_funnel(jr.key(62), 4, 2.2, 1.2)),
    ("funnel_hard", make_funnel(jr.key(63), 4, 2.8, 1.5)),
    ("gmm_d4", sample_target(jr.key(64), "gmm", 4)),
    ("warp_d4", sample_target(jr.key(65), "warp", 4)),
]

ARMS = ["std", "std_x3", "range", "range_x3"]


def arm_sigma(ctx, arm):
    x = np.asarray(ctx.x_raw, np.float64)
    if arm.startswith("std"):
        s = x.std(axis=0) + 1e-6
    else:
        s = (x.max(axis=0) - x.min(axis=0)) / 2.0 + 1e-6
    if arm.endswith("_x3"):
        s = 3.0 * s
    return jnp.asarray(s), jnp.asarray(x.mean(axis=0))


def run_arm(name, target, arm, seed):
    d = target.d
    ctx = generate_context_for_target(jr.key(seed), target, K=128)
    sigma, mu = arm_sigma(ctx, arm)
    x_all = sample_x(jr.key(seed + 1), target, 60_000)
    x_tr = ((x_all - mu) / sigma).astype(jnp.float32)

    model = TimeConditionedMLP(hidden_dims=(256, 256), output_dim=d)
    params = model.init(jr.key(seed + 2), jnp.ones((1, d), jnp.float32),
                        jnp.ones((1,), jnp.float32))["params"]
    tx = optax.adam(optax.cosine_decay_schedule(2e-3, STEPS))
    opt = tx.init(params)

    @jax.jit
    def step(params, opt, key):
        kb, kl = jr.split(key)
        idx = jr.randint(kb, (BATCH,), 0, x_tr.shape[0])
        loss, g = jax.value_and_grad(cfm_loss)(params, x_tr[idx], kl, model)
        up, opt = tx.update(g, opt)
        return optax.apply_updates(params, up), opt, loss

    for k in jr.split(jr.key(seed + 3), STEPS):
        params, opt, _ = step(params, opt, k)

    params64 = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64), params)
    s_white = cfm_sample(model, params64, jr.key(seed + 4), (8192, d),
                         n_steps=100, solver="heun")
    x_gen = np.asarray(mu + sigma * s_white)
    fresh = np.asarray(sample_x(jr.key(seed + 5), target, 8192), np.float64)
    fresh2 = np.asarray(sample_x(jr.key(seed + 6), target, 8192), np.float64)
    sw2 = sliced_w2_squared(x_gen, fresh, n_proj=128, rng=np.random.default_rng(seed))
    floor = sliced_w2_squared(fresh2, fresh, n_proj=128,
                              rng=np.random.default_rng(seed + 1))
    row = dict(target=name, arm=arm, sw2=float(sw2), floor=float(floor),
               ratio=float(sw2 / floor))
    if name.startswith("funnel"):
        def velocity_fn(x, t):
            t_b = jnp.full((x.shape[0],), t, dtype=x.dtype)
            return model.apply({"params": params64}, x, t_b)
        logq_white = cnf_logpdf(velocity_fn, s_white, n_steps=100)
        logp_white = logpdf(target, mu + sigma * s_white) + jnp.log(sigma).sum()
        cert = snis_certificate(np.asarray(logp_white), np.asarray(logq_white))
        row.update(ess=cert["ess_frac_2n"], stable=cert["stable"], logz=cert["logz"])
    return row


def main():
    t0 = time.time()
    rows = []
    for i, (name, target) in enumerate(TARGETS):
        for j, arm in enumerate(ARMS):
            row = run_arm(name, target, arm, 90_000 + 100 * i + 10 * j)
            rows.append(row)
            print(json.dumps(row), flush=True)
    worst = {arm: max(r["ratio"] for r in rows if r["arm"] == arm) for arm in ARMS}
    out = dict(rows=rows, worst_case_ratio=worst,
               winner=min(worst, key=worst.get), seconds=round(time.time() - t0, 1))
    with open(os.path.join(os.path.dirname(__file__), "..", "results",
                           "whitening_study.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("worst-case ratios:", worst)
    print("WINNER:", out["winner"])


if __name__ == "__main__":
    main()
