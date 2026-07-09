"""Karpathy gate (i): overfit ONE zoo target with the FM head, no context.

Trains an unconditional velocity MLP (jax_flows) on exact samples of a single
d=4 GMM target, then closes the certificate loop end-to-end: heun samples ->
CNF log-density (exact divergence) -> SNIS vs the true zoo log-density ->
ESS at N/2N + stability + logZ-hat + sliced-W2^2.

PASS criteria (pre-registered in log/2026-07-09-toy-gate1.md):
  ESS/N(2N) >= 0.05, stable, |logZ-hat| <= 0.1, SW2^2 <= 3x same-p floor.
Writes results/gate1.json.
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
from ics.zoo import logpdf, sample_target, sample_x
from stage0.sliced_w2 import sliced_w2_squared

D = 4
N_TRAIN = 100_000
STEPS = 20_000
BATCH = 1024
LR = 2e-3
N_EVAL = 8192  # certificate uses 2N = 16384


def main():
    t0 = time.time()
    target = sample_target(jr.key(1042), "gmm", D)
    x_train = sample_x(jr.key(1043), target, N_TRAIN).astype(jnp.float32)

    model = TimeConditionedMLP(hidden_dims=(256, 256, 256), output_dim=D)
    params = model.init(
        jr.key(1044), jnp.ones((1, D), jnp.float32), jnp.ones((1,), jnp.float32)
    )["params"]
    tx = optax.adam(optax.cosine_decay_schedule(LR, STEPS))
    opt_state = tx.init(params)

    @jax.jit
    def step(params, opt_state, key):
        kb, kl = jr.split(key)
        idx = jr.randint(kb, (BATCH,), 0, N_TRAIN)
        loss, grads = jax.value_and_grad(cfm_loss)(params, x_train[idx], kl, model)
        updates, opt_state = tx.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    keys = jr.split(jr.key(1045), STEPS)
    loss0 = float(cfm_loss(params, x_train[:4096], jr.key(1046), model))
    for i in range(STEPS):
        params, opt_state, loss = step(params, opt_state, keys[i])
        if (i + 1) % 4000 == 0:
            print(f"step {i+1}: loss {float(loss):.4f}", flush=True)
    loss_end = float(cfm_loss(params, x_train[:4096], jr.key(1046), model))
    print(f"trained in {time.time()-t0:.0f}s; loss {loss0:.3f} -> {loss_end:.3f}")

    # ---- eval in f64: samples, CNF logq, certificate, SW2
    params64 = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64), params)
    x_2n = cfm_sample(model, params64, jr.key(1047), (2 * N_EVAL, D), n_steps=200,
                      solver="heun")

    def velocity_fn(x, t):
        t_b = jnp.full((x.shape[0],), t, dtype=x.dtype)
        return model.apply({"params": params64}, x, t_b)

    logq = cnf_logpdf(velocity_fn, x_2n, n_steps=200)
    logp = logpdf(target, x_2n)
    cert = snis_certificate(np.asarray(logp), np.asarray(logq))

    fresh = sample_x(jr.key(1048), target, 2 * N_EVAL)
    fresh2 = sample_x(jr.key(1049), target, 2 * N_EVAL)
    sw2 = sliced_w2_squared(np.asarray(x_2n, np.float64), np.asarray(fresh, np.float64),
                            n_proj=128, rng=np.random.default_rng(0))
    floor = sliced_w2_squared(np.asarray(fresh2, np.float64), np.asarray(fresh, np.float64),
                              n_proj=128, rng=np.random.default_rng(1))

    passed = (
        cert["ess_frac_2n"] >= 0.05
        and cert["stable"]
        and abs(cert["logz"]) <= 0.1
        and sw2 <= 3.0 * floor
    )
    out = dict(
        gate="i", passed=bool(passed), loss0=loss0, loss_end=loss_end,
        sw2=float(sw2), sw2_floor=float(floor), seconds=round(time.time() - t0, 1),
        **cert,
    )
    print(json.dumps(out, indent=2))
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "gate1.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("GATE1-PASS" if passed else "GATE1-FAIL")


if __name__ == "__main__":
    main()
