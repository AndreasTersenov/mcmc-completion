"""Backpressure (c): the FM-head single-target overfit gate as an executable,
deterministic test. This is the CPU-scale twin of Karpathy gate (i): a small
velocity MLP trained on exact samples of ONE fixed 2-d GMM must (1) cut its
CFM loss by >60% from the untrained value and (2) produce samples whose
sliced-W2^2 against fresh exact samples is < 0.15 (same-p floor ~0.005; an
untrained head sits around ~1-10 on this target).

Runtime budget: well under the Stop-hook limit on 8 login cores.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from jax_flows import TimeConditionedMLP, cfm_loss, cfm_sample

from ics.zoo import sample_target, sample_x
from stage0.sliced_w2 import sliced_w2_squared

N_STEPS = 2500
BATCH = 512


def test_fm_single_target_overfit_gate():
    target = sample_target(jr.key(42), "gmm", 2)  # 3 comps, well separated
    x_train = sample_x(jr.key(43), target, 8192).astype(jnp.float32)

    model = TimeConditionedMLP(hidden_dims=(256, 256), output_dim=2)
    params = model.init(jr.key(44), jnp.ones((1, 2), jnp.float32), jnp.ones((1,), jnp.float32))["params"]
    tx = optax.adam(optax.cosine_decay_schedule(2e-3, N_STEPS))
    opt_state = tx.init(params)

    @jax.jit
    def step(params, opt_state, key):
        kb, kl = jr.split(key)
        idx = jr.randint(kb, (BATCH,), 0, x_train.shape[0])
        loss, grads = jax.value_and_grad(cfm_loss)(params, x_train[idx], kl, model)
        updates, opt_state = tx.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    keys = jr.split(jr.key(45), N_STEPS)
    loss0 = float(cfm_loss(params, x_train[:2048], jr.key(46), model))
    loss = None
    for i in range(N_STEPS):
        params, opt_state, loss = step(params, opt_state, keys[i])
    loss_end = float(cfm_loss(params, x_train[:2048], jr.key(46), model))

    assert loss_end < 0.4 * loss0, f"loss did not drop enough: {loss0=} {loss_end=}"

    samples = cfm_sample(model, params, jr.key(47), (4096, 2), n_steps=100, solver="heun")
    fresh = sample_x(jr.key(48), target, 4096)
    sw2 = sliced_w2_squared(np.asarray(samples, np.float64), np.asarray(fresh, np.float64),
                            n_proj=128, rng=np.random.default_rng(0))
    assert sw2 < 0.15, f"overfit gate failed: sliced-W2^2 = {sw2:.4f}"
