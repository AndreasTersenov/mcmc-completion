"""Smoke test for the zoo training pipeline: shapes, finiteness, and that a
few steps run and reduce loss on a tiny dataset (CPU-scale)."""

import jax.numpy as jnp
import jax.random as jr
import optax

from ics.models import ICSModel
from ics.train import build_zoo_dataset, make_train_step
from ics.zoo import DMAX


def test_pipeline_smoke():
    specs = [("gmm", 2), ("funnel", 4)]
    targets, ctxs, data = build_zoo_dataset(jr.key(0), specs, n_ctx=2, K=8, n_pool=256)
    assert data.tokens.shape[:3] == (2, 2, 8)
    assert data.pool.shape == (2, 256, DMAX)
    assert float(data.dim_mask[0].sum()) == 2 and float(data.dim_mask[1].sum()) == 4

    model = ICSModel(enc_dim=32, enc_hidden=64, head_hidden=(64, 64))
    params = model.init(
        jr.key(1), jnp.ones((2, DMAX), jnp.float32), jnp.ones((2,), jnp.float32),
        data.tokens[:, 0][:2],
    )["params"]
    tx = optax.adam(1e-3)
    opt = tx.init(params)
    step = make_train_step(model, tx, batch=64, n_targets=2, n_ctx=2, n_pool=256)
    losses = []
    for i in range(30):
        params, opt, loss = step(params, opt, jr.key(10 + i), data)
        losses.append(float(loss))
    assert all(jnp.isfinite(jnp.asarray(losses)))
    assert min(losses[-5:]) < losses[0]
