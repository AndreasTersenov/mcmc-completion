"""Zoo training pipeline: dataset builder (targets x contexts x sample pools,
padded to DMAX with per-target dim masks), jitted train step, checkpointing.

Whitening is per-context (frozen tokenization choice), so raw pools are
stored once per target and whitened on the fly inside the train step against
the drawn context's (mu, sigma); padding dims get FRESH N(0,1) noise every
visit (the head learns honest standard-normal marginals there, conditioned on
the d-feature).
"""

import pickle
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .context import generate_context
from .zoo import DMAX, logpdf, sample_target, sample_x


class ZooData(NamedTuple):
    tokens: jnp.ndarray   # (T, C, K, F) float32
    mu: jnp.ndarray       # (T, C, DMAX) float32 (pad dims: 0)
    sigma: jnp.ndarray    # (T, C, DMAX) float32 (pad dims: 1)
    pool: jnp.ndarray     # (T, N, DMAX) float32 raw samples, pad dims 0
    dim_mask: jnp.ndarray  # (T, DMAX) float32 1 for real dims
    d: jnp.ndarray        # (T,) int32


def build_zoo_dataset(key, specs, n_ctx, K, n_pool, temperature=1.0):
    """specs: list of (family, d) or (family, d, seed_offset). Returns
    (targets list, contexts list-of-lists, ZooData)."""
    targets, ctxs = [], []
    tok, mus, sigmas, pools, masks, ds = [], [], [], [], [], []
    for i, spec in enumerate(specs):
        family, d = spec[0], spec[1]
        off = spec[2] if len(spec) > 2 else 0
        kt, kc, kp = jr.split(jr.fold_in(key, 1000 * i + off), 3)
        t = sample_target(kt, family, d)
        targets.append(t)
        fn = lambda x, _t=t: logpdf(_t, x)
        cs = [generate_context(k, fn, d, K=K, temperature=temperature)
              for k in jr.split(kc, n_ctx)]
        ctxs.append(cs)
        tok.append(np.stack([np.asarray(c.tokens, np.float32) for c in cs]))
        mus.append(np.stack([
            np.concatenate([np.asarray(c.mu, np.float32), np.zeros(DMAX - d, np.float32)])
            for c in cs
        ]))
        sigmas.append(np.stack([
            np.concatenate([np.asarray(c.sigma, np.float32), np.ones(DMAX - d, np.float32)])
            for c in cs
        ]))
        x = np.asarray(sample_x(kp, t, n_pool), np.float32)
        pools.append(np.concatenate([x, np.zeros((n_pool, DMAX - d), np.float32)], axis=1))
        masks.append(np.concatenate([np.ones(d, np.float32), np.zeros(DMAX - d, np.float32)]))
        ds.append(d)
    data = ZooData(
        tokens=jnp.asarray(np.stack(tok)),
        mu=jnp.asarray(np.stack(mus)),
        sigma=jnp.asarray(np.stack(sigmas)),
        pool=jnp.asarray(np.stack(pools)),
        dim_mask=jnp.asarray(np.stack(masks)),
        d=jnp.asarray(np.array(ds, np.int32)),
    )
    return targets, ctxs, data


def make_train_step(model, tx, batch, n_targets, n_ctx, n_pool):
    from .cfm import cond_cfm_loss

    @jax.jit
    def step(params, opt_state, key, data):
        kt, kc, ki, kn, kl = jr.split(key, 5)
        ti = jr.randint(kt, (batch,), 0, n_targets)
        ci = jr.randint(kc, (batch,), 0, n_ctx)
        xi = jr.randint(ki, (batch,), 0, n_pool)
        x_raw = data.pool[ti, xi]                      # (B, DMAX)
        mu, sg = data.mu[ti, ci], data.sigma[ti, ci]
        mask = data.dim_mask[ti]
        noise = jr.normal(kn, x_raw.shape, dtype=x_raw.dtype)
        x1 = jnp.where(mask > 0, (x_raw - mu) / sg, noise)
        toks = data.tokens[ti, ci]                     # (B, K, F)
        loss, grads = jax.value_and_grad(cond_cfm_loss)(params, model, x1, toks, kl)
        import optax
        updates, opt_state = tx.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    return step


def save_checkpoint(path, params, opt_state, step_idx):
    with open(path, "wb") as f:
        pickle.dump(
            {
                "params": jax.tree_util.tree_map(np.asarray, params),
                "opt_state": jax.tree_util.tree_map(
                    lambda a: np.asarray(a) if hasattr(a, "shape") else a, opt_state
                ),
                "step": int(step_idx),
            },
            f,
        )


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)
