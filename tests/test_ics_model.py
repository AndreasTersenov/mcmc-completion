"""ICS model plumbing: permutation-invariant encoder, conditional CFM, and the
CNF log-density used by the certificate (this last one is validated against a
closed form BEFORE it is ever used on an open case — stage-0 hard rule).

API under test (ics.models / ics.cfm):
  ContextEncoder(...)(tokens (B,K,F)) -> (B,C)
  ICSVelocity(...)  .apply(vars, x (B,DMAX), t (B,), cond (B,C)) -> (B,DMAX)
  ics.cfm.cond_cfm_loss(params, model, x1 (B,DMAX), tokens (B,K,F), key)
  ics.cfm.cond_cfm_sample(model, params, tokens (K,F), key, n, n_steps) -> (n,DMAX)
  ics.cfm.cnf_logpdf(velocity_fn, x (n,D), n_steps) -> (n,)   [generic]
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ics.cfm import cnf_logpdf, cond_cfm_loss, cond_cfm_sample
from ics.context import TOKEN_DIM
from ics.models import ContextEncoder, ICSModel
from ics.zoo import DMAX

MU = jnp.array([1.5, -0.5])
S = 1.7


def _toy_model():
    model = ICSModel(enc_dim=32, enc_hidden=64, head_hidden=(64, 64))
    tokens = jr.normal(jr.key(0), (2, 8, TOKEN_DIM))
    x = jr.normal(jr.key(1), (2, DMAX))
    t = jnp.array([0.3, 0.7])
    params = model.init(jr.key(2), x, t, tokens)["params"]
    return model, params, tokens, x, t


def test_encoder_permutation_invariance():
    enc = ContextEncoder(enc_dim=32, hidden=64)
    tokens = jr.normal(jr.key(3), (2, 16, TOKEN_DIM))
    params = enc.init(jr.key(4), tokens)["params"]
    out1 = enc.apply({"params": params}, tokens)
    perm = jr.permutation(jr.key(5), 16)
    out2 = enc.apply({"params": params}, tokens[:, perm, :])
    np.testing.assert_allclose(np.asarray(out1), np.asarray(out2), rtol=1e-5, atol=1e-6)


def test_velocity_depends_on_context():
    model, params, tokens, x, t = _toy_model()
    v1 = model.apply({"params": params}, x, t, tokens)
    v2 = model.apply({"params": params}, x, t, tokens + 1.0)
    assert v1.shape == (2, DMAX)
    assert float(jnp.abs(v1 - v2).max()) > 1e-4


def test_cond_loss_grads_reach_encoder_and_head():
    model, params, tokens, _, _ = _toy_model()
    x1 = jr.normal(jr.key(6), (2, DMAX))
    loss, grads = jax.value_and_grad(cond_cfm_loss)(params, model, x1, tokens, jr.key(7))
    assert bool(jnp.isfinite(loss))
    leaves = jax.tree_util.tree_leaves_with_path(grads)
    enc_norm = sum(
        float(jnp.abs(g).sum()) for p, g in leaves if "encoder" in jax.tree_util.keystr(p)
    )
    head_norm = sum(
        float(jnp.abs(g).sum()) for p, g in leaves if "encoder" not in jax.tree_util.keystr(p)
    )
    assert enc_norm > 0 and head_norm > 0


def test_cond_sample_shape_finite():
    model, params, tokens, _, _ = _toy_model()
    s = cond_cfm_sample(model, params, tokens[0], jr.key(8), n=64, n_steps=16)
    assert s.shape == (64, DMAX)
    assert bool(jnp.isfinite(s).all())


def test_cnf_logpdf_matches_analytic_gaussian():
    # closed-form gate for the certificate's q-density: analytic OT velocity
    # for N(MU, S^2 I_2) must yield log q == Gaussian logpdf
    def velocity_fn(x, t):
        v_t = (1.0 - t) ** 2 + (t * S) ** 2
        return MU + (t * S**2 - (1.0 - t)) / v_t * (x - t * MU)

    x = MU + S * jr.normal(jr.key(9), (256, 2), dtype=jnp.float64)
    lq = cnf_logpdf(velocity_fn, x, n_steps=400)
    d = 2
    lp = -0.5 * jnp.sum((x - MU) ** 2, axis=1) / S**2 - 0.5 * d * jnp.log(
        2 * jnp.pi * S**2
    )
    np.testing.assert_allclose(np.asarray(lq), np.asarray(lp), rtol=1e-3, atol=2e-3)


def test_attention_encoder_permutation_invariance_and_mask():
    enc = ContextEncoder(enc_dim=32, hidden=64, n_attn=2)
    tokens = jr.normal(jr.key(30), (2, 16, TOKEN_DIM))
    params = enc.init(jr.key(31), tokens)["params"]
    out1 = enc.apply({"params": params}, tokens)
    perm = jr.permutation(jr.key(32), 16)
    out2 = enc.apply({"params": params}, tokens[:, perm, :])
    np.testing.assert_allclose(np.asarray(out1), np.asarray(out2), rtol=2e-4, atol=1e-5)
    # masking out tokens must equal dropping them
    mask = jnp.ones((2, 16)).at[:, 10:].set(0.0)
    out_masked = enc.apply({"params": params}, tokens, mask)
    out_dropped = enc.apply({"params": params}, tokens[:, :10, :])
    np.testing.assert_allclose(np.asarray(out_masked), np.asarray(out_dropped),
                               rtol=2e-4, atol=1e-5)


def test_aux_summary_tokens():
    from ics.context import generate_context_for_target
    from ics.zoo import sample_target

    t = sample_target(jr.key(33), "gmm", 4)
    c = generate_context_for_target(jr.key(34), t, K=64, aux_tokens=True)
    assert c.tokens.shape == (68, TOKEN_DIM)  # 64 chain points + 4 summaries
    flags = np.asarray(c.tokens[:, -1])
    assert flags[:64].max() == 0.0 and flags[64:].min() == 1.0
