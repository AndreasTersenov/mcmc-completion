"""Context generation & tokenization (frozen protocol: 4 MALA chains x K/4
steps, overdispersed inits, context-CENTERED energies; whitening is free
periphery but must round-trip exactly).

API under test (ics.context):
  generate_context(key, logpdf_fn, d, K, n_chains=4, temperature=1.0) -> Context
  Context fields: x_raw (K,d), energy (K,), grad (K,d), mu (d,), sigma (d,),
                  tokens (K, TOKEN_DIM), accept (scalar)
  TOKEN_DIM, token layout: [x_white pad DMAX | E_centered/scale | grad feature
                            pad DMAX | log e_scale, log g_scale, d/DMAX]
  whiten_apply / whiten_invert for sample-space round-trips.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ics.context import TOKEN_DIM, generate_context, whiten_invert
from ics.zoo import DMAX, logpdf, sample_target


def _logpdf_fn(target):
    return lambda x: logpdf(target, x)


def test_shapes_and_determinism():
    t = sample_target(jr.key(0), "gmm", 4)
    c1 = generate_context(jr.key(1), _logpdf_fn(t), 4, K=64)
    c2 = generate_context(jr.key(1), _logpdf_fn(t), 4, K=64)
    assert c1.x_raw.shape == (64, 4) and c1.energy.shape == (64,)
    assert c1.grad.shape == (64, 4) and c1.tokens.shape == (64, TOKEN_DIM)
    np.testing.assert_array_equal(np.asarray(c1.tokens), np.asarray(c2.tokens))
    assert 0.0 < float(c1.accept) <= 1.0


def test_energy_offset_invariance():
    # frozen rule: tokens must not change if the target energy shifts by a constant
    t = sample_target(jr.key(2), "funnel", 4)
    base = _logpdf_fn(t)
    c_a = generate_context(jr.key(3), base, 4, K=32)
    c_b = generate_context(jr.key(3), lambda x: base(x) + 137.5, 4, K=32)
    np.testing.assert_allclose(
        np.asarray(c_a.tokens), np.asarray(c_b.tokens), rtol=1e-9, atol=1e-9
    )


def test_energy_tokens_are_centered_and_scaled():
    t = sample_target(jr.key(4), "warp", 8)
    c = generate_context(jr.key(5), _logpdf_fn(t), 8, K=128)
    e_tok = np.asarray(c.tokens[:, DMAX])
    assert abs(e_tok.mean()) < 1e-6
    assert abs(e_tok.std() - 1.0) < 0.05


def test_whitening_roundtrip_and_grad_feature():
    t = sample_target(jr.key(6), "gmm", 4)
    fn = _logpdf_fn(t)
    c = generate_context(jr.key(7), fn, 4, K=32)
    x_white = np.asarray(c.tokens[:, :4])
    x_rec = whiten_invert(jnp.asarray(x_white), c.mu, c.sigma)
    np.testing.assert_allclose(np.asarray(x_rec), np.asarray(c.x_raw), rtol=1e-6, atol=1e-8)
    # padding dims are zero
    assert np.abs(np.asarray(c.tokens[:, 4:DMAX])).max() == 0.0
    # gradient of ENERGY = -grad logpdf, in whitened coords, shared scalar scale
    g_energy = -np.asarray(
        jax.vmap(jax.grad(lambda xi: fn(xi[None, :])[0]))(c.x_raw)
    )
    g_white = np.asarray(c.sigma)[None, :] * g_energy
    g_scale = g_white.std() + 1e-8
    np.testing.assert_allclose(
        np.asarray(c.tokens[:, DMAX + 1 : DMAX + 1 + 4]),
        g_white / g_scale,
        rtol=1e-4, atol=1e-6,
    )


def test_chain_structure_is_4_chains():
    # K tokens = 4 chains x K/4 steps; chains from overdispersed inits differ
    t = sample_target(jr.key(8), "dwell", 2)
    c = generate_context(jr.key(9), _logpdf_fn(t), 2, K=16)
    xs = np.asarray(c.x_raw).reshape(4, 4, 2)  # (chains, steps, d)
    starts = xs[:, 0, :]
    assert np.unique(starts, axis=0).shape[0] == 4
