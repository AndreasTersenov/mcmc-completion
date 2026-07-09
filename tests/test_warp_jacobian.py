"""Backpressure (b): warped-Gaussian log-det-Jacobian checks.

The warp families (train: sinh-arcsinh diagonal warp between rotations;
held-out: banana/Rosenbrock shear chain) must satisfy, at random points:
  1. forward(inverse(x)) == x and inverse(forward(z)) == z (roundtrip)
  2. logpdf computed via change of variables == N(z) - logdet(J_forward(z))
     with logdet cross-checked against jax.jacfwd's explicit determinant
  3. banana shear chain has logdet == 0 (unit-triangular Jacobian)

API under test (ics.warps):
  make_warp(key, d, kind)         kind in {"sinharcsinh", "banana"}
  warp_forward(w, z) -> x
  warp_inverse(w, x) -> z
  warp_logdet(w, z) -> (n,)       log|det dx/dz| at z
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from ics.warps import make_warp, warp_forward, warp_inverse, warp_logdet

KINDS = ["sinharcsinh", "banana"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("d", [2, 4, 8])
def test_roundtrip(kind, d):
    w = make_warp(jr.key(hash((kind, d)) % 2**31), d, kind)
    z = jr.normal(jr.key(1), (128, d), dtype=jnp.float64)
    x = warp_forward(w, z)
    z_back = warp_inverse(w, x)
    np.testing.assert_allclose(np.asarray(z_back), np.asarray(z), rtol=1e-8, atol=1e-8)
    x2 = warp_forward(w, z_back)
    np.testing.assert_allclose(np.asarray(x2), np.asarray(x), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("d", [2, 4, 8])
def test_logdet_matches_numeric_jacobian(kind, d):
    w = make_warp(jr.key(100 + hash((kind, d)) % 2**31), d, kind)
    z = jr.normal(jr.key(2), (16, d), dtype=jnp.float64) * 1.5

    def fwd_single(zi):
        return warp_forward(w, zi[None, :])[0]

    J = jax.vmap(jax.jacfwd(fwd_single))(z)  # (16, d, d)
    sign, numeric = np.linalg.slogdet(np.asarray(J))
    assert (sign > 0).all(), "warp must be orientation-preserving"
    analytic = np.asarray(warp_logdet(w, z))
    np.testing.assert_allclose(analytic, numeric, rtol=1e-8, atol=1e-8)


def test_banana_logdet_is_zero_before_scaling():
    # the shear chain itself is unit-triangular; any nonzero logdet must come
    # only from the diagonal scale factors, which make_warp stores explicitly
    w = make_warp(jr.key(5), 4, "banana")
    z = jr.normal(jr.key(6), (32, 4), dtype=jnp.float64)
    ld = np.asarray(warp_logdet(w, z))
    expected = np.full(32, float(jnp.log(w.scale).sum()))
    np.testing.assert_allclose(ld, expected, rtol=1e-10, atol=1e-10)
