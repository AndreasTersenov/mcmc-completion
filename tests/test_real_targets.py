"""Validation gates for the Readout-C real targets (CLAUDE.md: every estimator
validated against a closed form before use on an open case).

- eight-schools: logpdf at a hand-computable point matches an independent
  scipy computation term by term.
- gym banana: logpdf matches the change-of-variables closed form, and exact
  transform samples have the analytic moments.
- real-target augmented logpdf: identical to ics.eval.augmented_true_logpdf
  on a zoo target wrapped as a callable (the certificate path is unchanged).
- WL surrogate: only validated here if the fitted npz exists (its own build
  script enforces the held-out tolerance gate); skipped otherwise.
"""

import os

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy import stats

from ics.context import generate_context
from ics.eval import augmented_true_logpdf
from ics.real import (EIGHT_SCHOOLS_SIGMA, EIGHT_SCHOOLS_Y, eight_schools_logpdf,
                      gym_banana_logpdf, gym_banana_sample,
                      real_augmented_logpdf)
from ics.zoo import DMAX, logpdf as zoo_logpdf, sample_target


def test_eight_schools_logpdf_closed_form():
    x = np.zeros((1, 10))
    x[0, 0], x[0, 1] = 2.0, 0.5  # mu=2, log tau=0.5
    x[0, 2:] = np.linspace(-1, 1, 8)
    mu, tau, z = 2.0, np.exp(0.5), np.linspace(-1, 1, 8)
    want = stats.norm.logpdf(mu, 0, 5)
    want += stats.halfcauchy.logpdf(tau, scale=5) + 0.5  # + log tau Jacobian
    want += stats.norm.logpdf(z).sum()
    want += stats.norm.logpdf(np.asarray(EIGHT_SCHOOLS_Y), mu + tau * z,
                              np.asarray(EIGHT_SCHOOLS_SIGMA)).sum()
    got = float(eight_schools_logpdf(jnp.asarray(x))[0])
    assert abs(got - want) < 1e-8


def test_gym_banana_logpdf_and_moments():
    pts = np.array([[0.0, 3.0], [10.0, -2.0], [-15.0, 5.0]])
    y2 = pts[:, 1] + 0.03 * (pts[:, 0] ** 2 - 100.0)
    want = stats.norm.logpdf(pts[:, 0], 0, 10) + stats.norm.logpdf(y2, 0, 1)
    got = np.asarray(gym_banana_logpdf(jnp.asarray(pts)))
    assert np.allclose(got, want, atol=1e-10)

    s = np.asarray(gym_banana_sample(jr.key(0), 200_000))
    # closed-form moments: E[x1]=0, Var[x1]=100; E[x2]=E[y2]-0.03(E[x1^2]-100)=0
    assert abs(s[:, 0].mean()) < 0.15
    assert abs(s[:, 0].var() / 100.0 - 1.0) < 0.02
    assert abs(s[:, 1].mean()) < 0.05
    # Var[x2] = 1 + 0.03^2 Var[x1^2] = 1 + 0.0009 * 2*100^2 = 19
    assert abs(s[:, 1].var() / 19.0 - 1.0) < 0.05


def test_real_augmented_logpdf_matches_zoo_path():
    t = sample_target(jr.key(7), "gmm", 2)
    fn = lambda x: zoo_logpdf(t, x)
    ctx = generate_context(jr.key(8), fn, 2, K=32, temperature=1.0,
                           aux_tokens=True)
    x_full = jr.normal(jr.key(9), (16, DMAX), jnp.float64)
    want = np.asarray(augmented_true_logpdf(t, ctx, x_full))
    got = np.asarray(real_augmented_logpdf(fn, 2, ctx, x_full))
    assert np.allclose(got, want, atol=1e-10)


def test_wl_surrogate_if_built():
    path = os.path.join(os.path.dirname(__file__), "..", "results",
                        "wl_surrogate.npz")
    if not os.path.exists(path):
        pytest.skip("WL surrogate not built yet (grid job pending)")
    from ics.real import WLBandpower

    wl = WLBandpower(path)
    assert wl.d == 3
    lp = wl.logpdf(jnp.zeros((2, 3), jnp.float64))
    assert np.all(np.isfinite(np.asarray(lp)))
    meta = np.load(path)
    assert float(meta["heldout_max_rel_err"]) < 0.01  # the pre-registered gate
