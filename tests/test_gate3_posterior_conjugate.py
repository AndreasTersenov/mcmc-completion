"""Validation gate 3 (CLAUDE.md): SMC / grid posterior matches the conjugate
closed form on a linear-Gaussian family.

Model: theta ~ N(m0, tau0^2) per dim (independent), y_i | theta ~ N(theta, sigma^2).
Conjugate posterior per dim: N(m_n, tau_n^2) with
    tau_n^2 = 1 / (1/tau0^2 + n/sigma^2)
    m_n     = tau_n^2 * (m0/tau0^2 + sum(y)/sigma^2)

Both posterior engines used in M2 are gated here:
  - grid_posterior (1-d and 2-d grids)
  - smc_posterior  (adaptive-tempering SMC with MH rejuvenation)
"""

import numpy as np
import pytest

from stage0.posterior import grid_posterior, smc_posterior

M0, TAU0 = 1.0, 2.0
SIGMA = 1.5


def _conjugate(y, m0=M0, tau0=TAU0, sigma=SIGMA):
    n = len(y)
    tau_n2 = 1.0 / (1.0 / tau0**2 + n / sigma**2)
    m_n = tau_n2 * (m0 / tau0**2 + y.sum() / sigma**2)
    return m_n, np.sqrt(tau_n2)


def _loglik_1d(theta, y):
    # theta: (m, 1) grid/particle array -> (m,)
    r = y[None, :] - theta[:, 0:1]
    return -0.5 * (r**2).sum(axis=1) / SIGMA**2


def _loglik_2d(theta, y1, y2):
    r1 = y1[None, :] - theta[:, 0:1]
    r2 = y2[None, :] - theta[:, 1:2]
    return -0.5 * ((r1**2).sum(axis=1) + (r2**2).sum(axis=1)) / SIGMA**2


def _logprior_1d(theta):
    return -0.5 * ((theta[:, 0] - M0) / TAU0) ** 2


def _logprior_2d(theta):
    return -0.5 * (((theta - M0) / TAU0) ** 2).sum(axis=1)


def test_grid_posterior_1d_matches_conjugate():
    rng = np.random.default_rng(7)
    y = 2.0 + SIGMA * rng.standard_normal(10)
    m_n, s_n = _conjugate(y)
    res = grid_posterior(
        axes=[np.linspace(-10, 10, 4001)],
        loglik_fn=lambda th: _loglik_1d(th, y),
        logprior_fn=_logprior_1d,
    )
    assert res["mean"][0] == pytest.approx(m_n, abs=1e-3)
    assert res["std"][0] == pytest.approx(s_n, rel=1e-3)


def test_grid_posterior_2d_matches_conjugate():
    rng = np.random.default_rng(8)
    y1 = 2.0 + SIGMA * rng.standard_normal(12)
    y2 = -1.0 + SIGMA * rng.standard_normal(12)
    m1, s1 = _conjugate(y1)
    m2, s2 = _conjugate(y2)
    res = grid_posterior(
        axes=[np.linspace(-8, 8, 801), np.linspace(-8, 8, 801)],
        loglik_fn=lambda th: _loglik_2d(th, y1, y2),
        logprior_fn=_logprior_2d,
    )
    assert res["mean"] == pytest.approx(np.array([m1, m2]), abs=2e-3)
    assert res["std"] == pytest.approx(np.array([s1, s2]), rel=2e-3)


def test_smc_posterior_matches_conjugate_2d():
    rng = np.random.default_rng(9)
    y1 = 2.0 + SIGMA * rng.standard_normal(12)
    y2 = -1.0 + SIGMA * rng.standard_normal(12)
    m1, s1 = _conjugate(y1)
    m2, s2 = _conjugate(y2)
    res = smc_posterior(
        prior_sample=lambda r, n: M0 + TAU0 * r.standard_normal((n, 2)),
        logprior_fn=_logprior_2d,
        loglik_fn=lambda th: _loglik_2d(th, y1, y2),
        n_particles=4000,
        rng=np.random.default_rng(10),
    )
    # MC tolerance: mean within 0.15 posterior sd, sd within 15%
    assert abs(res["mean"][0] - m1) < 0.15 * s1
    assert abs(res["mean"][1] - m2) < 0.15 * s2
    assert res["std"][0] == pytest.approx(s1, rel=0.15)
    assert res["std"][1] == pytest.approx(s2, rel=0.15)
