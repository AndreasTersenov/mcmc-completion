"""Closed-form validation for every auxiliary estimator M1/M2 rely on
(CLAUDE.md hard rule: validate every estimator against a closed-form case
before using it on an open one).

Covered here:
  - logsumexp vs scipy reference
  - streaming is_summary == direct estimators on the same sample stream
  - target log-densities: analytic gradients vs finite differences
  - target samplers: moments vs closed forms (incl. funnel via the
    variance-normalized identity E[x_i^2 / (c^2 e^v)] = 1, which is
    finite-variance unlike the raw moment)
  - MALA: Gaussian target moments, temperature scaling of the stationary law
  - sliced W2: mean-shifted isotropic Gaussians, SW2^2 = ||delta||^2 / d
  - energy-context posterior: with exact in-family energy observations the
    posterior MAP recovers theta* (peak-at-truth sanity, Gaussian-location family)
"""

import numpy as np
import pytest
import scipy.special

from stage0.utils import logsumexp
from stage0.is_estimators import ess_frac_from_logw, logz_from_logw, is_summary
from stage0.targets import Gaussian, GMM, Funnel, DoubleWell
from stage0.mala import mala_chains
from stage0.sliced_w2 import sliced_w2_squared
from stage0.posterior import grid_posterior, energy_context_loglik


# ---------- logsumexp ----------

def test_logsumexp_matches_scipy():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((7, 11)) * 50
    assert logsumexp(a) == pytest.approx(scipy.special.logsumexp(a), rel=1e-13)
    np.testing.assert_allclose(
        logsumexp(a, axis=1), scipy.special.logsumexp(a, axis=1), rtol=1e-13
    )
    assert np.isinf(logsumexp(np.array([-np.inf, -np.inf])))


# ---------- streaming summary ----------

def test_is_summary_matches_direct():
    d, eps, n = 4, 0.3, 100_000

    def sample_fn(rng, m):
        return rng.standard_normal((m, d)) + eps

    def logw_fn(x):
        return (eps**2 / 2.0) * d - eps * x.sum(axis=1)

    direct_x = sample_fn(np.random.default_rng(77), n)
    direct_logw = logw_fn(direct_x)
    res = is_summary(sample_fn, logw_fn, n=n, batch=n // 4, seed=77)
    # chunked standard_normal draws reproduce the same stream -> near-exact match
    assert res["ess_frac"] == pytest.approx(ess_frac_from_logw(direct_logw), rel=1e-10)
    assert res["logz"] == pytest.approx(logz_from_logw(direct_logw), abs=1e-10)
    assert res["n"] == n


# ---------- gradients vs finite differences ----------

def _check_grad(target, x, rtol=1e-4):
    g = target.grad_logpdf_u(x)
    h = 1e-5
    for i in range(x.shape[1]):
        xp, xm = x.copy(), x.copy()
        xp[:, i] += h
        xm[:, i] -= h
        fd = (target.logpdf_u(xp) - target.logpdf_u(xm)) / (2 * h)
        np.testing.assert_allclose(g[:, i], fd, rtol=rtol, atol=1e-6)


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(3)
    x2 = rng.standard_normal((5, 2)) * 1.5
    x4 = rng.standard_normal((5, 4)) * 1.5
    _check_grad(Gaussian(mean=np.array([0.5, -1.0]), var=1.7), x2)
    _check_grad(
        GMM(means=np.array([[-2.0, 0.0], [2.0, 1.0]]),
            weights=np.array([0.4, 0.6]), var=np.array([1.0, 0.7])),
        x2,
    )
    _check_grad(Funnel(d=4, sigma_v=3.0, cond_scale=1.2), x4)
    _check_grad(DoubleWell(d=2, a=1.5, b=2.0), x2)


# ---------- sampler moments ----------

def test_gmm_sampler_moments():
    rng = np.random.default_rng(4)
    gmm = GMM(means=np.array([[-3.0, 0.0], [3.0, 1.0]]),
              weights=np.array([0.3, 0.7]), var=np.array([1.0, 2.0]))
    x = gmm.sample(rng, 400_000)
    mean_cf = 0.3 * np.array([-3.0, 0.0]) + 0.7 * np.array([3.0, 1.0])
    np.testing.assert_allclose(x.mean(axis=0), mean_cf, atol=0.02)
    # E[x0^2] = sum_j w_j (var_j + mu_j0^2)
    ex2_cf = 0.3 * (1.0 + 9.0) + 0.7 * (2.0 + 9.0)
    assert (x[:, 0] ** 2).mean() == pytest.approx(ex2_cf, rel=0.02)


def test_funnel_sampler_structure():
    rng = np.random.default_rng(5)
    f = Funnel(d=4, sigma_v=3.0, cond_scale=1.2)
    x = f.sample(rng, 300_000)
    v = x[:, 0]
    assert v.std() == pytest.approx(3.0, rel=0.02)
    # variance-normalized conditional check (finite-variance identity)
    z2 = x[:, 1:] ** 2 / (1.2**2 * np.exp(v))[:, None]
    assert z2.mean() == pytest.approx(1.0, rel=0.02)


def test_double_well_sampler_matches_quadrature():
    rng = np.random.default_rng(6)
    dw = DoubleWell(d=2, a=1.5, b=2.0)
    x = dw.sample(rng, 300_000)
    # quadrature reference for E[x1^2] under exp(-a (x1^2-b)^2)
    t = np.linspace(-6, 6, 200_001)
    w = np.exp(-1.5 * (t**2 - 2.0) ** 2)
    ex2 = np.trapezoid(t**2 * w, t) / np.trapezoid(w, t)
    assert (x[:, 0] ** 2).mean() == pytest.approx(ex2, rel=0.02)
    # second coordinate is standard normal
    assert x[:, 1].std() == pytest.approx(1.0, rel=0.02)
    assert np.isfinite(dw.log_Z)


# ---------- MALA ----------

def test_mala_samples_gaussian():
    target = Gaussian(mean=np.array([1.0, -2.0]), var=1.0)
    xs, acc = mala_chains(
        target, x0=np.zeros((8, 2)), n_steps=4000, step=0.6,
        rng=np.random.default_rng(11),
    )
    keep = xs[:, 1000:, :].reshape(-1, 2)
    assert 0.3 < acc < 0.98
    np.testing.assert_allclose(keep.mean(axis=0), [1.0, -2.0], atol=0.08)
    np.testing.assert_allclose(keep.std(axis=0), [1.0, 1.0], atol=0.08)


def test_mala_temperature_scales_variance():
    # p^(1/T) for N(0,1) is N(0, T): std must be sqrt(T)
    target = Gaussian(mean=np.zeros(1), var=1.0)
    xs, acc = mala_chains(
        target, x0=np.zeros((8, 1)), n_steps=6000, step=1.0, temperature=4.0,
        rng=np.random.default_rng(12),
    )
    keep = xs[:, 1500:, :].ravel()
    assert keep.std() == pytest.approx(2.0, rel=0.06)


# ---------- sliced W2 ----------

def test_sliced_w2_shifted_gaussians():
    rng = np.random.default_rng(13)
    d, n = 8, 8192
    delta = np.zeros(d)
    delta[0] = 1.5
    x = rng.standard_normal((n, d))
    y = rng.standard_normal((n, d)) + delta
    sw2 = sliced_w2_squared(x, y, n_proj=1024, rng=np.random.default_rng(14))
    assert sw2 == pytest.approx(1.5**2 / d, abs=0.05)
    same = sliced_w2_squared(
        rng.standard_normal((n, d)), rng.standard_normal((n, d)),
        n_proj=1024, rng=np.random.default_rng(15),
    )
    assert same < 0.01


# ---------- energy-context posterior: peak at truth ----------

def test_energy_context_posterior_peaks_at_truth():
    # Gaussian-location family in d=2, theta-dim=2: E_theta(x) = ||x-theta||^2/2.
    rng = np.random.default_rng(16)
    theta_star = np.array([1.2, -0.7])
    xs = rng.standard_normal((32, 2)) * 2.0
    energies = 0.5 * ((xs - theta_star) ** 2).sum(axis=1)
    grads = xs - theta_star  # grad_x E

    def family_energy(theta, x):
        # theta (m,2), x (K,2) -> E (m,K), gradE (m,K,2)
        diff = x[None, :, :] - theta[:, None, :]
        return 0.5 * (diff**2).sum(axis=2), diff

    def loglik(theta):
        return energy_context_loglik(
            family_energy, theta, xs, energies, sigma_e=0.05,
            grads=grads, sigma_g=0.05,
        )

    res = grid_posterior(
        axes=[np.linspace(-4, 4, 401), np.linspace(-4, 4, 401)],
        loglik_fn=loglik,
    )
    np.testing.assert_allclose(res["mean"], theta_star, atol=0.05)
    np.testing.assert_allclose(res["map"], theta_star, atol=0.05)
