"""Backpressure (a): every zoo family's exact sampler must agree with its
normalized log-density.

Strategy (stage-0 discipline: closed forms where they exist, exact identities
where possible, end-to-end at low d):

- d=2, ALL families: (i) dense-grid integration of exp(logpdf) == 1
  (normalization, end-to-end); (ii) the identity E_p[r/p] = 1 with a
  sample-centered mixture reference r (sampler<->density consistency).
  [A KDE-style reference is only sound at low d — in d=16 no 64-center
  mixture covers a target, and the identity's estimator dies for CORRECT
  code; high-d correctness is carried by the structural checks below plus
  the d-agnostic code paths.]
- d in {8, 16}, per family: exact structural checks — GMM closed-form
  mean/cov; warp/banana base recovery z = f^-1(x) ~ N(0, I) with full
  covariance (exact given the Jacobian tests); dwell per-well quadrature
  moments + unit-Gaussian rest; funnel v-std + variance-normalized
  conditional identity; funnelmix closed-form mean.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from ics.warps import warp_inverse
from ics.zoo import (
    _DW_GRID,
    FAMILIES_HELDOUT,
    FAMILIES_TRAIN,
    logpdf,
    mode_centers,
    sample_target,
    sample_x,
)

ALL_FAMILIES = FAMILIES_TRAIN + FAMILIES_HELDOUT


# --------------------------------------------------------------- d=2, all families

@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_grid_normalization_d2(family):
    t = sample_target(jr.key(hash(("norm", family)) % 2**31), family, 2)
    x = np.asarray(sample_x(jr.key(1), t, 20_000), dtype=np.float64)
    lo = x.min(axis=0) - 4.0
    hi = x.max(axis=0) + 4.0
    g0 = np.linspace(lo[0], hi[0], 1201)
    g1 = np.linspace(lo[1], hi[1], 1201)
    X, Y = np.meshgrid(g0, g1, indexing="ij")
    pts = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=1))
    lp = np.asarray(logpdf(t, pts), dtype=np.float64).reshape(1201, 1201)
    z = np.trapezoid(np.trapezoid(np.exp(lp), g1, axis=1), g0)
    assert abs(z - 1.0) < 2e-2, f"{family}: exp(logpdf) integrates to {z:.4f}"


# The generic identity below is only variance-stable for light-tailed
# families: for funnel/funnelmix/dwell geometries ANY fixed-bandwidth
# Gaussian-mixture reference gives a divergent-variance estimator (the same
# math as stage-0's "no Gaussian proposal certifies a funnel"). Those three
# families get exact structural checks at d=2/8/16 instead (below), plus the
# d=2 grid-normalization test above which covers all six families.
KDE_IDENTITY_FAMILIES = ("gmm", "warp", "banana")


@pytest.mark.parametrize("family", KDE_IDENTITY_FAMILIES)
def test_sampler_density_consistency_d2(family):
    """E_p[r/p] = 1, r = mixture of Gaussians at independent sample centers."""
    n, m_centers, h = 200_000, 64, 0.5
    key = jr.key(hash((family, 2)) % 2**31)
    k1, k_eval, k_ctr = jr.split(key, 3)
    t = sample_target(k1, family, 2)
    x = np.asarray(sample_x(k_eval, t, n), dtype=np.float64)
    centers = np.asarray(sample_x(k_ctr, t, m_centers), dtype=np.float64)
    sig = h * (x.std(axis=0) + 1e-9)
    diff = (x[:, None, :] - centers[None, :, :]) / sig[None, None, :]
    log_comp = -0.5 * (diff**2).sum(axis=2) - np.log(sig).sum() - np.log(2 * np.pi)
    mx = log_comp.max(axis=1, keepdims=True)
    log_r = mx[:, 0] + np.log(np.exp(log_comp - mx).sum(axis=1)) - np.log(m_centers)
    log_p = np.asarray(logpdf(t, jnp.asarray(x)), dtype=np.float64)
    logw = log_r - log_p
    shift = logw.max()
    w = np.exp(logw - shift)
    zhat = float(w.mean() * np.exp(shift))
    stderr = float(np.exp(shift) * w.std(ddof=1) / np.sqrt(n))
    ess = w.sum() ** 2 / (n * (w**2).sum())
    assert ess > 1e-3, f"{family}: consistency estimator degenerate ({ess=})"
    assert abs(zhat - 1.0) < 5.0 * stderr + 5e-3, f"{family}: {zhat=} {stderr=}"


# --------------------------------------------------------------- structural, d=8/16

@pytest.mark.parametrize("d", [8, 16])
def test_gmm_closed_form_moments(d):
    t = sample_target(jr.key(100 + d), "gmm", d)
    x = np.asarray(sample_x(jr.key(101), t, 500_000), dtype=np.float64)
    mask = np.asarray(t.k_mask)
    w = np.exp(np.asarray(t.log_weights))[mask]
    mu = np.asarray(t.means)[mask]
    chol = np.asarray(t.chols)[mask]
    covs = np.einsum("kij,klj->kil", chol, chol)
    mean_cf = (w[:, None] * mu).sum(0)
    cov_cf = np.einsum("k,kil->il", w, covs) + np.einsum(
        "k,ki,kl->il", w, mu, mu
    ) - np.outer(mean_cf, mean_cf)
    np.testing.assert_allclose(x.mean(0), mean_cf, atol=0.05)
    np.testing.assert_allclose(np.cov(x.T), cov_cf, atol=0.15 * max(1.0, cov_cf.max()))


@pytest.mark.parametrize("family", ["warp", "banana"])
@pytest.mark.parametrize("d", [8, 16])
def test_warp_base_recovery(family, d):
    """z = f^-1(samples) must be exactly N(0, I): sampler and density share f,
    and f's inverse/logdet are pinned by the Jacobian tests."""
    t = sample_target(jr.key(hash((family, d, "base")) % 2**31), family, d)
    x = sample_x(jr.key(3), t, 300_000)
    z = np.asarray(warp_inverse(t.warp, x), dtype=np.float64)
    np.testing.assert_allclose(z.mean(0), np.zeros(d), atol=0.02)
    np.testing.assert_allclose(np.cov(z.T), np.eye(d), atol=0.03)
    lp = np.asarray(logpdf(t, x[:1000]))
    assert np.isfinite(lp).all()


@pytest.mark.parametrize("d", [2, 8, 16])
def test_dwell_quadrature_moments(d):
    t = sample_target(jr.key(200 + d), "dwell", d)
    x = np.asarray(sample_x(jr.key(201), t, 400_000), dtype=np.float64)
    z = x @ np.asarray(t.Q)
    grid = np.asarray(_DW_GRID)
    for i in range(t.n_well):
        a, b = float(t.a[i]), float(t.b[i])
        w = np.exp(-a * (grid**2 - b) ** 2)
        ez2 = np.trapezoid(grid**2 * w, grid) / np.trapezoid(w, grid)
        assert abs((z[:, i] ** 2).mean() - ez2) < 0.02 * max(1.0, ez2)
    rest = z[:, t.n_well:]
    np.testing.assert_allclose(rest.mean(0), 0.0, atol=0.02)
    np.testing.assert_allclose(rest.std(0), 1.0, atol=0.02)


@pytest.mark.parametrize("d", [2, 8, 16])
def test_funnel_conditional_identity(d):
    # stage-0 trick: E[ y_i^2 / (c^2 e^v) ] = 1 for the conditional dims
    t = sample_target(jr.key(300 + d), "funnel", d)
    x = np.asarray(sample_x(jr.key(301), t, 300_000), dtype=np.float64)
    z = x @ np.asarray(t.Q)
    v = z[:, 0]
    assert abs(v.std() - float(t.sigma_v)) < 0.05 * float(t.sigma_v)
    ratio = z[:, 1:] ** 2 / (float(t.c) ** 2 * np.exp(v))[:, None]
    assert abs(ratio.mean() - 1.0) < 0.03


def test_funnelmix_closed_form_mean():
    t = sample_target(jr.key(400), "funnelmix", 8)
    x = np.asarray(sample_x(jr.key(401), t, 500_000), dtype=np.float64)
    w = np.exp(np.asarray(t.log_weights))
    mean_cf = (w[:, None] * np.asarray(t.shifts)).sum(0)  # funnel comp mean = shift
    np.testing.assert_allclose(x.mean(0), mean_cf, atol=0.15)


def test_funnelmix_logpdf_matches_component_composition():
    """Algebraic cross-check: mixture logpdf == logsumexp over independently
    constructed single-FunnelTarget logpdfs at the component params/shifts."""
    from ics.zoo import FunnelTarget

    t = sample_target(jr.key(402), "funnelmix", 4)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(256, 4)) * 3.0)
    comps = []
    for j in range(t.log_weights.shape[0]):
        ft = FunnelTarget(Q=t.Qs[j], sigma_v=t.sigma_vs[j], c=t.cs[j])
        comps.append(float(t.log_weights[j]) + np.asarray(logpdf(ft, x - t.shifts[j])))
    ref = np.asarray(jax.scipy.special.logsumexp(jnp.asarray(np.stack(comps)), axis=0))
    np.testing.assert_allclose(np.asarray(logpdf(t, x)), ref, rtol=1e-10, atol=1e-10)


# --------------------------------------------------------------- generic plumbing

@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_logpdf_is_finite_and_shapes(family):
    key = jr.key(0)
    target = sample_target(key, family, 4)
    x = sample_x(jr.key(1), target, 512)
    lp = logpdf(target, x)
    assert x.shape == (512, 4) and lp.shape == (512,)
    assert bool(jnp.isfinite(lp).all()) and bool(jnp.isfinite(x).all())
    g = jax.vmap(jax.grad(lambda xi: logpdf(target, xi[None, :])[0]))(x[:64])
    assert bool(jnp.isfinite(g).all())


def test_mode_centers_are_local_maxima():
    for family in ("gmm", "funnelmix", "dwell"):
        t = sample_target(jr.key(11), family, 4)
        mc = mode_centers(t)
        if mc is None:
            continue
        mc = jnp.asarray(mc)
        lp_c = logpdf(t, mc)
        rng = np.random.default_rng(0)
        for _ in range(3):
            pert = jnp.asarray(rng.normal(size=mc.shape) * 0.15)
            lp_p = logpdf(t, mc + pert)
            if family == "funnelmix":
                continue  # cluster labels, not sharp maxima (documented)
            assert bool((lp_c >= lp_p - 1e-6).all()), family


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_target_sampling_deterministic(family):
    t1 = sample_target(jr.key(3), family, 8)
    t2 = sample_target(jr.key(3), family, 8)
    x1 = sample_x(jr.key(4), t1, 64)
    x2 = sample_x(jr.key(4), t2, 64)
    np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))
