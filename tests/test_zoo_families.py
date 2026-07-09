"""Backpressure (a): every zoo family's exact sampler must agree with its
normalized log-density.

Generic sampler<->density consistency test (uses the stage-0 identity in
reverse): draw x ~ p via the family sampler; for a normalized reference
density r whose support p covers (a shrunk Gaussian at the sample moments),
E_p[ r(x) / p(x) ] = 1 exactly. A wrong sampler OR a wrong/unnormalized
log-density breaks the identity. We check |Zhat - 1| < 5 * stderr and demand
a healthy ESS so the test has teeth.

Family-specific closed-form moment checks are added where available.
API under test (ics.zoo):
  sample_target(key, family, d) -> target   (pytree with .family, .d)
  sample_x(key, target, n) -> (n, d)
  logpdf(target, x) -> (n,)                 (normalized; float64 in tests)
  mode_centers(target) -> (M, d) array or None
  FAMILIES_TRAIN, FAMILIES_HELDOUT, DMAX
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from ics.zoo import (
    DMAX,
    FAMILIES_HELDOUT,
    FAMILIES_TRAIN,
    logpdf,
    mode_centers,
    sample_target,
    sample_x,
)

ALL_FAMILIES = FAMILIES_TRAIN + FAMILIES_HELDOUT
N_CONSISTENCY = 200_000


def _consistency_check(target, key, n=N_CONSISTENCY):
    """E_p[r/p] = 1 for shrunk-Gaussian reference r (support inside p's)."""
    x = np.asarray(sample_x(key, target, n), dtype=np.float64)
    mu = x.mean(axis=0)
    cov = np.cov(x.T) + 1e-9 * np.eye(x.shape[1])
    # shrink the reference so p's tails dominate r's
    cov_r = 0.25 * cov
    d = x.shape[1]
    chol = np.linalg.cholesky(cov_r)
    diff = x - mu
    y = np.linalg.solve(chol, diff.T).T
    log_r = -0.5 * (y**2).sum(axis=1) - np.log(np.diag(chol)).sum() - 0.5 * d * np.log(2 * np.pi)
    log_p = np.asarray(logpdf(target, jnp.asarray(x)), dtype=np.float64)
    logw = log_r - log_p
    w = np.exp(logw - logw.max())
    zhat = w.mean() * np.exp(logw.max())
    stderr = np.exp(logw.max()) * w.std(ddof=1) / np.sqrt(n)
    ess_frac = w.sum() ** 2 / (n * (w**2).sum())
    return zhat, stderr, ess_frac


@pytest.mark.parametrize("family", ALL_FAMILIES)
@pytest.mark.parametrize("d", [2, 8, 16])
def test_sampler_density_consistency(family, d):
    key = jr.key(hash((family, d)) % 2**31)
    k1, k2 = jr.split(key)
    target = sample_target(k1, family, d)
    zhat, stderr, ess_frac = _consistency_check(target, k2)
    assert ess_frac > 0.001, f"reference too mismatched for a meaningful test ({ess_frac=})"
    assert abs(zhat - 1.0) < 5.0 * stderr + 5e-3, f"{zhat=} {stderr=} {ess_frac=}"


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


def test_gmm_moments_match_params():
    key = jr.key(7)
    t = sample_target(key, "gmm", 8)
    x = np.asarray(sample_x(jr.key(8), t, 400_000))
    w = np.exp(np.asarray(t.log_weights))
    w = np.where(np.asarray(t.k_mask), w, 0.0)
    w = w / w.sum()
    mean_cf = (w[:, None] * np.asarray(t.means)[:, :8]).sum(axis=0)
    np.testing.assert_allclose(x.mean(axis=0), mean_cf, atol=0.05)


def test_funnel_conditional_identity():
    # stage-0 trick: E[ (u.x)^2 / (c^2 e^v) ] = 1 for the conditional dims
    key = jr.key(9)
    t = sample_target(key, "funnel", 8)
    x = np.asarray(sample_x(jr.key(10), t, 300_000), dtype=np.float64)
    Q = np.asarray(t.Q)
    z = x @ Q  # rotate back: z[:,0] = v, z[:,1:] conditional
    v = z[:, 0]
    assert abs(v.std() - float(t.sigma_v)) < 0.05 * float(t.sigma_v)
    ratio = z[:, 1:] ** 2 / (float(t.c) ** 2 * np.exp(v))[:, None]
    assert abs(ratio.mean() - 1.0) < 0.03


def test_mode_centers_are_local_maxima():
    # declared mode structure: gradient at each mode center ~ 0 (GMM well-separated case may
    # have tiny cross-terms; tolerance loose) and center logpdf beats nearby perturbations
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
            assert bool((lp_c >= lp_p - 1e-6).all()), family


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_target_sampling_deterministic(family):
    t1 = sample_target(jr.key(3), family, 8)
    t2 = sample_target(jr.key(3), family, 8)
    x1 = sample_x(jr.key(4), t1, 64)
    x2 = sample_x(jr.key(4), t2, 64)
    np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))
