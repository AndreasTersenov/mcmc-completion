"""Validation for the M2 zoo-family adapters (vectorized-over-theta energy
functions) against the FD-validated stage0.targets implementations, and for
the centered energy pseudo-likelihood.

Contract:
  family.theta_dim, family.d, family.prior_lo/hi (arrays, uniform box prior)
  family.target(theta_row) -> stage0.targets object for that theta
  family.energy(theta (M,p), xs (K,d)) -> (E (M,K), gradE (M,K,d))
     with E = -logpdf_u and gradE = -grad_logpdf_u of the matching target
  family.log_z(theta (M,p)) -> (M,)  log normalizer of exp(-E) (0 if normalized)
  centered_energy_loglik: invariant to adding a constant to observed energies.
"""

import numpy as np
import pytest

from stage0.m2_families import (
    FAMILIES,
    Gmm2Family,
    centered_energy_loglik,
)


@pytest.mark.parametrize("famname", list(FAMILIES))
def test_energy_matches_target_object(famname):
    fam = FAMILIES[famname]
    rng = np.random.default_rng(20)
    thetas = fam.prior_lo + (fam.prior_hi - fam.prior_lo) * rng.uniform(
        size=(5, fam.theta_dim)
    )
    xs = rng.standard_normal((7, fam.d)) * 2.0
    E, G = fam.energy(thetas, xs)
    assert E.shape == (5, 7) and G.shape == (5, 7, fam.d)
    for i, th in enumerate(thetas):
        t = fam.target(th)
        np.testing.assert_allclose(E[i], -t.logpdf_u(xs), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(G[i], -t.grad_logpdf_u(xs), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("famname", list(FAMILIES))
def test_log_z_matches_target_object(famname):
    fam = FAMILIES[famname]
    rng = np.random.default_rng(21)
    thetas = fam.prior_lo + (fam.prior_hi - fam.prior_lo) * rng.uniform(
        size=(3, fam.theta_dim)
    )
    lz = fam.log_z(thetas)
    for i, th in enumerate(thetas):
        assert lz[i] == pytest.approx(fam.target(th).log_Z, abs=1e-6)


def test_centered_loglik_offset_invariant_and_peaks_at_truth():
    fam = Gmm2Family()
    rng = np.random.default_rng(22)
    theta_star = np.array([6.0, 0.35])
    xs = fam.target(theta_star).sample(rng, 64)
    E_obs, G_obs = fam.energy(theta_star[None, :], xs)
    E_obs, G_obs = E_obs[0], G_obs[0]

    grid = np.stack(
        np.meshgrid(np.linspace(2, 10, 81), np.linspace(0.2, 0.8, 61), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    ll = centered_energy_loglik(fam, grid, xs, E_obs, sigma_e=0.05,
                                grads=G_obs, sigma_g=0.05)
    ll_shifted = centered_energy_loglik(fam, grid, xs, E_obs + 37.5, sigma_e=0.05,
                                        grads=G_obs, sigma_g=0.05)
    np.testing.assert_allclose(ll, ll_shifted, rtol=1e-9, atol=1e-7)
    assert np.linalg.norm(grid[np.argmax(ll)] - theta_star) < 0.15
