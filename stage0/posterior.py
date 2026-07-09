"""Posterior engines for M2: dense-grid posterior (theta-dim <= 2-3) and
adaptive-tempering SMC (theta-dim up to ~8), plus the energy-observation
pseudo-likelihood that turns a context C = {(x_k, E_k[, grad E_k])} into
log p(C | theta).
"""

import numpy as np

from .is_estimators import ess_frac_from_logw
from .utils import logsumexp


def grid_posterior(axes, loglik_fn, logprior_fn=None):
    """Dense-grid posterior on a uniform tensor grid.

    axes: list of 1-d arrays (uniform spacing assumed; cell measure cancels).
    loglik_fn / logprior_fn: (M, p) -> (M,).
    """
    mesh = np.meshgrid(*axes, indexing="ij")
    theta = np.column_stack([m.ravel() for m in mesh])
    ll = np.asarray(loglik_fn(theta), dtype=float)
    if logprior_fn is not None:
        ll = ll + np.asarray(logprior_fn(theta), dtype=float)
    logpost = ll - logsumexp(ll)
    w = np.exp(logpost)
    mean = w @ theta
    std = np.sqrt(w @ (theta - mean) ** 2)
    return {
        "theta": theta,
        "logpost": logpost.reshape(mesh[0].shape),
        "w": w,
        "mean": mean,
        "std": std,
        "map": theta[int(np.argmax(ll))],
        "axes": axes,
    }


def _systematic_resample(w, rng):
    n = w.size
    positions = (rng.uniform() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w), positions), n - 1)


def smc_posterior(prior_sample, logprior_fn, loglik_fn, n_particles, rng,
                  ess_threshold=0.5, n_mcmc=5, max_stages=200):
    """Adaptive-tempering SMC: pi_beta ∝ prior * lik^beta, beta 0 -> 1.

    Each stage: choose the largest delta-beta keeping incremental-weight
    ESS/N >= ess_threshold (bisection), systematic-resample, then n_mcmc
    random-walk MH sweeps with proposal cov = 2.38^2/p * particle cov.
    """
    th = np.asarray(prior_sample(rng, n_particles), dtype=float)
    p = th.shape[1]
    ll = np.asarray(loglik_fn(th), dtype=float)
    beta, stages = 0.0, 0
    while beta < 1.0 and stages < max_stages:
        stages += 1
        hi = 1.0 - beta
        if ess_frac_from_logw(hi * ll) >= ess_threshold:
            delta = hi
        else:
            lo = 0.0
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if ess_frac_from_logw(mid * ll) >= ess_threshold:
                    lo = mid
                else:
                    hi = mid
            delta = max(lo, 1e-8)
        beta += delta
        lw = delta * ll
        w = np.exp(lw - logsumexp(lw))
        idx = _systematic_resample(w, rng)
        th, ll = th[idx], ll[idx]
        cov = np.atleast_2d(np.cov(th.T)) + 1e-12 * np.eye(p)
        chol = np.linalg.cholesky((2.38**2 / p) * cov)
        lpost = np.asarray(logprior_fn(th), dtype=float) + beta * ll
        for _ in range(n_mcmc):
            prop = th + rng.standard_normal(th.shape) @ chol.T
            ll_p = np.asarray(loglik_fn(prop), dtype=float)
            lpost_p = np.asarray(logprior_fn(prop), dtype=float) + beta * ll_p
            acc = np.log(rng.uniform(size=n_particles)) < lpost_p - lpost
            th[acc], ll[acc], lpost[acc] = prop[acc], ll_p[acc], lpost_p[acc]
    return {
        "particles": th,
        "mean": th.mean(axis=0),
        "std": th.std(axis=0, ddof=1),
        "n_stages": stages,
        "beta": beta,
    }


def energy_context_loglik(family_energy_fn, theta, xs, energies, sigma_e,
                          grads=None, sigma_g=None):
    """log p(C | theta) under a Gaussian observation model on energy values
    (and optionally gradient values) at the context points xs.

    family_energy_fn(theta (m,p), xs (K,d)) -> (E (m,K), gradE (m,K,d)).
    energies: (K,) observed energies; grads: (K,d) observed gradients or None.
    Additive constants (in theta) are dropped.
    """
    E, G = family_energy_fn(theta, xs)
    ll = -0.5 * (((E - energies[None, :]) / sigma_e) ** 2).sum(axis=1)
    if grads is not None:
        ll = ll - 0.5 * (((G - grads[None, :, :]) / sigma_g) ** 2).sum(axis=(1, 2))
    return ll
