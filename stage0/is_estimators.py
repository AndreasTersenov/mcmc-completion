"""Importance-sampling estimators and the Gaussian closed forms that gate them.

Conventions:
  x ~ q (normalized proposal), logw = log p_tilde(x) - log q(x).
  Zhat = mean(w) is unbiased for the normalizer Z of p_tilde.
  ESS/N = (sum w)^2 / (N sum w^2)  (self-normalized; invariant to scaling of w).
"""

import numpy as np

from .utils import logsumexp


def ess_frac_from_logw(logw):
    logw = np.asarray(logw, dtype=float)
    return float(np.exp(2.0 * logsumexp(logw) - logsumexp(2.0 * logw) - np.log(logw.size)))


def logz_from_logw(logw):
    logw = np.asarray(logw, dtype=float)
    return float(logsumexp(logw) - np.log(logw.size))


def is_summary(sample_fn, logw_fn, n, batch=100_000, seed=0):
    """Streaming ESS/N and logZ over n samples drawn in batches.

    sample_fn(rng, m) -> (m, d) samples; logw_fn(x) -> (m,) log-weights.
    Accumulates log(sum w) and log(sum w^2) across batches, so memory is O(batch).
    """
    rng = np.random.default_rng(seed)
    log_s1, log_s2 = -np.inf, -np.inf
    done = 0
    while done < n:
        m = min(batch, n - done)
        lw = np.asarray(logw_fn(sample_fn(rng, m)), dtype=float)
        log_s1 = np.logaddexp(log_s1, logsumexp(lw))
        log_s2 = np.logaddexp(log_s2, logsumexp(2.0 * lw))
        done += m
    return {
        "ess_frac": float(np.exp(2.0 * log_s1 - log_s2 - np.log(n))),
        "logz": float(log_s1 - np.log(n)),
        "n": n,
    }


# ---- closed forms: q proposal vs standard-Gaussian target, per-dim mismatch ----
# ESS/N -> 1/E_q[w^2] = exp(-D2(p||q)) where D2 is the Renyi-2 divergence.

def gaussian_shift_ess_frac(eps, d):
    """q = N(eps*1, I) vs p = N(0, I): ESS/N = exp(-d eps^2)."""
    return float(np.exp(-d * eps**2))


def gaussian_scale_ess_frac(eps, d):
    """q = N(0, (1+eps) I) vs p = N(0, I): ESS/N = (1+2eps)^(d/2)/(1+eps)^d.

    Requires eps > -1/2 (E[w^2] diverges otherwise); returns 0.0 beyond.
    """
    if eps <= -0.5:
        return 0.0
    return float((1.0 + 2.0 * eps) ** (d / 2.0) / (1.0 + eps) ** d)


def renyi2_gaussian_shift(eps, d):
    return float(d * eps**2)


def renyi2_gaussian_scale(eps, d):
    if eps <= -0.5:
        return float("inf")
    return float(d * np.log((1.0 + eps) / np.sqrt(1.0 + 2.0 * eps)))
