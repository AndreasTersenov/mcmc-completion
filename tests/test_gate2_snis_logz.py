"""Validation gate 2 (CLAUDE.md): SNIS log-Z estimator unbiased within
Monte-Carlo error on a mixture with known Z.

Target: p_tilde(x) = Z_TRUE * [w1 N(mu1, s1^2 I) + w2 N(mu2, s2^2 I)] in d=2,
so the true normalizer is Z_TRUE by construction. Proposal: wide Gaussian.

Zhat = mean(p_tilde/q) is exactly unbiased for Z (q normalized), so the
replicate-mean of Zhat must sit within Monte-Carlo error of Z_TRUE.
log Zhat has an O(1/N) Jensen bias; at large N it must sit within MC error
of log Z_TRUE.
"""

import numpy as np
import pytest

from stage0.is_estimators import ess_frac_from_logw, logz_from_logw
from stage0.targets import GMM, Gaussian
from stage0.utils import logsumexp

Z_TRUE = 7.3
D = 2

MIX = GMM(
    means=np.array([[-2.0, -1.0], [3.0, 2.0]]),
    weights=np.array([0.3, 0.7]),
    var=np.array([0.5**2, 1.5**2]),
)
PROPOSAL = Gaussian(mean=np.zeros(D), var=4.0**2)


def _logw(rng, n):
    x = PROPOSAL.sample(rng, n)
    return np.log(Z_TRUE) + MIX.logpdf(x) - PROPOSAL.logpdf(x)


def test_zhat_unbiased_over_replicates():
    rng = np.random.default_rng(42)
    reps, n = 250, 4000
    zhats = np.empty(reps)
    for r in range(reps):
        logw = _logw(rng, n)
        zhats[r] = np.exp(logsumexp(logw) - np.log(n))
    stderr = zhats.std(ddof=1) / np.sqrt(reps)
    assert abs(zhats.mean() - Z_TRUE) < 4.0 * stderr
    # and the MC error itself must be small enough for the test to have teeth
    assert stderr < 0.05 * Z_TRUE


def test_logz_within_mc_error_at_large_n():
    rng = np.random.default_rng(43)
    n = 200_000
    logw = _logw(rng, n)
    ess_frac = ess_frac_from_logw(logw)
    sd = np.sqrt((1.0 / ess_frac - 1.0) / n)  # delta-method sd of logZhat
    assert abs(logz_from_logw(logw) - np.log(Z_TRUE)) < 5.0 * sd + 1e-4
