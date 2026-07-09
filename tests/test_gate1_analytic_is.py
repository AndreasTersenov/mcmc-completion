"""Validation gate 1 (CLAUDE.md): analytic importance sampling.

ESS and log-Z for a Gaussian proposal vs a shifted/scaled Gaussian target must
match the closed form.

Conventions under test (the contract for stage0.is_estimators):
  - samples x ~ q (proposal, normalized), logw = log p_tilde(x) - log q(x)
  - ess_frac_from_logw(logw) = (sum w)^2 / (N * sum w^2)  in (0, 1]
  - logz_from_logw(logw) = logsumexp(logw) - log N   (unbiased Z-hat = mean w)

Closed forms (derived independently here; see log/2026-07-09-gates.md):
  mean shift eps in EVERY dim, unit variances:
      E_q[w^2] = exp(d * eps^2)          => ESS/N -> exp(-d * eps^2)
  proposal variance (1+eps) per dim, target unit variance:
      per-dim rho = (1+eps)/sqrt(1+2*eps)  (requires eps > -1/2)
      => ESS/N -> (1+2*eps)^(d/2) / (1+eps)^d
"""

import numpy as np
import pytest

from stage0.is_estimators import (
    ess_frac_from_logw,
    logz_from_logw,
    gaussian_shift_ess_frac,
    gaussian_scale_ess_frac,
)


def _logw_shift(rng, n, d, eps):
    """x ~ q = N(eps*1, I); target p = N(0, I). Returns logw."""
    x = rng.standard_normal((n, d)) + eps
    # log p - log q = sum_i [ -x_i^2/2 + (x_i-eps)^2/2 ] = sum_i [ eps^2/2 - eps*x_i ]
    return (eps**2 / 2.0) * d - eps * x.sum(axis=1)


def _logw_scale(rng, n, d, eps):
    """x ~ q = N(0, (1+eps) I); target p = N(0, I). Returns logw."""
    s2 = 1.0 + eps
    x = rng.standard_normal((n, d)) * np.sqrt(s2)
    # log p - log q = sum_i [ -x_i^2/2 + x_i^2/(2 s2) + 0.5*log(s2) ]
    return 0.5 * d * np.log(s2) - 0.5 * (1.0 - 1.0 / s2) * (x**2).sum(axis=1)


# Strict-match cases are restricted to where the ESS estimator itself is
# reliable: its relative sd is sqrt((E w^4/(E w^2)^2 - 1)/N) per the 4th-moment
# of the weights. For mean shift that factor is exp(4 d eps^2); for variance
# scaling s^2=1+eps the per-dim 4th moment s^3/sqrt(4-3/s^2) DIVERGES for
# s^2 <= 3/4. All cases below keep estimator noise < 1% at N=3e5.
SHIFT_CASES = [(2, 0.3), (8, 0.1), (8, 0.3), (32, 0.15)]
SCALE_CASES = [(8, 0.3), (8, -0.15), (32, 0.1)]


@pytest.mark.parametrize("d,eps", SHIFT_CASES)
def test_ess_matches_closed_form_mean_shift(d, eps):
    n = 300_000
    rng = np.random.default_rng(1000 + 10 * d + int(1000 * eps))
    logw = _logw_shift(rng, n, d, eps)
    target = np.exp(-d * eps**2)  # independent inline reference
    est = ess_frac_from_logw(logw)
    assert est == pytest.approx(target, rel=0.08)
    # library closed-form helper must agree with the inline formula exactly
    assert gaussian_shift_ess_frac(eps, d) == pytest.approx(target, rel=1e-12)


@pytest.mark.parametrize("d,eps", SCALE_CASES)
def test_ess_matches_closed_form_cov_scaling(d, eps):
    n = 300_000
    rng = np.random.default_rng(2000 + 10 * d + int(1000 * eps))
    logw = _logw_scale(rng, n, d, eps)
    target = (1.0 + 2.0 * eps) ** (d / 2.0) / (1.0 + eps) ** d
    est = ess_frac_from_logw(logw)
    assert est == pytest.approx(target, rel=0.08)
    assert gaussian_scale_ess_frac(eps, d) == pytest.approx(target, rel=1e-12)


def test_deep_tail_regime_ess_is_order_of_magnitude_only():
    """At (d=16, eps=0.5) the closed form gives ESS/N = e^-4 ~ 1.8%, but the
    ESS ESTIMATOR's relative sd at N=3e5 is sqrt((e^16-1)/3e5) ~ 5.4 (540%):
    the empirical ESS in the near-vacuous regime is order-of-magnitude only
    (typically an OVERestimate, since the rare dominating weights are missed).
    This is a known caveat that M1 must respect when choosing N per grid
    point; here we only pin the order of magnitude.
    """
    n = 300_000
    rng = np.random.default_rng(4000)
    logw = _logw_shift(rng, n, 16, 0.5)
    est = ess_frac_from_logw(logw)
    cf = np.exp(-16 * 0.25)
    assert cf / 10.0 < est < cf * 10.0


@pytest.mark.parametrize("kind,d,eps", [("shift", 8, 0.3), ("scale", 8, 0.3), ("shift", 32, 0.15)])
def test_logz_matches_closed_form(kind, d, eps):
    # Target is a NORMALIZED Gaussian treated as unnormalized => true log Z = 0.
    n = 300_000
    rng = np.random.default_rng(3000 + 10 * d)
    if kind == "shift":
        logw = _logw_shift(rng, n, d, eps)
        ess_frac = np.exp(-d * eps**2)
    else:
        logw = _logw_scale(rng, n, d, eps)
        ess_frac = (1.0 + 2.0 * eps) ** (d / 2.0) / (1.0 + eps) ** d
    # sd(logZhat) ~ sd(Zhat)/Z = sqrt((E w^2 / (E w)^2 - 1)/n) = sqrt((1/ess_frac - 1)/n)
    sd = np.sqrt((1.0 / ess_frac - 1.0) / n)
    assert abs(logz_from_logw(logw)) < 5.0 * sd + 1e-4
