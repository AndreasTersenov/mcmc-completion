"""Validation gate 4 (CLAUDE.md): every reported number passes an N-doubling
stability check (double N, move < tol).

Contract for stage0.stability.n_doubling:
  n_doubling(estimate_fn, n, tol, rel=False, seed=0) -> dict with keys
    'passed' (bool), 'v_n', 'v_2n', 'delta'
  estimate_fn(n, seed) -> float; the two evaluations use independent seeds.
"""

import numpy as np

from stage0.stability import n_doubling


def _mean_estimator(n, seed):
    rng = np.random.default_rng(seed)
    return float(rng.standard_normal(n).mean())


def test_stable_estimator_passes():
    res = n_doubling(_mean_estimator, n=200_000, tol=0.01, seed=123)
    assert res["passed"]
    assert res["delta"] == abs(res["v_2n"] - res["v_n"])


def test_unstable_tolerance_fails():
    # Same estimator, absurdly tight tolerance: must fail (MC noise >> tol).
    res = n_doubling(_mean_estimator, n=1000, tol=1e-9, seed=123)
    assert not res["passed"]


def test_relative_mode():
    def biased_up(n, seed):
        return 100.0 + _mean_estimator(n, seed)

    res = n_doubling(biased_up, n=200_000, tol=1e-3, rel=True, seed=5)
    assert res["passed"]  # |delta| ~ 5e-3 abs, but 5e-5 relative to ~100
