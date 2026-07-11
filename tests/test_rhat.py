"""Validation gates for the split-R-hat estimator (Readout C reference chains).
Closed-form behaviors: iid same-distribution chains give R-hat ~ 1; a
mean-shifted chain drives R-hat well above 1; a within-chain trend (first half
vs second half) is caught by the SPLIT (that's the point of splitting)."""

import numpy as np

from ics.real import split_rhat


def test_rhat_iid_chains_near_one():
    rng = np.random.default_rng(0)
    chains = rng.normal(size=(4, 4000, 3))
    r = split_rhat(chains)
    assert r.shape == (3,)
    assert np.all(r < 1.01)


def test_rhat_shifted_chain_flags():
    rng = np.random.default_rng(1)
    chains = rng.normal(size=(4, 4000, 2))
    chains[0, :, 0] += 3.0  # one chain stuck in a different mode
    r = split_rhat(chains)
    assert r[0] > 1.5
    assert r[1] < 1.01


def test_rhat_split_catches_trend():
    rng = np.random.default_rng(2)
    chains = rng.normal(size=(4, 4000, 1))
    chains[:, :, 0] += np.linspace(0.0, 3.0, 4000)  # same trend in every chain:
    r = split_rhat(chains)                          # unsplit R-hat would miss it
    assert r[0] > 1.2
