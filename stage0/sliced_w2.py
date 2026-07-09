"""Sliced Wasserstein-2 between sample sets.

SW2^2(p, q) = E_u[ W2^2(u#p, u#q) ] over uniform directions u on the sphere.
Closed-form gate: for N(mu1, I) vs N(mu2, I), SW2^2 = ||mu1 - mu2||^2 / d.
"""

import numpy as np


def sliced_w2_squared(x, y, n_proj=128, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    d = x.shape[1]
    n = min(len(x), len(y))
    if len(x) > n:
        x = x[rng.choice(len(x), n, replace=False)]
    if len(y) > n:
        y = y[rng.choice(len(y), n, replace=False)]
    u = rng.standard_normal((n_proj, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    px = np.sort(x @ u.T, axis=0)
    py = np.sort(y @ u.T, axis=0)
    return float(((px - py) ** 2).mean())
