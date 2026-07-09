"""MALA, vectorized over parallel chains, with tempering (targets p_u^(1/T))."""

import numpy as np


def mala_chains(target, x0, n_steps, step, temperature=1.0, rng=None):
    """Run C parallel MALA chains on log p_u / T.

    x0: (C, d) initial states. Proposal: x' = x + step * grad + sqrt(2 step) xi.
    Returns (states (C, n_steps, d), mean acceptance rate).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.array(np.atleast_2d(x0), dtype=float)
    C, d = x.shape
    T = float(temperature)
    lp = target.logpdf_u(x) / T
    g = target.grad_logpdf_u(x) / T
    out = np.empty((C, n_steps, d))
    n_acc = 0
    for t in range(n_steps):
        prop = x + step * g + np.sqrt(2.0 * step) * rng.standard_normal((C, d))
        lp_p = target.logpdf_u(prop) / T
        g_p = target.grad_logpdf_u(prop) / T
        log_fwd = -((prop - x - step * g) ** 2).sum(axis=1) / (4.0 * step)
        log_bwd = -((x - prop - step * g_p) ** 2).sum(axis=1) / (4.0 * step)
        log_alpha = lp_p - lp + log_bwd - log_fwd
        acc = np.log(rng.uniform(size=C)) < log_alpha
        x[acc] = prop[acc]
        lp[acc] = lp_p[acc]
        g[acc] = g_p[acc]
        n_acc += int(acc.sum())
        out[:, t, :] = x
    return out, n_acc / (C * n_steps)
