"""Frozen eval loop for the ICS: sample from the conditional flow, compute the
CNF log-density, and run the stage-0 certificate against the TRUE energy.

Space bookkeeping: the flow lives in whitened+padded DMAX space. The
certificate compares against the AUGMENTED true density
    p_aug(x_white, pad) = p(mu + sigma * x_white) * prod(sigma) * N(pad; 0, I)
(the augmented normalizer equals the true one, so logZ-hat is unbiased for
log Z = 0; padding mismatch honestly deflates ESS).
"""

import jax
import jax.numpy as jnp
import numpy as np

from .cfm import cnf_logpdf, cond_cfm_sample, make_velocity_fn
from .certificate import snis_certificate
from .context import whiten_invert
from .zoo import DMAX, logpdf


def augmented_true_logpdf(target, ctx, x_full):
    """log p_aug at whitened+padded points x_full (n, DMAX)."""
    d = target.d
    x_white, pad = x_full[:, :d], x_full[:, d:]
    x_raw = whiten_invert(x_white, ctx.mu, ctx.sigma)
    lp = logpdf(target, x_raw) + jnp.log(ctx.sigma).sum()
    lp_pad = -0.5 * (pad**2).sum(axis=1) - 0.5 * (DMAX - d) * jnp.log(2 * jnp.pi)
    return lp + lp_pad


def ics_evaluate(model, params, target, ctx, key, n_eval=8192, n_ode=200):
    """Returns (certificate dict, de-whitened samples (2*n_eval, d))."""
    tokens = ctx.tokens.astype(jnp.float64)
    params64 = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64), params)
    x_full = cond_cfm_sample(model, params64, tokens, key, n=2 * n_eval, n_steps=n_ode)
    velocity_fn = make_velocity_fn(model, params64, tokens)
    logq = cnf_logpdf(velocity_fn, x_full, n_steps=n_ode)
    logp = augmented_true_logpdf(target, ctx, x_full)
    cert = snis_certificate(np.asarray(logp), np.asarray(logq))
    x_raw = whiten_invert(x_full[:, : target.d], ctx.mu, ctx.sigma)
    return cert, np.asarray(x_raw)


def mode_recovery(target, x_gen):
    """Fraction of DECLARED modes recovered (nearest-center assignment; mode j
    recovered if its sample share >= w_j / 4). None when the family declares
    no mode structure. Mandatory column in gate reports (reconvene 2026-07-10:
    the P1 ESS clause alone cannot adjudicate multimodal targets)."""
    from .zoo import mode_centers

    mc = mode_centers(target)
    if mc is None:
        return None
    mc = np.asarray(mc)
    lw = getattr(target, "log_weights", None)
    if lw is not None:
        lw = np.asarray(lw)
        w = np.exp(lw[np.isfinite(lw)])
        w = w / w.sum()
    else:
        w = np.full(len(mc), 1.0 / len(mc))
    if len(w) != len(mc):  # dwell: equal-weight well combinations
        w = np.full(len(mc), 1.0 / len(mc))
    d2 = ((x_gen[:, None, :] - mc[None, :, :]) ** 2).sum(-1)
    share = np.bincount(d2.argmin(1), minlength=len(mc)) / len(x_gen)
    return float((share >= w / 4.0).mean())
