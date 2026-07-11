"""Real inference targets + reference diagnostics for the Readout-C usefulness
barometer (phase 1b amendment; pre-registration: log/2026-07-11-phase1b.md).

Every target is a batched float64 logpdf: (n, d) -> (n,), autodiff-able, so the
standard context protocol (x, E, grad E) and the certificate path apply
unchanged. Zoo coupling is isolated to real_augmented_logpdf /
ics_evaluate_fn, which mirror ics.eval for callable targets (equivalence is
pinned by tests/test_real_targets.py).
"""

import jax
import jax.numpy as jnp
import numpy as np

from .cfm import cnf_logpdf, cond_cfm_sample, make_velocity_fn
from .certificate import snis_certificate
from .context import whiten_invert
from .zoo import DMAX

_LOG2PI = float(np.log(2.0 * np.pi))


def _norm_logpdf(x, loc, scale):
    return -0.5 * ((x - loc) / scale) ** 2 - jnp.log(scale) - 0.5 * _LOG2PI


# ---------------------------------------------------------------- eight schools
# Rubin (1981) classic data; non-centered parameterization, d = 10:
# x = [mu, log tau, z_1..z_8];  mu ~ N(0, 5^2), tau ~ HalfCauchy(5) (via log tau
# + Jacobian), theta_j = mu + tau z_j, z ~ N(0,1), y_j ~ N(theta_j, sigma_j^2).
EIGHT_SCHOOLS_Y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
EIGHT_SCHOOLS_SIGMA = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])


def eight_schools_logpdf(x):
    x = jnp.asarray(x)
    mu, lt, z = x[:, 0], x[:, 1], x[:, 2:]
    tau = jnp.exp(lt)
    lp = _norm_logpdf(mu, 0.0, 5.0)
    lp += jnp.log(2.0 / (jnp.pi * 5.0)) - jnp.log1p((tau / 5.0) ** 2) + lt
    lp += _norm_logpdf(z, 0.0, 1.0).sum(axis=1)
    theta = mu[:, None] + tau[:, None] * z
    lp += _norm_logpdf(EIGHT_SCHOOLS_Y[None, :], theta,
                       EIGHT_SCHOOLS_SIGMA[None, :]).sum(axis=1)
    return lp


# ------------------------------------------------------------------ gym banana
# inference-gym Banana, ndims=2, curvature 0.03 (Haario form): closed-form
# transform of N(0, diag(100, 1)) — exactly portable, exactly sampleable.
_BANANA_B = 0.03


def gym_banana_logpdf(x):
    x = jnp.asarray(x)
    y2 = x[:, 1] + _BANANA_B * (x[:, 0] ** 2 - 100.0)
    return _norm_logpdf(x[:, 0], 0.0, 10.0) + _norm_logpdf(y2, 0.0, 1.0)


def gym_banana_sample(key, n):
    import jax.random as jr

    k1, k2 = jr.split(key)
    y1 = 10.0 * jr.normal(k1, (n,), jnp.float64)
    y2 = jr.normal(k2, (n,), jnp.float64)
    return jnp.stack([y1, y2 - _BANANA_B * (y1**2 - 100.0)], axis=1)


# ----------------------------------------------------- WL band-power posterior
class WLBandpower:
    """d=3 (Om, s8, ns) Euclid-like band-power posterior on the polynomial
    surrogate (results/wl_surrogate.npz from scripts/wl_surrogate.py).

    Sampling space: u in R^3, theta = center + hw * tanh(u / 2.5) (practitioner
    adapter — no extrapolation possible), prior u ~ N(0, I), Gaussian band-power
    likelihood with the fixed Knox covariance. Fully autodiff-able."""

    d = 3

    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.coef = jnp.asarray(z["coef"], jnp.float64)        # (ncoef, 120)
        self.powers = jnp.asarray(z["powers"], jnp.int32)      # (ncoef, 3)
        self.center = jnp.asarray(z["center"], jnp.float64)
        self.hw = jnp.asarray(z["hw"], jnp.float64)
        self.d_obs = jnp.asarray(z["d_obs"], jnp.float64)      # (120,)
        self.chol_w = jnp.asarray(z["chol_w"], jnp.float64)    # L^-1 whitener
        self.theta_fid = np.asarray(z["theta_fid"])

    def theta_of_u(self, u):
        return self.center + self.hw * jnp.tanh(u / 2.5)

    def _cl(self, t_scaled):
        feats = jnp.prod(t_scaled[None, :] ** self.powers, axis=1)  # (ncoef,)
        return feats @ self.coef                                    # (120,)

    def logpdf(self, u):
        u = jnp.asarray(u)

        def one(ui):
            th = jnp.tanh(ui / 2.5)
            t_scaled = th  # scaled coords in [-1, 1] = (theta - center)/hw
            resid = self.d_obs - self._cl(t_scaled)
            w = self.chol_w @ resid
            log_j = jnp.sum(jnp.log(self.hw) + jnp.log1p(-th**2) - jnp.log(2.5))
            prior = -0.5 * jnp.sum(ui**2) - 1.5 * _LOG2PI
            return -0.5 * jnp.sum(w**2) + log_j + prior

        return jax.vmap(one)(u)


# ------------------------------------------------- eval path for real targets
def real_augmented_logpdf(logpdf_fn, d, ctx, x_full):
    """Mirror of ics.eval.augmented_true_logpdf for a callable target."""
    x_white, pad = x_full[:, :d], x_full[:, d:]
    x_raw = whiten_invert(x_white, ctx.mu, ctx.sigma)
    lp = logpdf_fn(x_raw) + jnp.log(ctx.sigma).sum()
    lp_pad = -0.5 * (pad**2).sum(axis=1) - 0.5 * (DMAX - d) * jnp.log(2 * jnp.pi)
    return lp + lp_pad


def ics_evaluate_fn(model, params, logpdf_fn, d, ctx, key, n_eval=8192,
                    n_ode=200):
    """Mirror of ics.eval.ics_evaluate for a callable target.
    Returns (certificate dict, de-whitened samples (2*n_eval, d))."""
    tokens = ctx.tokens.astype(jnp.float64)
    params64 = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64), params)
    x_full = cond_cfm_sample(model, params64, tokens, key, n=2 * n_eval,
                             n_steps=n_ode)
    velocity_fn = make_velocity_fn(model, params64, tokens)
    logq = cnf_logpdf(velocity_fn, x_full, n_steps=n_ode)
    logp = real_augmented_logpdf(logpdf_fn, d, ctx, x_full)
    cert = snis_certificate(np.asarray(logp), np.asarray(logq))
    x_raw = whiten_invert(x_full[:, :d], ctx.mu, ctx.sigma)
    return cert, np.asarray(x_raw)


# ------------------------------------------------------------------- split R^
def split_rhat(chains):
    """Split-R-hat per dimension. chains: (n_chains, n_draws, d) -> (d,).
    Each chain is split in half (catching within-chain trends), then the
    standard Gelman-Rubin variance-ratio estimate is applied to the 2m halves."""
    c = np.asarray(chains, np.float64)
    m, n, d = c.shape
    half = n // 2
    h = np.concatenate([c[:, :half], c[:, half:2 * half]], axis=0)  # (2m, half, d)
    W = h.var(axis=1, ddof=1).mean(axis=0)
    B = half * h.mean(axis=1).var(axis=0, ddof=1)
    var_hat = (half - 1) / half * W + B / half
    return np.sqrt(var_hat / np.maximum(W, 1e-300))
