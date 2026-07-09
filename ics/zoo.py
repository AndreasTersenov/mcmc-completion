"""Zoo v1 (frozen core): train families {gmm, dwell, funnel, warp}, held-out
families {banana, funnelmix}. d in {2,4,8,16}. Every family exactly
sampleable, log-density normalized by construction (log Z = 0); double-well
1-d factors normalized by dense quadrature (validated in tests).

Priors over theta (free periphery, documented here):
  gmm      k ~ U{1..8}; means ~ N(0, 3^2 I); cov = s^2 (0.3 I + G G^T / d),
           s ~ U(0.6, 1.4); Dirichlet(2) weights.
  dwell    n_well ~ U{1..min(3,d)} rotated coords with energy a (z^2-b)^2,
           a ~ U(0.5, 2.5), b ~ U(1.0, 2.25); remaining coords N(0,1);
           random rotation Q couples them in x-space.
  funnel   sigma_v ~ U(0.5, 3) (frozen range), c ~ U(0.5, 1.5), random
           rotation (v-axis = first column of Q).
  warp     sinh-arcsinh warped standard Gaussian (ics.warps priors).
  banana   Rosenbrock shear-warped standard Gaussian (held out).
  funnelmix J ~ U{2,3} rotated funnels at shifts ~ N(0, 3^2 I), Dirichlet(2).
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .warps import BananaWarp, SinhArcsinhWarp, make_warp, random_rotation, warp_forward, warp_inverse, warp_logdet

DMAX = 16
KMAX = 8
FAMILIES_TRAIN = ("gmm", "dwell", "funnel", "warp")
FAMILIES_HELDOUT = ("banana", "funnelmix")

_LOG2PI = float(np.log(2.0 * np.pi))

# dense grid for double-well 1-d quadrature + inverse-CDF (stage-0 approach)
_DW_GRID = jnp.linspace(-6.0, 6.0, 8193)


class GMMTarget(NamedTuple):
    means: jnp.ndarray        # (KMAX, d), inactive rows zero
    chols: jnp.ndarray        # (KMAX, d, d) lower cholesky, inactive = I
    log_weights: jnp.ndarray  # (KMAX,), -inf at inactive comps
    k_mask: jnp.ndarray       # (KMAX,) bool
    family: str = "gmm"

    @property
    def d(self):
        return self.means.shape[1]


class DwellTarget(NamedTuple):
    Q: jnp.ndarray        # (d, d) rotation
    a: jnp.ndarray        # (nw,)
    b: jnp.ndarray        # (nw,)
    log_z1d: jnp.ndarray  # (nw,) 1-d normalizers of exp(-a (z^2-b)^2)
    cdf: jnp.ndarray      # (nw, G) inverse-CDF tables on _DW_GRID
    family: str = "dwell"

    @property
    def d(self):
        return self.Q.shape[0]

    @property
    def n_well(self):
        return self.a.shape[0]


class FunnelTarget(NamedTuple):
    Q: jnp.ndarray  # (d, d) rotation; v-axis = Q[:, 0]
    sigma_v: jnp.ndarray  # scalar
    c: jnp.ndarray        # scalar
    family: str = "funnel"

    @property
    def d(self):
        return self.Q.shape[0]


class WarpTarget(NamedTuple):
    warp: SinhArcsinhWarp
    family: str = "warp"

    @property
    def d(self):
        return self.warp.shift.shape[0]


class BananaTarget(NamedTuple):
    warp: BananaWarp
    family: str = "banana"

    @property
    def d(self):
        return self.warp.shift.shape[0]


class FunnelMixTarget(NamedTuple):
    Qs: jnp.ndarray       # (J, d, d)
    sigma_vs: jnp.ndarray  # (J,)
    cs: jnp.ndarray        # (J,)
    shifts: jnp.ndarray    # (J, d)
    log_weights: jnp.ndarray  # (J,)
    family: str = "funnelmix"

    @property
    def d(self):
        return self.shifts.shape[1]


# ---------------------------------------------------------------- constructors

def _sample_gmm(key, d):
    k1, k2, k3, k4, k5 = jr.split(key, 5)
    k = int(jr.randint(k1, (), 1, KMAX + 1))
    means = jnp.zeros((KMAX, d)).at[:k].set(3.0 * jr.normal(k2, (k, d)))
    gs = jr.normal(k3, (KMAX, d, d))
    ss = jr.uniform(k4, (KMAX,), minval=0.6, maxval=1.4)
    covs = ss[:, None, None] ** 2 * (
        0.3 * jnp.eye(d)[None] + gs @ jnp.swapaxes(gs, 1, 2) / d
    )
    chols = jnp.linalg.cholesky(covs)
    chols = jnp.where(jnp.arange(KMAX)[:, None, None] < k, chols, jnp.eye(d)[None])
    w = jr.dirichlet(k5, 2.0 * jnp.ones(k))
    log_weights = jnp.full(KMAX, -jnp.inf).at[:k].set(jnp.log(w))
    return GMMTarget(means=means, chols=chols, log_weights=log_weights,
                     k_mask=jnp.arange(KMAX) < k)


def _sample_dwell(key, d):
    k1, k2, k3, k4 = jr.split(key, 4)
    nw = int(jr.randint(k1, (), 1, min(3, d) + 1))
    a = jr.uniform(k2, (nw,), minval=0.5, maxval=2.5)
    b = jr.uniform(k3, (nw,), minval=1.0, maxval=2.25)
    lw = -a[:, None] * (_DW_GRID[None, :] ** 2 - b[:, None]) ** 2  # (nw, G)
    w = jnp.exp(lw)
    z1d = jnp.trapezoid(w, _DW_GRID, axis=1)
    cdf = jnp.cumsum(w, axis=1)
    cdf = cdf / cdf[:, -1:]
    return DwellTarget(Q=random_rotation(k4, d), a=a, b=b,
                       log_z1d=jnp.log(z1d), cdf=cdf)


def _sample_funnel(key, d):
    k1, k2, k3 = jr.split(key, 3)
    return FunnelTarget(
        Q=random_rotation(k1, d),
        sigma_v=jr.uniform(k2, (), minval=0.5, maxval=3.0),
        c=jr.uniform(k3, (), minval=0.5, maxval=1.5),
    )


def _sample_funnelmix(key, d):
    k1, k2, k3, k4, k5, k6 = jr.split(key, 6)
    j = int(jr.randint(k1, (), 2, 4))
    Qs = jnp.stack([random_rotation(k, d) for k in jr.split(k2, j)])
    return FunnelMixTarget(
        Qs=Qs,
        sigma_vs=jr.uniform(k3, (j,), minval=0.5, maxval=2.0),
        cs=jr.uniform(k4, (j,), minval=0.5, maxval=1.2),
        shifts=3.0 * jr.normal(k5, (j, d)),
        log_weights=jnp.log(jr.dirichlet(k6, 2.0 * jnp.ones(j))),
    )


def sample_target(key, family, d):
    if family == "gmm":
        return _sample_gmm(key, d)
    if family == "dwell":
        return _sample_dwell(key, d)
    if family == "funnel":
        return _sample_funnel(key, d)
    if family == "warp":
        return WarpTarget(warp=make_warp(key, d, "sinharcsinh"))
    if family == "banana":
        return BananaTarget(warp=make_warp(key, d, "banana"))
    if family == "funnelmix":
        return _sample_funnelmix(key, d)
    raise ValueError(f"unknown family: {family}")


# ---------------------------------------------------------------- log-densities

def _gauss_logpdf_chol(x, mean, chol):
    d = x.shape[-1]
    y = jax.scipy.linalg.solve_triangular(chol, (x - mean).T, lower=True).T
    return (
        -0.5 * (y**2).sum(-1)
        - jnp.log(jnp.diagonal(chol)).sum()
        - 0.5 * d * _LOG2PI
    )


def _funnel_logpdf_z(z, sigma_v, c):
    v, y = z[:, 0], z[:, 1:]
    k = z.shape[1] - 1
    var_y = c**2 * jnp.exp(v)
    lp_v = -0.5 * v**2 / sigma_v**2 - 0.5 * jnp.log(2 * jnp.pi * sigma_v**2)
    lp_y = -0.5 * (y**2).sum(-1) / var_y - 0.5 * k * (_LOG2PI + jnp.log(var_y))
    return lp_v + lp_y


def logpdf(target, x):
    x = jnp.atleast_2d(x)
    if isinstance(target, GMMTarget):
        comp = jax.vmap(lambda m, L: _gauss_logpdf_chol(x, m, L))(
            target.means, target.chols
        )  # (KMAX, n)
        return jax.scipy.special.logsumexp(
            target.log_weights[:, None] + comp, axis=0
        )
    if isinstance(target, DwellTarget):
        z = x @ target.Q
        nw = target.n_well
        zw, zg = z[:, :nw], z[:, nw:]
        lp_w = (
            -target.a[None, :] * (zw**2 - target.b[None, :]) ** 2
            - target.log_z1d[None, :]
        ).sum(-1)
        lp_g = -0.5 * (zg**2).sum(-1) - 0.5 * (z.shape[1] - nw) * _LOG2PI
        return lp_w + lp_g
    if isinstance(target, FunnelTarget):
        return _funnel_logpdf_z(x @ target.Q, target.sigma_v, target.c)
    if isinstance(target, (WarpTarget, BananaTarget)):
        z = warp_inverse(target.warp, x)
        lp_z = -0.5 * (z**2).sum(-1) - 0.5 * z.shape[1] * _LOG2PI
        return lp_z - warp_logdet(target.warp, z)
    if isinstance(target, FunnelMixTarget):
        def comp_lp(Q, sv, c, sh, lw):
            return lw + _funnel_logpdf_z((x - sh) @ Q, sv, c)
        lps = jax.vmap(comp_lp)(target.Qs, target.sigma_vs, target.cs,
                                target.shifts, target.log_weights)
        return jax.scipy.special.logsumexp(lps, axis=0)
    raise TypeError(f"unknown target type: {type(target)}")


# ---------------------------------------------------------------- exact samplers

def sample_x(key, target, n):
    if isinstance(target, GMMTarget):
        k1, k2 = jr.split(key)
        w = jnp.exp(target.log_weights)
        w = w / w.sum()
        comp = jr.choice(k1, KMAX, (n,), p=w)
        eps = jr.normal(k2, (n, target.d))
        x = target.means[comp] + jnp.einsum("nij,nj->ni", target.chols[comp], eps)
        return x
    if isinstance(target, DwellTarget):
        k1, k2 = jr.split(key)
        nw, d = target.n_well, target.d
        u = jr.uniform(k1, (n, nw))
        zw = jax.vmap(lambda cdf_i, u_i: jnp.interp(u_i, cdf_i, _DW_GRID),
                      in_axes=(0, 1), out_axes=1)(target.cdf, u)
        zg = jr.normal(k2, (n, d - nw))
        z = jnp.concatenate([zw, zg], axis=1)
        return z @ target.Q.T
    if isinstance(target, FunnelTarget):
        z = _sample_funnel_z(key, target.sigma_v, target.c, target.d, n)
        return z @ target.Q.T
    if isinstance(target, (WarpTarget, BananaTarget)):
        z = jr.normal(key, (n, target.d))
        return warp_forward(target.warp, z)
    if isinstance(target, FunnelMixTarget):
        k1, k2 = jr.split(key)
        j = target.log_weights.shape[0]
        comp = jr.choice(k1, j, (n,), p=jnp.exp(target.log_weights))
        zs = jnp.stack(
            [_sample_funnel_z(k, target.sigma_vs[i], target.cs[i], target.d, n)
             for i, k in enumerate(jr.split(k2, j))]
        )  # (J, n, d)
        xs = jnp.einsum("jni,jki->jnk", zs, target.Qs) + target.shifts[:, None, :]
        return xs[comp, jnp.arange(n)]
    raise TypeError(f"unknown target type: {type(target)}")


def _sample_funnel_z(key, sigma_v, c, d, n):
    k1, k2 = jr.split(key)
    v = sigma_v * jr.normal(k1, (n,))
    y = c * jnp.exp(v / 2)[:, None] * jr.normal(k2, (n, d - 1))
    return jnp.concatenate([v[:, None], y], axis=1)


# ---------------------------------------------------------------- mode structure

def mode_centers(target) -> Optional[np.ndarray]:
    """Declared mode structure for the mode-recovery metric. None when the
    family does not declare modes (funnel, warp, banana: unimodal or
    structureless for this metric)."""
    if isinstance(target, GMMTarget):
        k = int(np.asarray(target.k_mask).sum())
        return np.asarray(target.means)[:k]
    if isinstance(target, DwellTarget):
        nw, d = target.n_well, target.d
        b = np.asarray(target.b)
        signs = np.array(np.meshgrid(*([[-1.0, 1.0]] * nw))).T.reshape(-1, nw)
        z = np.zeros((signs.shape[0], d))
        z[:, :nw] = signs * np.sqrt(b)[None, :]
        return z @ np.asarray(target.Q).T
    if isinstance(target, FunnelMixTarget):
        # component centers: shift + Q e1 * v_peak, v_peak = -(d-1) sv^2 / 2
        sv = np.asarray(target.sigma_vs)
        vpk = -(target.d - 1) * sv**2 / 2.0
        vpk = np.clip(vpk, -2.0 * sv, 0.0)  # cluster label, not a sharp peak
        return np.asarray(target.shifts) + vpk[:, None] * np.asarray(target.Qs)[:, :, 0]
    return None
