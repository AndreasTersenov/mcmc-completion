"""Smooth invertible warps with exact inverses and explicit log-det-Jacobians.

Train-family warp ("sinharcsinh"): rotation -> per-dim sinh-arcsinh (skew +
tail-weight) -> rotation -> diag scale + shift. Held-out warp ("banana"):
rotation -> Rosenbrock shear chain (unit-triangular, log-det 0) -> diag scale
+ shift. Both orientation-preserving by construction.

Conventions: points are (n, d); rotations act as z @ R.T.
"""

from typing import NamedTuple

import jax.numpy as jnp
import jax.random as jr


class SinhArcsinhWarp(NamedTuple):
    R1: jnp.ndarray  # (d, d) rotation
    R2: jnp.ndarray  # (d, d) rotation
    s: jnp.ndarray   # (d,) tail-weight params, > 0
    t: jnp.ndarray   # (d,) skew params
    scale: jnp.ndarray  # (d,) > 0
    shift: jnp.ndarray  # (d,)


class BananaWarp(NamedTuple):
    R: jnp.ndarray      # (d, d) rotation
    bcoef: jnp.ndarray  # (d-1,) shear strengths
    scale: jnp.ndarray  # (d,) > 0
    shift: jnp.ndarray  # (d,)


def random_rotation(key, d):
    a = jr.normal(key, (d, d))
    q, r = jnp.linalg.qr(a)
    q = q * jnp.sign(jnp.diag(r))[None, :]
    # force det = +1 (orientation-preserving)
    det = jnp.linalg.det(q)
    q = q.at[:, 0].multiply(jnp.sign(det))
    return q


def make_warp(key, d, kind):
    k1, k2, k3, k4, k5, k6 = jr.split(key, 6)
    if kind == "sinharcsinh":
        return SinhArcsinhWarp(
            R1=random_rotation(k1, d),
            R2=random_rotation(k2, d),
            s=jnp.exp(jr.uniform(k3, (d,), minval=jnp.log(0.7), maxval=jnp.log(1.5))),
            t=jr.uniform(k4, (d,), minval=-0.5, maxval=0.5),
            scale=jnp.exp(jr.uniform(k5, (d,), minval=jnp.log(0.5), maxval=jnp.log(2.0))),
            shift=1.5 * jr.normal(k6, (d,)),
        )
    if kind == "banana":
        sign = jnp.where(jr.bernoulli(k2, 0.5, (d - 1,)), 1.0, -1.0)
        return BananaWarp(
            R=random_rotation(k1, d),
            bcoef=sign * jr.uniform(k3, (d - 1,), minval=0.3, maxval=1.0),
            scale=jnp.exp(jr.uniform(k5, (d,), minval=jnp.log(0.5), maxval=jnp.log(2.0))),
            shift=1.5 * jr.normal(k6, (d,)),
        )
    raise ValueError(f"unknown warp kind: {kind}")


def _sas_fwd(u, s, t):
    return jnp.sinh(s * jnp.arcsinh(u) + t)


def _sas_inv(w, s, t):
    return jnp.sinh((jnp.arcsinh(w) - t) / s)


def _sas_logderiv(u, s, t):
    # d/du sinh(s asinh(u) + t) = cosh(s asinh(u) + t) * s / sqrt(1 + u^2)
    return (
        jnp.log(jnp.cosh(s * jnp.arcsinh(u) + t))
        + jnp.log(s)
        - 0.5 * jnp.log1p(u**2)
    )


def _shear_fwd(u, b):
    # pairwise (NON-compounding) banana shears on the INPUT coords:
    #   v_0 = u_0 ; v_i = u_i + b_{i-1} (u_{i-1}^2 - 1)
    # Chaining on outputs (v_{i-1}) compounds quadratics and blows up ~1e20
    # by d=16; pairwise keeps values O(u^2) while staying unit-triangular.
    d = u.shape[-1]
    cols = [u[:, 0]]
    for i in range(1, d):
        cols.append(u[:, i] + b[i - 1] * (u[:, i - 1] ** 2 - 1.0))
    return jnp.stack(cols, axis=-1)


def _shear_inv(v, b):
    d = v.shape[-1]
    cols = [v[:, 0]]
    for i in range(1, d):
        cols.append(v[:, i] - b[i - 1] * (cols[i - 1] ** 2 - 1.0))
    return jnp.stack(cols, axis=-1)


def warp_forward(w, z):
    if isinstance(w, SinhArcsinhWarp):
        u = z @ w.R1.T
        y = _sas_fwd(u, w.s, w.t)
        return w.shift + w.scale * (y @ w.R2.T)
    u = z @ w.R.T
    v = _shear_fwd(u, w.bcoef)
    return w.shift + w.scale * v


def warp_inverse(w, x):
    if isinstance(w, SinhArcsinhWarp):
        y = ((x - w.shift) / w.scale) @ w.R2
        u = _sas_inv(y, w.s, w.t)
        return u @ w.R1
    v = (x - w.shift) / w.scale
    u = _shear_inv(v, w.bcoef)
    return u @ w.R


def warp_logdet(w, z):
    """log|det d warp_forward / dz| evaluated at z, shape (n,)."""
    if isinstance(w, SinhArcsinhWarp):
        u = z @ w.R1.T
        return _sas_logderiv(u, w.s, w.t).sum(axis=-1) + jnp.log(w.scale).sum()
    # shear chain is unit-triangular: only the diagonal scale contributes
    return jnp.full(z.shape[0], jnp.log(w.scale).sum(), dtype=z.dtype)
