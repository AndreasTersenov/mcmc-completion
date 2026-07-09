"""Conditional CFM on top of the verified jax_flows core, plus the CNF
log-density used by the SNIS certificate (exact divergence, fixed-step Heun;
validated against a closed form in tests before any open-case use).
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from jax_flows import ot_interpolate

from .zoo import DMAX


def cond_cfm_loss(params, model, x1, tokens, key):
    """L = E || v(x_t, t, tokens) - (x1 - x0) ||^2, per-row contexts."""
    kt, kn = jr.split(key)
    b = x1.shape[0]
    t = jr.uniform(kt, (b,), dtype=x1.dtype)
    x0 = jr.normal(kn, x1.shape, dtype=x1.dtype)
    x_t = ot_interpolate(x0, x1, t)
    v = model.apply({"params": params}, x_t, t, tokens)
    return jnp.mean((v - (x1 - x0)) ** 2)


def make_velocity_fn(model, params, tokens):
    """Encode ONE context once; return velocity_fn(x (n, DMAX), t scalar)."""
    cond1 = model.apply({"params": params}, tokens[None], method="encode")

    def velocity_fn(x, t):
        cond = jnp.broadcast_to(cond1, (x.shape[0], cond1.shape[-1]))
        t_b = jnp.full((x.shape[0],), t, dtype=x.dtype)
        return model.apply({"params": params}, x, t_b, cond, method="velocity")

    return velocity_fn


def _heun_flow(velocity_fn, x0, n_steps):
    dt = 1.0 / n_steps
    ts = jnp.arange(n_steps) * dt

    def step(x, t):
        v1 = velocity_fn(x, t)
        v2 = velocity_fn(x + dt * v1, t + dt)
        return x + 0.5 * dt * (v1 + v2), None

    x1, _ = jax.lax.scan(step, x0, ts)
    return x1


def cond_cfm_sample(model, params, tokens, key, n, n_steps=100):
    """Sample n points (in whitened DMAX space) for a single context."""
    velocity_fn = make_velocity_fn(model, params, tokens)
    x0 = jr.normal(key, (n, DMAX), dtype=tokens.dtype)
    return _heun_flow(velocity_fn, x0, n_steps)


def cnf_logpdf(velocity_fn, x, n_steps=200):
    """log q(x) for the flow defined by velocity_fn, via backward Heun
    integration of (x, integral of div v). Exact divergence (jacfwd trace).

    d/dt log p_t(x(t)) = -div v  =>  log q(x1) = log N(x0) - int_0^1 div dt.
    """
    d = x.shape[1]
    dt = 1.0 / n_steps

    def div_fn(xb, t):
        def v_single(xi):
            return velocity_fn(xi[None, :], t)[0]

        jac = jax.vmap(jax.jacfwd(v_single))(xb)
        return jnp.trace(jac, axis1=1, axis2=2)

    def step(carry, k):
        xk, integ = carry
        t = 1.0 - k * dt
        v1 = velocity_fn(xk, t)
        d1 = div_fn(xk, t)
        x_pred = xk - dt * v1
        v2 = velocity_fn(x_pred, t - dt)
        d2 = div_fn(x_pred, t - dt)
        x_new = xk - 0.5 * dt * (v1 + v2)
        integ = integ + 0.5 * dt * (d1 + d2)
        return (x_new, integ), None

    (x0, integral), _ = jax.lax.scan(
        step, (x, jnp.zeros(x.shape[0], dtype=x.dtype)), jnp.arange(n_steps)
    )
    log_p0 = -0.5 * (x0**2).sum(axis=1) - 0.5 * d * jnp.log(2 * jnp.pi)
    return log_p0 - integral
