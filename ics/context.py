"""Context generation (frozen protocol): 4 MALA chains x K/4 steps from
overdispersed inits N(0, 5^2 I), exact untempered (E, grad E) recorded at the
visited points, energies CENTERED within the context (offset invariance is
mandatory).

Free-periphery choices (logged): per-context whitening of x and gradients
(the sampler head works in whitened space; stage-0 showed relative scales are
the honest readout model); a 3-candidate step-size probe so one protocol
serves all zoo scales; energy/gradient features normalized by their context
spread with the log-scales appended as global token features.

Token layout (TOKEN_DIM = 2*DMAX + 4):
  [ x_white (DMAX, zero-padded) | (E - mean E)/sd(E) |
    sigma*gradE / sd (DMAX, zero-padded) | log sd(E) | log sd(grad) | d/DMAX ]
"""

from functools import partial
from typing import NamedTuple

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr

from .zoo import DMAX, logpdf

TOKEN_DIM = 2 * DMAX + 4
N_CHAINS = 4
INIT_SCALE = 5.0
STEP_CANDIDATES = (0.5, 0.15, 0.05)


class Context(NamedTuple):
    x_raw: jnp.ndarray   # (K, d)
    energy: jnp.ndarray  # (K,)
    grad: jnp.ndarray    # (K, d) gradient of the ENERGY (-grad logpdf)
    mu: jnp.ndarray      # (d,)
    sigma: jnp.ndarray   # (d,)
    tokens: jnp.ndarray  # (K, TOKEN_DIM)
    accept: jnp.ndarray  # scalar mean acceptance
    step: jnp.ndarray    # scalar step size used


def whiten_apply(x, mu, sigma):
    return (x - mu) / sigma


def whiten_invert(x_white, mu, sigma):
    return mu + sigma * x_white


def _run_mala(key, ld_single, d, n_steps, step, temperature):
    tempered = lambda xi: ld_single(xi) / temperature
    mala = blackjax.mala(tempered, step)
    k_init, k_run = jr.split(key)
    x0 = INIT_SCALE * jr.normal(k_init, (N_CHAINS, d))
    states = jax.vmap(mala.init)(x0)

    def one_step(states, k):
        ks = jr.split(k, N_CHAINS)
        states, infos = jax.vmap(mala.step)(ks, states)
        return states, (states.position, infos.is_accepted)

    keys = jr.split(k_run, n_steps)
    _, (pos, acc) = jax.lax.scan(one_step, states, keys)
    # pos: (n_steps, N_CHAINS, d) -> chain-major (chain0 steps..., chain1 ...)
    x = pos.transpose(1, 0, 2).reshape(N_CHAINS * n_steps, d)
    return x, acc.mean()


@partial(jax.jit, static_argnames=("n_steps",))
def _mala_target(target, key, n_steps, step, temperature):
    """Own MALA (stage-0 algorithm, jax twin) with the TARGET AS A PYTREE
    ARGUMENT: compiles once per (family type, d, n_steps) instead of once per
    target — mandatory for zoo-scale context generation."""
    d = target.d
    ld = lambda xi: logpdf(target, xi[None, :])[0] / temperature
    grad_ld = jax.grad(ld)
    k_init, k_run = jr.split(key)
    x = INIT_SCALE * jr.normal(k_init, (N_CHAINS, d))
    lp = jax.vmap(ld)(x)
    g = jax.vmap(grad_ld)(x)

    def one_step(carry, k):
        x, lp, g = carry
        kn, ku = jr.split(k)
        noise = jr.normal(kn, x.shape)
        prop = x + step * g + jnp.sqrt(2.0 * step) * noise
        lp_p = jax.vmap(ld)(prop)
        g_p = jax.vmap(grad_ld)(prop)
        fwd = -((prop - x - step * g) ** 2).sum(-1) / (4.0 * step)
        bwd = -((x - prop - step * g_p) ** 2).sum(-1) / (4.0 * step)
        log_alpha = lp_p - lp + bwd - fwd
        acc = jnp.log(jr.uniform(ku, (N_CHAINS,))) < log_alpha
        x = jnp.where(acc[:, None], prop, x)
        lp = jnp.where(acc, lp_p, lp)
        g = jnp.where(acc[:, None], g_p, g)
        return (x, lp, g), (x, acc)

    _, (pos, accs) = jax.lax.scan(one_step, (x, lp, g), jr.split(k_run, n_steps))
    xs = pos.transpose(1, 0, 2).reshape(N_CHAINS * n_steps, d)
    return xs, accs.mean()


def generate_context_for_target(key, target, K, temperature=1.0):
    """Zoo-scale context generation: jit-cached per (family, d, K)."""
    assert K % N_CHAINS == 0
    k_probe, k_run = jr.split(key)
    step = STEP_CANDIDATES[-1]
    for cand in STEP_CANDIDATES:
        _, acc = _mala_target(target, k_probe, 8, cand, temperature)
        if float(acc) > 0.25:
            step = cand
            break
    x_raw, accept = _mala_target(target, k_run, K // N_CHAINS, step, temperature)
    energy = -logpdf(target, x_raw)
    ld_single = lambda xi: logpdf(target, xi[None, :])[0]
    grad_e = -jax.vmap(jax.grad(ld_single))(x_raw)
    return _build_context(x_raw, energy, grad_e, accept, step, target.d)


def generate_context(key, logpdf_fn, d, K, n_chains=N_CHAINS, temperature=1.0):
    assert n_chains == N_CHAINS, "frozen protocol: 4 chains"
    assert K % N_CHAINS == 0
    n_steps = K // N_CHAINS
    ld_single = lambda xi: logpdf_fn(xi[None, :])[0]

    # deterministic step-size probe: largest candidate with accept > 0.25
    k_probe, k_run = jr.split(key)
    step = STEP_CANDIDATES[-1]
    for cand in STEP_CANDIDATES:
        _, acc = _run_mala(k_probe, ld_single, d, 8, cand, temperature)
        if float(acc) > 0.25:
            step = cand
            break

    x_raw, accept = _run_mala(k_run, ld_single, d, n_steps, step, temperature)

    energy = -logpdf_fn(x_raw)
    grad_e = -jax.vmap(jax.grad(ld_single))(x_raw)
    return _build_context(x_raw, energy, grad_e, accept, step, d)


def _build_context(x_raw, energy, grad_e, accept, step, d):
    mu = x_raw.mean(axis=0)
    sigma = x_raw.std(axis=0) + 1e-6
    x_white = whiten_apply(x_raw, mu, sigma)

    e_scale = energy.std() + 1e-8
    e_tok = (energy - energy.mean()) / e_scale
    g_white = sigma[None, :] * grad_e
    g_scale = g_white.std() + 1e-8
    g_tok = g_white / g_scale

    K_ = x_raw.shape[0]
    pad = lambda a: jnp.concatenate(
        [a, jnp.zeros((K_, DMAX - d), dtype=a.dtype)], axis=1
    )
    const = jnp.tile(
        jnp.array([jnp.log(e_scale), jnp.log(g_scale), d / DMAX])[None, :], (K_, 1)
    )
    tokens = jnp.concatenate([pad(x_white), e_tok[:, None], pad(g_tok), const], axis=1)
    return Context(x_raw=x_raw, energy=energy, grad=grad_e, mu=mu, sigma=sigma,
                   tokens=tokens, accept=accept, step=jnp.asarray(step))


def pad_to_dmax(key, x_white):
    """Pad whitened d-dim samples to DMAX with fresh N(0,1) draws (the FM head
    always works in DMAX dims; padding dims are honest standard normals)."""
    n, d = x_white.shape
    if d == DMAX:
        return x_white
    noise = jr.normal(key, (n, DMAX - d), dtype=x_white.dtype)
    return jnp.concatenate([x_white, noise], axis=1)
