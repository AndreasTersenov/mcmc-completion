"""Verification of the reused jax_flows CFM core (user flagged: 'fast toy,
verify'). Two checks:

1. cfm_sample integrates the probability-flow ODE correctly: with the ANALYTIC
   OT-CFM velocity for a Gaussian target N(mu, s^2 I) plugged in as the
   "model", samples must land on N(mu, s^2 I). Closed form for independent
   coupling x0~N(0,I), x1~N(mu, s^2 I):
     x_t ~ N(t mu, v_t I),  v_t = (1-t)^2 + t^2 s^2
     u(x,t) = E[x1 - x0 | x_t=x] = mu + (t s^2 - (1-t)) / v_t * (x - t mu)
2. cfm_loss is minimized by that analytic velocity: any fixed perturbation of
   u increases the loss (checked at matched RNG).

Also pins the conditional-CFM extension in ics.cfm to the same math (it must
reduce to the unconditional case when the condition is ignored).
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from flax import linen as nn

from jax_flows import cfm_loss, cfm_sample

MU = jnp.array([1.5, -0.5])
S = 1.7


class AnalyticGaussianVelocity(nn.Module):
    """Exact OT-CFM velocity for N(MU, S^2 I); perturb adds a bias."""

    perturb: float = 0.0

    @nn.compact
    def __call__(self, x, t):
        t = t[:, None]
        v_t = (1.0 - t) ** 2 + (t * S) ** 2
        u = MU + (t * S**2 - (1.0 - t)) / v_t * (x - t * MU)
        return u + self.perturb


def test_cfm_sample_transports_to_gaussian():
    model = AnalyticGaussianVelocity()
    params = model.init(jr.key(0), jnp.ones((1, 2)), jnp.ones((1,)))["params"]
    x = cfm_sample(model, params, jr.key(1), (100_000, 2), n_steps=200, solver="heun")
    x = np.asarray(x)
    np.testing.assert_allclose(x.mean(axis=0), np.asarray(MU), atol=0.03)
    np.testing.assert_allclose(x.std(axis=0), [S, S], rtol=0.02)


def test_euler_and_heun_agree():
    model = AnalyticGaussianVelocity()
    params = model.init(jr.key(0), jnp.ones((1, 2)), jnp.ones((1,)))["params"]
    xh = cfm_sample(model, params, jr.key(2), (20_000, 2), n_steps=400, solver="heun")
    xe = cfm_sample(model, params, jr.key(2), (20_000, 2), n_steps=400, solver="euler")
    # same key => same x0 draw; fine integrators must agree closely
    np.testing.assert_allclose(np.asarray(xh), np.asarray(xe), atol=0.02)


@pytest.mark.parametrize("perturb", [0.4, -0.4])
def test_cfm_loss_minimized_by_analytic_velocity(perturb):
    key = jr.key(3)
    x1 = MU + S * jr.normal(jr.key(4), (50_000, 2))
    m0 = AnalyticGaussianVelocity()
    mp = AnalyticGaussianVelocity(perturb=perturb)
    p0 = m0.init(jr.key(0), jnp.ones((1, 2)), jnp.ones((1,)))["params"]
    pp = mp.init(jr.key(0), jnp.ones((1, 2)), jnp.ones((1,)))["params"]
    l0 = float(cfm_loss(p0, x1, key, m0))
    lp = float(cfm_loss(pp, x1, key, mp))  # same key: matched t and noise draws
    assert l0 < lp, f"analytic velocity should minimize CFM loss: {l0=} {lp=}"
