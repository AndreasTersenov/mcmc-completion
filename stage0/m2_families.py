"""M2 zoo v0: parametric families with explicit uniform box priors, plus
vectorized-over-theta energy functions and the centered energy
pseudo-likelihood.

Observation model (implementation choice, logged in
log/2026-07-09-m2-oracle.md): a context C = {(x_k, E_k[, grad E_k])} enters
the posterior through Gaussian pseudo-noise on energy VALUES CENTERED WITHIN
THE CONTEXT (energies are only defined up to an additive constant across
targets/families — the model must not rely on absolute offsets) and, when
present, raw gradient values (offset-free already).

Energy convention matches stage0.targets: E = -logpdf_u, gradE = -grad_logpdf_u.
"""

import numpy as np

from .targets import GMM, DoubleWell, Funnel

LOG2PI = np.log(2.0 * np.pi)


class _Family:
    def prior_sample(self, rng, n):
        return self.prior_lo + (self.prior_hi - self.prior_lo) * rng.uniform(
            size=(n, self.theta_dim)
        )

    def logprior(self, theta):
        theta = np.atleast_2d(theta)
        inside = np.all((theta >= self.prior_lo) & (theta <= self.prior_hi), axis=1)
        return np.where(inside, 0.0, -np.inf)

    @property
    def prior_std(self):
        return (self.prior_hi - self.prior_lo) / np.sqrt(12.0)

    def log_z(self, theta):
        return np.zeros(len(np.atleast_2d(theta)))


class Gmm2Family(_Family):
    """d=2, theta = (separation, weight1): modes at (+-sep/2, 0), unit var."""

    name, d, theta_dim = "gmm2", 2, 2
    prior_lo = np.array([2.0, 0.2])
    prior_hi = np.array([10.0, 0.8])

    def target(self, theta):
        sep, w = theta
        mu = np.array([sep / 2.0, 0.0])
        return GMM(np.stack([mu, -mu]), np.array([w, 1.0 - w]))

    def energy(self, theta, xs):
        theta, xs = np.atleast_2d(theta), np.atleast_2d(xs)
        mu = theta[:, 0:1] / 2.0                      # (M,1)
        w = theta[:, 1:2]                             # (M,1)
        x0, x1 = xs[:, 0][None, :], xs[:, 1][None, :]  # (1,K)
        d1 = (x0 - mu) ** 2 + x1**2
        d2 = (x0 + mu) ** 2 + x1**2
        lc1 = np.log(w) - 0.5 * d1 - LOG2PI
        lc2 = np.log(1.0 - w) - 0.5 * d2 - LOG2PI
        logp = np.logaddexp(lc1, lc2)
        r1 = np.exp(lc1 - logp)
        r2 = 1.0 - r1
        G = np.empty(logp.shape + (2,))
        G[:, :, 0] = r1 * (x0 - mu) + r2 * (x0 + mu)  # -d logp/dx0
        G[:, :, 1] = np.broadcast_to(x1, logp.shape)
        return -logp, G


class Funnel2Family(_Family):
    """d=2, theta = (sigma_v, cond_scale)."""

    name, d, theta_dim = "funnel2", 2, 2
    prior_lo = np.array([1.0, 0.5])
    prior_hi = np.array([4.0, 2.0])

    def target(self, theta):
        return Funnel(2, sigma_v=theta[0], cond_scale=theta[1])

    def energy(self, theta, xs):
        theta, xs = np.atleast_2d(theta), np.atleast_2d(xs)
        sv = theta[:, 0:1]                             # (M,1)
        c = theta[:, 1:2]
        v, y = xs[:, 0][None, :], xs[:, 1][None, :]    # (1,K)
        var_y = c**2 * np.exp(v)                       # (M,K)
        logp = (
            -0.5 * v**2 / sv**2 - 0.5 * np.log(2 * np.pi * sv**2)
            - 0.5 * y**2 / var_y - 0.5 * (LOG2PI + np.log(var_y))
        )
        G = np.empty(logp.shape + (2,))
        G[:, :, 0] = -(-v / sv**2 + 0.5 * y**2 / var_y - 0.5)
        G[:, :, 1] = y / var_y
        return -logp, G


class Dwell2Family(_Family):
    """d=2, theta = (a, b): E = a (x1^2-b)^2 + x2^2/2 (unnormalized)."""

    name, d, theta_dim = "dwell2", 2, 2
    prior_lo = np.array([0.5, 0.5])
    prior_hi = np.array([3.0, 2.5])
    _t = np.linspace(-8.0, 8.0, 100_001)

    def target(self, theta):
        return DoubleWell(2, a=theta[0], b=theta[1])

    def energy(self, theta, xs):
        theta, xs = np.atleast_2d(theta), np.atleast_2d(xs)
        a, b = theta[:, 0:1], theta[:, 1:2]
        x1, x2 = xs[:, 0][None, :], xs[:, 1][None, :]
        E = a * (x1**2 - b) ** 2 + 0.5 * x2**2
        G = np.empty(E.shape + (2,))
        G[:, :, 0] = 4.0 * a * x1 * (x1**2 - b)
        G[:, :, 1] = np.broadcast_to(x2, E.shape)
        return E, G

    def log_z(self, theta):
        theta = np.atleast_2d(theta)
        out = np.empty(len(theta))
        for start in range(0, len(theta), 64):
            chunk = theta[start:start + 64]
            w = np.exp(-chunk[:, 0:1] * (self._t[None, :] ** 2 - chunk[:, 1:2]) ** 2)
            out[start:start + 64] = np.log(np.trapezoid(w, self._t, axis=1))
        return out + 0.5 * LOG2PI  # x2 factor


class Gmm8Family(_Family):
    """d=8, theta = mu in R^8: equal-weight modes at +-mu, unit var.

    Prior restricts mu_1 >= 0 to break the exact mu <-> -mu symmetry of the
    energy (otherwise the posterior is bimodal by construction and
    'contraction' is undefined).
    """

    name, d, theta_dim = "gmm8", 8, 8
    prior_lo = np.array([0.0] + [-3.0] * 7)
    prior_hi = np.array([3.0] + [3.0] * 7)

    def target(self, theta):
        theta = np.asarray(theta, dtype=float)
        return GMM(np.stack([theta, -theta]), np.array([0.5, 0.5]))

    def energy(self, theta, xs):
        theta, xs = np.atleast_2d(theta), np.atleast_2d(xs)
        x2 = (xs**2).sum(axis=1)[None, :]              # (1,K)
        m2 = (theta**2).sum(axis=1)[:, None]           # (M,1)
        xm = theta @ xs.T                              # (M,K)
        d1 = x2 + m2 - 2.0 * xm
        d2 = x2 + m2 + 2.0 * xm
        lc1, lc2 = -0.5 * d1, -0.5 * d2
        logp = np.logaddexp(lc1, lc2) + np.log(0.5) - 0.5 * self.d * LOG2PI
        r1 = np.exp(lc1 - np.logaddexp(lc1, lc2))
        # G = r1 (x - mu) + r2 (x + mu) = x - (2 r1 - 1) mu
        G = xs[None, :, :] - (2.0 * r1 - 1.0)[:, :, None] * theta[:, None, :]
        return -logp, G


FAMILIES = {
    "gmm2": Gmm2Family(),
    "funnel2": Funnel2Family(),
    "dwell2": Dwell2Family(),
    "gmm8": Gmm8Family(),
}


def centered_energy_loglik(family, theta, xs, energies, sigma_e, grads=None, sigma_g=None):
    """log p(C | theta) with energies centered within the context (offset-
    invariant); gradient observations, when given, enter uncentered."""
    E, G = family.energy(theta, xs)
    e_c = E - E.mean(axis=1, keepdims=True)
    obs_c = energies - energies.mean()
    ll = -0.5 * (((e_c - obs_c[None, :]) / sigma_e) ** 2).sum(axis=1)
    if grads is not None:
        ll = ll - 0.5 * (((G - grads[None, :, :]) / sigma_g) ** 2).sum(axis=(1, 2))
    return ll
