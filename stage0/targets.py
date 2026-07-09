"""Target families for M1/M2.

Every target exposes:
  logpdf_u(x)      : possibly-unnormalized log density, x (n, d) -> (n,)
  grad_logpdf_u(x) : gradient of logpdf_u, (n, d) -> (n, d)
  log_Z            : log normalizer of exp(logpdf_u) (0.0 when already normalized)
  sample(rng, n)   : exact samples (where available)
Energy convention: E(x) = -logpdf_u(x).
"""

import numpy as np

from .utils import logsumexp


class Gaussian:
    def __init__(self, mean, var):
        self.mean = np.atleast_1d(np.asarray(mean, dtype=float))
        self.d = self.mean.size
        self.var = np.broadcast_to(np.asarray(var, dtype=float), (self.d,)).copy()
        self.log_Z = 0.0

    def logpdf(self, x):
        x = np.atleast_2d(x)
        quad = (((x - self.mean) ** 2) / self.var).sum(axis=1)
        return -0.5 * quad - 0.5 * np.log(2 * np.pi * self.var).sum()

    logpdf_u = logpdf

    def grad_logpdf_u(self, x):
        return -(np.atleast_2d(x) - self.mean) / self.var

    def sample(self, rng, n):
        return self.mean + np.sqrt(self.var) * rng.standard_normal((n, self.d))


class GMM:
    """Isotropic-component Gaussian mixture; var is scalar or per-component (J,)."""

    def __init__(self, means, weights, var=1.0):
        self.means = np.atleast_2d(np.asarray(means, dtype=float))
        self.J, self.d = self.means.shape
        w = np.asarray(weights, dtype=float)
        self.weights = w / w.sum()
        self.var = np.broadcast_to(np.asarray(var, dtype=float), (self.J,)).copy()
        self.log_Z = 0.0

    def _comp_logpdfs(self, x):
        x = np.atleast_2d(x)
        diff = x[:, None, :] - self.means[None, :, :]
        return (
            -0.5 * (diff**2).sum(axis=2) / self.var[None, :]
            - 0.5 * self.d * np.log(2 * np.pi * self.var)[None, :]
        )

    def logpdf(self, x):
        lc = np.log(self.weights)[None, :] + self._comp_logpdfs(x)
        return logsumexp(lc, axis=1)

    logpdf_u = logpdf

    def grad_logpdf_u(self, x):
        x = np.atleast_2d(x)
        lc = np.log(self.weights)[None, :] + self._comp_logpdfs(x)
        resp = np.exp(lc - logsumexp(lc, axis=1, keepdims=True))
        comp_grad = -(x[:, None, :] - self.means[None, :, :]) / self.var[None, :, None]
        return (resp[:, :, None] * comp_grad).sum(axis=1)

    def sample(self, rng, n):
        comp = rng.choice(self.J, size=n, p=self.weights)
        eps = rng.standard_normal((n, self.d))
        return self.means[comp] + np.sqrt(self.var)[comp][:, None] * eps


class Funnel:
    """Neal-style funnel: v = x[:,0] ~ N(0, sigma_v^2),
    x_i | v ~ N(0, cond_scale^2 * e^v) for i >= 1. Normalized (log_Z = 0)."""

    def __init__(self, d, sigma_v=3.0, cond_scale=1.0):
        assert d >= 2
        self.d = d
        self.sigma_v = float(sigma_v)
        self.c = float(cond_scale)
        self.log_Z = 0.0

    def logpdf(self, x):
        x = np.atleast_2d(x)
        v, y = x[:, 0], x[:, 1:]
        k = self.d - 1
        var_y = self.c**2 * np.exp(v)
        lp_v = -0.5 * v**2 / self.sigma_v**2 - 0.5 * np.log(2 * np.pi * self.sigma_v**2)
        lp_y = -0.5 * (y**2).sum(axis=1) / var_y - 0.5 * k * (np.log(2 * np.pi) + np.log(var_y))
        return lp_v + lp_y

    logpdf_u = logpdf

    def grad_logpdf_u(self, x):
        x = np.atleast_2d(x)
        v, y = x[:, 0], x[:, 1:]
        k = self.d - 1
        var_y = self.c**2 * np.exp(v)
        g = np.empty_like(x)
        g[:, 0] = -v / self.sigma_v**2 + 0.5 * (y**2).sum(axis=1) / var_y - 0.5 * k
        g[:, 1:] = -y / var_y[:, None]
        return g

    def sample(self, rng, n):
        v = self.sigma_v * rng.standard_normal(n)
        y = self.c * np.exp(v / 2)[:, None] * rng.standard_normal((n, self.d - 1))
        return np.column_stack([v, y])


class DoubleWell:
    """E(x) = a (x1^2 - b)^2 + sum_{i>=2} x_i^2 / 2 (unnormalized).

    log_Z from 1-d quadrature on the x1 factor; exact sampling of x1 by
    inverse-CDF on a dense grid, remaining coords standard normal.
    """

    def __init__(self, d, a, b, x1_range=8.0, n_grid=100_001):
        self.d, self.a, self.b = d, float(a), float(b)
        t = np.linspace(-x1_range, x1_range, n_grid)
        w = np.exp(-self.a * (t**2 - self.b) ** 2)
        self.log_Z = float(np.log(np.trapezoid(w, t)) + 0.5 * (d - 1) * np.log(2 * np.pi))
        cdf = np.cumsum(w)
        self._t, self._cdf = t, cdf / cdf[-1]

    def logpdf_u(self, x):
        x = np.atleast_2d(x)
        return -self.a * (x[:, 0] ** 2 - self.b) ** 2 - 0.5 * (x[:, 1:] ** 2).sum(axis=1)

    def logpdf(self, x):
        return self.logpdf_u(x) - self.log_Z

    def grad_logpdf_u(self, x):
        x = np.atleast_2d(x)
        g = np.empty_like(x)
        g[:, 0] = -4.0 * self.a * x[:, 0] * (x[:, 0] ** 2 - self.b)
        g[:, 1:] = -x[:, 1:]
        return g

    def sample(self, rng, n):
        x1 = np.interp(rng.uniform(size=n), self._cdf, self._t)
        rest = rng.standard_normal((n, self.d - 1))
        return np.column_stack([x1, rest])
