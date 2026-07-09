import numpy as np


def logsumexp(a, axis=None, keepdims=False):
    """Numerically stable log(sum(exp(a))). Handles all-(-inf) slices."""
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        s = np.log(np.sum(np.exp(a - m_safe), axis=axis, keepdims=True)) + m_safe
    if keepdims:
        return s
    if axis is None:
        return float(s.reshape(()))
    return np.squeeze(s, axis=axis)
