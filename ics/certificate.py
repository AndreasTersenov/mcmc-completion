"""The shipped certificate (frozen protocol): SNIS against the true energy,
ESS at N and 2N with the stability flag. All estimator math reuses the
gate-tested stage-0 module — do not reimplement (PLAN free-periphery rule).
"""

import numpy as np

from stage0.is_estimators import ess_frac_from_logw, logz_from_logw

STABILITY_LOG10_TOL = 0.2  # stage-0 M2 protocol: >1.6x movement on doubling flags


def snis_certificate(logp_target, logq):
    """logp_target, logq: (2N,) numpy arrays on the SAME 2N samples ~ q.

    Returns the frozen certificate: ESS/N at N (first half) and 2N (all),
    stability flag, logZ-hat, and D2-hat = -ln(ESS/N at 2N).
    """
    logw = np.asarray(logp_target, dtype=np.float64) - np.asarray(logq, dtype=np.float64)
    n2 = logw.size
    assert n2 % 2 == 0
    ess_n = ess_frac_from_logw(logw[: n2 // 2])
    ess_2n = ess_frac_from_logw(logw)
    stable = bool(
        abs(np.log10(max(ess_2n, 1e-300)) - np.log10(max(ess_n, 1e-300)))
        < STABILITY_LOG10_TOL
    )
    return {
        "ess_frac_n": float(ess_n),
        "ess_frac_2n": float(ess_2n),
        "stable": stable,
        "logz": float(logz_from_logw(logw)),
        "d2_hat": float(-np.log(max(ess_2n, 1e-300))),
        "n2": int(n2),
    }
