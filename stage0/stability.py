"""Gate 4: N-doubling stability check for every reported number."""


def n_doubling(estimate_fn, n, tol, rel=False, seed=0):
    """estimate_fn(n, seed) -> float. Evaluates at n and 2n (independent
    seeds); passes if the move is below tol (absolute, or relative to the
    larger magnitude when rel=True)."""
    v1 = float(estimate_fn(n, seed))
    v2 = float(estimate_fn(2 * n, seed + 1))
    delta = abs(v2 - v1)
    denom = max(abs(v1), abs(v2)) if rel else 1.0
    return {"passed": bool(delta <= tol * denom), "v_n": v1, "v_2n": v2, "delta": delta}
