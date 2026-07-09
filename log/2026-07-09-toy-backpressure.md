# 2026-07-09 (evening) — phase 1: backpressure suite + ics package

## Hypothesis
Zoo v1, context protocol, and the conditional-FM stack can be built on the
verified stage-0 + jax_flows foundations with every piece gated by a test
before any training job runs (PLAN backpressure a–d).

## Setup
`ics/` package: `zoo.py` (6 families, exact samplers, normalized logpdfs),
`warps.py` (sinh-arcsinh + banana warps, explicit log-dets), `context.py`
(frozen 4-chain MALA protocol via blackjax; centered energies; per-context
whitening — free-periphery choice; deterministic 3-candidate step probe),
`models.py` (deep-sets encoder, velocity head with x1000-scaled time
embedding per the jax_flows audit), `cfm.py` (conditional CFM reusing
jax_flows ot_interpolate; scan-based Heun; CNF log-density with exact
divergence — the certificate's q-density), `certificate.py` (thin wrapper
over stage-0 estimators, ESS at N/2N + stability flag).

## Expectation
All tests green without weakening; failures during bring-up get diagnosed to
root cause, not tolerance-bumped.

## Result — 102 passed. Three design corrections found by the tests:
1. **Banana warp redesigned**: chained Rosenbrock shears (v_i depends on
   v_{i-1}) compound quadratics — values hit ~1e20 by d=16, breaking f64
   roundtrips outright. Replaced with PAIRWISE shears on input coords
   (v_i = u_i + b(u_{i-1}^2 - 1)): still unit-triangular (log-det from scale
   only), still non-Gaussian held-out geometry, values stay O(u^2).
2. **Overfit-gate recipe was underfit**, not broken: 1200 steps flat-LR gave
   sliced-W2^2 = 0.28; cosine decay + (256,256) + 2500 steps gives 0.027
   (same-p floor 0.018) in ~6 s CPU. Frozen into the executable test.
3. **The generic sampler<->density identity E_p[r/p]=1 is only
   variance-stable for light-tailed families.** For funnel/funnelmix/dwell,
   ANY fixed-bandwidth Gaussian-mixture reference r yields a
   divergent-variance estimator — the same mathematics as stage-0's "no
   Gaussian proposal certifies a funnel", now met from the testing side.
   Those families are covered instead by exact structural checks at every d
   (quadrature moments, conditional identities, algebraic component
   composition) plus d=2 dense-grid normalization for all six families.
   gmm/warp/banana keep the generic identity at d=2.
   (Also: dwell was never wrong — first-round failures were the reference's;
   verified by 2-d grid integration = 1.0000 and exact quadrature moments.)

## Updated belief
The estimator/zoo stack is trustworthy at stage-0 grade. The Karpathy ladder
can start: gate (i) has a green CPU-scale executable twin; the GPU-scale
confirmation with the certificate loop closes it next.
