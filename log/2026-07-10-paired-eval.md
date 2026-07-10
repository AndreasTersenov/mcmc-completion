# 2026-07-10 — PAIRED EVAL (reconvene-endorsed deciding evidence)

## Pre-registration
Both existing checkpoints — b1 (10-target recipe) and gate3e (128-target,
T-mix) — on ONE common set of 24 FRESH targets (2 per cell, 4 families x
d {2,4,8}; seed stream jr.key(424242), disjoint from all training/eval
draws). Both (K,T) columns per target: (128,1) and (128,5); funnels K=512 in
both. SHARED per-target bespoke reference (T=1-context whitening) and SHARED
contexts (tokens built once; b1 receives the one-hot-stripped 37-dim view).
This removes the theta-draw and population confounds of the gate3e
comparison entirely.

## Pre-registered verdict rule (directive branches)
improvement := median(b1 T1 ratio) / median(z128 T1 ratio) over the 24.
- improvement >= 1.5x  -> PAIRED-B (amortization visible when paired): the
  gate3e P-sharp fail was a population/theta artifact; assemble the evidence
  package for the reconvene bar conversation and STOP (no bar movement here).
- improvement < 1.5x   -> PAIRED-A (flat-or-worse at 10x less compute per
  pair): compute-per-pair limitation unfalsified and cheap to test ->
  pre-register and launch the 2M-step resumable 3h-chain training on the
  128-zoo, then re-run the paired eval against the 2M checkpoint.
Secondary readouts (reported, not gating): T5 columns, composite pass counts
per model/column, per-family medians, paired per-target ratio distribution.

## Expectation
Genuinely uncertain (that is the point). Weak prior (55%) on PAIRED-A: the
gate3e run's heterogeneity (best-ever rows next to catastrophic ones) looks
like undertraining variance, and 260 effective steps/pair vs 2500 is a 10x
compute cut that SHOULD bite.
- submitted: job 15647929, config a01c711199b5
