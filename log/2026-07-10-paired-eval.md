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

## RESULT — PAIRED-B-AMORTIZATION (job 15647929, 5.7 min)
Pre-registered rule: improvement = 1.65x >= 1.5x. Full picture:
- The 128-model is better on 18/24 targets paired (median paired ratio
  0.62); medians: T=1 905 -> 549 (1.65x), T=5 855 -> 342 (2.5x) — at 10x
  LESS compute per (target,ctx) pair. AMORTIZATION IS REAL: more zoo, same
  compute, better per-target quality once theta-draw noise is removed.
- Losses cluster by family: the 6/24 regressions are scattered (2 dwell-d4,
  warp-d4/d8, funnel-d4, gmm-d2) with no single-family failure mode.
- Marginality note (honesty): 1.65x clears the 1.5x line but not by much on
  the primary column; the T=5 column (2.5x) and the 18/24 sign-test
  (p ~ 0.011, binomial) corroborate the direction independently.
- CRITICAL CONTEXT FOR THE BAR CONVERSATION: this eval is on FRESH theta —
  the in-family-generalization regime of frozen P1 — where BOTH models are
  far from the bars (pass 0-1/24; medians in the hundreds). Gate-(iii)'s
  trained-target regime and this regime must not be conflated. The evidence
  says: the amortization slope is positive and survives pairing; absolute
  P1-grade generalization plausibly needs the gate-(iv) scale zoo (1024
  targets) — which is precisely the amortization premise, now with a
  measured, paired scaling direction behind it.

## Evidence package for the reconvene (per directive: STOP, no bar movement)
1. Paired verdict + distribution (results/paired_eval.json, this log).
2. The gate3e P-sharp fail decomposition: registered metric confounded by
   scoring column + population + unpaired theta; the paired instrument
   reverses the sign of the conclusion.
3. Decision requested: (a) gate-(iii) redefinition at trained-target regime
   with the paired instrument as the sharpness metric, and/or (b) proceed to
   gate (iv) scale on the strength of the paired slope, with gate (iii)
   re-adjudicated at that scale. The 2M-compute question is now SECONDARY
  (staged scripts remain if the reconvene wants the compute axis measured).
