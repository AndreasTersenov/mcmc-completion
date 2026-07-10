# 2026-07-10 — reconvene review of gate-(iii) attempt 3 (job 15615537, results/gate3.json)

Headline: 1/10 vs the >=8/10 bar — but the failure decomposes into three distinct
findings requiring three distinct responses. Reviewed by the main (reconvene) session.

## What the data says

1. **Attention fix WORKED for mode structure:** mode_recovery = 1.0 on gmm/dwell at
   d=4,8 (was: "exactly the visited modes" in attempt 2). The encoder now infers
   unvisited modes. REMAINING real deficiency: d=2 multimodal rows still drop
   (recovery 0.67 gmm2 with logZ = -0.83 ~ ln covered mass; 0.5 dwell2) — and M2's
   oracle identifies gmm2 from K=8, so this is below a provable ceiling. Real bug.
2. **First genuine in-context sampling success:** warp d=2 ESS 94%, D2_hat 0.063,
   |logZ| 0.003 (stable). Graceful d-scaling on warp (94/42/12% at d=2/4/8).
   Across gmm/dwell, correct mode structure but ESS 0.1-1.4% => the conditional
   sharpness is the gap, not the concept. Training loss plateaued ~1.38-1.44 from
   step 10k: optimization/capacity suspect BEFORE architecture scale — run the cheap
   diagnostic first (single-pair conditional overfit with the NEW architecture; if it
   cannot overfit one pair sharply, the pathway/objective needs the fix, not width).
3. **Funnel rows: misspecified bar, not model failure.** Stage-0 M2: funnel sigma_v is
   near-unidentifiable at K=128 — the Bayes-optimal oracle is prior-wide until K=512.
   Demanding ESS>=5% at K=128 on funnels exceeds the oracle ceiling. The model
   refusing (low ESS, certificate flags) on unidentifiable context is the DESIGNED
   P3 behavior. Fix = eval design: evaluate funnel rows at K=512 AND/OR against the
   oracle-ceiling reference; count honest refusal separately as certificate success.

## Directives for attempt 4 (criteria = session periphery; FROZEN core untouched)

- Re-specify gate-(iii) criteria, pre-registered BEFORE the run, to mirror the frozen
  P1 composite it guards (>=80% of d<=8 in-family targets on the ESS+SW2 composite),
  with funnel rows handled per finding 3 (K=512 or oracle-ceiling-referenced) and
  mode_recovery mandatory as before. Justify the amendment in the log (this note).
- Order of work: (a) single-pair overfit diagnostic (new arch); (b) fix d=2 mode drop
  (hypotheses: shortK augmentation creating low-d ambiguity; d-embedding; per-family
  balance) — one variable at a time; (c) only then wider/deeper.
- Do NOT chase funnel ESS at K=128. Do NOT touch PLAN.md frozen core.

## Status

Budget ~1.5 H100-hours of 480. K-T1: distant. The gate ladder continues to pay for
itself: attempt 3 cost 32 min and returned one confirmed fix, one localized real bug,
one eval-design correction, and the project's first certified in-context samples.
