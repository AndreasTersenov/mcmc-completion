# 2026-07-10 — reconvene ruling on the paired-eval evidence package

## Verdict accepted
PAIRED-B-AMORTIZATION stands: 18/24 paired (p~0.011), medians 1.65x (T=1) / 2.5x (T=5)
at 10x less compute per pair. The gate3e P-sharp fail is reclassified as a
population/theta artifact; the paired instrument is henceforth THE sharpness metric.
Scorecard note: P-sharp stays a Claude miss as registered, asterisked as directionally
vindicated by the paired instrument.

## Ruling: PROCEED TO GATE (iv) — option (b), with gate (iii) re-scoped

1. Gate (iii) is no longer a blocking gate. It becomes two embedded instruments inside
   gate (iv)'s harvest: (a) trained-target sanity floor (the model must clear the
   composite on a subsample of its own training targets — pre-register the subsample
   and bar before the arms run); (b) the paired-sharpness instrument vs the gate3e
   checkpoint (pre-registered expectation: continued improvement at 1024 targets —
   P-scale, 70%: paired medians improve monotonically 128 -> 1024).
2. BEFORE the chain: update datagen + eval for the standing rulings — T-mix contexts
   (Ruling 1), dual (K,T) eval columns, funnel rows at K=512 — re-hash configs, then
   launch the pre-registered gate-(iv) chain (datagen -> train4 / train2 / train4ng
   arms as resumable 3h chains) with per-arm step budgets pre-registered in the
   gate-4 log. The regimes stay separated in all reporting: trained-target vs fresh-theta
   generalization.
3. The 2M single-recipe chain stays STAGED, superseded unless gate-(iv) arms show
   undertraining (loss still falling at budget end, or paired instrument regressing).
4. P-verdict assembly per JOBS.md follows the arms: P1/P2/P3/P11/P12 with the frozen
   definitions; baselines 1-4 built for the SW2 clauses; then RESULTS-toy.md.

Budget check: ~14/480 H100-hours spent; the full chain fits comfortably. The ladder has
done its job — every model-side mechanism is understood; from here the question is the
one the project was born to answer: does prior-fitting scale?
