# 2026-07-10 — reconvene directives for the gate-(iv) P-verdict assembly

Eval jsons landed (eval_train4/eval_train2); reconvene did a first-pass read ONLY
(crude cuts, known artifacts). Assembly belongs to the session's harness. Directives:

1. **Data audit first:** some ess_frac_n values are NaN — find and classify before any
   fraction is computed. Verify funnel rows actually used K=512 in these evals.
2. **Assembly:** per-family x per-d tables, both (K,T) columns, REGIMES SEPARATED
   (in-family fresh-theta vs held-out-family). Then adjudicate P1/P2/P3/P11/P12 with
   the FROZEN definitions only.
3. **Baselines 1-4 now** — required for P1's SW2 clause, P3's bad-SW2 definition, and
   P7. P11 additionally needs eval_train4ng on the held-out rows (run if missing).
4. **P12 metric question:** report BOTH the composite/ESS-fraction cut and the paired
   instrument; if they disagree, say so — the frozen P12 wording ("performance
   improves") is adjudicated on the pre-specified composite, with the paired result
   reported alongside as the mechanism evidence.
5. **Sanity-floor ruling:** the floors were absolute numbers imported from a different
   compute-per-target regime — mis-calibration acknowledged. Recalibrate as a RELATIVE
   instrument (each arm vs its own trained-target subsample at its own compute);
   document; do not retroactively gate on the old floors.
6. **RESULTS-toy.md** then gets written with whatever the assembly shows. If the story
   is "scaling direction positive, absolute bars unmet at toy compute," report exactly
   that — it is a publishable, pre-registration-consistent outcome, and the staged
   2M/longer-arm chain is the pre-registered follow-up. K-T1 assessment included
   (budget ~30/480 H100-h — 'honest effort within budget' is far from exhausted).
