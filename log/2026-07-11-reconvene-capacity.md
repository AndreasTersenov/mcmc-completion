# 2026-07-11 — RECONVENE RULING: capacity arm harvested — STRUCTURAL AGAIN. The pivot conversation is triggered.

Harvest note: eval_cap (15754027) completed 18:15; the executor session was not
active, so the reconvene harvested directly from results/eval_cap.json using the
pre-registered mirror rule and instruments (shared refs via --refs-from; no new
choices were required or made). Chain 15754023/24 did the work (links 25/26 no-oped
as designed).

## Mirror rule applied verbatim (numbers re-extracted from raw JSON)

- TT_cap: 0.25M .083 | 0.5M .042 | 1M **.042** → TT_cap@1M = 0.042 < 0.50.
- Plateau: (0.042 − 0.042)/0.042 = 0% < +10% over 0.5M→1M → plateaued.
- FQ_cap = 549 / 1039 = **0.53×** (< 4×) — fresh quality DEGRADED with width.
- FE_cap at 1M: best d=4 family = warp 2.1% (< 5%) → fail. (Reported honestly:
  warp touched 5.73% at the 0.5M checkpoint and fell back — a non-monotone blip on
  a 6-target median; under any reading it cannot change the branch, since ALIVE and
  MEMORIZE both require TT ≥ 50%.)
- T=5 column: TT_T5 = 0.000 at every capacity checkpoint (base model: nonzero).

**Branch: STRUCTURAL (second time). Per the frozen rule → the PIVOT conversation
with Andreas. No further runs are authorized on this branch.**

Width did not merely fail to help — it hurt on every instrument (TT halved, fresh
ratio 559→1039 at matched steps, T=5 collapsed to zero). Combined with phase 1b:
10× compute flat, 4× parameters negative. The per-target sharpness limitation is
neither compute- nor capacity-bound; the evidence points at the objective/eval
combination (prior-fitting the zoo does not concentrate per-target mass), i.e. the
H-starve program is closed, not merely paused.

Scorecard: reconvene STRUCTURAL-confirm 60% → HIT. Executor 55% → HIT.

## Pivot recommendation (reconvene's input to the conversation; the decision is Andreas's)

Recommendation: **PARK (option 4), with a salvage plan.** Reasons, from our own
measurements: (a) both structural levers exhausted by rule; (b) Readout B: 2/12
cost-crossover — no measured niche; (c) Readout C: the certificate-gated proposal
works only where classical samplers are already trivially fast (d=2 banana,
eight-schools), and it FAILED on the one target from Andreas's own field (WL
band-power, concentrated posterior); (d) the dogfooding north star fails — our
first real user would not use it; (e) 7–8 weeks to thesis defense makes attention
the binding budget.

Salvage (banked regardless): the ESS/N = exp(−D₂) identity + certificate frontier;
the SNIS overspread false-blessing blind spot (24%); the compute-buys-robustness-
not-sharpness dissociation with clean instruments; the paired-instrument
methodology. Cheap later deliverable: a short methods note (identity + blind spot +
dissociation) — workshop-grade, honest, no further compute. The certificate
machinery remains live infrastructure (already reused by eft-sbi's Readout-C
surrogate work; candidate fold-in for any future D2-style engine).

Not recommended: the d≤4 proposal-engine product (technically defensible, no user);
per-domain scoping (WL result argues directly against the nearest domain we care
about); 1024@2M (that lever belongs to the MEMORIZE branch, which never fired).

If Andreas prefers to keep a thread alive rather than park, the least-bad option is
the methods-note path (no training, no new claims requiring compute).

Awaiting Andreas's decision; the executor session gets its paste after that.
