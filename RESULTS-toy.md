# RESULTS-toy — phase 1, the in-context sampler (2026-07-10)

Audience: the reconvene. Assumes PLAN.md, not the code. Every number passed
the repo's gates; every run was pre-registered before submission; regimes
are separated throughout: **trained-target** (the model on targets it
trained on, fresh contexts) vs **fresh-θ** (never-seen targets — P1's
regime) vs **held-out-family** (never-seen families — P2/P3/P11's regime).
Context budgets are always (K, T) pairs; the T=1 column is retained
everywhere per Ruling 1.

## The headline

**Prior-fitting scales; the frozen absolute bars are not met at toy
compute.** Both statements are measured, and the first is the phase's
positive result: on a common set of fresh targets with shared references and
contexts (the *paired instrument*), median sample quality improves
monotonically as the training zoo grows 10 → 128 → 1024 targets at fixed
compute-per-pair — SW2²/bespoke medians 905 → 561 → 345 (T=1 column;
sharper on T=5). P-scale (registered 70%): **PASS** (1.62× at the 1024
step, better on 17/24 targets). The absolute levels remain far from the
frozen P1/P3 bars — a legitimate, pre-registration-consistent outcome whose
pre-registered follow-up (longer/larger arms) is staged.

## Frozen-prediction scorecard

| Prediction (registered confidence) | Verdict | Numbers |
|---|---|---|
| **P1** (75%): in-family fresh-θ, K=128+grads, d≤8: ESS ≥ 1% on ≥80% + SW2 within 2× bespoke | **NOT MET** (both clauses, final) | ESS clause: 39.6% of 144 (bar 80%). SW2 clause: 0/8 subset targets within 2× of even the under-trained B2 (a-fortiori vs full-strength; deviation documented) |
| **P2** (70%): zero-shot cross-family largely fails at mode level (≥50%) on the 2-family arm | **CONFIRMED at the boundary** | 55.4% (T=5 scoring column) / 49.4% (T=1) — column-dependence stated plainly |
| **P3** (65%): certificate flags ≥90% of bad cross-family targets; false-blessing <10% excl. mode-drop | **FAILED — new blind spot found** | 69% flagged; 24% non-mode-drop false blessings (robust to the "bad" cut): smoothly overspread mass keeps ESS ≥1% while SW2 is bad |
| **P11** (60%): gradient tokens help cross-family more than in-family | **CONFIRMED** (ordering) | grad/nograd median-ESS ratio: in-family 0.97×/1.04×, held-out 1.37×/1.18× (T1/T5) |
| **P12** (55%): held-out performance improves with 2→4 train families at fixed compute | **CONFIRMED** (composite + both SW2 columns; one flat ESS cell) | held-out ESS-clause 31.1% vs 27.0% (t5); median SW2 7.9 vs 11.4 (t1), 10.4 vs 16.1 (t5); paired mechanism: 1.86× on cross-cells |
| **P-scale** (70%, registered mid-phase): paired medians improve 128→1024 | **PASS** | 1.62× (T=1), 1.48× (T=5), 17/24 |
| **P7** (70%): MCLMC wins single targets; ICS wins many-target regime | **half-confirmed** | MCLMC wins 9/12 at matched sampling wall-clock (first half ✓); ICS is ~5× cheaper per new target (4.3 s all-in w/ certificate vs 18.7 s adaptation alone) but crossover count (cheaper AND better) = 3/12 — many-target claim not demonstrated at toy quality |
| P-sharp (70%, superseded) | registered MISS, asterisked | population/θ artifact; reversed by the paired instrument (reconvene-accepted) |

**K-T1 (kill): does NOT fire.** The ESS-clause shortfall is a level, not a
mechanism failure: the pathway overfits pairs at 80–98% ESS (gates i–ii),
the paired scaling slope is positive and unexhausted, and ~30/480 H100-hours
are spent. "Honest effort within the phase budget" is far from exhausted;
the staged follow-up is the pre-registered next step.

## What the model can do (fresh-θ capability map)

Median certified ESS per family × d (T=5 column; full table with both
columns: results/assembly_tables.txt; visual: results/p_dashboard.png,
results/sample_gallery.png):

| | d=2 | d=4 | d=8 | d=16 |
|---|---|---|---|---|
| gmm | 22% | 0.6% | 0.14% | 0.05% |
| dwell | 1.8% | 0.2% | 0.02% | 0.01% |
| funnel | 4.8% | 0.4% | 0.09% | 0.03% |
| warp | **85%** | 14% | 0.3% | 0.02% |
| banana (held-out) | **12%** | 0.05% | numeric† | numeric† |
| funnelmix (held-out) | **9%** | 0.2% | 0.12% | 0.03% |

Real zero-shot cross-family sampling exists at d=2 (banana 12%, funnelmix
9% — never trained on), and the **d-cliff** — roughly an order of magnitude
of ESS per dimension doubling in every family — is the single constraint
that most shapes the next phase. († banana d≥8: target-side numeric overflow
in the shear warp; 23 rows excluded by the audit; zoo-design fix required.)

## The certificate: what it catches, what it misses

- **It prices coverage exactly** — the phase's cleanest law, end-to-end from
  stage-0 synthetic (logẐ = ln ½ at −0.6921) through the trained system
  (logẐ = −0.770 = ln covered mass) to the tempered-context fix (−0.045):
  results/gate3_story.png panel A. Likely paper figure.
- **It refuses correctly on unidentifiable contexts** (funnel-σᵥ at K=128:
  refusal_ok on all tested rows — designed P3 behavior confirmed).
- **New blind spot (the P3 failure, a real discovery):** cross-family, mass
  that is smoothly overspread — covering the target but far too wide — keeps
  ESS moderate (≥1%, stable) while SW2 is bad; 24% of bad held-out targets
  are false-blessed this way (mode-drop, the known blind spot, is only 5 of
  23 cases). Visible by eye: results/sample_gallery.png, funnelmix panel.
  SNIS prices missing and misplaced mass, but is weak against diffuse
  covering error. Candidate mitigations (next phase, pre-register):
  a dispersion/moment check alongside ESS, or SW2-proxy diagnostics from
  weighted samples.

## Mechanisms found and fixed along the way (the gate-ladder dividend)

1. shortK augmentation was the mid-d sharpness killer (removed).
2. Three information-ceiling reclassifications — funnel@K=128 (σᵥ
   unidentifiable: bar moved to K=512), d=2 mode coverage (T=1 chains never
   hop; tempered contexts adopted, Ruling 1), and P-sharp's population
   artifact (paired instrument adopted) — "evaluation must respect what
   context can identify" is now a named method contribution.
3. The √T whitening squeeze: tempered contexts inflate σ_ctx, compressing
   whitened targets to the blur scale (closest fresh-θ mode pair: 0.18
   whitened units) and stretching errors 6.5× on de-whitening
   (results/whitening_diag.png). T-corrected whitening (σ/√T) is the staged
   candidate arm.
4. Sanity floors: absolute floors imported across compute regimes are
   invalid; recalibrated as a relative instrument (no broken-arm signal:
   arm spread <7×). Registration lesson: anchor floors to measured
   reference quantiles.

## Baselines (all four, as frozen — no cherry-picking)

12-target pre-registered subset (each family × d∈{2,4}); medians / notes:

| Baseline | SW2² (median) | vs ICS (t5) | Notes |
|---|---|---|---|
| B1 untrained head | 25–41 (full set) | ~6× worse | floor well-separated; P1-clause 0.0% |
| B2 bespoke per-target FM | **0.006** | ~300× better | ESS 99%+; run at ~70 s/target — a LOWER bound on the frozen 10-min spec (deviation documented; verdicts one-sided-safe) |
| B3 energy-fit + MALA | 19 | worse on 9/12 | catastrophic on heavy tails (funnelmix-d4 SW2 1.9·10⁵); the citable ablation is not competitive |
| B4 MCLMC (matched sampling wall-clock) | 1.7 | better on 9/12 | but: 18.7 s median adaptation per NEW target, no certificate, mode-drops (gmm-d2 recovery 0.43) |

The gap to B2 (~300× in SW2) is the honest size of the conditional-sharpness
problem at toy scale; the paired instrument says it shrinks with zoo size at
measured rate ~1.6× per 8× targets.

## Honest limits

- Fresh-θ competence is d≤4 today; the frozen d≤8 bar is not met.
- banana d≥8 is numerically unusable (zoo-design flaw, not model).
- The T=5 scoring column is harder for well-covered targets (whitening
  squeeze) — column choice materially moves medians; always report both.
- Trained-target composites are weak at 260 steps/pair across ALL
  generations — per-pair compute, not zoo size, appears to set that level
  (the staged 2M chain tests this directly).
- SW2 heavy-tail overflow rows and funnel-SW2 instability inherit stage-0's
  known estimator limits.

## Reproduction & artifacts

Figures: gate3_story.png (coverage law), scaling_result.png (amortization
curve + diversity split), p_dashboard.png (capability map, P3 confusion,
scorecard), sample_gallery.png (truth-vs-model densities), explainer.png,
whitening_diag.png. Tables: assembly_tables.txt. Raw: results/*.json. Logs:
log/2026-07-10-*.md (pre-registrations, rulings, job IDs, config hashes).
Suite: 107 tests green. Budget: ~30/480 H100-hours.
