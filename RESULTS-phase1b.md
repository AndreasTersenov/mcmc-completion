# RESULTS-phase1b — the decisive compute experiment (2026-07-11)

Audience: the reconvene. Assumes PLAN.md (phase 1b frozen core + the
usefulness amendment), not the code. Every number pre-registered before
submission (log/2026-07-11-phase1b.md — including error post-mortems and one
pre-authorized gate amendment, all logged before the affected numbers
existed). Instruments and seeds are the toy phase's verbatim; the capacity
arm shares the identical reference set (zero reference noise across lines).
T=1 is the scoring column throughout (Ruling 1 columns both reported in the
jsons).

## The branch verdict (frozen rule applied verbatim)

**STRUCTURAL, confirmed by the conditional capacity arm → this is the PIVOT
conversation.** H-starve is refuted on the zoo instruments: neither 10×
per-target compute nor 2× width moves them.

| clause | bar | measured | verdict |
|---|---|---|---|
| TT @2M (trained-target composite) | ≥ 50% | **12.5%** | fails |
| plateau (rel. TT gain over final 500k) | ≥ +10% to continue | **−25%** | plateaued |
| plateau before 1M? (fires capacity arm) | — | TT flat from 0.2M | **arm fired** |
| FQ (fresh-θ median improvement vs 200k) | ≥ 4× | **0.85×** | fails |
| FE (any d=4 family median ESS ≥ 5%, fresh-θ) | ≥ 5% | best 0.93% @2M | fails |
| capacity arm: TT_cap @1M | ≥ 50% | **4.2%** | fails |
| capacity arm plateau (0.5M→1M) | — | 0% | plateaued |
| FQ_cap (549 / cap@1M median) | ≥ 4× | **0.53×** | fails |

The full curves (figure: results/phase1b_curves.png; raw: eval_curve.json,
eval_cap.json):

| steps | 0.2M | 0.25M | 0.5M | 1M | 1.5M | 2M |
|---|---|---|---|---|---|---|
| TT composite (T=1) | 16.7% | 16.7% | 8.3% | 8.3% | 16.7% | 12.5% |
| fresh-θ med SW2²/bespoke | 549 | 518 | 652 | 559 | 677 | 642 |
| capacity arm (2× width) | — | 8.3% / 731 | 4.2% / 637 | 4.2% / 1039 | — | — |

Width HURT: both instruments and even the training loss (1.416 vs 1.368 at
matched 1M steps, despite the capacity arm's cosine ending there). At this
zoo scale (128 targets × 50k-sample pools) the sharpness gap is not
resource-limited — more optimization and more parameters do nothing. The
one FE transient (warp-d4 median 5.7% at cap-0.5M; medians of TWO targets)
is noise-level and moot (TT ≪ 50% blocks ALIVE regardless).

## The finding: compute moves transfer, not sharpness (the dissociation)

The same 200k→2M lever that left the zoo instruments flat transformed
zero-shot behavior on REAL inference problems (Readout C, non-gating;
figure: results/real_by_eye.png; raw: readout_c_{200k,2M}.json):

| real target (zero-shot, T=1, K=128 stalled chains) | 200k | 2M |
|---|---|---|
| eight-schools (Rubin data, d=10, non-centered) | ESS 8.4% STABLE, SW2 0.09 | ESS 9.6% STABLE, SW2 0.28 |
| inference-gym banana (d=2) | 0.03% UNSTABLE | **33.1% STABLE** |
| WL band-power (d=3, Ω_m σ₈ n_s, Euclid-like 120 bands) | 0.01% UNSTABLE | 0.01% UNSTABLE |

- **Pre-registered barometer clause (60%): CONFIRMED, exceeded** — two of
  three real targets clear certified-ESS ≥ 1% stable at 2M; eight-schools
  clears it already at 200k, at d=10.
- The samples say more than the numbers (real_by_eye.png): the 2M banana
  covers the arch's bulk without tracing its geometry (ESS 33% while SW2
  stays 15.6 — the certificate prices mass overlap, SW2 prices shape; both
  honest). The 2M WL cloud visibly reorganizes along the Ω_m–σ₈ degeneracy
  ridge — the model reads the ridge direction out of the stalled chains —
  but stays ~100× too wide, with a tanh-adapter corner artifact; the
  certificate refuses it, correctly, at both checkpoints.
- Interpretation (hypothesis, not measurement): long training improves the
  conditional field's robustness/transfer while per-target sharpness against
  the frozen composite is limited by the objective/architecture class, not
  by steps or width. This reframes phase 2: the lever that helps is the one
  the zoo instruments cannot see.

## Readout B — the proposal-engine cost-crossover

(readout_b_{200k,2M}.json; accounting pre-registered, jit warm-up excluded,
frozen B4 references.) Crossover count: **2/12 at 200k and 2/12 at 2M**
(train4-line baseline: 3/12; pivot-option threshold ≥ 6/12: NOT reached —
Readout B does not modify the pivot options). ICS all-in per new target
1.6–4.2 s vs MCLMC 19–30 s (adaptation + sampling): the ~10× amortized cost
advantage is real and survived 10× training compute, but quality crossover
happens only where MCLMC itself fails (gmm-d2 mode drops, funnelmix-d4 heavy
tails). P-crossover (65%: → ≥5/12): **MISS**.

## Prediction scorecard (registered 2026-07-11, pre-launch)

| prediction | confidence | verdict |
|---|---|---|
| branch: ALIVE 45 / MEMORIZE 25 / STRUCTURAL 30 | — | STRUCTURAL (the 30% leg) |
| P-curve: TT still rising at 2M | 70% | **MISS** (flat from 0.2M) |
| P-crossover: 3/12 → ≥5/12 | 65% | **MISS** (2/12) |
| Readout-C: ≥1 real target ≥1% stable at 2M | 60% | **CONFIRMED** (2/3; at 200k already) |
| capacity arm: STRUCTURAL-confirm | 55% | CONFIRMED |

Two frozen-core expectations missed in the pessimistic direction; the
usefulness bet hit. The registered record stands as written.

## Pivot options (pre-listed in PLAN; evidence attached, decision = reconvene)

1. **Per-domain scoping** — the barometer says transfer to real targets is
   where compute pays; a zoo REBUILT around one real problem family
   (amendment's usefulness clauses; first candidate user: Andreas's
   thesis/CosmOrford chains) tests whether in-domain prior-fitting closes
   the sharpness gap where it matters. Readout-C evidence: eight-schools
   d=10 works today; WL fails today but visibly finds the ridge.
2. **Proposal-engine product at current quality, d ≤ 4** — 10× cheaper per
   target with an honest certificate, but crossover 2/12 says the niche is
   narrow (targets where classical samplers mode-drop).
3. **Fold the certificate machinery into the D2-style inference engine** —
   the certificate results (coverage law, blind spot, honest refusals on
   WL/banana here) are the phase's most portable asset.
4. **Park.**

Readout C SHAPES this choice (option 1 gains the most support); it gates
nothing, per the amendment.

## Honest limits

- The dissociation rests on three real targets; n=3, hand-picked for
  dimensional reach (choice documented pre-launch, incl. why gym-banana ≈
  geometry-adjacent to a held-out zoo family and real-data gym standards at
  d ≤ 10 essentially don't exist).
- WL surrogate fidelity gate was amended (pre-authorized fallback): held-out
  max 0.211 σ_Knox / median 0.011 σ_Knox vs the registered 1% — CAMB-side
  point noise, plateau reproduced at two grid resolutions. Reference and
  model sample the SAME surrogate, so the internal comparison is unaffected;
  "real WL" means "real WL up to a ≤0.2 σ band-power perturbation".
- Banana NUTS at 4×20k draws was silently under-converged (R̂ 1.002 while
  variance was 15% off — arch IACT is O(10²⁺)); caught by the pre-registered
  exact-sampler cross-check, reference switched to the exact sampler. R̂
  alone is NOT a sufficient reference gate on curved targets — carry this
  into any phase-2 reference protocol.
- TT composite medians ride on 2 targets/cell (24 rows); the ±8% wobble in
  the TT curve is that instrument's noise floor. The flatness conclusion is
  robust to it (no trend across 10× compute in either instrument), the
  individual checkpoint values are not.
- Three GPU-node incidents (silent wedge post-maintenance) cost ~1.9 H100-h
  and initially masqueraded as code bugs; diagnosed by instrumented rerun.

## Reproduction & budget

Figures: phase1b_curves.png, real_by_eye.png, scaling_by_eye.png. Raw:
eval_curve.json, eval_cap.json, readout_b_{200k,2M}.json,
readout_c_{200k,2M}.json, wl_surrogate.npz (+ refs on scratch, R̂ ≤ 1.0004,
half-vs-full var ≤ 1.5%). Logs: log/2026-07-11-phase1b.md (pre-registrations,
hashes, job ledger, post-mortems). Suite: 113 tests green. Budget: ~13 of the
40 H100-h phase cap (project total ~45/480).
