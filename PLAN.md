# PLAN — Phase 1b: the decisive compute experiment (+ proposal-engine readout)

> **PROJECT STATUS: PARKED (2026-07-11, Andreas's decision at reconvene).**
> Phase 1b landed STRUCTURAL; the capacity arm landed STRUCTURAL again; the pivot
> conversation concluded: park the method, migrate the certificate machinery
> (already live in eft-sbi), optional methods note pending the novelty check
> (SPEC-novelty-identity.md). Decision record, salvage inventory, and pre-named
> revival conditions: log/2026-07-11-reconvene-park.md. No training runs are
> authorized. This plan is retained as the record of the decisive experiment.

**Status of the project entering this phase** (context for a compacted/fresh session —
re-anchor from disk, not memory: RESULTS-toy.md, log/2026-07-10-*.md, JOBS.md):
the toy phase closed with the scaling direction confirmed (P-scale: paired medians
905→561→345 across 10→128→1024 targets) and the absolute bars unmet (P1 NOT MET; ~300×
SW2 gap to bespoke). The zoo axis alone cannot close the gap (measured ~1.6× per 8×
targets). Two hypotheses remain for the gap, and THIS RUN decides between them:

- **H-starve:** per-target compute is the binding constraint (arms got ~260 steps/pair,
  ~100× less optimization than the bespoke baseline; trained-target quality was weak
  across ALL generations — the starvation signature; gates i–ii prove the architecture
  can represent sharp conditionals at 80–98% ESS).
- **H-structural:** the per-dim sharpness gap is intrinsic to this design at any
  feasible compute.

Additionally, the foundations review (2026-07-11) established the **proposal-engine
frame**: the mathematically strongest product is "learned proposals + exact IS/SMC
wrapper" — correctness classical at any proposal quality, quality = efficiency on a
cost curve. This run carries that as a co-primary readout, NOT as a change to the
system under test.

---

## FROZEN CORE — do not modify (pre-registered 2026-07-11; Andreas + reconvene)

### The run: ONE variable

Training: the gate3e recipe EXACTLY (128-target zoo from the gate3e datagen, T-mix
contexts [1,1,2,2,5,5], attn-2 + aux, no shortk, same seed) with steps **200k → 2M**
(cosine stretched proportionally — the one bundled necessity; mitigated by the curve
readout). Resumable 3h-chain (staged scripts/slurm/train128_2m.sh). Checkpoints saved
and evaluated at **0.25M, 0.5M, 1M, 1.5M, 2M**. No other change of any kind: no
whitening fix, no architecture change, no zoo change, no new families. (The √T
whitening arm and the certificate dispersion-check mitigation are PARKED for phase 2.)

### Readout A — the compute curve (decides H-starve vs H-structural)

At every checkpoint, with the frozen instruments:
- **Paired instrument** on the SAME common 24-target set (same contexts, same bespoke
  references as the toy-phase paired evals). Metrics: paired SW2²/bespoke medians;
  frozen P1-mirror composite fractions; per-dim D̂₂ = −ln(ESS)/d with reliability flags
  (stage-0's quartic law; decisions do not hinge on unreliable cells).
- **Trained-target subsample** (the relative sanity instrument from gate-iv, same
  24-target subsample): composite fraction.
- Both (K,T) columns reported; **T=1 is the primary scoring column for this phase**
  (sidesteps the known √T whitening squeeze without bundling a fix).

### Readout B — the cost-crossover (the proposal-engine frame)

On the 12-target subset with existing B2/B4 references, at 200k (baseline, exists) and
2M: **(i) crossover count** — number of targets where ICS-with-SNIS reaches
MCLMC-matched quality at lower end-to-end wall-clock per NEW target (all-in: forward
sampling + weights + doubling check vs adaptation + sampling); **(ii) certified
effective samples per second per new target** vs MCLMC's including adaptation. Metric
definitions frozen; procedure details are periphery. Baseline value: crossover 3/12.

### Decision rule — asymmetric and branch-complete (ambiguity = negative)

Let TT = trained-target composite fraction at 2M; FQ = paired fresh-θ median
SW2²/bespoke improvement vs the 200k checkpoint; FE = any d=4 family reaching median
certified ESS ≥ 5% (fresh-θ, T=1).

1. **ALIVE** (H-starve confirmed): TT ≥ 50% AND (FQ ≥ 4× OR FE). → Report and STOP.
   Phase-2 scoping is a reconvene decision (joint zoo×compute program + proposal-engine
   product). 1024-zoo@2M becomes authorized but NOT auto-launched.
2. **MEMORIZE** (compute helps training, not transfer): TT ≥ 50% AND neither FQ ≥ 4×
   nor FE. → Pre-assigned single follow-up: 1024-zoo@2M, same protocol, decision
   re-applied on the fresh-θ criteria only. (Tests whether generalization needs zoo AND
   compute jointly.)
3. **STRUCTURAL** (H-starve refuted): TT < 50% at 2M AND the trained-target curve has
   plateaued (<10% relative improvement over the final 500k). → Conditional capacity
   arm fires ONLY IF the plateau occurred before 1M: one run, 2× encoder+head width,
   1M steps, same rule once. If it also lands here → the PIVOT conversation (= the
   K-T1 conversation), with pre-listed options: per-domain scoping; proposal-engine
   product scoped to d ≤ 4 at current quality; fold the certificate machinery into the
   D2-style inference engine; park.
4. **Readout B modifies the pivot options, not the branch:** crossover ≥ 6/12 at 2M
   means the amortized proposal engine is already cost-competitive in a niche — that
   evidence enters any pivot conversation.

### Pre-registered predictions (Claude, 2026-07-11 — for the scorecard)

- Branch distribution: **ALIVE 45% / MEMORIZE 25% / STRUCTURAL 30%.**
- **P-curve (70%):** the trained-target curve is still visibly rising at 2M (no
  plateau by the rule above).
- **P-crossover (65%):** crossover count improves 3/12 → ≥5/12 at 2M.

### Budget & discipline

Phase cap: **40 H100-hours** (2M chain ≈ 6h; evals ≈ 1h; conditional arm ≈ 4h;
memorize-branch 1024@2M ≈ 30h only if that branch fires). Every chain link logged
pre-submission (job ID, config hash, expected outcome). The session REPORTS AND STOPS
at every branch point — phase-2 scoping decisions belong to the reconvene, not to this
run.

### Out of scope

Everything not named above. In particular: no new training tricks, no eval-bar edits,
no zoo edits (the banana-d≥8 numeric fix is parked with a note), no phase-2 work on an
ALIVE result.

---

## FREE PERIPHERY — implementer's choice

Chain bookkeeping details, checkpoint storage layout (scratch, per login rules), eval
batching, the exact wall-clock accounting procedure for Readout B (document it), plot
styles. Reuse the existing eval harness and stage-0 certificate code untouched.

## Deliverable

`RESULTS-phase1b.md`: the five-checkpoint curves (both regimes, both readouts), the
branch verdict with the rule applied verbatim, prediction verdicts, honest limits.
Written for the reconvene; assumes this PLAN, not the code.

---

## AMENDMENT (2026-07-11, pre-launch; Andreas's usefulness requirement) — FROZEN

### Readout C — the usefulness barometer (eval-only; NON-GATING)

At the 200k baseline and 2M checkpoints, evaluate the model ZERO-SHOT on 2–3 REAL
inference problems within dimensional reach, references from long exact chains
(NUTS/MCLMC, R̂-checked): (a) eight-schools (non-centered, d≈10 — deliberately
funnel-geometry); (b) one inference-gym standard with d ≤ 10; (c) a real weak-lensing
band-power posterior (d=3: Ω_m, σ₈, n_s) built from the eft-sbi phase-0.5 machinery
(~/software/eft-sbi — Gaussian band-power likelihood; coordinate with that repo
read-only). Contexts: short stalled chains on the real targets, (x, E, ∇E) via
autodiff, standard protocol. Metrics: certified ESS + doubling flag, SW2 vs reference,
and the Readout-B cost accounting per target. **These numbers do not gate any branch**
(zero-shot out-of-zoo at toy scale is expected to be partial); they are reported
prominently in RESULTS-phase1b as the usefulness barometer and they SHAPE the
phase-2/pivot conversation.

### Usefulness clauses (bind the follow-ups, not this run)

- ALIVE branch phase-2 scoping MUST be product-shaped: zoo rebuilt around real problem
  families (reference-chain ground truth), a blackjax/NumPyro "stalled-chain" adapter,
  and the certified-proposal wrapper as the product surface.
- The project's north star, superseding all bars: **accelerate one real inference
  problem end-to-end for a real user — the first candidate user is Andreas's own
  thesis/CosmOrford chains (dogfooding).** Any phase-2 plan must name its first real
  user problem explicitly.
- Claude pre-registration for Readout C (60%): at 2M, at least one real target (most
  likely the d=3 band-power posterior) yields stable certified ESS ≥ 1% zero-shot.
