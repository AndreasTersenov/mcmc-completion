# PLAN — Toy phase: the in-context sampler, first working version

**Phase 1 of the flagship.** Stage-0 passed (see `PLAN-stage0.md` + `RESULTS.md`,
2026-07-09): the certificate envelope is the exact identity ESS/N = exp(−D₂(p‖q)), the
oracle ceiling doesn't bind in-family, gradients are worth ≥4× in context. This phase
tests the core hypothesis stage-0 could not: **can a single trained network actually do
in-context sampling of unseen targets?** Everything here inherits the stage-0-derived
capability targets (RESULTS §end). Wider context:
`~/claude-notes/brainstorms/2026-07-09-dl-project-directions.md` (§D1, v3).

**Stack (binding, per CLAUDE.md):** JAX — blackjax (chains + MCLMC), distrax (zoo
families), flax or equinox + optax, Andreas's `~/software/jax_flows` as the
flow-matching core. GPU work via SLURM only (login-node rule).

---

## FROZEN CORE — do not modify (pre-registered 2026-07-09, evening)

### The system under test

One network: a permutation-invariant context encoder over tokens
(x, E_centered, ∇E) from short MALA chains (stage-0 protocol: 4 chains × K/4 steps,
overdispersed inits; **context-centered energies** — offset invariance is mandatory),
conditioning a flow-matching sampler head. Trained across a zoo of targets with exact
samples; zero per-target adaptation at test time. Output protocol: samples + SNIS
certificate **including the N-doubling check** (stage-0: point-ESS is not trustworthy;
instability is the malignancy signal).

### Zoo v1 (train) and held-out families (test)

- **Train families (4):** GMMs (k ∈ 1..8, random means/covs/weights); coupled
  double-wells; funnels (σᵥ ∈ [0.5, 3]); randomly-warped Gaussians (smooth invertible
  pushforwards — exactly sampleable, exact log-density via change of variables).
  d ∈ {2, 4, 8, 16}. Every family exactly sampleable with known log Z ≡ 0
  (normalized by construction) — no ground-truth bottleneck.
- **Held-out θ** within train families (in-family generalization) AND **held-out
  families (2)** never seen in training: banana/Rosenbrock-warped Gaussians;
  mixtures-of-funnels. Cross-family = the zero-shot test.
- Zoo-diversity arms for P12: identical compute, train on {2, 4} families (8 optional
  if time allows).

### Evaluation (frozen)

Metrics: sliced-W2 vs exact samples; D̂₂ = −ln(ESS/N) via the stage-0 identity; |logẐ|
error; certificate protocol = ESS at N and 2N + stability flag. Mode-recovery rate
measured explicitly against known zoo mode structure (the certificate's declared blind
spot — reported, never hidden).

Baselines (all four, no cherry-picking):
1. Prior/untrained head (floor).
2. Per-target FM sampler, trained 10 H100-minutes per target (the "bespoke" reference).
3. Samples→energy→Langevin pipeline in the spirit of arXiv:2406.12785 (inverse
   direction) — the citable ablation.
4. **MCLMC (blackjax) per-target at matched per-target wall-clock** — the P7 frame:
   report amortized marginal cost per new target, not single-target raw speed.

### Pre-registered predictions

- **P1 (75%):** in-family, unseen θ, K=128 with gradients: D̂₂ ≤ 4.6 (ESS/N ≥ 1%) for
  ≥80% of test targets at d ≤ 8; sliced-W2 within 2× of baseline-2.
- **P2 (70%):** zero-shot cross-family largely fails at the mode level (ESS/N < 0.1%
  or doubling-unstable on ≥50% of held-out-family targets) at the 2-family arm.
- **P3 (65%):** the certificate correctly refuses: among cross-family targets with bad
  sliced-W2, ≥90% are flagged by (ESS below threshold OR doubling instability);
  false-blessing rate <10% **excluding** mode-drop cases (reported separately as the
  known blind spot).
- **P11 (60%):** gradient tokens help cross-family transfer more than in-family
  (relative ESS gain larger on held-out families).
- **P12 (55%):** held-out-family performance improves monotonically 2→4 train families
  at fixed compute — the scaling-curve seed.
- **P7 (70%, carried):** MCLMC wins any single target at matched per-target budget;
  the ICS wins the many-targets regime (crossover count reported).

### Kill / gate criteria

- **K-T1 (kill):** if P1 fails after honest effort within the phase budget (below) —
  in-context conditioning doesn't work even in-family — STOP, document, reconvene
  (pivot options: D4-focus, or per-domain scoping à la TBG).
- **K-T2 (scope, not kill):** if P12 is flat (zoo diversity buys nothing cross-family),
  the "foundation sampler" story shrinks to per-domain amortization — reconvene on
  framing before any further scaling.
- **Budget cap: 20 H100-days total for this phase.** The Karpathy gate is mandatory
  and *ordered*: (i) overfit ONE target with the FM head (no context) → (ii) overfit
  one (context, target) pair → (iii) 10-target mini-zoo → (iv) full training. No
  multi-hour job before the previous gate is green and committed.

### Out of scope (do NOT build)

d > 16; lattice/discrete targets; adaptive probing policies; log-Z machinery beyond
SNIS; NumPyro/real-application showcases (phase 2); any per-target fine-tuning of the
ICS at test time; architecture innovation beyond what the interfaces above require.

---

## FREE PERIPHERY — implementer's choice

Architecture details (encoder depth/width, FM parameterization, embedding of d as a
variable), optimizer/schedule, curriculum, batching across targets, chain
implementation details (blackjax), tokenization specifics (subject to the frozen
centering rule), how zoo generation is parallelized, checkpoint cadence. Reuse
`jax_flows` rather than rewriting FM; reuse stage-0 `stage0/` estimator code for all
certificate math (it is gate-tested — do not reimplement SNIS/ESS).

## Backpressure additions (on top of existing gates, which stay)

New tests required before training: (a) each zoo family's exact sampler vs its
log-density (KS/analytic-moment test); (b) warped-Gaussian log-det-Jacobian check;
(c) FM head single-target overfit gate as an executable test (loss below threshold in
fixed steps, deterministic seed); (d) certificate module reused from stage-0 passes
its existing suite untouched. Training runs: deterministic seeding; every submitted
job logged in `log/` with job ID, config hash, and expected outcome BEFORE submission.

## Logging & deliverable

Same log discipline. Deliverable: `RESULTS-toy.md` — P1/P2/P3/P11/P12/P7 verdicts with
numbers, the zoo-diversity curve, the certificate confusion matrix (flagged vs actually
bad, mode-drop column separate), baseline table, and honest limits. Written for the
reconvene; assumes PLAN.md, not the code.
