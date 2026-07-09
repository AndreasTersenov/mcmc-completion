# PLAN — Stage-0 for the In-Context Sampler ("MCMC completion")

**Project claim (context, not this stage's target):** a single transformer, prior-fitted
on a synthetic zoo of energy landscapes, that takes a short unconverged MCMC trace
(positions, energies, optionally gradients) of a NEW unnormalized density as context and
emits equilibrium samples plus an importance-weight-certified log-Z — no per-target
training. Full background: `~/claude-notes/brainstorms/2026-07-09-dl-project-directions.md`
(§D1, v1.1 predictions, v2 decision record) and
`~/claude-notes/brainstorms/2026-07-09-novelty-sweep-RESULTS.md` (§D1). Novelty verified
2026-07-09; estimated scoop window 6–12 months. This repo's stage-0 must finish before any
model code exists.

**Stage-0 purpose:** measure the two model-free ceilings that bound the whole project.
No transformers, no training. numpy/jax (+ GPU where convenient; Rorqual has plenty).

---

## FROZEN CORE — do not modify (pre-registered 2026-07-09)

### Measurement M1 — the certificate envelope (PRIMARY; can kill the project)

The shipped certificate is self-normalized importance sampling of the model's samples
against the true energy. SNIS ESS decays ~exponentially in (dimension × mismatch). So
there is a computable frontier: how close must the learned sampler be to the target, per
dimension, for the certificate to remain non-vacuous?

Design: for target p* (start: standard Gaussian; then GMM with well-separated modes;
then funnel), construct controlled proposals q with known mismatch: mean shift ε per dim,
covariance scaling (1+ε), mode-weight distortion, one dropped mode. Compute
ESS/N vs (ε, d) for d ∈ {2, 4, 8, 16, 32, 64}, N up to 10^6. Also compute the resulting
log-Z estimator bias/variance at each grid point. Deliverable: the ESS(ε, d) surface +
the iso-ESS contour "capability targets" plot, plus the mode-drop row (does the
certificate SEE a missing mode? — expected: no from samples alone; document this
honestly as a known blind spot inherited from SNIS).

- **P8a (85%):** ESS decays ~exponentially in d at fixed per-dim mismatch; usable
  certification (ESS/N > 1%) at d=16 requires per-dim mismatch decreasing with d.
  The deliverable is the quantitative frontier, not this qualitative shape.

**Kill criterion K-M1:** if no achievable mismatch level (judge: mismatch comparable to
what per-target-trained neural samplers achieve in the literature at that d) yields
ESS/N > 0.1% at d = 8, the "certified equilibrium samples" headline is dead as scoped →
STOP; report; the fallback scopes (low-d only, or aggregate rather than per-batch
certification) get decided at reconvene, not unilaterally.

### Measurement M2 — the in-context oracle (what can context identify?)

Bayes-optimal in-context sampling = the posterior predictive q(x) = ∫ p(x|θ) p(θ|C) dθ
over the zoo prior (PFN theory). With exact (E, ∇E) values in context, in-family
identification of parametric θ is expected to be easy — the informative cases are the
imperfect ones. Design: small zoo v0 (GMMs with θ-dim ≤ 8; funnel family; double-well
family; explicit prior p(θ) per family). Contexts: short MALA/HMC chains from
overdispersed inits, K ∈ {8, 32, 128, 512} points, at temperature T ∈ {1, 2, 5}.
Posterior p(θ|C) via SMC or grid (θ-dim kept small on purpose). Measure:
(a) posterior contraction vs K — WITH and WITHOUT gradient values in context (this
answers the open gradient-value question); (b) sliced-W2(q, p*) vs K;
(c) **family mismatch**: context generated from family A, zoo prior = family B — measure
where the posterior concentrates and how overconfident q becomes; then verify SNIS
against the TRUE energy still catches it (this is the honesty story: the zoo can be
wrong, the certificate uses ground truth).

- **P8b (75%):** in-family, θ-posterior collapses with K = O(θ-dim) probes given exact
  (E, ∇E); under family mismatch the posterior concentrates confidently on wrong θ (no
  self-diagnosis) — and the true-energy SNIS certificate catches it (low ESS).
- **P-grad (60%):** gradients cut the K needed for contraction by ≥2× in d ≥ 8.

**Kill criterion K-M2:** none — M2 cannot kill the project (it shapes zoo/context
design). Only M1 kills.

### Out of scope for stage-0 (do NOT build)

Transformers or any trained model; lattice/discrete targets; d > 64; adaptive probing
policies; MCLMC comparisons; log-Z beyond SNIS; any "shared codebase" abstractions.

---

## FREE PERIPHERY — implementer's choice

Language (numpy vs jax), chain implementation (MALA vs HMC; use a library if preferred —
blackjax is fine), SMC details, plotting, file layout, GPU usage. Single-batch-overfit
discipline is meaningless here (no training) — the analog: validate every estimator on a
case with a closed-form answer before trusting it on a case without one (e.g., ESS
formulas against analytic Gaussian-vs-Gaussian IS; SMC posterior against conjugate
Gaussian case).

## Logging (required)

Every run appends to `log/YYYY-MM-DD-<slug>.md`: hypothesis → setup → expectation →
result → updated belief. Final deliverable: `RESULTS.md` — the two frontier plots, the
P8a/P8b/P-grad verdicts with numbers, kill-criterion status, and (if alive) the derived
quantitative capability targets for the toy-phase network. Written for a reconvene
session that has read PLAN.md but not the code.
