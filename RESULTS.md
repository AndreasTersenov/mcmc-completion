# RESULTS — stage-0 (2026-07-09)

Audience: the reconvene session. Assumes PLAN.md; does not assume the code.
Everything quantitative below passed the repo's validation gates (closed-form
checks on every estimator; N-doubling stability on every reported number —
where a number failed its stability check, that failure is reported instead
of the number). Full run-by-run narrative: `log/2026-07-09-{gates,m1-envelope,m2-oracle}.md`.

## Verdict summary

| Item | Pre-registered | Outcome |
|---|---|---|
| **K-M1** (kill) | fires if no achievable mismatch gives ESS/N > 0.1% at d=8 | **Does not fire.** The 0.1% frontier at d=8 sits at per-dim shift ε\*=0.93 (per-dim KL ≈ 0.43) — far looser than per-target-trained neural samplers achieve (FAB/CRAFT: ESS ≈ 88% at d=32, ≈ 93% at d=60 ⇒ D₂ ≈ 0.1–0.5; **now cited — see Caveats §K-M1 citation correction, DRAFT**). |
| **P8a** (85%) | ESS decays ~exp in d at fixed per-dim mismatch; d=16 needs mismatch shrinking with d | **Confirmed, and exact, not approximate**: ESS/N = exp(−D₂(p‖q)) (Rényi-2) on every in-family row measured; iso-ESS frontier ε\*(d) = √(ln(1/target)/d) ∝ 1/√d. |
| **P8b** (75%) | in-family θ collapses at K = O(θ-dim) with exact (E,∇E); mismatch → confident wrong θ; true-energy SNIS catches it | **Confirmed both halves**, with two refinements: identifiability can be parameter-anisotropic in-family (funnel σᵥ), and "catches it" operationally requires the N-doubling check, not a single ESS reading (see M2 §mismatch). |
| **P-grad** (60%) | gradients cut K needed for contraction ≥2× in d ≥ 8 | **Confirmed at d=8, θ-dim 8**: K ratio ≥ 4 (gradients: contraction already at K=8; energy-only: K ∈ (32,128]). At fixed K=8: 13× tighter posterior, 200× smaller θ-error. |
| **K-M2** | none (cannot kill) | — |

**Bottom line: the project survives stage-0.** Neither ceiling binds at the
pre-registered thresholds. The binding constraints that DID emerge are
structural, not metric — they become the toy-phase capability targets (§end).

---

## M1 — the certificate envelope

Setup: SNIS of proposal-q samples against the true (normalized, so log Z = 0)
target energy; controlled mismatch; d ∈ {2,…,64}; N = 10⁶ per grid point;
222 grid points. Figures: `results/m1_frontier.png` (ESS surface + iso-ESS
capability targets + variance asymmetry), `results/m1_universality.png`
(exp(−D₂) collapse + logẐ error vs ESS). Raw grid: `results/m1_grid.csv`.

**1. The frontier is one number: D₂(p‖q).** Measured ESS/N equals exp(−D₂)
exactly across Gaussian shift, Gaussian variance scaling, GMM mode-weight
distortion, and funnel conditional/v-scale mismatch (169/222 points reliable
+ stable; empirical/closed-form ∈ [0.946, 1.023]). A usable certificate at
target t requires **D₂ < ln(1/t)** — e.g. D₂ < 6.9 for 0.1%, < 4.6 for 1%.

**2. Shift frontier (per-dim, σ units):** ε\*(d) = √(ln(1/t)/d):

| ESS/N target | d=2 | d=8 | d=16 | d=32 | d=64 |
|---|---|---|---|---|---|
| 10% | 1.07 | 0.54 | 0.38 | 0.27 | 0.19 |
| 1% | 1.52 | 0.76 | 0.54 | 0.38 | 0.27 |
| 0.1% | 1.86 | 0.93 | 0.66 | 0.47 | 0.33 |

**3. Variance mismatch is strongly asymmetric.** At d=8, 1% target: proposal
may be up to 5.8× overdispersed but only ~2× underdispersed; E[w²] diverges
at variance ratio ½ regardless of d. **Capability rule: err wide.**

**4. Mode weights are cheap; mode support is fatal.** Distorting a 50/50 GMM
to 95/5 still leaves ESS/N = 19% (d-independent; matches 1/Σp_j²/q_j).
Dropping a mode entirely (separation Δ=8): exact analysis gives E[w] = 1 (Ẑ
technically unbiased) but true ESS/N = 4/(3+e^{Δ²}) ≈ 6·10⁻²⁸, while at any
practical N: **logẐ locks onto −ln 2 (the log covered mass) with sd ~10⁻³ —
stably, confidently wrong** — and the measured ESS is a seed lottery
(observed {0.87, 0.05, 0.68, 6·10⁻⁶, 0.88, 3·10⁻⁴} across the six d rows at
N=10⁶). N-doubling caught 4/6 rows; logẐ looked healthy in all. This is the
pre-registered blind spot, quantified: **the certificate prices what q
covers, and is silent about what q misses.**

**5. No Gaussian proposal can certify a funnel, at any d.** The funnel's
conditional variance e^v exceeds any fixed Gaussian tail with positive
probability ⇒ E[w²] = ∞. Empirically ESS/N reads 10⁻⁵–10⁻⁴ at d ≥ 4 and keeps
falling with N (doubling gate fails). Within-family funnel mismatch is
perfectly certifiable (row 1 closed forms). **Tail/conditional structure
dominates distance.** Caveat: at d=2 the same divergent-weight construction
read a stable 4.9% and *passed* doubling — finite checks cannot fully protect
against infinite-variance weights.

**6. Self-diagnosis is quartically harder than certification** (not
pre-registered). The ESS estimator's own relative sd is
√((E w⁴/(E w²)² − 1)/N); for shift-type mismatch that is √((e^{4D₂}−1)/N).
The certificate works when e^{D₂} ≪ N, but *measuring that it works* needs
e^{4D₂} ≪ N — at N=10⁶ the measured ESS is trustworthy only where true ESS/N
≳ 4.5% (shift geometry). Observed: d=64, ε=0.5 reads ESS 1.2·10⁻⁴ when truth
is 1.1·10⁻⁷. In the near-vacuous regime the diagnostic itself over-reports.
All in-family stability-gate failures lie exactly in the analytically-flagged
unreliable region.

**logẐ error tracks ESS as theory says** (delta-method sd √((1/essN−1)/N)),
with bias switching on below ESS/N ≈ 1% (at N=10⁴: bias −0.07 at cf-ESS
3·10⁻⁴; −0.39 at 10⁻⁷).

**K-M1 status: does not fire.** At d=8 even the 0.1% frontier allows per-dim
shift 0.93σ / variance ratio in [0.52, 9.7]. Judged against per-target-trained
neural samplers (FAB/CRAFT reach D₂ ≈ 0.1–0.5 at d = 32–60, i.e. ESS 88–93% —
now cited, see Caveats §K-M1 citation correction, DRAFT), the frontier is loose
by a factor ~14–90 in the D₂ budget. The honest counterweight: the frontier is
loose in *distance* but strict in *structure* (points 3–5).

---

## M2 — the in-context oracle

Setup: zoo v0 of four families with explicit uniform priors — gmm2 (d=2,
θ=(separation, weight)), funnel2 (d=2, θ=(σᵥ, conditional scale)), dwell2
(d=2, θ=(barrier a, well position b)), gmm8 (d=8, θ = one mode's mean ∈ R⁸).
Contexts: 4 short MALA chains × K/4 steps from overdispersed inits,
K ∈ {8,32,128,512}, T ∈ {1,2,5}; exact untempered (E, ∇E) recorded at visited
points. Posterior over θ: dense grid (θ-dim 2) / adaptive SMC (θ-dim 8) under
a Gaussian pseudo-likelihood on **context-centered energies** (energies are
defined only up to an additive constant across targets — the model must not
exploit offsets) with **relative readout precision** σ = 5% of the context's
energy/gradient spread. 6 replicates per cell (fresh θ\* ~ prior + fresh
chains). Figure: `results/m2_oracle.png`; raw:
`results/m2_{contraction,sw2,mismatch}.csv`.

**Observation-model deviation (logged mid-run):** the first sweep used
absolute σ = 0.05 energy units. For the funnel that is broken — neck energies
are O(10²–10³), the likelihood became a spike thinner than a grid cell, and
two grid resolutions "collapsed" onto *different* wrong θ (the smoking gun).
Numbers were discarded before reporting; the sweep re-ran with relative
precision. Lesson worth keeping: **what context can identify depends on the
assumed readout precision model**; absolute precision is indefensible for
heavy-scale energy families.

**1. In-family identification is fast — the oracle ceiling does not bind.**
Contraction (posterior sd / prior sd, median of 6): gmm2 reaches 0.07–0.10 at
K=8; dwell2 0.055 at K=8 with gradients; gmm8 0.035 at K=8 with gradients.
Posterior-predictive sliced-W2² against p\* sits at the same-p\* sampling
floor from K=8 for gmm2/dwell2 — once the posterior localizes, generation is
solved (in these exact-sampler families).

**2. P-grad — confirmed at d=8** (numbers in the verdict table). At θ-dim 2
gradients still help where the energy has a parameter ridge (dwell2: 4.3×
tighter at K=8; energy-only needs K > 128 to cross the 0.1 threshold).

**3. Identifiability is parameter-anisotropic (new).** funnel2's conditional
scale c is pinned (±0.005) at K=8 while σᵥ remains essentially prior-wide
until K=512, in BOTH arms — the v-marginal information sits below relative
readout precision when neck energies dominate the context's dynamic range.
Crucially the posterior is honestly *wide*, not confidently wrong; a
θ-posterior contraction diagnostic localizes which parameter directions the
context has and hasn't bought.

**4. Temperature: no resolvable effect** on in-family identification at R=6.
With exact value readout, where the chain probes matters much less than what
is read there.

**5. Mismatch: zoo posteriors lie confidently; the certificate, used
correctly, does not.** With context from family A and zoo B (K=128): 11/12
runs contract to ≤5% of prior sd around a wrong θ — typically a prior-box
corner (gmm2 zoo explains a funnel as the "single blob" corner sep=2/w=0.8;
funnel2 zoo explains a bimodal GMM as the maximal-spread corner σᵥ=4/c=2).
No self-diagnosis, exactly as predicted. The true-energy SNIS certificate:

| pair | ESS/N (n=16k) | stable under N-doubling? | logẐ error |
|---|---|---|---|
| funnel2→gmm2 (severe reps) | 4–6% | **no** (drops ~3× on doubling) | −0.42…−0.47 |
| funnel2→gmm2 (mild reps) | 11–13% | mixed | −0.12…−0.01 |
| dwell2→gmm2 (all 4) | 15–26% | yes | ≤ 0.02 |
| gmm2→funnel2 | 0.05–14% | **no** (3/4) | up to 0.40 |
| in-family controls | 72–100% | yes | ≤ 0.002 |

Reading: **malignant mismatch is caught, but by *instability*, not by a
single glance** — one-shot ESS can read a comforting 5–14% while collapsing
under N-doubling (the finite-N face of E[w²] = ∞, same mechanism as M1 §5).
Benign mismatch (a wrong family that still covers p\*: GMM covering a
double-well) passes with honestly small logẐ error — which is the *correct*
behavior for a coverage-priced certificate. The M1 mode-drop caveat still
stands: a q that cleanly misses mass is invisible to any of this.

---

## Derived capability targets for the toy-phase network

1. **Divergence budget:** produce samples with D₂(p‖q_model) ≲ 4.6 (1% ESS)
   at the working d; per-dim budget shrinks as 1/d (table above). This is the
   *easy* axis — trained samplers clear it by orders of magnitude in-family.
2. **Err wide:** bias the sampler/zoo toward overdispersion; underdispersion
   hits the E[w²] wall at variance ratio ½. Never let predicted conditionals
   get lighter-tailed than the target class allows (Gaussian-vs-funnel: ∞).
3. **Mode coverage is the hard constraint** — the certificate cannot see
   dropped modes (logẐ stably wrong by the log covered mass). Whatever
   guarantees coverage (overdispersed context chains, zoo priors over mode
   counts, explicit mode-search probes) must live OUTSIDE the certificate.
4. **Ship the N-doubling check as part of the certificate protocol.** A
   point-ESS is not trustworthy: in the vacuous regime it over-reports
   (needs e^{4D₂} ≪ N to be reliable — quartically harder than the
   certificate itself). Instability under sample doubling was the reliable
   malignancy signal in every family tested.
5. **Gradient tokens are worth ≥4× in context length at d=8** (and break
   parameter ridges at low d). Include (E, ∇E); the open gradient-value
   question resolves in favor of gradients.
6. **Expect anisotropic identification**: some θ directions (funnel σᵥ) cost
   64× more context than others under relative readout precision. A
   θ-posterior contraction probe on the zoo tells you which; the model's
   uncertainty head should be able to represent "this direction is still
   prior".

## Caveats & honest limits

- The K-M1 judge criterion originally referenced literature ESS values quoted
  from memory. **Now pinned to primary sources — see §K-M1 citation correction
  below (appended 2026-07-10, DRAFT).** The verdict is unchanged (K-M1 does not
  fire — if anything reinforced); the phrasing is narrowed.
- Finite stability checks can pass infinite-variance cases (funnel-vs-Gaussian
  at d=2 read a stable 4.9%). Structural tail-domination arguments must
  supplement empirical gates.
- Funnel-family sliced-W2 numbers failed their stability gate (heavy tails)
  and are reported qualitatively only.
- σ_readout = 5% (relative) is an assumption standing in for "transformer
  reading tokens"; contraction *floors* scale with it. The K-scalings and
  with/without-gradient comparisons — the pre-registered targets — are robust
  to it; absolute floor values are not.
- P-grad's K-ratio is a lower bound (≥4×): K < 8 was not in the frozen grid.
- d > 64, discrete targets, adaptive probing: out of scope by design.

## K-M1 citation correction (appended 2026-07-10 — DRAFT, verify camera-ready at reconvene)

The K-M1 verdict compared the certificate's distance-frontier to per-target
neural-sampler performance using ESS values quoted **from memory**. Those are now
pinned to primary sources. **The verdict is unchanged — K-M1 does not fire, if
anything reinforced; only the phrasing is narrowed and corrected.**

**Cleanest anchor — FAB** (Midgley et al., ICLR 2023, arXiv:2208.01893). FAB trains
a normalizing flow by minimizing the mass-covering **α-divergence with α = 2 — i.e.
importance-weight variance**, exactly the quantity our envelope identifies as D₂
(Rényi-2). Its reported ESS is the reweighted self-normalized-IS ESS of flow samples
against the true target — **the same object as our ESS/N = exp(−D₂(p‖q))**. So FAB is
directly commensurable with our certificate, unlike most of the lineage.

| Sampler | Target | d | Reported ESS | ⇒ D₂ = −ln(ESS/N) | Source |
|---|---|---|---|---|---|
| FAB (w/ buffer) | 40-mode GMM | 2 | 61.9 ± 8.0% | ≈ 0.48 | 2208.01893, Table 1 |
| FAB (w/ buffer) | Many-Well (16× double-well) | 32 | 87.8 ± 1.9% | ≈ 0.13 | 2208.01893, Table 5 |
| CRAFT (as reported in FAB) | Many-Well | 32 | 88.9 ± 0.3% | ≈ 0.12 | 2208.01893, Table 5 |
| FAB (w/ buffer) | alanine dipeptide | 60 | 92.8 ± 0.1% | ≈ 0.075 | 2208.01893, Table 2 |
| FAB (w/o buffer) | alanine dipeptide | 60 | 52.2 ± 1.3% | ≈ 0.65 | 2208.01893, Table 2 |

**Reading:** at d = 32–60 the best per-target flows sit at **D₂ ≈ 0.08–0.5**, a factor
**~14–90 inside** the D₂ < 6.9 budget the 0.1% certificate frontier permits (< 4.6 for
1%). The frontier is loose *in distance*, now with citations. The
structural-strictness counterweight (§M1 points 3–5) is untouched.

**Three corrections to the original from-memory phrasing:**

1. **The ESS band is FAB/CRAFT-specific, not "neural samplers" broadly.** The flagship
   dimension-spanning sampler **iDEM** (Akhound-Sadegh et al., ICML 2024,
   arXiv:2402.06121; DW-4 d=8, LJ-13 d=39, LJ-55 d=165) **reports NO ESS** — only NLL,
   total variation, and 2-Wasserstein (e.g. DW-4 𝒲₂ = 2.15, LJ-13 𝒲₂ = 4.35). Path
   Integral Sampler (2111.15141) and Denoising Diffusion Sampler (2302.13834) report
   **log-Ẑ, not ESS**. The literature does not uniformly report ESS — that metric
   heterogeneity is itself the Cornell position paper's point (evaluation without
   ground truth is unsettled).
2. **The targets are multimodal / particle systems, not "smooth."** Many-Well is a
   product of 16 double-wells; LJ-n are Lennard-Jones clusters. These are *harder* than
   the smooth d≈8 targets the original note implied — so "the frontier is loose" is
   strengthened, not weakened.
3. **Commensurability caveat.** Only FAB's ESS maps cleanly onto ESS/N = exp(−D₂)
   (shared α=2 objective + reweighted-IS definition). The D₂ column is *our* translation
   via the stage-0 identity; the papers report ESS, and iDEM/PIS/DDS do not report the
   comparable ESS at all. Do not read the D₂ column as quoted from the papers.

**Verify-at-reconvene (this note is DRAFT):** the FAB GMM-d2 = 61.9% and all table-cell
values were harvested from ar5iv HTML renders, not the camera-ready PDF (OpenReview
XCTVFJwS9LJ); the FAB predecessor "Bootstrap Your Flow" (2111.11510) reported 70.1% for
the GMM, so eyeball that one. CRAFT d=32 is FAB's *reproduction* of CRAFT, not the
original CRAFT paper. NB: FAB's canonical ID is **2208.01893** (ICLR 2023), *not*
2111.11510 (the 2021 predecessor). Companion deep-reads of the two conditioning-method
anchor papers (arXiv:2406.12785 in-context energy functions → baseline-3; arXiv:2205.09735
foundation posteriors → K1/K2 prior art) drafted same day under
`~/claude-notes/papers/deep-reads/` (also DRAFT).

```bibtex
@inproceedings{midgley2023fab,
  title={Flow Annealed Importance Sampling Bootstrap},
  author={Midgley, Laurence Illing and Stimper, Vincent and Simm, Gregor N. C. and Sch{\"o}lkopf, Bernhard and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}, eprint={2208.01893}, archivePrefix={arXiv}, primaryClass={stat.ML}
}
@inproceedings{akhoundsadegh2024idem,
  title={Iterated Denoising Energy Matching for Sampling from Boltzmann Densities},
  author={Akhound-Sadegh, Tara and Rector-Brooks, Jarrid and Bose, Avishek Joey and Mittal, Sarthak and Lemos, Pablo and Liu, Cheng-Hao and Sendera, Marcin and Ravanbakhsh, Siamak and Gidel, Gauthier and Bengio, Yoshua and Malkin, Nikolay and Tong, Alexander},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}, eprint={2402.06121}, archivePrefix={arXiv}, primaryClass={cs.LG}
}
```

## Reproduction

Deterministically seeded throughout. `~/wl-challenge-env` (numpy 2.4.2) +
pytest. Gates: `python -m pytest tests/` (38 tests, ~4 s). Measurements:
`python scripts/run_m1.py` (~4 min, 12 CPU workers),
`python scripts/run_m2.py` (~7 min), figures via
`scripts/make_plots_m{1,2}.py`. No GPU, no SLURM, nothing outside the repo
except the venv.
