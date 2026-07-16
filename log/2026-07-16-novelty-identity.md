# 2026-07-16 — Novelty kill-test: the ESS↔D₂ identity + the SNIS overspread blind spot

**Session type:** literature kill-test (web-search + direct-fetch only; NO code, NO
training). Executes `SPEC-novelty-identity.md` against the two claims below.
**Bottom line up front: both claims are known.** Claim 1 (the identity) is a one-line
restatement of published equations and must not be claimed as a result. Claim 2 (the
blind spot) is scooped at the mechanism and geometry level; only a narrow empirical
measurement (the 24% false-blessing *rate* on our specific grid, scored against SW2)
appears un-scooped, and under the SPEC's "ambiguity = assume scooped" rule that is a
measurement of a known phenomenon, not a new one.

## Claims under test (verbatim from SPEC)

1. **The identity.** For SNIS with proposal q and target p, the asymptotic ESS fraction
   satisfies **ESS/N = exp(−D₂(p‖q))**, D₂ = Rényi divergence of order 2. Used as an
   EXACT identity organizing a certificate frontier (log ẑ error bands from certified
   ESS), not as a bound.
2. **The blind spot.** Smoothly overspread proposals (mass-covering, correct support,
   inflated dispersion) are false-blessed by SNIS certificates: high measured ESS +
   stable under N-doubling, while SW2 to the target is large (24% false-blessing rate in
   our stage-0 grid). Claimed as a characterized failure mode WITH geometry (complement
   of the heavy-tail failures weight diagnostics target).

## Method / reliability caveat

All quotes below were obtained by direct fetch of the rendered paper (ar5iv HTML, arXiv
abstract page, or author case-study HTML) — none are from memory. Second-order caveat:
the fetch pipeline transcribes via a fast summarizer model, so exact glyphs/subscripts
may be lightly normalized; **equation NUMBERS are as printed in the rendered source**.
One body (Martino–Elvira–Louzada 1602.03572) did not render past its ar5iv wrapper — it
is used only as corroborating folklore, flagged [BODY-NOT-FETCHED], not as a load-bearing
quote. The mathematical identity D₂(p‖q) = log E_q[(p/q)²] = log(1+χ²(p‖q)) = log ρ is the
textbook definition of the Rényi divergence of order 2 and is used to translate every χ²/ρ
result below into the exp(−D₂) form.

---

## VERDICT TABLE — Claim 1 (the identity)

Ruling convention: SUBSUMES = the source contains our claim (we are scooped);
ADJACENT-BUT-DISTINCT = same territory, materially different object; UNRELATED.

| Source | Ruling | Verbatim pointer |
|---|---|---|
| **Agapiou, Papaspiliopoulos, Sanz-Alonso, Stuart** — *Importance Sampling: Intrinsic Dimension and Computational Cost* (arXiv:1511.06196, Statist. Sci. 2017) | **SUBSUMES** | Eq (2.2): "ρ := π(g²)/π(g)²" = "the second moment of the Radon–Nikodym derivative of the target with respect to the proposal". §2.3.2: "for large enough N … the strong law of large numbers gives **ess ≈ N/ρ**". Eq (2.4): "D_χ²(μ‖π) := π([g/π(g)−1]²) = **ρ−1**". Eq (2.5): "**ρ ≥ e^{D_KL(μ‖π)}**". |
| **Chatterjee & Diaconis** — *The sample size required in importance sampling* (arXiv:1511.01437, Ann. Appl. Probab. 2018) | **ADJACENT-BUT-DISTINCT** | Abstract: "a sample of size approximately **exp(D(ν‖μ))** is necessary and sufficient for accurate estimation … where D(ν‖μ) is the **Kullback–Leibler** divergence." |
| **Sanz-Alonso** — *Importance Sampling and Necessary Sample Size: an Information Theory Approach* (arXiv:1608.08814) | **ADJACENT-BUT-DISTINCT** | Thm 4.1: "D_f(ℙ‖ℚ) ≤ U_f(N,ε) + δ" (general f-divergence necessary condition). Thm 4.3(2): "If **N < (1+ε)⁻²(1+2ε+D_χ²−δ)**, then, with probability at least 1/2, [IS fails]." Scope: "Our presentation **does not cover autonormalized** importance sampling." |
| **Elvira, Martino, Robert** — *Rethinking the Effective Sample Size* (arXiv:1809.04129, Int. Stat. Rev. 2022) | **SUBSUMES** (corroborating) | Eq (30): "**E_q[W²] − 1 = χ²(π̃, q)**"; "the quantity **ρ = E_q[W²]** has been studied … due to the connection to the χ²(π̃,q) and for its relation with the ESS-hat." |
| **Martino, Elvira, Louzada** — *ESS for IS based on discrepancy measures* (arXiv:1602.03572) | **SUBSUMES** (corroborating, [BODY-NOT-FETCHED]) | Search-surfaced folklore: "a well-known connection between the χ²-divergence and the effective sample size defined as ESS := 1/Σ(w)²." Not load-bearing; body did not render. |

### Reduction that settles Claim 1
Agapiou (2.2) defines ρ as the **self-normalized** second moment of dp/dq (the /π(g)²
makes it normalization-invariant → this is exactly the SNIS efficiency). (2.4) gives the
**exact** identity ρ = 1 + χ²(p‖q). §2.3.2 gives ess ≈ N/ρ. Substituting the definition
D₂ = log(1+χ²) = log ρ:

> **ESS/N ≈ 1/ρ = 1/(1+χ²) = exp(−D₂(p‖q)).**

That is Claim 1 verbatim. It is not a bound in their treatment either — (2.4) is an
equality and (2.5) shows the authors already reason in "ρ = exp of a divergence" terms
(ρ ≥ e^{D_KL}). Elvira Eq (30) re-derives ρ = 1+χ² independently in the SNIS ESS-hat
setting. The exp-of-divergence *sample-size template* itself predates all of this
(Chatterjee–Diaconis, KL version, "necessary and sufficient"). **Nothing in Claim 1 as an
identity is new.**

---

## VERDICT TABLE — Claim 2 (the overspread blind spot)

| Source | Ruling | Verbatim pointer |
|---|---|---|
| **Dieng, Tran, Ranganath, Paisley, Blei** — *Variational Inference via χ Upper Bound Minimization* / CHIVI (arXiv:1611.00328, NeurIPS 2017) | **SUBSUMES (mechanism + geometry)** | Eq (1): "D_χ²(p‖q) = E_q[(p(z\|x)/q(z;λ))² − 1]". "Optimizing the χ-divergence leads to a variational distribution with a **zero-avoiding** behavior … the χ-divergence is **infinite whenever q=0 and p>0**"; KL is "**zero-forcing**." App. D: "minimizing this variance is equivalent to minimizing the quantity **E_q[(p(x,z)/q(z;λ))²]**." |
| **Li & Turner** — *Rényi Divergence Variational Inference* (arXiv:1602.02311, NeurIPS 2016) | **SUBSUMES (geometry axis)** | "Choosing different alpha values allows the approximation to balance between **zero-forcing (α→+∞ … mode-seeking)** and **mass-covering (α→−∞)** behaviour." Table 1: α=2 "**proportional to the χ²-divergence**." (No explicit ESS/IS-diagnostic link.) |
| **Vehtari, Simpson, Gelman, Yao, Gabry** — *Pareto Smoothed Importance Sampling* (arXiv:1507.02646, JMLR 2024) | **ADJACENT-BUT-DISTINCT (defines the complement)** | Abstract: "the resulting estimate can be highly variable when the importance ratios have a **heavy right tail**. This routinely occurs when there are aspects of the target distribution that are **not well captured** by the approximating distribution." k̂ = generalized-Pareto shape of the **upper tail** of the ratios. |
| **Elvira, Martino, Robert** — *Rethinking the Effective Sample Size* (arXiv:1809.04129) | **SUBSUMES (diagnostic unreliability + the overdispersed regime)** | Eq (7): "ESS-hat = 1/Σ_{n} w̄_n²". "Even when the sample approximation to the integral I is **very poor**, we have that **ESS-hat ≥ 1**." h-independence flaw: "the ESS conveys the efficiency … but this dependence [on h] is **completely lost** in the approximation." Fig 2 (variance mismatch, large σ_q): "for high values of N and for high values of σ_q, the ESS-hat now **underestimates** the ESS," and in that range "the **ESS is larger than N**." |
| Pareto-k̂ diagnostics case study (Vehtari, posterior/loo vignette) | (checked, negative) | Does not name the overdispersed/light-tailed pass explicitly; only "when k̂>0.7 both bias and variance grow so fast that Pareto smoothing rarely helps." Confirms k̂ is a heavy-tail diagnostic; the overspread blind spot is not stated there. |

### Why Claim 2 is scooped at the level that matters
- **The geometry** ("complement of the heavy-tail failures") is the zero-avoiding /
  zero-forcing asymmetry of the χ²/Rényi objective, stated outright by CHIVI ("infinite
  whenever q=0 and p>0") and Li–Turner (mass-covering ↔ mode-seeking axis, χ² = α=2).
  Under-dispersion (missing mass) sends χ² — hence the importance-ratio variance, hence
  1/ESS — to infinity; over-dispersion only mildly inflates it. So the ESS/D₂ certificate
  is *by construction* lenient toward overspread and merciless toward under-spread. That
  is exactly Claim 2's "geometry," and it is published.
- **The failure mode** ("overspread ⇒ high ESS but far from target") is the direct
  corollary: CHIVI App. D says the χ² objective *is* the IS-weight second moment, and
  minimizing it yields the mass-covering (overdispersed) proposal — i.e. the ESS-optimal
  proposal is overdispersed. PSIS's k̂ sees only the heavy right tail (under-dispersion),
  so it cannot flag overspread — its stated scope *implies* the complement.
- **The diagnostic-unreliability core** ("high measured ESS while the approximation is
  poor") is Elvira–Martino–Robert head-on: ESS-hat ≥ 1 even when the estimate is "very
  poor," ESS-hat is h-blind, and in the **overdispersed variance-mismatch regime** the
  true ESS can exceed N. That is the strongest single scoop of Claim 2's diagnostic side.
- **N-doubling stability** adds nothing new conceptually: an overspread proposal has
  bounded ratios ⇒ finite weight variance ⇒ a genuinely convergent ESS estimate, so
  "stable under N-doubling" is the expected, known behavior, not a surprise the gate
  fails to anticipate.

**What no fetched source does verbatim:** run the SNIS certificate as a *pass/fail gate*
(measured ESS ≥ threshold AND N-doubling-stable) against a **Wasserstein** (SW2) ground
truth and report a **false-blessing rate** (24%) over a proposal grid. That specific
ESS-gate-vs-SW2 quantification appears original — but it is a *measurement* of the
mass-covering asymmetry above, not a new phenomenon or new geometry.

---

## Three-sentence bottom line per claim

**Claim 1 (the identity) — KNOWN; do not claim it.** ESS/N = exp(−D₂(p‖q)) is a one-line
restatement of Agapiou et al. (2.2) ρ = second moment of dp/dq, (2.4) ρ = 1+χ², §2.3.2
ess ≈ N/ρ, via the textbook definition D₂ = log ρ, and it is independently re-derived by
Elvira–Martino–Robert (Eq 30) with the exp-of-divergence sample-size template itself owed
to Chatterjee–Diaconis (KL) and Sanz-Alonso (χ² lower bound). A workshop note may **not**
present the identity as a contribution; at most it may claim the **certificate-frontier
construction** — turning certified ESS into usable log-ẑ error bands as an engineering
artifact — and even that must cite Agapiou for the identity and Chatterjee–Diaconis /
Sanz-Alonso for the divergence-vs-sample-size lineage. Honest framing: "we *use* the known
ESS = exp(−D₂) identity to build a certificate frontier," never "we *derive*" it.

**Claim 2 (the blind spot) — SCOOPED on mechanism and geometry; a thin empirical sliver
survives.** The claim that SNIS/ESS certificates false-bless overspread proposals is a
corollary of the published mass-covering asymmetry of the χ²/Rényi-2 objective (CHIVI's
zero-avoiding vs zero-forcing; Li–Turner's mass-covering↔mode-seeking axis with χ² = α=2),
combined with PSIS's explicitly heavy-right-tail-only k̂ and Elvira–Martino–Robert's
demonstration that ESS-hat is high-but-misleading precisely in the overdispersed
variance-mismatch regime. The only un-scooped element is the *quantified* false-blessing
rate (24%) of a concrete ESS+doubling gate scored against SW2 on our grid, which is an
empirical measurement of a known effect. Under the SPEC's "ambiguity = assume scooped"
rule, a note may claim only "we *quantify* the false-blessing rate of an SNIS ESS+doubling
gate against SW2," explicitly citing CHIVI/Li–Turner for the mechanism, PSIS for the
tail-only complement, and Elvira–Martino–Robert for ESS-hat unreliability — it may **not**
claim to characterize a new failure mode or new geometry.

## Consequence for the parked project

Neither claim clears the novelty bar for a stand-alone methods result. If a note is
written at all, its honest, defensible contribution is **engineering + measurement**: (a)
the certificate-frontier construction as a reusable artifact (identity cited, not
claimed), and (b) the measured SW2-vs-ESS-gate false-blessing rate as an empirical
cautionary datapoint on a known blind spot. This matches the SPEC's pre-stated default
outcome ("known, cite and move on") and the park decision — the certificate machinery's
value is as tooling migrated into eft-sbi, not as a novelty claim. No further search is
warranted; both claims are settled against fetched primary sources.

## Sources (all fetched this session)

- Agapiou, Papaspiliopoulos, Sanz-Alonso, Stuart, arXiv:1511.06196 — https://arxiv.org/abs/1511.06196
- Chatterjee & Diaconis, arXiv:1511.01437 — https://arxiv.org/abs/1511.01437
- Sanz-Alonso, arXiv:1608.08814 — https://arxiv.org/abs/1608.08814
- Elvira, Martino, Robert, arXiv:1809.04129 — https://arxiv.org/abs/1809.04129
- Martino, Elvira, Louzada, arXiv:1602.03572 — https://arxiv.org/abs/1602.03572 [body not rendered]
- Dieng, Tran, Ranganath, Paisley, Blei (CHIVI), arXiv:1611.00328 — https://arxiv.org/abs/1611.00328
- Li & Turner, arXiv:1602.02311 — https://arxiv.org/abs/1602.02311
- Vehtari, Simpson, Gelman, Yao, Gabry (PSIS), arXiv:1507.02646 — https://arxiv.org/abs/1507.02646
- Vehtari Pareto-k̂ diagnostics case study — https://users.aalto.fi/~ave/casestudies/Pareto/pareto_diagnostics.html
