# SPEC — novelty check: the ESS identity + the SNIS overspread blind spot

**For a cheap-model session. Read this file and nothing else first. Budget: one
session, web-search + direct-fetch only, NO code, NO training. Deliverable:
log/YYYY-MM-DD-novelty-identity.md + a PR-style summary at the end of the session.**

## Claims under test (ours, stated exactly)

1. **The identity.** For self-normalized importance sampling with proposal q and
   target p, the asymptotic ESS fraction satisfies ESS/N = exp(−D₂(p‖q)) where D₂
   is the Rényi divergence of order 2. We use it as an EXACT identity organizing a
   certificate frontier (log ẑ error bands from certified ESS), not as a bound.
2. **The blind spot.** Smoothly overspread proposals (mass-covering, correct
   support, inflated dispersion) are false-blessed by SNIS certificates: high
   measured ESS + stable under N-doubling, while SW2 distance to the target is
   large (24% false-blessing rate in our stage-0 grid). Claimed as a
   characterized failure mode WITH geometry (it is the complement of the
   heavy-tail failures that weight diagnostics target).

## Kill-test protocol (adversarial: try to prove we are scooped)

For each source: fetch the actual abstract/PDF (arXiv), quote the relevant
equation/claim verbatim with its number, and rule IDENTICAL / SUBSUMES /
ADJACENT-BUT-DISTINCT / UNRELATED. From-memory quotes are DRAFT until fetched.
A hallucinated citation killed a finding once in this campaign — verify by fetch.

Priority queue:
- Sanz-Alonso, "Importance sampling and necessary sample size: an information
  theory approach" (arXiv:1608.08814) — the closest suspect for claim 1: relates
  necessary N to exp of a divergence. Is it D₂ exactly? Identity or bound? SNIS
  or IS?
- Agapiou, Papaspiliopoulos, Sanz-Alonso, Stuart, "Importance sampling:
  intrinsic dimension and computational cost" (arXiv:1511.06196) — ρ = second
  moment of dp/dq; note ESS/N ≈ 1/ρ and ρ = exp(D₂) is elementary — if that
  chain appears explicitly anywhere, claim 1 is KNOWN (the interesting question
  becomes only our certificate-frontier USE of it).
- Chatterjee & Diaconis, "The sample size required in importance sampling"
  (arXiv:1511.01437) — KL-based; check whether D₂ appears in their remarks.
- Vehtari, Simpson, Gelman, Yao, Gabry, "Pareto smoothed importance sampling"
  (arXiv:1507.02646, latest version) — for claim 2: does the PSIS k̂ literature
  already characterize the light-tailed/overspread regime as a diagnostic blind
  spot, or only heavy tails? Also check the "khat < 0.5 but biased" discussions.
- Rényi/α-divergence variational inference literature (Li & Turner 1602.02311)
  for the D₂–overdispersion link stated as a sampling-diagnostic failure.
- One free search lens of your choosing (e.g. "effective sample size chi-squared
  divergence identity" and "importance sampling diagnostic overdispersed
  proposal") — report what surfaced.

## Output format

Verdict table (claim × source × ruling × verbatim-quote pointer), then a
three-sentence bottom line per claim: is a workshop note honest, and what may it
claim (identity? framing? certificate-frontier construction? blind-spot
geometry?). No hedging beyond the evidence; ambiguity = assume scooped.

## Context (one paragraph, for calibration only)

These claims come from a parked project (see log/2026-07-11-reconvene-park.md).
The note only happens if something here is actually new — the default outcome is
"known, cite and move on," and that outcome is FINE. Do not motivated-reason
toward novelty.
