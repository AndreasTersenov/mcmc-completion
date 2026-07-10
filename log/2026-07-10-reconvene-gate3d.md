# 2026-07-10 — reconvene rulings on the gate3d decision packages

Evidence reviewed (log/2026-07-10-toy-gate3d.md + T=5 diagnostic). Both decisions ruled.

## Ruling 1 — tempered contexts: ADOPTED (train + eval)

Not a frozen-core amendment: PLAN froze "the stage-0 protocol," and stage-0 M2 swept
T in {1,2,5}; phase-1's T=1 was an implementation narrowing. Moreover stage-0
capability-target #3 named "overdispersed context chains" as the outside-the-certificate
mode-coverage mechanism — tempered probes ARE that mechanism. Zero-shot affirmative
evidence accepted (gmm-d2 logZ -0.770 -> -0.045, recovery 0.67 -> 0.83, ESS 18%).
Constraints: (a) context budget reported as (K, T) JOINTLY everywhere; (b) a T=1-only
eval column is retained — "at T=1, d=2 multimodal is context-coverage-limited" is a
finding, documented not hidden; (c) pre-register the training T-mix before the run.
RESULTS-toy gains a named theme: three instances now of bars exceeding the context's
information ceiling (funnel@K128, funnel-sigma_v, d=2 coverage) — evaluation must
respect what context can identify; the oracle/ceiling-referenced eval is part of the
method's contribution.

## Ruling 2 — sharpness: grow the mini-zoo; the bar does NOT move without data

Recalibrating the SW2 bar now = gate inflation. The amortization premise predicts
sharpness improves with zoo size; 10 targets is a micro-corpus. One pre-registered run
at ~128 targets (100-200, session's choice), gate-(iii) criteria re-pre-registered at
that scale (P1-mirror + funnel-K512 handling + (K,T) columns per Ruling 1).
**Pre-registration (Claude, 2026-07-10): P-sharp (70%) — median SW2/bespoke ratio
improves >= 2x going 10 -> 128 targets.** If it does NOT, that is a real architecture
finding and the bar conversation happens then, with evidence. Green at 128 -> gate (iv).

## Process

grep-verify rule endorsed; propagated to all three repos' CLAUDE.md by the reconvene
session. Lever bookkeeping accepted: shortK removed / 1a refuted on valid re-eval /
length & widths exonerated / pad-weight plausible-unproven (parked).
