# 2026-07-11 — DECISION RECORD: D1 PARKED (Andreas, at reconvene)

The pivot conversation ran its course the same day the capacity arm landed
STRUCTURAL. Decision maker: Andreas. Recommendation and expert-lens reasoning:
this entry + log/2026-07-11-reconvene-capacity.md.

## The decision

**PARK the method. Migrate the soul. Optionally write the honest note.**

- The method (prior-fitted in-context sampler) is parked: both structural levers
  were exhausted by the frozen rules (10× compute flat; 4× parameters negative on
  every instrument), the cost readout found no niche (2/12), and the usefulness
  barometer failed on the one target from Andreas's own field.
- The reasoning that settled it (the "expert-lens" test): the method requires
  cheap evaluable density + gradients — exactly the regime where NUTS/MCLMC are
  push-button — and does not apply at all in the simulator-only regime where
  practitioners actually hurt. The amortization it offers (across a hand-built
  family zoo) is not the amortization anyone needs (across datasets of a fixed
  forward model — already owned by amortized SBI). The PFN analogy fails on both
  preconditions (per-task inference is cheap here; the task distribution is
  artificial).

## What migrates (the soul)

- The SNIS certificate machinery + doubling/stability gates: LIVE infrastructure,
  already embedded in eft-sbi (Readout-C surrogate work reused its GLS/reference
  pipeline in reverse). Any future certified-ML-inside-exact-algorithms work
  (D2-style engine) starts from ics/ certificate code.
- The paired-instrument evaluation methodology (population-confound-proof).
- The findings, banked as candidate note material:
  1. compute-buys-robustness-not-sharpness dissociation (clean instruments,
     honest negative);
  2. the SNIS overspread false-blessing blind spot (24%);
  3. the ESS/N = exp(−D₂) identity — **novelty check REQUIRED before any claim**
     (Sanz-Alonso necessary-sample-size line; Agapiou et al. intrinsic dimension;
     Chatterjee–Diaconis; PSIS/Pareto-k̂ literature for the blind-spot claim).
     Spec: SPEC-novelty-identity.md (cheap-session task).

## Revival conditions (pre-named, so nothing reopens by vibe)

Revisit ONLY if one of these becomes true: (a) the novelty check finds the
identity + blind spot are new AND a venue-worthy note grows into a method claim;
(b) a real repeated-inference user appears whose targets are dense in a definable
family AND classical samplers measurably fail there; (c) a future project needs a
zero-shot proposal and is willing to accept certificate-gated 1–30% ESS. None of
these authorize training runs without a new phase plan and Andreas's sign-off.

## Final D1 scorecard (toy + phase 1b + capacity arm)

Executor and reconvene predictions both included; misses first, as always:
reconvene missed P-curve (70%), P-crossover (65%), modal branch ALIVE (45%), the
×3 whitening arm (toy phase), P-sharp confound (caught by the paired instrument);
hit RC clause (60%), STRUCTURAL-confirm (60%), and the kill discipline held —
no bar was moved, ever. Budget: ~19 H100-hours of the 40-hour phase cap;
~50/480 campaign-wide. Eleven days brainstorm-to-verdict.

## Repo state at park

Tests 114 green. All instruments frozen and documented. JOBS.md closed. PLAN.md
carries the PARKED banner. The repo is a complete, auditable record: stage-0 →
toy → phase 1b → capacity arm → park, with every prediction priced.
