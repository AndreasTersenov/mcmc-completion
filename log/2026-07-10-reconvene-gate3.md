# 2026-07-10 — reconvene review of the gate-(iii) failure report

Verdict from Andreas: diagnosis endorsed, attempt-3 plan approved, no
frozen-core violation. Four additions, all adopted:

1. **Err-wide inflation arm**: stage-0 measured the variance-mismatch
   asymmetry (5.8x overdispersion tolerated vs ~2x underdispersion fatal at
   the 1% target), so EVERY scale estimator tested gets a x3-inflated twin
   arm — near-free insurance with asymmetric payoff.
2. **Hybrid scale fix**: don't bet only on a hand-crafted estimator; also
   feed chain-summary statistics (per-chain spread, energy range, gradient
   magnitudes) as auxiliary tokens and let an attention encoder learn the
   scale correction. Hand-crafted robust init + learned correction.
3. **The 46.5% -> ln(0.46) result is a headline**: end-to-end experimental
   confirmation of the stage-0 mode-drop law in a TRAINED system ("the
   certificate prices exactly what the sampler covers"). Preserve
   prominently in RESULTS-toy; likely paper figure.
4. **P1-composite caution**: P1 = ESS >= 1% AND sliced-W2 within 2x of
   baseline-2. The ESS clause alone cannot adjudicate multimodal targets
   (mode drop evades it — our own diagnosis). The earlier "7/10 already
   clear P1's real bar" claim is hereby qualified to "clear P1's ESS
   clause"; mode-recovery is a mandatory column in every gate-(iii) report
   from now on.

Execution order (as directed): whitening first, one variable at a time,
criteria pre-registered before each re-run.
- **Attempt 3a** = whitening fix ONLY (winner of the scale study below).
  Expected: funnel rows heal; mode-drop rows (gmm/dwell d=2) still fail ->
  likely still <8/10, but the improvement is cleanly attributable.
- **Attempt 3b** = + attention encoder + auxiliary summary tokens +
  short-context training augmentation (the mode-completion fix).

## Whitening scale study (pre-registration)
Testbed: single-target UNCONDITIONAL FM (the clean diagnostic from the
gate-3 log), 3 funnels (sigma_v ~ 1.5/2.2/2.8) + 1 gmm d4 + 1 warp d4
control, 4k steps each. Scale estimators, all computed FROM THE CONTEXT
CHAIN ONLY (K=128, frozen protocol):
  A. sigma_std   = per-dim chain std (status quo, known bad on funnels)
  B. sigma_std x3            (err-wide arm)
  C. sigma_range = per-dim (max-min)/2 over the chain
  D. sigma_range x3          (err-wide arm)
Metric: SW2^2 of de-whitened samples vs exact, over same-p floor; plus the
certificate on the funnel rows. Decision rule (pre-registered): pick the arm
with the best worst-case (max over testbed targets of sw2/floor); ties go to
the more overdispersed arm (stage-0 asymmetry).

## Whitening study job (pre-submission record)
- script: scripts/slurm/whitening_study.sh -> scripts/whitening_study.py
- config hash: d0d705e03c3c; MIG h100_20gb, 1h cap
- expected outcome: results/whitening_study.json; arms C/D (range-based)
  beat A on funnels; the x3 arms don't hurt the gmm/warp controls (stage-0:
  overdispersion is cheap); a clear worst-case winner emerges.
- submitted: job ID 15612008
