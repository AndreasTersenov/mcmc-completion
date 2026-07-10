# JOBS.md — SLURM state + handoff (phase 1, toy)

Updated 2026-07-10 (paired eval harvested). **No jobs in flight. STOPPED
per directive** — evidence package assembled for the reconvene bar
conversation (log/2026-07-10-paired-eval.md):
- VERDICT: PAIRED-B-AMORTIZATION. 128-model better on 18/24 paired targets
  (sign-test p~0.011); medians T=1 905->549 (1.65x), T=5 855->342 (2.5x),
  at 10x less compute per pair. The gate3e P-sharp fail was a
  population/theta artifact; the paired instrument reverses the conclusion.
- CONTEXT: paired eval ran on FRESH theta (P1's generalization regime) where
  both models are far from bars — distinct from gate-(iii)'s trained-target
  regime. Decision requested from reconvene: redefine gate-(iii) sharpness
  via the paired instrument, and/or proceed to gate-(iv) scale (1024
  targets) on the measured scaling direction.
- The 2M-compute chain remains STAGED (scripts/slurm/train128_2m.sh +
  paired_eval_2m.sh) if the compute axis is wanted.
Story figure: results/gate3_story.png. Budget ~14/480 H100-hours.

## Gap inventory after the P1-mirror measurement (results/gate3_p1.json)
- warp-d2 PASSES (first certified target); ESS clause solved on 6/10 rows.
- (1) CONDITIONAL SHARPNESS is the dominant gap: SW2 30-400x bespoke refs.
- (2) d=2 mode drop: localized bug, survives all ablations.
- (3) funnels fail conditionally even at K=512 (bespoke passes with the same
  whitening) — conditional-model-specific. K=128 refusal behavior correct.
## Next levers IN ORDER (pre-register each in a new gate3d log):
(1) d=2 fix — d-embedding one-hot, then per-family balance (eval-only reuse
    of checkpoints where possible); (2) training length on the no-shortk
    recipe (b1 loss plateau ~1.38 may be real convergence — check with 400k);
    (3) head capacity for sharpness. On eventual GATE3-PASS: patch
    eval_full.py with funnel-K=512 handling, then the gate-(iv) chain.

## Environment (everything assumes this)
- venv `~/ics-env` (jax 0.9.1 + cuda12 plugin; distrax/diffrax dropped — see
  log/2026-07-09-toy-env.md). Jobs: `module load python/3.11.5 gcc cuda/12.6`.
- Login node: ALWAYS `JAX_PLATFORMS=cpu taskset -c 0-7` for python (cgroup
  pids.max=512; XLA thread pools). Never seed with builtin hash() (salted).
- `~/software/jax_flows` on local branch `rorqual-compat` (do not push).
- Full test suite: 105 green. Stop hook runs it via ics-env.

## Gate (iii) attempt-3 plan (APPROVED at reconvene 2026-07-10 with four
## additions — see log/2026-07-10-reconvene-gate3.md; err-wide x3 arms +
## hybrid learned scale correction added below)
Two independent fixes, test cheapest-first, ONE variable at a time:
1. **Whitening scale (funnel killer).** Proven: FM head hits the sampling
   floor on funnels when whitened with true pool moments (login diagnostic in
   the gate-3 log, 4k steps). Candidate fixes, in order:
   a. inflate context sigma by a per-dim factor inferred from energy DROP
      across the chain (the context knows how far downhill it went);
   b. heavier-tailed FM base (student-t, df~5) — cheap, composable;
   c. quick empirical study: does sigma_ctx accuracy improve with K=512 or
      larger INIT_SCALE? (frozen protocol allows K choice at train time.)
2. **Mode-subset completion (gmm/dwell d=2 killer).** The encoder completes
   only context-visited modes; a Bayes-optimal reader would recover the full
   in-family mode structure from 128 exact (x,E,gradE) tokens (stage-0 M2).
   Candidate fixes: (a) self-attention token mixer before pooling (2 blocks);
   (b) train-time context augmentation — pair each target with SHORT contexts
   (K=8/32) so "context under-covers, pool has more" is a learned pattern;
   (c) check gmm-d8+warp-d8 stability flags too (borderline rows).
Gate-iii criteria stand as amended (ESS>=5%, stable, |logZ|<=0.1,
SW2^2 <= max(3x floor, 0.1), >=8/10). Re-pre-register anything else BEFORE
the run, in log/2026-07-10-toy-gate3b.md.

## Gate (iv) submission recipe (ONLY after gate (iii) green + committed)
```bash
cd ~/software/mcmc-completion && mkdir -p jobout
GEN=$(sbatch --parsable scripts/slurm/gen_data.sh)
A1=$(sbatch --parsable -J train4 --dependency=afterok:$GEN scripts/slurm/train_arm.sh train4)
A2=$(sbatch --parsable -J train4 --dependency=afterany:$A1 scripts/slurm/train_arm.sh train4)
B1=$(sbatch --parsable -J train2 --dependency=afterok:$GEN scripts/slurm/train_arm.sh train2)
B2=$(sbatch --parsable -J train2 --dependency=afterany:$B1 scripts/slurm/train_arm.sh train2)
C1=$(sbatch --parsable -J train4ng --dependency=afterok:$GEN scripts/slurm/train_arm.sh train4 --nograd)
C2=$(sbatch --parsable -J train4ng --dependency=afterany:$C1 scripts/slurm/train_arm.sh train4 --nograd)
sbatch -J eval_train4   --dependency=afterany:$A2 scripts/slurm/eval_arm.sh train4
sbatch -J eval_train2   --dependency=afterany:$B2 scripts/slurm/eval_arm.sh train2
sbatch -J eval_train4ng --dependency=afterany:$C2 scripts/slurm/eval_arm.sh train4ng --nograd
```
Log every job ID + config hash + expected outcome in
log/2026-07-10-toy-gate4.md BEFORE submitting. If the encoder changes in
attempt 3, ICSModel defaults change → datagen is unaffected, but re-hash.

## After eval jsons land — P-verdict assembly
- P1: eval_train4, heldout=False, d<=8: fraction d2_hat <= 4.6 (ESS >= 1%)
  >= 80%; SW2 within 2x baseline-2. NOTE: gate-3 attempt 2 shows 7/10
  training targets clearing the ESS CLAUSE only — P1 is a composite (ESS AND
  SW2 vs baseline-2), and ESS alone cannot adjudicate multimodal targets
  (mode drop evades it). Reconvene 2026-07-10: mode-recovery column is
  mandatory in every gate report.
- P2/P3: heldout=True rows; P3 confusion matrix separates mode-drop via the
  mode_recovery column (gate-3 showed the blind spot behaves exactly as
  stage-0 predicted: dropped modes read ESS 19% STABLE with logZ = ln(mass)).
- P11: eval_train4 vs eval_train4ng. P12: eval_train2 vs eval_train4 heldout.
- P7 + baselines 1-4: NOT BUILT. baseline-1 = eval with untrained params;
  baseline-2 = per-target FM, 10 H100-min x ~12 targets; baseline-3 = energy
  MLP fit on context (x,E) + MALA; baseline-4 = blackjax MCLMC at matched
  wall-clock (SW2/mode-recovery only, no q-density). Then RESULTS-toy.md.

## Kill criteria status
K-T1: not close (P1's bar within reach in-family; < 1 H100-hour of the
20-day cap spent). K-T2: undetermined until the P12 arms run.

## Budget ledger
smoke 15582495 + gate1 15583499 (33 s) + gate2 15583637 + gate3 15583912 +
gate3-attempt2 15586126 (~13 min MIG) ≈ **< 1 H100-hour total spent**.
