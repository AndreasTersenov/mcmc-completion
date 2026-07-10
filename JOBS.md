# JOBS.md — SLURM state + handoff (phase 1, toy)

Updated 2026-07-10 (GATE (iv) LAUNCHED, authorized by
log/2026-07-10-reconvene-paired.md; pre-registration + full job table in
log/2026-07-10-toy-gate4.md). **21 jobs queued/running:**
datagen 15651223 (train4) / 15651225 (train2) / 15651226 (eval);
arm chains train4 15651227-29, train2 15651230-32, train4ng 15651233-35
(1.6M steps each = gate3e's 260 steps/pair held fixed at 6144 pairs);
per-arm post: evals 15651236-38 -> results/eval_<arm>.json (dual (K,T)
columns, fresh-theta regime), sanity floors 15651239-41 ->
results/sanity_<arm>.json (trained-target regime, floor >= 6/24 & med ESS
>= 1%), paired instruments 15651242-44 -> results/paired_<arm>.json
(P-scale: median(gate3e)/median(arm) > 1 on T=1).

## Harvest (in order; keep regimes SEPARATED in every report)
1. Check sanity floors first: a SANITY-FLOOR-FAIL invalidates that arm
   (diagnose + resubmit, do not reinterpret).
2. Paired instruments -> P-scale verdict (70% expectation logged).
3. P-assembly from eval jsons (fresh-theta): P1 (train4, heldout=False,
   d<=8: ESS>=1% clause now; SW2 clause NEEDS baselines below); P2/P3
   (heldout=True, mode_recovery column separates mode-drop); P11 (train4 vs
   train4ng); P12 (train2 vs train4 heldout rows).
4. Baselines 1-4 for the SW2 clauses: (1) untrained params via eval_full
   --ckpt on an init checkpoint; (2) per-target FM 10 H100-min x ~12
   representative eval targets; (3) energy-MLP-on-context + MALA; (4)
   blackjax MCLMC at matched wall-clock (SW2/mode-recovery only).
5. RESULTS-toy.md per PLAN: P-verdicts, zoo-diversity curve (P12),
   certificate confusion matrix (mode-drop column separate), baseline
   table, honest limits; carry-items: coverage law (paper figure,
   results/gate3_story.png), funnel conditional-specific gap, three
   information-ceiling reclassifications, PAIRED-B scaling direction.
2M chain stays staged/superseded unless arms show undertraining (loss still
falling at 1.6M or paired instrument regressing). Budget ~14/480 spent;
chain adds ~15-20 H100-hours.

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
