# JOBS.md — live SLURM state + harvest instructions (phase 1, toy)

Updated 2026-07-10 (early). Session context: gates (i)+(ii) GREEN and
committed; gate (iii) attempt 2 in flight; gate (iv) infrastructure committed
but NOT submitted (ladder rule: waits for gate (iii) green).

## Environment (everything assumes this)
- venv `~/ics-env` (jax 0.9.1 + cuda12 plugin; no distrax/diffrax — see
  log/2026-07-09-toy-env.md). Jobs: `module load python/3.11.5 gcc cuda/12.6`.
- Login node: ALWAYS `JAX_PLATFORMS=cpu taskset -c 0-7` for python (cgroup
  pids.max=512; XLA thread pools).
- `~/software/jax_flows` on local branch `rorqual-compat` (do not push it).

## In-flight / pending jobs
| job | what | expected outcome | harvest |
|-----|------|------------------|---------|
| 15586126 | gate (iii) attempt 2 (200k steps, amended SW2 bar — see log/2026-07-09-toy-gate3.md) | GATE3-PASS in jobout/gate3_15586126.out; >=8/10 targets pass | append verdict to log/2026-07-09-toy-gate3.md; commit results/gate3.json + checkpoint note; THEN submit gate (iv) below |

## Gate (iv) submission (ONLY after gate (iii) is green and committed)
```bash
cd ~/software/mcmc-completion && mkdir -p jobout
GEN=$(sbatch --parsable scripts/slurm/gen_data.sh)
# arms (resumable; 2-slot insurance chains on b1):
A1=$(sbatch --parsable -J train4 --dependency=afterok:$GEN scripts/slurm/train_arm.sh train4)
A2=$(sbatch --parsable -J train4 --dependency=afterany:$A1 scripts/slurm/train_arm.sh train4)
B1=$(sbatch --parsable -J train2 --dependency=afterok:$GEN scripts/slurm/train_arm.sh train2)
B2=$(sbatch --parsable -J train2 --dependency=afterany:$B1 scripts/slurm/train_arm.sh train2)
C1=$(sbatch --parsable -J train4ng --dependency=afterok:$GEN scripts/slurm/train_arm.sh train4 --nograd)
C2=$(sbatch --parsable -J train4ng --dependency=afterany:$C1 scripts/slurm/train_arm.sh train4 --nograd)
# evals (one per arm; nograd arm evaluates with grad tokens zeroed):
sbatch -J eval_train4   --dependency=afterany:$A2 scripts/slurm/eval_arm.sh train4
sbatch -J eval_train2   --dependency=afterany:$B2 scripts/slurm/eval_arm.sh train2
sbatch -J eval_train4ng --dependency=afterany:$C2 scripts/slurm/eval_arm.sh train4ng --nograd
```
Log every job ID + config hash (sha256 of the scripts) + expected outcome in
log/2026-07-10-toy-gate4.md BEFORE submitting (discipline per PLAN).
Expected: TRAIN-COMPLETE within one 3h slot per arm (300k steps ~ <1h on a
full H100); chains are insurance. Eval writes results/eval_<arm>.json.

## After eval jsons land — P-verdict assembly (next session)
- P1: eval_train4 rows, heldout=False, d<=8: fraction with d2_hat <= 4.6
  (ESS >= 1%) must be >= 80%; SW2 within 2x of baseline-2 (see baselines).
- P2/P3: heldout=True rows of eval_train2: ESS<0.1% OR !stable on >=50% (P2);
  certificate confusion matrix vs sw2-bad with mode-drop rows separated via
  mode_recovery column (P3).
- P11: eval_train4 vs eval_train4ng, relative ESS gain in-family vs held-out.
- P12: held-out rows eval_train2 vs eval_train4 (2 vs 4 families).
- P7 + baselines 1-4: NOT IMPLEMENTED YET (baseline-1 = eval with untrained
  params — trivial; baseline-2 = per-target FM, 10 H100-min each on ~12
  representative eval targets; baseline-3 = fit energy MLP on context (x,E)
  pairs + MALA on it; baseline-4 = blackjax MCLMC at matched wall-clock,
  compare SW2/mode-recovery — no q-density, so no SNIS row). This is the
  main remaining build work besides RESULTS-toy.md.

## Kill criteria watch
K-T1 fires only if P1 fails after honest effort (tuning within the 20
H100-day cap — spent so far: < 1 H100-hour total). K-T2 is a scoping
decision at reconvene if P12 is flat.

## Budget ledger (20 H100-day cap)
smoke 15582495 (~2 min MIG) + gate1 15583499 (33 s) + gate2 15583637 (~4 min)
+ gate3 15583912 (~9 min) + gate3 attempt-2 15586126 (~30 min MIG) ≈
well under one H100-hour. Gate (iv) adds ~3-6 H100-hours (3 arms + evals).
