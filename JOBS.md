# JOBS.md — SLURM state + handoff (phase 1b)

Updated 2026-07-11 ~14:30. Phase 1b (PLAN.md frozen core + amendment) is
EXECUTING. Pre-registrations, config hashes, post-mortems:
log/2026-07-11-phase1b.md. Everything below fires automatically; harvest
begins when the chain + dependent evals complete.

## In flight / queued

- **15738731 t2m link 2** RUNNING (resumed at 1.0M steps, ~113 steps/s,
  budget 9000 s — may end ~50k short of 2M; link 3 finishes it).
- **15738732 t2m link 3** (afterany link 2; no-ops if complete flag set).
- **15738734 evalcurve** (afterany link 3; REFUSES to run a partial curve —
  resubmit after chain completion if it exits 1).
- **15741585 rb2m** (afterany link 3; in-script guard on the 2M snapshot).
- **15741589 rce2m** (afterok rcrefs 15741587 [DONE] + afterany link 3; guard).

## Landed today (all committed; jsons in results/)

- Training milestones: ckpt_2m_step{250000,500000,1000000}.pkl on scratch;
  1.5M + 2M pending. Loss ~1.37 @1M (gate3e-200k level was ~1.30 on this zoo
  — noisy per-step, the eval curve is the real readout).
- readout_b_200k.json: **crossover 2/12** (128-zoo line; frozen train4
  baseline 3/12). Per-row all-in 1.6–2.7 s vs MCLMC 19–30 s.
- readout_c_200k.json: **eight-schools T1 zero-shot: ESS 8.4% STABLE, SW2
  0.091 vs NUTS ref** (pre-registered 60% barometer clause already met at
  200k); gym-banana partial (T5 1.3% stable); WL band-power fails at 200k
  (needle posterior; theta means off).
- References on scratch ics-refs/: eight_schools (4x100k, R-hat 1.0000),
  gym_banana (EXACT sampler + 4x1M NUTS cross-check 0.3%/1.3%),
  wl_bandpower (4x50k, R-hat 1.0001). wl_surrogate.npz: degree 5, held-out
  max 0.211 sigma / median 0.011 sigma, gate AMENDED (documented).
- results/scaling_by_eye.png (3 targets x 3 generations, honest captions).

## Harvest (when the chain + evals land) — task for the session

1. eval_curve.json → Readout A: TT/fresh curves, both columns, T=1 primary.
2. Apply the frozen branch rule VERBATIM (PLAN.md): TT@2M vs 50%; FQ = fresh
   med ratio(0.2M)/(2M) vs 4x; FE = any d=4 family fresh median ESS ≥ 5% T1;
   plateau = <10% relative TT improvement over the final 500k.
3. readout_b_2M.json → crossover count (rule: ≥6/12 modifies pivot options).
4. readout_c_2M.json → barometer table (NON-GATING), both columns.
5. RESULTS-phase1b.md per PLAN deliverable spec; prediction scorecard
   (branch dist 45/25/30, P-curve 70%, P-crossover 65%, Readout-C 60%).
6. REPORT AND STOP — phase-2/pivot scoping belongs to the reconvene.

## Environment (unchanged)

- venv ~/ics-env; jobs: module load python/3.11.5 gcc cuda/12.6. Login:
  JAX_PLATFORMS=cpu taskset -c 0-7. crc32 seeds. grep-verify edits.
- eft-sbi read-only; its .venv has camb; WL caches on $SCRATCH/ics-wl.
- Post-maintenance nodes can silently wedge GPU jobs (three timeouts today:
  rg31703 x2 co-scheduled, rg31604): if a GPU job prints nothing past the
  ptxas banner for minutes, suspect the node before the code — but verify
  with an instrumented rerun before blaming either.

## Budget

Phase-1b cap 40 H100-h. Spent so far: chain ~5 h (links 1–2) + evals/figures
~2 h + 3 wasted timeouts ~1.75 h ≈ 9 h. Well inside cap.
