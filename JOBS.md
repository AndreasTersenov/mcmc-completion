# JOBS.md — SLURM state + handoff

Updated 2026-07-11 ~18:30. **PHASE 1B COMPLETE. No jobs in flight.**
RESULTS-phase1b.md is FINAL: branch verdict **STRUCTURAL, capacity-arm
confirmed → the PIVOT conversation** (rule applied verbatim; full record in
log/2026-07-11-phase1b.md). The dissociation — zoo instruments flat under
10× compute and 2× width, real-target zero-shot transformed (banana
0.03%→33% certified stable; eight-schools d=10 working at both checkpoints;
WL needle refused honestly) — is the finding the reconvene should weigh.

## Awaiting reconvene (nothing runs before rulings)

- The pivot decision (options pre-listed in PLAN.md branch 3 + RESULTS;
  Readout B does not modify them at 2/12; Readout C shapes toward
  per-domain scoping around a real problem family — usefulness clauses bind
  any ALIVE-shaped phase 2).
- Parked from phase 1: √T whitening arm, certificate dispersion-check (P3
  blind spot), banana-warp d≥8 zoo fix.

## Artifacts

- RESULTS-phase1b.md (deliverable), RESULTS-toy.md (phase 1), RESULTS.md
  (stage 0).
- Figures: phase1b_curves.png, real_by_eye.png, scaling_by_eye.png (+ toy
  set). Raw jsons in results/; references + checkpoints on $SCRATCH
  (ics-zoo: 2M line five snapshots + cap line three; ics-refs: R̂-gated).
- Suite: 113 tests green. Budget: phase-1b ~13/40 H100-h; project ~45/480.

## Environment (unchanged; hard-won notes)

- venv ~/ics-env; jobs `module load python/3.11.5 gcc cuda/12.6`. Login:
  `JAX_PLATFORMS=cpu taskset -c 0-7`. crc32 seeds, never builtin hash().
  grep-verify scripted edits. CPU-smoke every new job path.
- **CPU-only jobs → --account=def-lplevass** (rrg has no CPU allocation).
- Post-maintenance GPU nodes can silently wedge (zero stdout past the ptxas
  banner): instrument + rerun before blaming code; --exclude the node.
- R̂ < 1.01 is NOT sufficient for reference chains on curved targets
  (gym-banana IACT O(10²⁺)): use exact samplers where available, else the
  half-vs-full variance stability gate (both now standard in readout_c).
- eft-sbi read-only; its .venv has camb; WL grid/caches on $SCRATCH/ics-wl;
  wl_surrogate gate runs AMENDED (0.5σ sanity bound; see log post-mortem #5).

## PROJECT PARKED — 2026-07-11

No jobs are running or authorized. Final chain: 15754023-27 (capacity arm,
harvested at reconvene → STRUCTURAL again → park decision). Scratch checkpoints
under $SCRATCH/ics-zoo/ may be reclaimed after 2026-08-01 EXCEPT gate3e-200k,
ckpt_2m_step2000000, and ckpt_cap_step1000000 (archive-grade: the three verdict
checkpoints). See log/2026-07-11-reconvene-park.md.
