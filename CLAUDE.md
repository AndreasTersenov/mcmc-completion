# CLAUDE.md — mcmc-completion

Stage-0 falsification study for the in-context sampler project (D1). Read `PLAN.md`
first, always. Wider context (only if needed):
`~/claude-notes/brainstorms/2026-07-09-dl-project-directions.md` and
`...novelty-sweep-RESULTS.md` (§D1).

## Hard rules

- `PLAN.md` "FROZEN CORE" (measurements, predictions, kill criteria) is immutable in this
  session. If it seems wrong, write the objection to `log/` and `RESULTS.md` — do not
  silently redesign. Implementation details are yours.
- A kill criterion firing means STOP + document; the go/no-go decision happens at a
  reconvene session with Andreas, not here.
- Validate every estimator against a closed-form case before using it on an open one.

## Environment & conventions

- Rorqual (Compute Canada). Big outputs (grids, sample arrays) → `$SCRATCH` (via
  `~/links/scratch`), NOT `$HOME` (quota). Small artifacts (plots, csv summaries) can
  live in-repo under `results/`.
- GPU jobs via SLURM, account `rrg-lplevass`; consult the `rorqual-jobs` skill for queue
  strategy. CPU is fine for most of stage-0.
- Python env: create a fresh venv or reuse `~/wl-challenge-env` if compatible; prefer
  jax or numpy — no torch needed for stage-0.
- Log format: `log/YYYY-MM-DD-<slug>.md`, entries = hypothesis → setup → expectation →
  result → updated belief. Commit early and often; this repo is local-only (no remote yet).
