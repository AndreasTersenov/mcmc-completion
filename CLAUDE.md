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

## Git discipline (decided 2026-07-09)

- **Commit after every meaningful unit**: a validated estimator, a completed measurement
  grid, a log entry, a RESULTS.md section. WIP commits are fine; uncommitted work at
  session end is not.
- **Push after committing** if a remote is configured (`git remote -v`) — currently
  local-only; Andreas is setting up SSH auth + private GitHub remotes. Never force-push;
  never rewrite pushed history.
- Worktrees: NOT used in stage-0 (one sequential agent per repo). They become the tool in
  the toy phase for parallel variant exploration (note: local-only repos need
  `worktree.baseRef: "head"` since there is no origin/HEAD yet).

## Backpressure (non-negotiable)

- **Tests-first**: before implementing any estimator, write its validation test in
  `tests/`. A Stop hook (`.claude/settings.json`) runs pytest and blocks session
  completion while tests fail — this is deliberate; fix or xfail-with-justification.
- A number plotted or written into RESULTS.md whose validation test is not green does
  not exist.
- Validation gates for THIS repo:
  1. Analytic IS: ESS and log-Z for Gaussian proposal vs shifted/scaled Gaussian target match the closed form.
  2. SNIS log-Z estimator unbiased within Monte-Carlo error on a known-Z mixture.
  3. SMC/grid posterior matches the conjugate closed form on a linear-Gaussian family.
  4. Every reported number passes an N-doubling stability check (double N, move < tol).

## Compact instructions

When compacting, preserve: modified file paths, test commands and their latest status,
the measurement/grid currently running, SLURM job IDs, and any deviation-from-PLAN notes.

## Long jobs

Prefer Bash run_in_background or the Monitor tool to babysit SLURM jobs within a session;
/loop for periodic in-session polling. Consult the rorqual-jobs skill before submitting.

## Stack (decided 2026-07-09 — binds from toy phase; stage-0 numpy code stands as-is)

JAX-first: **blackjax** (chains/SMC + the reference MCLMC implementation — the P7
benchmark ships in our dependency), **distrax** (exact-sampleable zoo families),
**flax or equinox + optax** (transformer + FM sampler head), **numpyro +
inference-gym** (realistic Bayesian targets & community benchmark suite), Andreas's
**jax_flows** (flow-matching core). Rule: call torch-only baselines (e.g. iDEM) through
numpy boundaries; never port them.

## Login-node rule (post-stage-0)

Login node = tests and short validation runs only. Anything beyond ~1 core-minute
(grids, bootstraps, sweeps, training) goes through SLURM — sbatch for batch, salloc for
interactive iteration, account rrg-lplevass; consult the rorqual-jobs skill first.
Note: remotes are live on GitHub (private) — push after every commit; earlier
"local-only" notes above are outdated.
