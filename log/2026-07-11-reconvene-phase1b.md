# 2026-07-11 — RECONVENE RULING: phase-1b 2M harvest audited; STRUCTURAL stands; capacity arm legitimate

Inputs: log/2026-07-11-phase1b.md (branch adjudication + capacity-arm prereg),
results/eval_curve.json, readout_b_{200k,2M}.json, readout_c_{200k,2M}.json,
phase1b_curves.png, real_by_eye.png. Audit: suite re-run green (114/114, repo
interpreter); diff a7e1d96..HEAD — NO hook/test/conftest changes; decisive numbers
re-extracted independently from the JSONs and they match the log verbatim
(TT T=1: .167/.167/.083/.083/.167/.125; fresh med: 549/518/652/559/677/642;
RC@2M: eight-schools 9.55% stable, gym-banana 33.10% stable, WL 0.012% unstable;
crossover 2/12).

## Ruling 1 — the frozen rule was applied correctly. Branch = STRUCTURAL.

TT@2M = 0.125 < 0.50; final-500k improvement −25% < +10% → plateaued; H-starve
REFUTED. Capacity-arm provision: the rule fires it "only if the plateau occurred
before 1M" — the curve never rose at all (TT(0.5M)=TT(1M)=.083 ≤ TT(0.2M)), which
satisfies the provision under its intent (don't buy width while the curve still
moves late). Firing the arm was the rule's own budgeted provision, correctly
pre-registered with a mirror rule and shared references before submission. Budget:
~7h spent of the 40 H100-h cap; arm fits comfortably.

Scorecard (reconvene's misses first): modal branch call ALIVE 45% → MISS
(STRUCTURAL, my 30% tail). P-curve 70% → MISS. P-crossover 65% → MISS. RC 60%
clause → HIT. Three misses, one hit this readout — the zoo-instrument optimism was
wrong across the board and is now priced.

## Ruling 2 — the dissociation is endorsed as THE finding, with the visual caveat attached.

10× per-pair compute moved nothing on every zoo instrument (TT flat, fresh ratio
flat, zoo family ESS DOWN: warp 3.66%→0.93%) while transforming real-target
zero-shot behavior (gym-banana 0.03% UNSTABLE → 33.1% STABLE certified;
eight-schools stable ~9.5% throughout; WL fails at both). The executor's hypothesis
(compute buys conditional-field robustness/transfer, not per-target sharpness
against the frozen composite) is accepted AS HYPOTHESIS — it is the capacity arm's
job to separate capacity from objective/eval limitation.

The raw-evidence caveat (real_by_eye.png, stated plainly so nobody oversells this
again): the raw zero-shot samples do NOT visually trace the targets — banana's
cloud sits on the high-mass region but misses the arch arms; eight-schools is
slightly mislocated; SW2 actually WORSENS 200k→2M on both passing targets while
certified ESS rises. What improved is the model as a CERTIFIED IMPORTANCE PROPOSAL
(coverage + reweighting), not as a direct posterior sampler. That is exactly the
proposal-engine framing ("learned proposals, exact algorithms") and it is the only
framing Readout C supports. Any write-up of phase 1b must quote certified-ESS after
reweighting, never present raw sample clouds as posterior approximations.

WL band-power fail is unsurprising in hindsight and diagnostic: an extremely
concentrated d=3 posterior (data-dominated) gives a zero-shot proposal ~no overlap;
the certificate correctly refuses it (0.01% UNSTABLE). "The certificate knows when
it doesn't work" is a selling point of the reframe — log it as such, not as a
partial success.

## Ruling 3 — what happens next (no session action needed until eval_cap lands).

- Capacity arm (chain 15754023–26 + eval 15754027) runs to completion; the
  pre-registered mirror rule is applied verbatim; REPORT AND STOP either way.
- If STRUCTURAL again → the PIVOT conversation with Andreas, per the frozen rule's
  pre-listed options. Reconvene's preparation, recorded now so the conversation
  starts from evidence: Readout B (2/12 at both checkpoints) does NOT support the
  cost-competitive-niche claim today; Readout C supports a scoped proposal-engine
  pivot (d ≤ 10, diffuse-to-moderate posteriors, certificate-gated) and the
  eight-schools/banana results are the dogfooding argument. The K-T1 option should
  be argued from THESE numbers, not from the toy-phase warp result.
- Pre-registered (reconvene, before eval_cap): capacity arm lands
  STRUCTURAL-confirm 60% / TT ≥ 50% 15% / mixed-ambiguous 25% (close to the
  executor's 55/20/25; the dissociation makes objective/eval limitation more likely
  than capacity).
- Out of scope stays out of scope: no new training tricks, no eval-bar edits on an
  ambiguous capacity result — ambiguity = negative = STRUCTURAL.
