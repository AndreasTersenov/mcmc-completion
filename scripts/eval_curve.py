"""Phase 1b Readout A: the five-checkpoint compute curve (+ gate3e-200k anchor),
both frozen instruments, seeds identical to the toy phase so every column is
directly comparable (pre-registration: log/2026-07-11-phase1b.md).

- TT block: gate3e eval protocol verbatim — 24 trained targets (first 2 per cell
  of the jr.key(3131) 128-zoo), ctx seeds fold_in(key(909000+jj), T), eval seeds
  800000+100jj+T, bespoke refs 700000+100jj; funnels K=512.
- Fresh-theta block: paired_eval protocol verbatim — 24 fresh targets
  (key(424242)), ctx key(515151), refs 600000+1000ci+100i, eval seeds = the z128
  slot (900000+1000ci+100i+T+10).
Refs computed ONCE and stored; missing checkpoints skipped LOUDLY.
Writes results/eval_curve.json incrementally. ICS_SMOKE=1 = CPU path test only.
"""

import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jax_flows import TimeConditionedMLP, cfm_loss, cfm_sample

from ics.context import generate_context_for_target
from ics.eval import ics_evaluate, mode_recovery
from ics.models import ICSModel
from ics.train import load_checkpoint
from ics.zoo import sample_target, sample_x
from stage0.sliced_w2 import sliced_w2_squared

FAMS, DS = ("gmm", "dwell", "funnel", "warp"), (2, 4, 8)
PER_CELL = {c: (11 if i < 8 else 10) for i, c in enumerate(
    [(f, d) for f in FAMS for d in DS])}
N_EVAL, K = 4096, 128
REF_STEPS = 4000
SMOKE = os.environ.get("ICS_SMOKE") == "1"
if SMOKE:
    N_EVAL, REF_STEPS = 64, 100

R = os.path.join(os.path.dirname(__file__), "..", "results")
SCR = os.path.join(os.environ.get("SCRATCH", "/tmp"), "ics-zoo")
CKPTS = [("0.2M", os.path.join(R, "gate3e_params.pkl"))] + [
    (tag, os.path.join(SCR, f"ckpt_2m_step{n}.pkl"))
    for tag, n in (("0.25M", 250_000), ("0.5M", 500_000), ("1M", 1_000_000),
                   ("1.5M", 1_500_000), ("2M", 2_000_000))]
if SMOKE:
    CKPTS = [("0.2M", os.path.join(R, "gate3e_params.pkl")),
             ("smk2", os.path.join(R, "gate3e_params.pkl"))]


def bespoke_ref_sw2(target, ctx, seed):
    # verbatim gate3e/paired_eval reference recipe
    d = target.d
    x_tr = ((sample_x(jr.key(seed), target, 60_000) - ctx.mu) / ctx.sigma
            ).astype(jnp.float32)
    m = TimeConditionedMLP(hidden_dims=(256, 256), output_dim=d)
    p = m.init(jr.key(seed + 1), jnp.ones((1, d), jnp.float32),
               jnp.ones((1,), jnp.float32))["params"]
    tx = optax.adam(optax.cosine_decay_schedule(2e-3, REF_STEPS))
    o = tx.init(p)

    @jax.jit
    def st(p, o, k):
        kb, kl = jr.split(k)
        idx = jr.randint(kb, (512,), 0, x_tr.shape[0])
        loss, gr = jax.value_and_grad(cfm_loss)(p, x_tr[idx], kl, m)
        up, o = tx.update(gr, o)
        return optax.apply_updates(p, up), o, loss

    for k in jr.split(jr.key(seed + 2), REF_STEPS):
        p, o, _ = st(p, o, k)
    p64 = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64), p)
    s = cfm_sample(m, p64, jr.key(seed + 3), (2 * N_EVAL, d), n_steps=100,
                   solver="heun")
    x_gen = np.asarray(ctx.mu + ctx.sigma * s, np.float64)
    fresh = np.asarray(sample_x(jr.key(seed + 4), target, 2 * N_EVAL), np.float64)
    return float(sliced_w2_squared(x_gen, fresh, n_proj=128,
                                   rng=np.random.default_rng(seed)))


def eval_one(model, params, target, ctx, seed, ref):
    cert, x_gen = ics_evaluate(model, params, target, ctx, jr.key(seed),
                               n_eval=N_EVAL, n_ode=100)
    fresh = np.asarray(sample_x(jr.key(seed + 1), target, 2 * N_EVAL), np.float64)
    sw2 = float(sliced_w2_squared(x_gen, fresh, n_proj=128,
                                  rng=np.random.default_rng(seed)))
    return dict(sw2=sw2, ratio=sw2 / max(ref, 1e-9),
                passed=bool(cert["ess_frac_2n"] >= 0.01 and cert["stable"]
                            and sw2 <= max(2.0 * ref, 0.1)),
                mode_recovery=mode_recovery(target, x_gen), **cert)


def tt_units():
    """(jj, family, d, target) for the gate3e 24-target trained subsample."""
    specs = [(f, d, i) for (f, d), n in PER_CELL.items() for i in range(n)]
    units, seen = [], {}
    for j, (f, d, i) in enumerate(specs):
        if seen.get((f, d), 0) < 2:
            kt, _, _ = jr.split(jr.fold_in(jr.key(3131), 1000 * j + i), 3)
            units.append((len(units), f, d, sample_target(kt, f, d)))
            seen[(f, d)] = seen.get((f, d), 0) + 1
    return units


def fresh_units():
    units = []
    for ci, (f, d) in enumerate([(f, d) for f in FAMS for d in DS]):
        for i in range(2):
            t = sample_target(jr.fold_in(jr.key(424242), 100 * ci + i), f, d)
            units.append((ci, i, f, d, t))
    return units


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default=None,
                    help="comma list tag=path; default = the frozen 2M-line six")
    ap.add_argument("--wide", action="store_true",
                    help="2x width model (the conditional capacity arm)")
    ap.add_argument("--out", default="eval_curve.json")
    ap.add_argument("--refs-from", default=None,
                    help="reuse the refs dict from this results json (removes "
                         "reference-training noise from cross-line comparisons)")
    args = ap.parse_args()
    global CKPTS
    if args.ckpts:
        CKPTS = [tuple(p.split("=", 1)) for p in args.ckpts.split(",")]

    t0 = time.time()
    out_path = (os.path.join(os.environ.get("TMPDIR", "/tmp"), "eval_curve_smoke.json")
                if SMOKE else os.path.join(R, args.out))
    out = (json.load(open(out_path)) if os.path.exists(out_path) and not SMOKE
           else {"refs": {}, "blocks": {}, "skipped_ckpts": []})
    if args.refs_from:
        out["refs"] = json.load(open(os.path.join(R, args.refs_from)))["refs"]
        print(f"refs preloaded from {args.refs_from} ({len(out['refs'])})", flush=True)

    tt = tt_units()[:1 if SMOKE else None]
    fr = fresh_units()[:1 if SMOKE else None]
    model = (ICSModel(enc_dim=256, enc_hidden=512, head_hidden=(512, 512, 512),
                      n_attn=2) if args.wide else ICSModel(n_attn=2))

    # contexts (regenerated deterministically each run — cheap) + refs (stored)
    tt_ctx, fr_ctx = {}, {}
    for jj, f, d, t in tt:
        k_eval = 512 if f == "funnel" else K
        for tag, temp in (("T1", 1.0), ("T5", 5.0)):
            tt_ctx[(jj, tag)] = generate_context_for_target(
                jr.fold_in(jr.key(909_000 + jj), int(temp)), t, K=k_eval,
                temperature=temp, aux_tokens=True)
        rk = f"tt{jj}"
        if rk not in out["refs"]:
            out["refs"][rk] = bespoke_ref_sw2(t, tt_ctx[(jj, "T1")], 700_000 + 100 * jj)
            print(f"ref {rk} = {out['refs'][rk]:.4f} [{time.time()-t0:.0f}s]", flush=True)
    for ci, i, f, d, t in fr:
        k_eval = 512 if f == "funnel" else K
        for tag, temp in (("T1", 1.0), ("T5", 5.0)):
            fr_ctx[(ci, i, tag)] = generate_context_for_target(
                jr.fold_in(jr.key(515151), 100 * ci + 10 * i + int(temp)), t,
                K=k_eval, temperature=temp, aux_tokens=True)
        rk = f"fr{ci}_{i}"
        if rk not in out["refs"]:
            out["refs"][rk] = bespoke_ref_sw2(t, fr_ctx[(ci, i, "T1")],
                                              600_000 + 1000 * ci + 100 * i)
            print(f"ref {rk} = {out['refs'][rk]:.4f} [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(out_path, "w"))

    for ck_tag, ck_path in CKPTS:
        if not os.path.exists(ck_path):
            print(f"!! MISSING CHECKPOINT {ck_tag} at {ck_path} — SKIPPED", flush=True)
            if ck_tag not in out["skipped_ckpts"]:
                out["skipped_ckpts"].append(ck_tag)
            continue
        blk = out["blocks"].setdefault(ck_tag, {"tt": [], "fresh": []})
        if len(blk["tt"]) == len(tt) and len(blk["fresh"]) == len(fr):
            print(f"{ck_tag}: already complete, skipping", flush=True)
            continue
        params = jax.tree_util.tree_map(jnp.asarray,
                                        load_checkpoint(ck_path)["params"])
        blk["tt"], blk["fresh"] = [], []
        for jj, f, d, t in tt:
            row = dict(family=f, d=d, ref=out["refs"][f"tt{jj}"])
            for tag, temp in (("T1", 1.0), ("T5", 5.0)):
                row[tag] = eval_one(model, params, t, tt_ctx[(jj, tag)],
                                    800_000 + 100 * jj + int(temp), row["ref"])
            blk["tt"].append(row)
        for ci, i, f, d, t in fr:
            row = dict(family=f, d=d, idx=i, ref=out["refs"][f"fr{ci}_{i}"])
            for tag, temp in (("T1", 1.0), ("T5", 5.0)):
                s0 = 900_000 + 1000 * ci + 100 * i + int(temp)
                row[tag] = eval_one(model, params, t, fr_ctx[(ci, i, tag)],
                                    s0 + 10, row["ref"])
            blk["fresh"].append(row)
        for col in ("T1", "T5"):
            tt_frac = float(np.mean([r[col]["passed"] for r in blk["tt"]]))
            med = float(np.median([r[col]["ratio"] for r in blk["fresh"]]))
            blk[f"tt_frac_{col}"] = tt_frac
            blk[f"fresh_med_ratio_{col}"] = med
        json.dump(out, open(out_path, "w"))
        print(f"[{ck_tag}] tt_frac T1={blk['tt_frac_T1']:.3f} T5={blk['tt_frac_T5']:.3f} "
              f"| fresh med ratio T1={blk['fresh_med_ratio_T1']:.0f} "
              f"T5={blk['fresh_med_ratio_T5']:.0f} [{time.time()-t0:.0f}s]", flush=True)

    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(out_path, "w"), indent=1)
    print("EVAL-CURVE-DONE", flush=True)


if __name__ == "__main__":
    main()
