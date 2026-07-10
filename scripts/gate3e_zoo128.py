"""Gate (iii) at 128 targets — the single pre-registered run of reconvene
Ruling 2 (log/2026-07-10-reconvene-gate3d.md; pre-registration in
log/2026-07-10-toy-gate3e.md).

Zoo: 4 train families x d in {2,4,8} (12 cells, 128 targets), 6 contexts per
target with the pre-registered T-mix [1,1,2,2,5,5] (Ruling 1). Recipe: b1
(attn + aux, narrow, no shortk), 200k steps. Eval on 24 targets (first 2 per
cell), each in TWO (K,T) columns per Ruling 1: (128,1) documented column and
(128,5) scoring column; funnels use K=512 in both (finding 3).
GATE: >= 20/24 pass the P1-mirror composite on the T=5 column.
P-SHARP (70%): median SW2/bespoke over the 24 <= 31 (>= 2x better than the
10-target run's 62). Writes results/gate3e.json + checkpoint.
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
from ics.train import build_zoo_dataset, make_train_step, save_checkpoint
from ics.zoo import DMAX, sample_x
from stage0.sliced_w2 import sliced_w2_squared

FAMS = ("gmm", "dwell", "funnel", "warp")
DS = (2, 4, 8)
PER_CELL = {c: (11 if i < 8 else 10) for i, c in enumerate(
    [(f, d) for f in FAMS for d in DS])}  # 8*11 + 4*10 = 128
T_MIX = [1.0, 1.0, 2.0, 2.0, 5.0, 5.0]
N_CTX, K, N_POOL = 6, 128, 50_000
STEPS, BATCH, LR = 200_000, 512, 1e-3
N_EVAL = 4096
P_SHARP_BAR = 31.0


def bespoke_ref_sw2(target, ctx, seed):
    d = target.d
    x_tr = ((sample_x(jr.key(seed), target, 60_000) - ctx.mu) / ctx.sigma
            ).astype(jnp.float32)
    m = TimeConditionedMLP(hidden_dims=(256, 256), output_dim=d)
    p = m.init(jr.key(seed + 1), jnp.ones((1, d), jnp.float32),
               jnp.ones((1,), jnp.float32))["params"]
    tx = optax.adam(optax.cosine_decay_schedule(2e-3, 4000))
    o = tx.init(p)

    @jax.jit
    def st(p, o, k):
        kb, kl = jr.split(k)
        idx = jr.randint(kb, (512,), 0, x_tr.shape[0])
        loss, gr = jax.value_and_grad(cfm_loss)(p, x_tr[idx], kl, m)
        up, o = tx.update(gr, o)
        return optax.apply_updates(p, up), o, loss

    for k in jr.split(jr.key(seed + 2), 4000):
        p, o, _ = st(p, o, k)
    p64 = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64), p)
    s = cfm_sample(m, p64, jr.key(seed + 3), (2 * N_EVAL, d), n_steps=100,
                   solver="heun")
    x_gen = np.asarray(ctx.mu + ctx.sigma * s, np.float64)
    fresh = np.asarray(sample_x(jr.key(seed + 4), target, 2 * N_EVAL), np.float64)
    return float(sliced_w2_squared(x_gen, fresh, n_proj=128,
                                   rng=np.random.default_rng(seed)))


def eval_col(model, params, target, ctx, seed, ref_sw2):
    cert, x_gen = ics_evaluate(model, params, target, ctx, jr.key(seed),
                               n_eval=N_EVAL, n_ode=100)
    fresh = np.asarray(sample_x(jr.key(seed + 1), target, 2 * N_EVAL), np.float64)
    sw2 = float(sliced_w2_squared(x_gen, fresh, n_proj=128,
                                  rng=np.random.default_rng(seed)))
    ok = (cert["ess_frac_2n"] >= 0.01 and cert["stable"]
          and sw2 <= max(2.0 * ref_sw2, 0.1))
    return dict(passed=bool(ok), sw2=sw2, ratio=sw2 / max(ref_sw2, 1e-9),
                mode_recovery=mode_recovery(target, x_gen), **cert)


def main():
    t0 = time.time()
    specs = [(f, d, i) for (f, d), n in PER_CELL.items() for i in range(n)]
    print(f"building 128-target dataset (T-mix {T_MIX})...", flush=True)
    targets, ctxs, data = build_zoo_dataset(
        jr.key(3131), specs, N_CTX, K, N_POOL, temperature=T_MIX, aux_tokens=True)
    print(f"dataset built in {time.time()-t0:.0f}s", flush=True)

    model = ICSModel(n_attn=2)
    params = model.init(
        jr.key(32), jnp.ones((2, DMAX), jnp.float32), jnp.ones((2,), jnp.float32),
        data.tokens[:2, 0],
    )["params"]
    tx = optax.adam(optax.cosine_decay_schedule(LR, STEPS))
    opt_state = tx.init(params)
    step = make_train_step(model, tx, BATCH, len(specs), N_CTX, N_POOL)
    for i in range(STEPS):
        params, opt_state, loss = step(params, opt_state, jr.fold_in(jr.key(33), i),
                                       data)
        if (i + 1) % 20_000 == 0:
            print(f"step {i+1}: loss {float(loss):.4f} [{time.time()-t0:.0f}s]",
                  flush=True)

    # eval: first 2 targets of each cell, two (K,T) columns
    eval_idx = []
    seen = {}
    for j, (f, d, i) in enumerate(specs):
        if seen.get((f, d), 0) < 2:
            eval_idx.append(j)
            seen[(f, d)] = seen.get((f, d), 0) + 1
    out = {"gate": "iii-128", "targets": [], "t_mix": T_MIX}
    n_pass = 0
    ratios = []
    for jj, j in enumerate(eval_idx):
        f, d, _ = specs[j]
        t = targets[j]
        k_eval = 512 if f == "funnel" else K
        cols = {}
        ref = None
        for tag, temp in (("T1", 1.0), ("T5", 5.0)):
            ctx = generate_context_for_target(
                jr.fold_in(jr.key(909_000 + jj), int(temp)), t, K=k_eval,
                temperature=temp, aux_tokens=True)
            if ref is None:
                ref = bespoke_ref_sw2(t, ctx, 700_000 + 100 * jj)
            cols[tag] = eval_col(model, params, t, ctx, 800_000 + 100 * jj +
                                 int(temp), ref)
        row = dict(family=f, d=d, sw2_ref=ref, t1=cols["T1"], t5=cols["T5"])
        n_pass += cols["T5"]["passed"]
        ratios.append(cols["T5"]["ratio"])
        out["targets"].append(row)
        print(json.dumps(row), flush=True)

    med_ratio = float(np.median(ratios))
    out.update(n_pass=n_pass, n_eval_targets=len(eval_idx),
               passed=bool(n_pass >= 20), median_sw2_ratio=med_ratio,
               p_sharp=bool(med_ratio <= P_SHARP_BAR),
               seconds=round(time.time() - t0, 1))
    base = os.path.join(os.path.dirname(__file__), "..", "results")
    with open(os.path.join(base, "gate3e.json"), "w") as f:
        json.dump(out, f, indent=2)
    save_checkpoint(os.path.join(base, "gate3e_params.pkl"), params, opt_state, STEPS)
    print(f"median SW2/bespoke = {med_ratio:.1f} (P-sharp bar {P_SHARP_BAR})")
    print(("GATE3-PASS" if out["passed"] else "GATE3-FAIL")
          + (" P-SHARP-PASS" if out["p_sharp"] else " P-SHARP-FAIL"))


if __name__ == "__main__":
    main()
