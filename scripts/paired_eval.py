"""PAIRED EVAL (reconvene-endorsed deciding evidence; pre-registration in
log/2026-07-10-paired-eval.md): both existing checkpoints — b1 (10-target,
37-dim tokens) and gate3e (128-target, 41-dim tokens) — evaluated on ONE
common set of 24 fresh targets (2 per cell, 4 families x d {2,4,8}), both
(K,T) columns ((128,1) and (128,5); funnels K=512), SHARED bespoke
references and SHARED contexts (tokens built once at 41 dims; b1 receives
the one-hot-stripped view). Writes results/paired_eval.json.
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

FAMS = ("gmm", "dwell", "funnel", "warp")
DS = (2, 4, 8)
N_EVAL = 4096
K = 128


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


def strip_onehot(ctx):
    t_ = jnp.concatenate([ctx.tokens[:, :-5], ctx.tokens[:, -1:]], axis=1)
    return ctx._replace(tokens=t_)


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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--z128-ckpt", default=None, help="override 128-model ckpt path")
    ap.add_argument("--out", default="paired_eval.json")
    args = ap.parse_args()
    t0 = time.time()
    base = os.path.join(os.path.dirname(__file__), "..", "results")
    model = ICSModel(n_attn=2)  # identical arch for both checkpoints
    pb1 = jax.tree_util.tree_map(
        jnp.asarray, load_checkpoint(os.path.join(base, "gate3_noshortk_params.pkl"))["params"])
    p128 = jax.tree_util.tree_map(
        jnp.asarray, load_checkpoint(args.z128_ckpt or os.path.join(base, "gate3e_params.pkl"))["params"])

    rows = []
    n = 0
    for ci, (f, d) in enumerate([(f, d) for f in FAMS for d in DS]):
        for i in range(2):
            t = sample_target(jr.fold_in(jr.key(424242), 100 * ci + i), f, d)
            k_eval = 512 if f == "funnel" else K
            ref = None
            row = dict(family=f, d=d, idx=i)
            for tag, temp in (("T1", 1.0), ("T5", 5.0)):
                ctx = generate_context_for_target(
                    jr.fold_in(jr.key(515151), 100 * ci + 10 * i + int(temp)),
                    t, K=k_eval, temperature=temp, aux_tokens=True)
                if ref is None:
                    ref = bespoke_ref_sw2(t, ctx, 600_000 + 1000 * ci + 100 * i)
                    row["sw2_ref"] = ref
                s0 = 900_000 + 1000 * ci + 100 * i + int(temp)
                row[f"b1_{tag}"] = eval_one(model, pb1, t, strip_onehot(ctx), s0, ref)
                row[f"z128_{tag}"] = eval_one(model, p128, t, ctx, s0 + 10, ref)
            rows.append(row)
            n += 1
            print(f"[{n}/24] {f}-d{d}#{i} ref={ref:.4f} "
                  f"b1T1={row['b1_T1']['ratio']:.0f} z128T1={row['z128_T1']['ratio']:.0f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    def med(model_tag, col):
        return float(np.median([r[f"{model_tag}_{col}"]["ratio"] for r in rows]))

    paired_t1 = [r["z128_T1"]["ratio"] / max(r["b1_T1"]["ratio"], 1e-9) for r in rows]
    out = dict(
        rows=rows,
        med_b1_T1=med("b1", "T1"), med_z128_T1=med("z128", "T1"),
        med_b1_T5=med("b1", "T5"), med_z128_T5=med("z128", "T5"),
        med_paired_T1=float(np.median(paired_t1)),
        pass_b1_T1=sum(r["b1_T1"]["passed"] for r in rows),
        pass_z128_T1=sum(r["z128_T1"]["passed"] for r in rows),
        pass_b1_T5=sum(r["b1_T5"]["passed"] for r in rows),
        pass_z128_T5=sum(r["z128_T5"]["passed"] for r in rows),
        seconds=round(time.time() - t0, 1),
    )
    with open(os.path.join(base, args.out), "w") as fo:
        json.dump(out, fo, indent=2)
    print({k: v for k, v in out.items() if k != "rows"})
    # pre-registered verdict rule (log/2026-07-10-paired-eval.md)
    improv = out["med_b1_T1"] / max(out["med_z128_T1"], 1e-9)
    verdict = "PAIRED-B-AMORTIZATION" if improv >= 1.5 else "PAIRED-A-COMPUTE"
    print(f"paired T1 median improvement (b1/z128) = {improv:.2f}x -> {verdict}")


if __name__ == "__main__":
    main()
