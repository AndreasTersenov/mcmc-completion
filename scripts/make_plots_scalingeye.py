"""The scaling story BY EYE: results/scaling_by_eye.png
Rows = three paired-eval targets; cols = truth, then samples from the
10-target (b1), 128-target (gate3e), 1024-target (train4) checkpoints,
conditioned on the SAME (K=128, T=1) paired-eval contexts. Panel captions
carry each model's measured SW2^2/bespoke ratio from the paired jsons."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, jax.random as jr
from ics.cfm import cond_cfm_sample
from ics.context import generate_context_for_target, whiten_invert
from ics.models import ICSModel
from ics.train import load_checkpoint
from ics.zoo import sample_target, sample_x

R = os.path.join(os.path.dirname(__file__), "..", "results")
SC = os.environ["SCRATCH"]
SURFACE, INK, INK2, MUTED, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": BASE, "text.color": INK,
    "axes.labelcolor": INK2, "axes.grid": False, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9})

model = ICSModel(n_attn=2)
def load(p):
    return jax.tree_util.tree_map(jnp.asarray, load_checkpoint(p)["params"])
P = {"10": load(os.path.join(R, "gate3_noshortk_params.pkl")),
     "128": load(os.path.join(R, "gate3e_params.pkl")),
     "1024": load(os.path.join(SC, "ics-zoo", "ckpt_train4.pkl"))}
STRIP = {"10"}  # 37-dim tokens

pe = json.load(open(os.path.join(R, "paired_eval.json")))       # b1 + gate3e(z128 slot)
p4 = json.load(open(os.path.join(R, "paired_train4.json")))     # gate3e(b1 slot) + train4
FAMS, DS = ("gmm", "dwell", "funnel", "warp"), (2, 4, 8)
cells = [(f, d) for f in FAMS for d in DS]
def prow(js, fam, d, i):
    return next(r for r in js["rows"] if (r["family"], r["d"], r.get("idx", None)) ==
                (fam, d, i) or (r["family"] == fam and r["d"] == d and js["rows"].index(r) % 2 == i))

import os as _os
PICKS = ([("gmm", 2, 0)] if _os.environ.get("PREVIEW") else
         [("gmm", 2, 0), ("dwell", 2, 0), ("warp", 4, 0)])
fig, axes = plt.subplots(len(PICKS), 4, figsize=(13.6, 3.5 * len(PICKS)))
axes = np.atleast_2d(axes)
for row_i, (fam, d, i) in enumerate(PICKS):
    ci = cells.index((fam, d))
    t = sample_target(jr.fold_in(jr.key(424242), 100 * ci + i), fam, d)
    ctx = generate_context_for_target(
        jr.fold_in(jr.key(515151), 100 * ci + 10 * i + 1), t, K=128,
        temperature=1.0, aux_tokens=True)
    truth = np.asarray(sample_x(jr.key(777), t, 20_000))
    lo = np.quantile(truth[:, :2], 0.002, axis=0) - 1.0
    hi = np.quantile(truth[:, :2], 0.998, axis=0) + 1.0
    bins = [np.linspace(lo[k], hi[k], 76) for k in (0, 1)]
    axes[row_i, 0].hist2d(truth[:, 0], truth[:, 1], bins=bins, cmap="Greys")
    axes[row_i, 0].set_title(f"{fam}-d{d} — TRUTH", fontsize=9, color=INK2)
    # measured ratios
    r_pe = [r for r in pe["rows"] if r["family"] == fam and r["d"] == d][i]
    r_p4 = [r for r in p4["rows"] if r["family"] == fam and r["d"] == d][i]
    ratios = {"10": r_pe["b1_T1"]["ratio"], "128": r_p4["b1_T1"]["ratio"],
              "1024": r_p4["z128_T1"]["ratio"]}
    for col_j, tag in enumerate(["10", "128", "1024"], start=1):
        toks = ctx.tokens
        if tag in STRIP:
            toks = jnp.concatenate([toks[:, :-5], toks[:, -1:]], axis=1)
        xw = cond_cfm_sample(model, P[tag], toks.astype(jnp.float64),
                             jr.key(31_000 + row_i), n=int(_os.environ.get("NSAMP", 2500)), n_steps=int(_os.environ.get("NSTEP", 60)))
        xs = np.asarray(whiten_invert(xw[:, :d], ctx.mu, ctx.sigma))
        ax = axes[row_i, col_j]
        ax.hist2d(xs[:, 0], xs[:, 1], bins=bins, cmap="Blues")
        ax.set_title(f"{tag}-target zoo — SW2²/bespoke: {ratios[tag]:.0f}",
                     fontsize=9, color=INK2)
    for ax in axes[row_i]:
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("The amortization curve, by eye — same targets, same contexts (K=128, T=1),\n"
             "three generations of training zoo; captions = measured paired ratios",
             fontsize=11, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(R, _os.environ.get("OUTNAME", "scaling_by_eye.png")), dpi=160)
print("wrote scaling_by_eye.png")
