"""Sample gallery: results/sample_gallery.png — the train4 model's samples,
BY EYE, on fresh eval targets (median-ESS representative per cell),
conditioned on the exact stored T=5 contexts the eval measured."""
import json, os, pickle, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, jax.random as jr
from ics.cfm import cond_cfm_sample
from ics.context import Context, whiten_invert
from ics.models import ICSModel
from ics.train import load_checkpoint
from ics.zoo import logpdf, sample_x

R = os.path.join(os.path.dirname(__file__), "..", "results")
SURFACE, INK, INK2, MUTED, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
COL = {"gmm": "#2a78d6", "dwell": "#eda100", "funnel": "#1baf7a",
       "warp": "#4a3aa7", "banana": "#e34948", "funnelmix": "#d55181"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": BASE,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": False, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9})

rows_meta = json.load(open(os.path.join(R, "eval_train4.json")))
evalset = pickle.load(open(os.path.join(os.environ["SCRATCH"], "ics-zoo", "eval",
                                        "eval_set.pkl"), "rb"))
model = ICSModel(n_attn=2)
params = jax.tree_util.tree_map(
    jnp.asarray, load_checkpoint(os.path.join(os.environ["SCRATCH"],
                                              "ics-zoo", "ckpt_train4.pkl"))["params"])

CELLS = [("gmm", 2), ("warp", 2), ("dwell", 2), ("banana", 2),
         ("funnelmix", 2), ("gmm", 4)]

def median_idx(fam, d):
    cand = [r for r in rows_meta if r["family"] == fam and r["d"] == d
            and np.isfinite(r["t5"]["sw2"])]
    cand.sort(key=lambda r: r["t5"]["ess_frac_2n"])
    return cand[len(cand) // 2]

fig, axes = plt.subplots(3, 4, figsize=(14.5, 10.2))
pairs = [(axes[i // 2, 2 * (i % 2)], axes[i // 2, 2 * (i % 2) + 1])
         for i in range(6)]
for (axT, axM), (fam, d) in zip(pairs, CELLS):
    meta = median_idx(fam, d)
    row = next(r for r in evalset if (r["family"], r["d"], r["idx"]) ==
               (fam, d, meta["idx"]))
    t = row["target"]
    ctx = Context(**{k: jnp.asarray(v) for k, v in row["context_t5"]._asdict().items()})
    xw = cond_cfm_sample(model, params, ctx.tokens.astype(jnp.float64),
                         jr.key(4242), n=4000, n_steps=60)
    xs = np.asarray(whiten_invert(xw[:, :d], ctx.mu, ctx.sigma))
    truth = np.asarray(sample_x(jr.key(4243), t, 20_000))
    # robust shared limits: truth quantiles padded (model outliers can explode)
    lo = np.quantile(truth[:, :2], 0.002, axis=0) - 1.5
    hi = np.quantile(truth[:, :2], 0.998, axis=0) + 1.5
    bins = [np.linspace(lo[k], hi[k], 81) for k in (0, 1)]
    axT.hist2d(truth[:, 0], truth[:, 1], bins=bins, cmap="Greys")
    axM.hist2d(xs[:, 0], xs[:, 1], bins=bins, cmap="Blues")
    inside = float(np.mean((xs[:, 0] >= lo[0]) & (xs[:, 0] <= hi[0])
                           & (xs[:, 1] >= lo[1]) & (xs[:, 1] <= hi[1])))
    held = " (NEVER TRAINED ON)" if meta["heldout"] else ""
    c5 = meta["t5"]
    tcol = INK2 if not meta["heldout"] else COL["banana"]
    axT.set_title(f"{fam}-d{d}{held}\nTRUTH (exact samples)", fontsize=8.5, color=tcol)
    axM.set_title(f"MODEL — ESS {c5['ess_frac_2n']*100:.2g}%, logẐ {c5['logz']:+.2f},"
                  f" SW2² {c5['sw2']:.2f}\n({100*inside:.0f}% of model mass in frame)",
                  fontsize=8.5, color=tcol)
    for a in (axT, axM):
        a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1])
        a.set_xticks([]); a.set_yticks([])
fig.suptitle("TRUTH vs MODEL as densities — fresh targets, median-ESS pick per cell,\n"
             "same axes per pair; measured certificate per panel (T=5 column)",
             fontsize=11, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(R, "sample_gallery.png"), dpi=160)
print("wrote sample_gallery.png")
