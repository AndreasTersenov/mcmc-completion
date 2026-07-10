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

fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.6))
for ax, (fam, d) in zip(axes.ravel(), CELLS):
    meta = median_idx(fam, d)
    row = next(r for r in evalset if (r["family"], r["d"], r["idx"]) ==
               (fam, d, meta["idx"]))
    t = row["target"]
    ctx = Context(**{k: jnp.asarray(v) for k, v in row["context_t5"]._asdict().items()})
    xw = cond_cfm_sample(model, params, ctx.tokens.astype(jnp.float64),
                         jr.key(4242), n=800, n_steps=60)
    xs = np.asarray(whiten_invert(xw[:, :d], ctx.mu, ctx.sigma))
    truth = np.asarray(sample_x(jr.key(4243), t, 800))
    lo = np.minimum(truth.min(0), xs.min(0))[:2] - 1
    hi = np.maximum(truth.max(0), xs.max(0))[:2] + 1
    if d == 2:
        g0, g1 = np.linspace(lo[0], hi[0], 161), np.linspace(lo[1], hi[1], 161)
        X, Y = np.meshgrid(g0, g1, indexing="ij")
        Z = np.exp(np.asarray(logpdf(t, jnp.asarray(np.stack([X.ravel(), Y.ravel()], 1)))
                              ).reshape(161, 161))
        ax.contour(X, Y, Z, levels=np.max(Z) * np.array([0.03, 0.15, 0.5]),
                   colors=MUTED, linewidths=0.8)
    else:
        ax.plot(truth[:, 0], truth[:, 1], "o", ms=2.6, alpha=0.35, color=MUTED)
    ax.plot(xs[:, 0], xs[:, 1], "o", ms=2.8, alpha=0.5, color=COL[fam])
    held = " (NEVER TRAINED ON)" if meta["heldout"] else ""
    c5 = meta["t5"]
    ax.set_title(f"{fam}-d{d}{held}\nmeasured: ESS {c5['ess_frac_2n']*100:.2g}%, "
                 f"logẐ {c5['logz']:+.2f}, SW2² {c5['sw2']:.2f}",
                 fontsize=9, color=INK2 if not meta["heldout"] else COL["banana"])
    if d > 2:
        ax.set_xlabel("first 2 of %d coords (grey = exact samples)" % d, fontsize=8)
fig.suptitle("The 1024-target model on FRESH targets, by eye — median-ESS pick per cell,\n"
             "conditioned on the exact contexts the eval measured (T=5 column)",
             fontsize=11, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(R, "sample_gallery.png"), dpi=160)
print("wrote sample_gallery.png")
