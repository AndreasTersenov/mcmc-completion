"""Diagnostic: are panel-3's off-target groupings the model's, or the
whitening geometry's? Compare T=5 samples in whitened space (the model's
working coordinates) vs whitened truth, alongside raw space with the
whitening ellipse. results/whitening_diag.png"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, jax.random as jr
from ics.cfm import cond_cfm_sample
from ics.context import generate_context, whiten_apply
from ics.models import ICSModel
from ics.train import build_zoo_dataset, load_checkpoint
from ics.zoo import logpdf, sample_x

R = os.path.join(os.path.dirname(__file__), "..", "results")
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BASE, AQUA, RED = "#c3c2b7", "#1baf7a", "#e34948"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": BASE,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": False, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "legend.frameon": False})

targets, _, _ = build_zoo_dataset(jr.key(31), [("gmm", 2)], n_ctx=1, K=8, n_pool=8)
t = targets[0]
model = ICSModel(n_attn=2)
params = jax.tree_util.tree_map(
    jnp.asarray, load_checkpoint(os.path.join(R, "gate3_noshortk_params.pkl"))["params"])
fn = lambda x: logpdf(t, x)
ctx5 = generate_context(jr.key(9000), fn, 2, K=128, temperature=5.0, aux_tokens=True)
toks = jnp.concatenate([ctx5.tokens[:, :-5], ctx5.tokens[:, -1:]], axis=1).astype(jnp.float64)
xw = np.asarray(cond_cfm_sample(model, params, toks, jr.key(62), n=1500, n_steps=100))[:, :2]
truth = np.asarray(sample_x(jr.key(99), t, 1500))
truth_w = np.asarray(whiten_apply(jnp.asarray(truth), ctx5.mu, ctx5.sigma))
mu, sg = np.asarray(ctx5.mu), np.asarray(ctx5.sigma)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8),
                         gridspec_kw={'width_ratios': [1, 2.2]})
ax = axes[0]
ax.plot(truth_w[:, 0], truth_w[:, 1], "o", ms=2.4, alpha=0.4, color=MUTED,
        label="truth (whitened)")
ax.plot(xw[:, 0], xw[:, 1], "o", ms=2.4, alpha=0.4, color=AQUA,
        label="model output (native space)")
ax.set_aspect("equal")
# adjacent-mode separation in the model's coordinates
from ics.zoo import mode_centers
mcw = (np.asarray(mode_centers(t)) - mu) / sg
# the ACTUAL closest mode pair in whitened space (computed, not hand-picked)
dists = np.linalg.norm(mcw[:, None, :] - mcw[None, :, :], axis=-1)
np.fill_diagonal(dists, np.inf)
i, j = np.unravel_index(np.argmin(dists), dists.shape)
ax.annotate("", xy=tuple(mcw[j]), xytext=tuple(mcw[i]),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
ax.annotate(f"closest mode pair:\n{dists[i, j]:.2f} whitened units —\n"
            "comparable to the blur",
            (0.02, 0.02), xycoords="axes fraction", color=RED, fontsize=8.5,
            va="bottom")
ax.legend(fontsize=8.5, loc="upper left")
ax.set_title("The model's OWN coordinates (equal aspect)\nwhitening SQUEEZES "
             f"the target: σ = [{sg[0]:.1f}, {sg[1]:.1f}]", fontsize=9.5, color=INK)
ax.set_xlabel("whitened x₁"); ax.set_ylabel("whitened x₂")
ax = axes[1]
xr = mu + sg * xw
ax.plot(truth[:, 0], truth[:, 1], "o", ms=2.4, alpha=0.4, color=MUTED,
        label="truth (raw)")
ax.plot(xr[:, 0], xr[:, 1], "o", ms=2.4, alpha=0.4, color=AQUA,
        label="model (de-whitened)")
th = np.linspace(0, 2*np.pi, 100)
ax.plot(mu[0] + sg[0]*np.cos(th), mu[1] + sg[1]*np.sin(th), "--", lw=1.4,
        color=RED, label="context whitening ellipse (1σ)")
ax.set_aspect("equal")
ax.legend(fontsize=8.5, loc="upper left")
ax.set_title("Raw space (equal aspect): SAME clouds, de-whitened —\nevery "
             "error stretched 6.5× horizontally, 2.3× vertically",
             fontsize=9.5, color=INK)
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
fig.tight_layout()
fig.savefig(os.path.join(R, "whitening_diag.png"), dpi=170)
print("wrote whitening_diag.png")
