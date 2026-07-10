"""Explainer figure: results/explainer.png — what the in-context sampler IS,
shown on the real gmm-d2 diagnostic target with the real b1 checkpoint.

A: the problem — a target density the model has never seen, and the ONLY
   thing it receives: 4 short MALA chains (positions + energies).
B: what the model emits given that context (T=1) — and how the certificate
   prices the missing modes (measured numbers from the gate runs).
C: same model, tempered probes (T=5) — coverage, and the certificate agrees.
D: the campaign map in plain language.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

from ics.cfm import cond_cfm_sample
from ics.context import generate_context_for_target, whiten_invert
from ics.models import ICSModel
from ics.train import build_zoo_dataset, load_checkpoint
from ics.zoo import logpdf, mode_centers

R = os.path.join(os.path.dirname(__file__), "..", "results")
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, RED, AQUA, VIOLET = "#2a78d6", "#e34948", "#1baf7a", "#4a3aa7"
CHAIN_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def strip_onehot_tokens(tokens):
    return jnp.concatenate([tokens[:, :-5], tokens[:, -1:]], axis=1)


def main():
    # the real diagnostic target (6-mode gmm-d2) + the real b1 checkpoint
    targets, _, _ = build_zoo_dataset(jr.key(31), [("gmm", 2)], n_ctx=1, K=8,
                                      n_pool=8)
    t = targets[0]
    model = ICSModel(n_attn=2)
    params = jax.tree_util.tree_map(
        jnp.asarray,
        load_checkpoint(os.path.join(R, "gate3_noshortk_params.pkl"))["params"])

    ctx1 = generate_context_for_target(jr.key(9000), t, K=128, temperature=1.0,
                                       aux_tokens=True)
    ctx5 = generate_context_for_target(jr.key(9000), t, K=128, temperature=5.0,
                                       aux_tokens=True)

    def model_samples(ctx, seed):
        toks = strip_onehot_tokens(ctx.tokens).astype(jnp.float64)
        xw = cond_cfm_sample(model, params, toks, jr.key(seed), n=1500, n_steps=80)
        return np.asarray(whiten_invert(xw[:, :2], ctx.mu, ctx.sigma))

    s1 = model_samples(ctx1, 61)
    s5 = model_samples(ctx5, 62)

    # density grid
    g = np.linspace(-9, 9, 301)
    X, Y = np.meshgrid(g, g, indexing="ij")
    Z = np.exp(np.asarray(logpdf(t, jnp.asarray(np.stack([X.ravel(), Y.ravel()], 1)))
                          ).reshape(301, 301))
    mc = np.asarray(mode_centers(t))
    lv = np.max(Z) * np.array([0.02, 0.1, 0.35, 0.7])

    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.5))

    # ---- A: the problem
    ax = axes[0]
    ax.contour(X, Y, Z, levels=lv, colors=MUTED, linewidths=0.9)
    pts = np.asarray(ctx1.x_raw).reshape(4, 32, 2)
    for c_i in range(4):
        ax.plot(pts[c_i, :, 0], pts[c_i, :, 1], "-", color=CHAIN_COLORS[c_i],
                lw=0.9, alpha=0.7)
        ax.plot(pts[c_i, :, 0], pts[c_i, :, 1], "o", color=CHAIN_COLORS[c_i],
                ms=3.2)
        ax.plot(*pts[c_i, 0], "s", color=CHAIN_COLORS[c_i], ms=7,
                markeredgecolor=SURFACE)
    for m in mc:
        ax.plot(*m, "x", color=INK, ms=7, markeredgewidth=1.6)
    ax.annotate("x = true modes\n(unknown to the model)", (0.03, 0.97),
                xycoords="axes fraction", va="top", fontsize=8.5, color=INK2)
    ax.set_title("THE TASK — a never-seen target (contours)\n"
                 "model receives ONLY: 4 short chains, 32 steps each\n"
                 "(positions + energies + gradients; squares = starts)",
                 fontsize=9.5, color=INK)
    ax.set_xlim(-9, 9); ax.set_ylim(-9, 9)

    # ---- B: what it emits (T=1 context) + certificate
    ax = axes[1]
    ax.contour(X, Y, Z, levels=lv, colors=MUTED, linewidths=0.9)
    ax.plot(s1[:, 0], s1[:, 1], "o", color=BLUE, ms=2.2, alpha=0.45)
    for m in mc:
        ax.plot(*m, "x", color=INK, ms=7, markeredgewidth=1.6)
    ax.annotate("certificate (measured):\nESS = 8%,  logẐ = −0.77\n"
                "= ln(46%) — it PRICES the\nmass the samples miss",
                (0.03, 0.03), xycoords="axes fraction", fontsize=8.5,
                color=RED, va="bottom")
    ax.set_title("WHAT IT EMITS — 1500 samples, zero retraining\n"
                 "chains visited 4/6 modes → model covers those 4\n"
                 "(the certificate sees exactly what's missing)",
                 fontsize=9.5, color=INK)
    ax.set_xlim(-9, 9); ax.set_ylim(-9, 9)

    # ---- C: tempered probes fix coverage
    ax = axes[2]
    ax.contour(X, Y, Z, levels=lv, colors=MUTED, linewidths=0.9)
    ax.plot(s5[:, 0], s5[:, 1], "o", color=AQUA, ms=2.2, alpha=0.45)
    for m in mc:
        ax.plot(*m, "x", color=INK, ms=7, markeredgewidth=1.6)
    ax.annotate("certificate (measured):\nESS = 18%,  logẐ = −0.045 ≈ 0\n"
                "coverage fixed → certificate\nagrees, zero-shot",
                (0.03, 0.03), xycoords="axes fraction", fontsize=8.5,
                color="#0a7a4d", va="bottom")
    ax.set_title("SAME MODEL, HOTTER PROBES (T=5)\n"
                 "tempered chains hop between modes →\n"
                 "context covers → samples cover",
                 fontsize=9.5, color=INK)
    ax.set_xlim(-9, 9); ax.set_ylim(-9, 9)

    # ---- D: the campaign map
    ax = axes[3]
    ax.axis("off")
    rows = [
        ("stage-0", "PASSED", "the physics: certificate honesty is exactly\n"
         "ESS/N = e^(-D2); it prices covered mass", AQUA),
        ("gate (i)", "PASSED", "can the flow learn ONE target?\nESS 98%", AQUA),
        ("gate (ii)", "PASSED", "does context routing work?\nwrong context → 1000× worse", AQUA),
        ("gate (iii)", "RE-SCOPED", "10→128-target studies; found: shortK bug,\n"
         "3 information ceilings, paired instrument", "#eda100"),
        ("PAIRED-B", "KEY RESULT", "more training targets → better on EACH\n"
         "target (18/24), at same total compute", VIOLET),
        ("gate (iv)", "RUNNING NOW", "1024 targets, 3 arms + baselines:\n"
         "does prior-fitting SCALE? (the thesis)", RED),
    ]
    y = 0.98
    for name, status, desc, c in rows:
        ax.text(0.02, y, name, fontsize=10, fontweight="bold", color=INK,
                va="top", transform=ax.transAxes)
        ax.text(0.30, y, status, fontsize=9, fontweight="bold", color=c,
                va="top", transform=ax.transAxes)
        ax.text(0.02, y - 0.045, desc, fontsize=8.2, color=INK2, va="top",
                transform=ax.transAxes)
        y -= 0.165
    ax.set_title("THE CAMPAIGN — one trained network that samples\n"
                 "new distributions from a glimpse, with honest\n"
                 "error bars it cannot fake", fontsize=9.5, color=INK)

    fig.tight_layout()
    out = os.path.join(R, "explainer.png")
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main()
