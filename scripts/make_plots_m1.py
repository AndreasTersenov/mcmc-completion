"""M1 deliverable figures.

fig 1  results/m1_frontier.png   — ESS(eps,d) surface (closed form + reliable
       empirical points) and the iso-ESS "capability targets" contours.
fig 2  results/m1_universality.png — ESS/N vs Renyi-2 collapse across all
       in-family mismatch kinds, plus logZ error vs ESS at N=1e4.

Only closed forms and reliable+stable empirical points (gate 4 + 4th-moment
flag) are drawn; that filter is the backpressure rule, not cosmetics.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# reference palette (dataviz skill), light mode
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
# ordinal blue ramp for ordered d (steps 250..700)
D_RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#1c5cab", "#0d366b"]
# categorical slots, fixed order, for mismatch kinds
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "font.size": 10,
})

DS = [2, 4, 8, 16, 32, 64]
df = pd.read_csv(os.path.join(RESULTS, "m1_grid.csv"))
reliable = (df.relsd_analytic < 0.1) & df.ess_dbl_pass


def fig_frontier():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    eps_fine = np.linspace(0.01, 1.05, 300)
    for c, d in zip(D_RAMP, DS):
        ax.plot(eps_fine, np.exp(-d * eps_fine**2), color=c, lw=1.8)
        sub = df[(df.kind == "shift") & (df.d == d) & reliable]
        ax.plot(sub.eps, sub.ess_emp, "o", color=c, ms=5.5,
                markeredgecolor=SURFACE, markeredgewidth=0.8)
        y_end = np.exp(-d * 1.05**2)
        if y_end > 2e-6:
            ax.annotate(f"d={d}", (1.06, y_end), color=c, fontsize=9,
                        va="center", annotation_clip=False)
        else:
            x_lab = np.sqrt(np.log(1 / 2e-6) / d)
            ax.annotate(f"d={d}", (x_lab - 0.03, 3.0e-6), color=c, fontsize=9,
                        ha="right")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.6)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("per-dim mean shift ε (σ units)")
    ax.set_ylabel("ESS / N")
    ax.set_title("ESS surface: mean shift\nlines = exp(−dε²), dots = measured (N=10⁶)",
                 fontsize=10, color=INK)

    ax = axes[1]
    d_fine = np.linspace(2, 64, 400)
    targets = [(0.10, "10%"), (0.01, "1%"), (0.001, "0.1%")]
    shades = ["#cde2fb", "#9ec5f4", "#6da7ec"]
    prev = np.zeros_like(d_fine)
    for (t, lab), shade in zip(targets, shades):
        curve = np.sqrt(np.log(1 / t) / d_fine)
        ax.fill_between(d_fine, prev, curve, color=shade, alpha=0.55, lw=0)
        ax.plot(d_fine, curve, color="#256abf", lw=1.8)
        k = 120
        ax.annotate(f"ESS/N > {lab}", (d_fine[k], curve[k] - 0.028),
                    color="#104281", fontsize=9, ha="left", va="top")
        prev = curve
    ax.plot(8, np.sqrt(np.log(1000) / 8), "o", color="#e34948", ms=7,
            markeredgecolor=SURFACE, markeredgewidth=1)
    ax.annotate("K-M1 test point:\nd=8, 0.1% ⇒ ε* = 0.93", (9, 0.96),
                color="#e34948", fontsize=9)
    ax.set_xlim(2, 64)
    ax.set_ylim(0, 1.35)
    ax.set_xscale("log", base=2)
    ax.set_xticks(DS, [str(d) for d in DS])
    ax.set_xlabel("dimension d")
    ax.set_ylabel("max per-dim shift ε*")
    ax.set_title("Capability targets: iso-ESS frontier\ncertifiable below each curve",
                 fontsize=10, color=INK)

    ax = axes[2]
    from scipy.optimize import brentq

    def log_scale_ess(e, d):
        return (d / 2) * np.log1p(2 * e) - d * np.log1p(e)

    dgrid = np.linspace(2, 64, 200)
    for (t, lab), c in zip(targets, ["#86b6ef", "#3987e5", "#0d366b"]):
        lt = np.log(t)
        pos = np.array([brentq(lambda e: log_scale_ess(e, d) - lt, 1e-9, 1e8)
                        for d in dgrid])
        neg = np.array([brentq(lambda e: log_scale_ess(e, d) - lt, -0.4999999, -1e-9)
                        for d in dgrid])
        ax.plot(dgrid, 1 + pos, color=c, lw=1.8)
        ax.plot(dgrid, 1 + neg, color=c, lw=1.8)
        ax.annotate(f"ESS/N > {lab}", (2.12, (1 + pos[0]) * 1.18), color=c,
                    fontsize=9, va="bottom")
    ax.annotate("underdispersed branch (all three targets)", (5.5, 0.80),
                color=INK2, fontsize=8.5)
    ax.axhline(1.0, color=BASE, lw=0.8, ls=":")
    ax.axhline(0.5, color="#e34948", lw=1.2, ls="--")
    ax.annotate("E[w²] = ∞ below ratio ½", (2.2, 0.40), color="#e34948",
                fontsize=9, va="top")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(DS, [str(d) for d in DS])
    ax.set_xlim(2, 64)
    ax.set_ylim(0.3, 4000)
    ax.set_xlabel("dimension d")
    ax.set_ylabel("proposal/target variance ratio")
    ax.set_title("Variance-mismatch frontier is asymmetric\nerr wide, never narrow",
                 fontsize=10, color=INK)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "m1_frontier.png"), dpi=170)
    plt.close(fig)


def fig_universality():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    kinds = [("shift", "Gaussian: mean shift"), ("scale", "Gaussian: variance"),
             ("weight_distort", "GMM: mode weights"), ("cond_scale", "Funnel: conditionals"),
             ("v_scale", "Funnel: v-scale")]
    x = np.linspace(0, 8, 100)
    ax.plot(x, np.exp(-x), color=INK2, lw=1.4, ls="--")
    ax.annotate("exp(−D₂)", (5.9, np.exp(-5.6)), color=INK2, fontsize=9)
    for (kind, lab), c in zip(kinds, CAT):
        sub = df[(df.kind == kind) & reliable & np.isfinite(df.d2_cf)]
        ax.plot(sub.d2_cf, sub.ess_emp, "o", color=c, ms=5.5, label=lab,
                markeredgecolor=SURFACE, markeredgewidth=0.7)
    ax.axvspan(np.log(1e6) / 4, 8, color="#f0efec", zorder=0)
    ax.annotate("ESS estimate itself unreliable\nat N=10⁶ (e^{4D₂} > N)",
                (3.6, 2e-1), fontsize=8.5, color=MUTED)
    ax.set_yscale("log")
    ax.set_xlim(0, 8)
    ax.set_ylim(1e-4, 2)
    ax.set_xlabel("Rényi-2 divergence D₂(p ∥ q)")
    ax.set_ylabel("measured ESS / N")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("One frontier: ESS/N = exp(−D₂)\nacross target families and mismatch types",
                 fontsize=10, color=INK)

    ax = axes[1]
    sub = df[df.kind.isin(["shift", "scale"]) & np.isfinite(df.ess_cf) & (df.ess_cf > 0)]
    ax.plot(sub.ess_cf, sub.logz_sd_1e4, "o", color=CAT[0], ms=5,
            markeredgecolor=SURFACE, markeredgewidth=0.7, label="sd of log Ẑ")
    ax.plot(sub.ess_cf, np.abs(sub.logz_bias_1e4), "o", color=CAT[2], ms=5,
            markeredgecolor=SURFACE, markeredgewidth=0.7, label="|bias| of log Ẑ")
    ef = np.logspace(-7, 0, 100)
    ax.plot(ef, np.sqrt((1 / ef - 1) / 1e4), color=CAT[0], lw=1.4, ls="--")
    ax.axvline(1e-2, color="#e34948", lw=1.0, ls=":")
    ax.annotate("1% ESS", (1.25e-2, 3e-4), color="#e34948", fontsize=9, rotation=90)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-7, 1.5)
    ax.set_ylim(1e-4, 30)
    ax.set_xlabel("true ESS / N (closed form)")
    ax.set_ylabel("log Ẑ error at N = 10⁴")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("Certificate error vs ESS\ndashes: delta-method √((1/essN−1)/N)",
                 fontsize=10, color=INK)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "m1_universality.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_frontier()
    fig_universality()
    print("wrote m1_frontier.png, m1_universality.png")
