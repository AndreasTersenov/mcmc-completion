"""M2 deliverable figure: results/m2_oracle.png
  A: posterior contraction vs K, grid families (theta-dim 2), with/without grads
  B: gmm8 (theta-dim 8, d=8, SMC) — the P-grad readout
  C: sliced-W2^2(posterior predictive, p*) vs K with same-p* baseline
Family-mismatch numbers stay a table in RESULTS.md (three numbers per row tell
the story; no chart needed).
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
CAT = {"gmm2": "#2a78d6", "funnel2": "#1baf7a", "dwell2": "#eda100",
       "gmm8": "#4a3aa7"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "font.size": 10,
})

KS = [8, 32, 128, 512]
con = pd.read_csv(os.path.join(RESULTS, "m2_contraction.csv"))
sw2 = pd.read_csv(os.path.join(RESULTS, "m2_sw2.csv"))


def med_iqr(sub, col="contract_ratio"):
    g = sub.groupby("K")[col]
    return g.median(), g.quantile(0.25), g.quantile(0.75)


def panel_contraction(ax, families, title):
    for fam in families:
        c = CAT[fam]
        for grads, ls in [(True, "-"), (False, "--")]:
            sub = con[(con.family == fam) & (con.T == 1) & (con.grads == grads)]
            med, lo, hi = med_iqr(sub)
            ax.plot(med.index, med.values, ls, color=c, lw=1.8,
                    marker="o", ms=5.5, markeredgecolor=SURFACE,
                    markeredgewidth=0.8)
            ax.fill_between(med.index, lo.values, hi.values, color=c,
                            alpha=0.12, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.set_xlabel("context length K (probes)")
    ax.set_ylabel("posterior std / prior std")
    ax.set_title(title, fontsize=10, color=INK)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    panel_contraction(ax, ["gmm2", "funnel2", "dwell2"],
                      "In-family identification, θ-dim 2 (grid)\n"
                      "solid = with ∇E, dashed = E only; band = IQR over 6 reps")
    for fam, lab in [("gmm2", "GMM"), ("funnel2", "funnel"), ("dwell2", "double-well")]:
        sub = con[(con.family == fam) & (con.T == 1) & con.grads]
        med = sub.groupby("K")["contract_ratio"].median()
        ax.annotate(lab, (KS[-1] * 1.15, med.iloc[-1]), color=CAT[fam],
                    fontsize=9, va="center", annotation_clip=False)

    ax = axes[1]
    panel_contraction(ax, ["gmm8"],
                      "gmm8: θ-dim 8, d=8 (SMC) — the P-grad case\n"
                      "solid = with ∇E, dashed = E only")
    ax.axhline(0.1, color="#e34948", lw=1.0, ls=":")
    ax.annotate("contraction threshold 0.1", (8.6, 0.113), color="#e34948",
                fontsize=8.5)

    ax = axes[2]
    for fam, lab in [("gmm2", "GMM"), ("funnel2", "funnel"), ("dwell2", "double-well")]:
        c = CAT[fam]
        sub = sw2[sw2.family == fam]
        med, lo, hi = med_iqr(sub, "sw2")
        ax.plot(med.index, med.values, "-", color=c, lw=1.8, marker="o",
                ms=5.5, markeredgecolor=SURFACE, markeredgewidth=0.8, label=lab)
        ax.fill_between(med.index, lo.values, hi.values, color=c, alpha=0.12, lw=0)
        base = sub.groupby("K")["sw2_baseline"].median()
        ax.plot(base.index, base.values, ":", color=c, lw=1.2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.set_xlabel("context length K (probes)")
    ax.set_ylabel("sliced-W2²(q, p*)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("Posterior-predictive quality vs K (T=1, with ∇E)\n"
                 "dotted = same-p* sampling floor", fontsize=10, color=INK)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "m2_oracle.png"), dpi=170)
    print("wrote m2_oracle.png")


if __name__ == "__main__":
    main()
