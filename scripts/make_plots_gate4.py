"""Gate-(iv) status figure: results/gate4_status.png
A: the paired instrument (PAIRED-B) — per-target b1-vs-128 ratios, log-log.
B: live arm training curves (train4/train2/train4ng slots concatenated) vs
   the gate3e 128-target reference.
C: the amortization curve so far — paired median SW2/bespoke at 10 and 128
   targets (common instrument), 1024-target arms pending.
"""

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results")
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
CAT = {"gmm": "#2a78d6", "dwell": "#eda100", "funnel": "#1baf7a", "warp": "#4a3aa7"}
RED = "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "font.size": 10,
})


def loss_curve(pattern):
    steps, losses = [], []
    for path in sorted(glob.glob(os.path.join(ROOT, "jobout", pattern))):
        for m in re.finditer(r"step (\d+): loss ([0-9.]+)", open(path).read()):
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    order = np.argsort(steps)
    s, l = np.asarray(steps)[order], np.asarray(losses)[order]
    keep = np.concatenate([[True], np.diff(s) > 0])  # dedupe resumed overlaps
    return s[keep], l[keep]


def main():
    pe = json.load(open(os.path.join(R, "paired_eval.json")))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # ---- A: the paired instrument
    ax = axes[0]
    for row in pe["rows"]:
        c = CAT[row["family"]]
        ax.plot(row["b1_T1"]["ratio"], row["z128_T1"]["ratio"], "o", ms=7,
                color=c, markeredgecolor=SURFACE, markeredgewidth=0.8)
    lim = np.logspace(0, 4.8, 10)
    ax.plot(lim, lim, ls="--", lw=1.3, color=INK2)
    ax.fill_between(lim, lim * 1e-4, lim, color="#cde2fb", alpha=0.35, lw=0)
    ax.annotate("128-target model better\n(18/24, sign-test p≈0.011)",
                (2.2, 1.8e-1 * 20), color="#104281", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 6e4)
    ax.set_ylim(1, 6e4)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f)
               for f, c in CAT.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.set_xlabel("10-target model:  SW2² / bespoke")
    ax.set_ylabel("128-target model:  SW2² / bespoke")
    ax.set_title("The paired instrument (PAIRED-B)\nsame 24 fresh targets, "
                 "shared contexts & references", fontsize=10, color=INK)

    # ---- B: live arm loss curves
    ax = axes[1]
    curves = [
        ("train4 (live)", "train_train4_*.out", "#104281"),
        ("train2 (live)", "train_train2_*.out", "#1baf7a"),
        ("train4ng (live)", "train_train4ng_*.out", "#eda100"),
        ("gate3e ref (128 targets, 200k)", "gate3e_*.out", MUTED),
    ]
    for lab, pat, c in curves:
        s, l = loss_curve(pat)
        if len(s):
            ls = ":" if "ref" in lab else "-"
            ax.plot(s / 1e6, l, ls, lw=1.6, color=c, label=lab)
    ax.axvline(1.6, ls="--", lw=1.0, color=BASE)
    ax.annotate("step budget", (1.615, 1.52), color=MUTED, fontsize=8.5,
                rotation=90)
    ax.set_xlabel("training step (millions)")
    ax.set_ylabel("conditional CFM loss")
    ax.set_ylim(1.2, 1.6)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Gate-(iv) arms, live\n1024 targets, 260 steps/pair held at "
                 "gate3e level", fontsize=10, color=INK)

    # ---- C: the amortization curve so far
    ax = axes[2]
    zoo = [10, 128]
    t1 = [pe["med_b1_T1"], pe["med_z128_T1"]]
    t5 = [pe["med_b1_T5"], pe["med_z128_T5"]]
    ax.plot(zoo, t1, "-o", color="#2a78d6", lw=1.8, ms=8,
            markeredgecolor=SURFACE, label="(K=128, T=1) column")
    ax.plot(zoo, t5, "-o", color=RED, lw=1.8, ms=8,
            markeredgecolor=SURFACE, label="(K=128, T=5) column")
    for y, c in ((t1[-1] / 1.65, "#2a78d6"), (t5[-1] / 2.5, RED)):
        ax.plot(1024, y, "o", ms=9, markerfacecolor=SURFACE,
                markeredgecolor=c, markeredgewidth=1.6)
    ax.annotate("1024: arms running\n(P-scale, 70%: keeps falling)",
                (1024, t5[-1] / 2.5 * 0.45), color=INK2, fontsize=8.5,
                ha="right")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([10, 128, 1024], ["10", "128", "1024"])
    ax.set_xlabel("zoo size (targets), compute per pair fixed")
    ax.set_ylabel("paired median  SW2² / bespoke")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("The amortization curve (the thesis question)\n"
                 "paired medians on the common fresh-θ set", fontsize=10,
                 color=INK)

    fig.tight_layout()
    out = os.path.join(R, "gate4_status.png")
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main()
