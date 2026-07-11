"""Phase-1b decision figure: results/phase1b_curves.png — json-only, no GPU.
A: the trained-target composite curve (the branch decider) + the frozen 50% bar.
B: fresh-theta paired median SW2^2/bespoke across checkpoints (log) + the FQ=4x
   improvement target derived from the 0.2M baseline.
C: the usefulness barometer — certified ESS on the three REAL targets at
   200k vs 2M (the dissociation: A/B flat, C jumps).
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, RED, AQUA, VIOLET, GOLD = "#2a78d6", "#e34948", "#1baf7a", "#4a3aa7", "#eda100"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "font.size": 10,
})

TAGS = ["0.2M", "0.25M", "0.5M", "1M", "1.5M", "2M"]
STEPS = [0.2, 0.25, 0.5, 1.0, 1.5, 2.0]


def main():
    ec = json.load(open(os.path.join(R, "eval_curve.json")))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    # ---- A: TT composite (the branch decider)
    ax = axes[0]
    for col, c, lab in (("T1", BLUE, "(K=128, T=1) — scoring column"),
                        ("T5", MUTED, "(K=128, T=5)")):
        y = [100 * ec["blocks"][t][f"tt_frac_{col}"] for t in TAGS]
        ax.plot(STEPS, y, "-o", color=c, lw=1.8, ms=6.5,
                markeredgecolor=SURFACE, label=lab)
    ax.axhline(50, ls="--", lw=1.2, color=RED)
    ax.annotate("ALIVE/MEMORIZE needed TT ≥ 50%", (0.22, 52), color=RED,
                fontsize=8.5)
    ax.annotate("measured: flat, never rises\n→ STRUCTURAL (plateau before 1M)\n"
                "→ capacity arm fired (2× width)", (0.6, 30), color=INK2,
                fontsize=8.5)
    ax.set_ylim(0, 60)
    ax.set_xlabel("training steps (millions), one variable")
    ax.set_ylabel("trained-target composite pass %")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("A — 10× compute does NOT move trained-target\nquality (the branch decider)",
                 fontsize=10, color=INK)

    # ---- B: fresh-theta paired median
    ax = axes[1]
    for col, c, lab in (("T1", BLUE, "T=1"), ("T5", RED, "T=5")):
        y = [ec["blocks"][t][f"fresh_med_ratio_{col}"] for t in TAGS]
        ax.plot(STEPS, y, "-o", color=c, lw=1.8, ms=6.5,
                markeredgecolor=SURFACE, label=lab)
    y0 = ec["blocks"]["0.2M"]["fresh_med_ratio_T1"]
    ax.axhline(y0 / 4, ls="--", lw=1.2, color=RED)
    ax.annotate(f"FQ target: 4× better than 0.2M ({y0:.0f}→{y0/4:.0f})",
                (0.22, y0 / 4 * 0.78), color=RED, fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel("fresh-θ median SW2² / bespoke")
    ax.legend(fontsize=8)
    ax.set_title("B — fresh-target sharpness: also flat\n(measured FQ = 0.85×; zoo axis was 1.6×/8×)",
                 fontsize=10, color=INK)

    # ---- C: the barometer (real targets)
    ax = axes[2]
    c200 = json.load(open(os.path.join(R, "readout_c_200k.json")))["targets"]
    c2m = json.load(open(os.path.join(R, "readout_c_2M.json")))["targets"]
    names = [("eight_schools", "eight-schools\n(real data, d=10)"),
             ("gym_banana", "gym banana\n(d=2)"),
             ("wl_bandpower", "WL band-power\n(d=3, needle)")]
    x = np.arange(3)
    for off, src, c, lab in ((-0.18, c200, MUTED, "200k"),
                             (0.18, c2m, BLUE, "2M")):
        vals = [100 * src[n]["T1"]["ess_frac_2n"] for n, _ in names]
        stab = [src[n]["T1"]["stable"] for n, _ in names]
        bars = ax.bar(x + off, np.maximum(vals, 0.006), width=0.34, color=c,
                      label=lab, edgecolor=SURFACE)
        for xi, v, s in zip(x + off, vals, stab):
            ax.annotate(f"{v:.2f}%" + ("" if s else "\nunstable"),
                        (xi, max(v, 0.006) * (1.25 if s else 1.6)), ha="center",
                        fontsize=6.8, color=INK if s else RED, rotation=0 if s else 30)
    ax.axhline(1.0, ls="--", lw=1.2, color=AQUA)
    ax.annotate("barometer clause: ≥1% certified & stable", (0.6, 1.25),
                color="#0a7a4d", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(0.005, 300)
    ax.set_xticks(x, [n for _, n in names], fontsize=8.5)
    ax.set_ylabel("certified ESS (T=1 column), %")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("C — same compute lever, REAL targets zero-shot:\nbanana 0.03%→33% — the dissociation",
                 fontsize=10, color=INK)

    fig.tight_layout()
    out = os.path.join(R, "phase1b_curves.png")
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main()
