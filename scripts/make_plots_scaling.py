"""The scaling result: results/scaling_result.png
A: the amortization curve, all three points real (10 -> 128 -> 1024).
B: where the gain lives — family-split paired medians (diversity, P12/P11)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, RED, AQUA, YEL = "#2a78d6", "#e34948", "#1baf7a", "#eda100"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": BASE,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "legend.frameon": False})

pe = json.load(open(os.path.join(R, "paired_eval.json")))
p4 = json.load(open(os.path.join(R, "paired_train4.json")))
p2 = json.load(open(os.path.join(R, "paired_train2.json")))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
ax = axes[0]
zoo = [10, 128, 1024]
t1 = [pe["med_b1_T1"], p4["med_b1_T1"], p4["med_z128_T1"]]
t5 = [pe["med_b1_T5"], p4["med_b1_T5"], p4["med_z128_T5"]]
ax.plot(zoo, t1, "-o", color=BLUE, lw=2, ms=9, markeredgecolor=SURFACE,
        markeredgewidth=1, label="(K=128, T=1) column")
ax.plot(zoo, t5, "-o", color=RED, lw=2, ms=9, markeredgecolor=SURFACE,
        markeredgewidth=1, label="(K=128, T=5) column")
for x, y, lab in [(10, t1[0], "905"), (128, t1[1], "561"), (1024, t1[2], "345")]:
    ax.annotate(lab, (x, y * 1.13), ha="center", fontsize=9, color=BLUE)
ax.annotate("P-scale (registered 70%): PASS\n1.62× at the 1024 step",
            (0.5, 0.16), xycoords="axes fraction", fontsize=9, color=INK2)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(zoo, [str(z) for z in zoo])
ax.set_xlabel("training-zoo size (targets), compute per pair FIXED")
ax.set_ylabel("paired median  SW2² / bespoke  (fresh θ)")
ax.legend(fontsize=8.5, loc="upper right")
ax.set_title("The amortization curve — all points measured\n"
             "same 24 fresh targets, shared refs, paired", fontsize=10, color=INK)

ax = axes[1]
def med(rows, fams, key):
    return float(np.median([r[key]["ratio"] for r in rows if r["family"] in fams]))
groups = [("in-family\n(gmm + funnel)", ("gmm", "funnel")),
          ("dwell + warp\n(train2 never saw these)", ("dwell", "warp"))]
models = [("gate3e (128, 4 fams)", "b1_T1", p4, MUTED),
          ("train4 (1024, 4 fams)", "z128_T1", p4, BLUE),
          ("train2 (1024, 2 fams)", "z128_T1", p2, YEL)]
x = np.arange(2)
for i, (lab, key, src, c) in enumerate(models):
    vals = [med(src["rows"], fams, key) for _, fams in groups]
    ax.bar(x + (i - 1) * 0.26, vals, 0.24, color=c, label=lab,
           edgecolor=SURFACE, linewidth=0.6)
ax.set_yscale("log")
ax.set_xticks(x, [g for g, _ in groups])
ax.set_ylabel("paired median  SW2² / bespoke")
ax.legend(fontsize=8.5, loc="upper left")
ax.set_title("Where scaling helps — and what diversity buys\n"
             "2-family arm: fine in-family, 2× worse cross-family (P12)",
             fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(R, "scaling_result.png"), dpi=170)
print("wrote scaling_result.png")
