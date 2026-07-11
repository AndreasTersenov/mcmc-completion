"""Phase-1 closing figures: results/final_landscape.png
A: the method plane — cost per NEW target vs quality (P7 visual).
B: the d-cliff — median certified ESS vs d per family, with untrained floor.
C: P11 — gradient effect by regime (the confirmed ordering)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, RED, AQUA, YEL, VIO = "#2a78d6", "#e34948", "#1baf7a", "#eda100", "#4a3aa7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": BASE,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "legend.frameon": False})

B = json.load(open(os.path.join(R, "baselines.json")))
e4 = json.load(open(os.path.join(R, "eval_train4.json")))
eng = json.load(open(os.path.join(R, "eval_train4ng.json")))
eu = json.load(open(os.path.join(R, "eval_untrained.json")))
e4m = {(r["family"], r["d"], r["idx"]): r for r in e4}
subset = [(r["family"], r["d"], r["idx"]) for r in B["rows"]]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

# ---- A: method plane (12-target subset medians)
ax = axes[0]
ics_sw2 = np.median([e4m[k]["t5"]["sw2"] for k in subset])
b1_sw2 = np.median([r["t5"]["sw2"] for r in eu
                    if (r["family"], r["d"], r["idx"]) in subset])
methods = [
    ("ICS (ours)", B["ics_seconds"], ics_sw2, BLUE, "o", "certificate: YES"),
    ("B1 untrained", B["ics_seconds"], b1_sw2, MUTED, "o", "floor"),
    ("B2 bespoke FM", 70, np.median([r["b2"]["sw2"] for r in B["rows"]]), AQUA, "s",
     "per-target training\n(lower bound of 10-min spec)"),
    ("B3 energy-fit+MALA", 15, np.median([r["b3"]["sw2"] for r in B["rows"]]), YEL, "D",
     "cost est."),
    ("B4 MCLMC", 18.7 + B["ics_seconds"], np.median([r["b4"]["sw2"] for r in B["rows"]]),
     VIO, "^", "no certificate,\nmode-drops"),
]
for name, cost, q, c, mk, note in methods:
    ax.plot(cost, q, mk, ms=11, color=c, markeredgecolor=SURFACE, markeredgewidth=1)
    ax.annotate(f"{name}\n{note}", (cost * 1.15, q), fontsize=7.8, color=c, va="center")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(2, 700)
ax.set_xlabel("cost per NEW target (seconds, H100)")
ax.set_ylabel("median SW2² (12-target subset, fresh θ)")
ax.set_title("The method plane (P7)\nbottom-left is the goal; ICS is cheapest-with-\n"
             "certificate, not yet sharpest", fontsize=10, color=INK)

# ---- B: d-cliff curves
ax = axes[1]
fams = [("gmm", BLUE), ("dwell", YEL), ("funnel", AQUA), ("warp", VIO),
        ("banana", RED), ("funnelmix", "#d55181")]
ds = [2, 4, 8, 16]
for f, c in fams:
    v = []
    for d in ds:
        rows = [r["t5"]["ess_frac_2n"] for r in e4
                if r["family"] == f and r["d"] == d and np.isfinite(r["t5"]["sw2"])]
        v.append(np.median(rows) if rows else np.nan)
    ls = "--" if f in ("banana", "funnelmix") else "-"
    ax.plot(ds, v, ls + "o", color=c, lw=1.7, ms=6, label=f,
            markeredgecolor=SURFACE, markeredgewidth=0.7)
uv = [np.median([r["t5"]["ess_frac_2n"] for r in eu if r["d"] == d
                 and np.isfinite(r["t5"]["sw2"])]) for d in ds]
ax.plot(ds, uv, ":", color=MUTED, lw=1.6, label="untrained floor")
ax.axhline(0.01, color=INK2, lw=1.0, ls="--")
ax.annotate("P1 clause (1%)", (8.6, 0.012), fontsize=8, color=INK2)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(ds, [str(d) for d in ds])
ax.set_xlabel("dimension d")
ax.set_ylabel("median certified ESS (fresh θ, T=5)")
ax.legend(fontsize=7.5, ncol=2, loc="upper right")
ax.set_title("The d-cliff\n~10× ESS lost per dimension doubling;\n"
             "dashed = never-trained families", fontsize=10, color=INK)

# ---- C: P11 bars
ax = axes[2]
def med_ess(ee, ho, tag):
    return np.median([r[tag]["ess_frac_2n"] for r in ee
                      if r["heldout"] == ho and np.isfinite(r[tag]["sw2"])])
groups = [("in-family", False), ("held-out\nfamilies", True)]
x = np.arange(2)
for i, (tag, shade) in enumerate([("t1", 0), ("t5", 1)]):
    ratios = [med_ess(e4, ho, tag) / med_ess(eng, ho, tag) for _, ho in groups]
    ax.bar(x + (i - 0.5) * 0.32, ratios, 0.3,
           color=[BLUE, RED][i], edgecolor=SURFACE, linewidth=0.6,
           label=f"(K=128, T={'1' if tag=='t1' else '5'}) column")
ax.axhline(1.0, color=INK, lw=1.2)
ax.set_xticks(x, [g for g, _ in groups])
ax.set_ylabel("median-ESS ratio:  with ∇E / without ∇E")
ax.legend(fontsize=8.5, loc="upper left")
ax.set_title("P11 CONFIRMED — gradients matter where\nit's hardest: cross-family "
             "transfer\n(in-family ≈ 1×, held-out 1.2–1.4×)", fontsize=10, color=INK)

fig.tight_layout()
fig.savefig(os.path.join(R, "final_landscape.png"), dpi=170)
print("wrote final_landscape.png")
