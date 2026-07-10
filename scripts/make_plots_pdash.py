"""P-dashboard: results/p_dashboard.png
A: capability map — median certified ESS per (family, d), train4, T=5 column.
B: P3 confusion scatter — held-out targets, certificate verdict vs SW2 truth.
C: prediction scorecard — measured vs registered bars."""
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
    "text.color": INK, "axes.grid": False, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "legend.frameon": False})

e4 = json.load(open(os.path.join(R, "eval_train4.json")))
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), gridspec_kw={"width_ratios": [1.1, 1.2, 1]})

# A: capability heatmap (median t5 ESS per family x d)
ax = axes[0]
fams = ["gmm", "dwell", "funnel", "warp", "banana", "funnelmix"]
ds = [2, 4, 8, 16]
M = np.full((len(fams), len(ds)), np.nan)
for i, f in enumerate(fams):
    for j, d in enumerate(ds):
        v = [r["t5"]["ess_frac_2n"] for r in e4 if r["family"] == f and r["d"] == d]
        if v: M[i, j] = np.median(v)
im = ax.imshow(np.log10(np.maximum(M, 1e-5)), cmap="Blues", vmin=-5, vmax=0,
               aspect="auto")
for i in range(len(fams)):
    for j in range(len(ds)):
        v = M[i, j]
        txt = f"{v*100:.2g}%" if np.isfinite(v) else "-"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                color=INK if (np.isfinite(v) and v < 0.05) else SURFACE)
ax.axhline(3.5, color=RED, lw=1.6)
ax.annotate("never seen in training", (3.45, 4.1), color=RED, fontsize=8.5,
            ha="right")
ax.set_xticks(range(len(ds)), [f"d={d}" for d in ds])
ax.set_yticks(range(len(fams)), fams)
ax.set_title("Capability map — median certified ESS\n(train4 model, fresh θ, T=5 column)",
             fontsize=10, color=INK)

# B: P3 confusion scatter (held-out rows)
ax = axes[1]
rows = [r for r in e4 if r["heldout"] and np.isfinite(r["t5"]["sw2"])]
for r in rows:
    ess = max(r["t5"]["ess_frac_2n"], 1e-6)
    q = r["t5"]["sw2"] / max(r["sw2_floor"], 1e-6)
    md = (r["t5"]["mode_recovery"] or 1.0) < 0.999
    flagged = ess < 0.01 or not r["t5"]["stable"]
    c = MUTED if flagged else (YEL if md else RED)
    ax.plot(q, ess, "o", ms=5.5, color=c, alpha=0.8,
            markeredgecolor=SURFACE, markeredgewidth=0.5)
ax.axhline(0.01, color=INK2, lw=1.1, ls="--")
ax.axvline(10, color=INK2, lw=1.1, ls=":")
ax.set_xscale("log"); ax.set_yscale("log")
ax.annotate("FALSE BLESSINGS (24%):\nbad samples, certificate calm\n(the new blind spot, P3 FAIL)",
            (0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
            fontsize=8.5, color=RED)
ax.annotate("flagged (correct refusals)", (0.03, 0.05), xycoords="axes fraction",
            fontsize=8.5, color=MUTED)
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
           [(MUTED, "flagged by certificate"), (RED, "blessed but bad (new)"),
            (YEL, "blessed but bad (mode-drop, known)")]]
ax.legend(handles=handles, fontsize=7.5, loc="lower right")
ax.set_xlabel("SW2² / sampling floor (truth quality; right of ⋮ = bad)")
ax.set_ylabel("certified ESS (below -- = flagged)")
ax.set_title("P3 confusion — held-out families\ncertificate flags 69% of bad targets (bar: 90%)",
             fontsize=10, color=INK)

# C: scorecard
ax = axes[2]
items = [
    ("P-scale (70%)", 1.62, 1.0, "×  improvement", True),
    ("P2 (70%)", 65.6, 50, "%  mode-fail", True),
    ("P12 dir (55%)", 1.54, 1.0, "×  4-vs-2 fams", True),
    ("P1 (75%)", 39.6, 80, "%  ESS clause", False),
    ("P3 flag (65%)", 68.9, 90, "%  flagged", False),
]
y = np.arange(len(items))[::-1]
for yi, (name, val, bar, unit, ok) in zip(y, items):
    c = AQUA if ok else RED
    ax.barh(yi, val / bar, height=0.55, color=c, edgecolor=SURFACE)
    ax.text(val / bar + 0.04, yi, f"{val:g} vs bar {bar:g} {unit}",
            va="center", fontsize=8)
ax.axvline(1.0, color=INK, lw=1.2)
ax.set_yticks(y, [i[0] for i in items])
ax.set_xlim(0, 2.4)
ax.set_xlabel("measured / registered bar  (1 = exactly at bar)")
ax.set_title("Frozen-prediction scorecard\n(registered confidence in parens)",
             fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(R, "p_dashboard.png"), dpi=170)
print("wrote p_dashboard.png")
