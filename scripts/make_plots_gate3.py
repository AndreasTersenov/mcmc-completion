"""Gate-(iii) story figure: results/gate3_story.png

A: the coverage law — logZ-hat vs ln(sampler-covered mass), from stage-0's
   synthetic mode-drop through the trained system at T=1 to the tempered fix.
B: why T (not K) buys coverage — mode visit counts for the gmm-d2 eval
   context at (K,T) in {(128,1),(512,1),(128,5)}.
C: the sharpness gap under the P1-mirror bar — conditional SW2 vs bespoke
   reference per target (gate3_p1.json), motivating the 128-target ruling.
D: the shortK lever — per-target ESS across attempt 2 (deep-sets), attempt 3
   (+attn+shortk), and b1 (no shortk).
Sources: results/*.json on disk + two historical jsons resolved from git.
"""

import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")
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


def git_json(msg_pattern, path="results/gate3.json"):
    line = subprocess.run(
        f'git log --format="%H %s" | grep -m1 "{msg_pattern}"',
        shell=True, capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    ).stdout.strip()
    sha = line.split()[0]
    blob = subprocess.run(["git", "show", f"{sha}:{path}"], capture_output=True,
                          text=True, cwd=os.path.join(os.path.dirname(__file__), "..")).stdout
    return json.loads(blob)


def main():
    p1 = json.load(open(os.path.join(R, "gate3_p1.json")))
    t5 = json.load(open(os.path.join(R, "gate3_t5.json")))
    b1 = json.load(open(os.path.join(R, "gate3_noshortk.json")))
    a2 = git_json("attempt 2 FAIL: two structural")  # attempt 2 (deep-sets)
    a3 = git_json("attempt 3: 1/10")               # attempt 3 (+attn +shortk)

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.1))

    # ---- A: the coverage law
    ax = axes[0]
    # (label, ln covered mass of sampler-recovered modes, logZ-hat, color)
    pts = [
        ("stage-0 synthetic\nmode drop", np.log(0.5), -0.6921, INK2),
        ("trained ICS, T=1\n(modes 1-4)", np.log(0.465), -0.770, CAT["gmm"]),
        ("trained ICS, T=5\n(5/6 modes)", np.log(0.923), -0.045, RED),
    ]
    line = np.linspace(-0.95, 0.05, 10)
    ax.plot(line, line, ls="--", lw=1.4, color=BASE)
    ax.annotate("logZ-hat = ln(covered mass)", (-0.93, -0.88), color=MUTED,
                fontsize=8.5, rotation=38)
    for lab, x, y, c in pts:
        ax.plot(x, y, "o", ms=9, color=c, markeredgecolor=SURFACE, markeredgewidth=1)
        ax.annotate(lab, (x + 0.03, y - 0.015), color=c, fontsize=8.5, va="top")
    ax.annotate("", xy=(np.log(0.923) - 0.02, -0.06), xytext=(np.log(0.465) + 0.04, -0.74),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2, ls=":"))
    ax.annotate("tempered\ncontexts", (-0.48, -0.38), color=RED, fontsize=8.5)
    ax.set_xlabel("ln(mass of modes the sampler covers)")
    ax.set_ylabel("certificate logẐ")
    ax.set_xlim(-1.0, 0.1)
    ax.set_ylim(-1.0, 0.1)
    ax.set_title("The certificate prices what the sampler covers\n"
                 "same law: synthetic (stage-0) and trained (phase 1)",
                 fontsize=10, color=INK)

    # ---- B: T not K buys coverage (gmm-d2 eval context)
    ax = axes[1]
    counts = {
        "K=128, T=1": [0, 49, 44, 20, 14, 1],
        "K=512, T=1": [1, 201, 94, 154, 61, 1],
        "K=128, T=5": [14, 50, 36, 1, 0, 27],
    }
    weights = [0.249, 0.144, 0.014, 0.230, 0.077, 0.286]
    x = np.arange(6)
    shades = ["#9ec5f4", "#3987e5", RED]
    for i, (lab, c) in enumerate(zip(counts, shades)):
        frac = np.asarray(counts[lab], float) / sum(counts[lab])
        ax.bar(x + (i - 1) * 0.27, frac, 0.25, color=c, label=lab,
               edgecolor=SURFACE, linewidth=0.5)
    ax.plot(x, weights, "_", ms=22, color=INK, label="true mode weight")
    ax.set_xticks(x, [f"m{i}" for i in x])
    ax.set_xlabel("mode of the 6-component gmm-d2 target")
    ax.set_ylabel("fraction of context points")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Why temperature, not K, buys coverage\n"
                 "T=1 chains never hop; T=5 touches 5/6 modes",
                 fontsize=10, color=INK)

    # ---- C: sharpness gap vs the P1-mirror bar
    ax = axes[2]
    for t in p1["targets"]:
        fam = t["family"] if t["family"] in CAT else "funnel"
        fam = {"funnelmix": "funnel", "banana": "warp"}.get(t["family"], t["family"])
        ref, sw2 = t["sw2_ref"], t["sw2"]
        ax.plot(ref, sw2, "o", ms=8, color=CAT[fam], markeredgecolor=SURFACE,
                markeredgewidth=0.9)
        off = 1.25 if (t["d"] in (2, 8)) else 0.78
        ax.annotate(f"{t['family']}-d{t['d']}", (ref * 1.15, sw2 * off), fontsize=7.5,
                    color=CAT[fam], va="bottom" if off > 1 else "top")
    grid_x = np.logspace(-3.3, 0.2, 10)
    ax.plot(grid_x, grid_x, ls="--", lw=1.2, color=BASE)
    ax.plot(grid_x, 2 * grid_x, ls="-", lw=1.2, color=INK2)
    ax.axhline(0.1, ls=":", lw=1.0, color=INK2)
    ax.annotate("pass bar: max(2× bespoke, 0.1)", (1.3e-3, 0.115), fontsize=8,
                color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bespoke per-target FM  SW2² (the reference)")
    ax.set_ylabel("conditional ICS  SW2²")
    ax.set_title("The sharpness gap under the P1-mirror bar\n"
                 "10-target zoo: only warp-d2 passes → grow to 128 (ruling 2)",
                 fontsize=10, color=INK)

    # ---- D: the shortK lever
    ax = axes[3]
    rows = [("gmm", 4), ("dwell", 4), ("gmm", 8), ("warp", 2)]
    runs = [("attempt 2\ndeep-sets", a2, "#9ec5f4"),
            ("attempt 3\n+attn +shortK", a3, "#3987e5"),
            ("b1 recipe\n+attn, no shortK", b1, "#104281")]
    x = np.arange(len(rows))
    for i, (lab, data, c) in enumerate(runs):
        m = {(t["family"], t["d"]): t for t in data["targets"]}
        vals = [max(m[r]["ess_frac_2n"], 2e-4) for r in rows]
        ax.bar(x + (i - 1) * 0.27, vals, 0.25, color=c, label=lab.replace("\n", " "),
               edgecolor=SURFACE, linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.axhline(0.01, ls=":", lw=1.0, color=RED)
    ax.annotate("P1 ESS clause (1%)", (-0.42, 0.013), fontsize=8, color=RED)
    ax.set_xticks(x, [f"{f}-d{d}" for f, d in rows])
    ax.set_ylabel("certified ESS / N")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("The shortK lever: sharpness lost, recovered\n"
                 "(mode coverage kept throughout)",
                 fontsize=10, color=INK)

    fig.tight_layout()
    out = os.path.join(R, "gate3_story.png")
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main()
