"""Real targets BY EYE: results/real_by_eye.png — 3 rows x 3 cols.
Rows: gym banana (x1,x2), eight-schools (mu, log tau), WL band-power (Om, s8).
Cols: reference (exact/NUTS, gray) | ICS @200k (blue) | ICS @2M (blue).
Model samples are drawn through the SAME code path and SAME context seeds as
the MEASURED readout_c numbers (777000+idx fold T=1), so panels correspond to
the quoted certificates. Captions quote the measured certified ESS / SW2.
GPU job (CNF sampling of two checkpoints on three targets)."""

import json
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

from ics.context import generate_context
from ics.models import ICSModel
from ics.real import (WLBandpower, eight_schools_logpdf, gym_banana_logpdf,
                      ics_evaluate_fn)
from ics.train import load_checkpoint

R = os.path.join(os.path.dirname(__file__), "..", "results")
REFS = os.path.join(os.environ["SCRATCH"], "ics-refs")
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BASE, RED = "#c3c2b7", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})

N = int(os.environ.get("NSAMP", 4096))
N_ODE = int(os.environ.get("NSTEP", 100))
OUTNAME = os.environ.get("OUTNAME", "real_by_eye.png")


def panel(ax, pts, dims, lims, cmap, title, note=None, note_color=INK2):
    x, y = pts[:, dims[0]], pts[:, dims[1]]
    inside = ((x >= lims[0][0]) & (x <= lims[0][1])
              & (y >= lims[1][0]) & (y <= lims[1][1])).mean()
    ax.hist2d(np.clip(x, *lims[0]), np.clip(y, *lims[1]), bins=70,
              range=lims, cmap=cmap)
    ax.set_title(title + f"\n({100*inside:.0f}% of mass in frame)",
                 fontsize=8.8, color=INK)
    if note:
        ax.annotate(note, (0.03, 0.03), xycoords="axes fraction", fontsize=8,
                    color=note_color, va="bottom")
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])


def main():
    wl = WLBandpower(os.path.join(R, "wl_surrogate.npz"))
    targets = [
        ("gym_banana", gym_banana_logpdf, 2, (0, 1), [(-32, 32), (-12, 8)],
         ("x1", "x2"), 1),
        ("eight_schools", eight_schools_logpdf, 10, (0, 1),
         [(-12, 22), (-6, 5)], ("mu", "log tau"), 0),
        ("wl_bandpower", wl.logpdf, 3, (0, 1), [(0.15, 0.39), (0.68, 0.98)],
         ("Omega_m", "sigma_8"), 2),
    ]
    model = ICSModel(n_attn=2)
    ckpts = [("200k", os.path.join(R, "gate3e_params.pkl")),
             ("2M", os.path.join(os.environ["SCRATCH"], "ics-zoo",
                                 "ckpt_2m_step2000000.pkl"))]
    params = {tag: jax.tree_util.tree_map(jnp.asarray, load_checkpoint(p)["params"])
              for tag, p in ckpts}
    cjson = {tag: json.load(open(os.path.join(R, f"readout_c_{tag}.json")))["targets"]
             for tag, _ in ckpts}

    fig, axes = plt.subplots(3, 3, figsize=(13.2, 12.6))
    for row, (name, fn, d, dims, lims, labels, idx) in enumerate(targets):
        ref = np.load(os.path.join(REFS, f"{name}.npz"))["draws"].reshape(-1, d)
        ref = ref[np.random.default_rng(5).choice(len(ref), N, replace=False)]
        ref_pts = np.asarray(wl.theta_of_u(jnp.asarray(ref, jnp.float64))) \
            if name == "wl_bandpower" else ref
        panel(axes[row, 0], ref_pts, dims, lims, "Greys",
              f"{name} — REFERENCE" + (" (exact sampler)" if name == "gym_banana"
                                       else " (NUTS, R-hat checked)"))
        axes[row, 0].set_xlabel(labels[0])
        axes[row, 0].set_ylabel(labels[1])

        for col, (tag, _) in enumerate(ckpts, start=1):
            ctx = generate_context(jr.fold_in(jr.key(777_000 + idx), 1), fn, d,
                                   K=128, temperature=1.0, aux_tokens=True)
            cert, x_gen = ics_evaluate_fn(model, params[tag], fn, d, ctx,
                                          jr.key(778_000 + 100 * idx + 1),
                                          n_eval=N // 2, n_ode=N_ODE)
            pts = np.asarray(wl.theta_of_u(jnp.asarray(x_gen))) \
                if name == "wl_bandpower" else x_gen
            m = cjson[tag][name]["T1"]
            ok = m["stable"] and m["ess_frac_2n"] >= 0.01
            note = (f"measured: ESS {100*m['ess_frac_2n']:.2f}% "
                    f"{'STABLE' if m['stable'] else 'UNSTABLE'}, "
                    f"SW2 {m['sw2_vs_ref']:.2f}")
            panel(axes[row, col], pts, dims, lims, "Blues",
                  f"ICS @{tag} — zero-shot, T=1",
                  note=note, note_color=("#0a7a4d" if ok else RED))
            axes[row, col].set_xlabel(labels[0])
    fig.suptitle("Real inference problems, zero-shot — reference vs the model "
                 "before/after the compute lever (contexts = the measured ones)",
                 fontsize=11, color=INK, y=0.995)
    fig.tight_layout()
    out = os.path.join(R, OUTNAME)
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
