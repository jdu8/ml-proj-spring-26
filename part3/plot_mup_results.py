"""
Generate Part 3 comparison plots.

Plots produced (in plots/):
    mup_lr_sweep.png      — µP tiny LR sweep vs SP sweep on same axes
    compare_wide.png      — SP vs µP wide series (both nominal LRs)
    compare_original.png  — SP vs µP original series (best µP run)
    training_curves_mup.png — µP wide series training curves (out2, lr=3e-3)
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SP_DIR   = Path("../part2/out")
MUP_1E2  = Path("out")       # µP nominal lr=1e-2 (sweep-optimal)
MUP_3E3  = Path("out2")      # µP nominal lr=3e-3
PLOT_DIR = Path("plots")

ORIGINAL = ["tiny", "small", "medium", "large", "xl"]
WIDE     = ["tiny", "small_wide", "medium_wide", "large_wide", "xl_wide"]

LABELS = {
    "tiny": "Tiny", "small": "Small", "medium": "Medium",
    "large": "Large", "xl": "XL",
    "small_wide": "Small-W", "medium_wide": "Medium-W",
    "large_wide": "Large-W", "xl_wide": "XL-W",
}

_D_MODEL = {
    "tiny": 128, "small": 192, "medium": 384, "large": 512, "xl": 768,
    "small_wide": 192, "medium_wide": 384, "large_wide": 512, "xl_wide": 768,
}
_D_COLORS = {128: "#555555", 192: "#4C72B0", 384: "#55A868", 512: "#DD8452", 768: "#C44E52"}


def cmap(name):
    return _D_COLORS[_D_MODEL[name]]


def power_law(N, a, alpha, c):
    return a * N ** (-alpha) + c


def fit_pl(pts):
    N = np.array([p["n_params"] for p in pts], float)
    L = np.array([p["val_loss"] for p in pts], float)
    try:
        popt, pcov = curve_fit(
            power_law, N, L,
            p0=[5.0, 0.1, max(L.min() - 0.5, 0.01)],
            bounds=([0, 1e-4, 0], [1e8, 5.0, L.min() - 1e-4]),
            maxfev=20000,
        )
        return popt, np.sqrt(np.diag(pcov))
    except Exception:
        return None, None


def load(base_dir, configs):
    pts = []
    for name in configs:
        p = Path(base_dir) / name / "results.json"
        if p.exists():
            r = json.load(open(p))
            pts.append({"name": name, "n_params": r["n_params"],
                        "val_loss": r["best_val_loss"]})
    return pts


def load_log(base_dir, name):
    steps, train, val_steps, val = [], [], [], []
    p = Path(base_dir) / name / "log.csv"
    if not p.exists():
        return None
    for row in csv.DictReader(open(p)):
        steps.append(int(row["step"]))
        train.append(float(row["train_loss"]))
        if row["val_loss"]:
            val_steps.append(int(row["step"]))
            val.append(float(row["val_loss"]))
    return {"steps": steps, "train": train, "val_steps": val_steps, "val": val}


# ── Plot 1: LR sweep comparison ───────────────────────────────────────────────

def plot_lr_sweep():
    sp_path  = SP_DIR / "sweep" / "sweep_summary.json"
    mup_path = MUP_1E2 / "sweep" / "sweep_summary_mup.json"
    if not sp_path.exists() or not mup_path.exists():
        print("  Skipping mup_lr_sweep.png — missing sweep summary files")
        return

    sp_sweep  = json.load(open(sp_path))
    mup_sweep = json.load(open(mup_path))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for sweep, label, color, marker in [
        (sp_sweep,  "SP (no LR scaling)",  "steelblue", "o"),
        (mup_sweep, "µP (MuAdamW scales LR by width)", "crimson", "D"),
    ]:
        res = sorted(sweep["results"], key=lambda x: x["lr"])
        lrs  = [r["lr"]       for r in res]
        vals = [r["val_loss"] for r in res]
        best = sweep["best_lr"]
        ax.plot(lrs, vals, f"{marker}-", color=color, linewidth=2,
                markersize=8, zorder=5, label=label)
        ax.axvline(best, color=color, linestyle="--", alpha=0.5,
                   label=f"Best {label.split()[0]} LR = {best:.0e}")
        for lr, val in zip(lrs, vals):
            ax.annotate(f"{val:.3f}", (lr, val),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7.5, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Nominal Learning Rate (log scale)", fontsize=11)
    ax.set_ylabel("Best Validation Loss — Tiny Model", fontsize=11)
    ax.set_title("LR Sweep: SP vs µP (Tiny Model)", fontsize=12)
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mup_lr_sweep.png", dpi=150)
    plt.close()
    print("  saved plots/mup_lr_sweep.png")


# ── Shared: scaling comparison ────────────────────────────────────────────────

def _scaling_compare(configs, sp_pts, mup1_pts, mup2_pts,
                     fit_sp, fit_mup, title, out_name):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    series = [
        (sp_pts,   "SP (LR=3e-3, fixed)",       "steelblue", "o", "-"),
        (mup2_pts, "µP (nominal 3e-3)",          "crimson",   "D", "--"),
        (mup1_pts, "µP (nominal 1e-2, sweep-opt)", "darkorange", "^", ":"),
    ]

    for pts, label, color, marker, ls in series:
        if not pts:
            continue
        N = [p["n_params"] for p in pts]
        L = [p["val_loss"]  for p in pts]
        names = [p["name"]  for p in pts]

        ax.scatter(N, L, color=color, s=80, zorder=6, marker=marker)
        ax.plot(N, L, ls=ls, color=color, linewidth=1.4, alpha=0.7, label=label)
        for x, y, nm in zip(N, L, names):
            ax.annotate(LABELS[nm], (x, y),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=8, color=color)

    # power law fit on fit_sp subset for SP
    sp_fit_pts = [p for p in sp_pts if p["name"] in fit_sp]
    popt, _ = fit_pl(sp_fit_pts)
    if popt is not None:
        Ns = [p["n_params"] for p in sp_pts]
        N_fit = np.logspace(np.log10(min(Ns) * 0.5), np.log10(max(Ns) * 2), 300)
        a, alpha, c = popt
        ax.plot(N_fit, power_law(N_fit, *popt), color="steelblue", linewidth=0.8, alpha=0.5,
                linestyle="-")
        print(f"  SP fit ({out_name}): α={alpha:.3f}")

    # power law fit on monotone µP (3e-3) subset
    mup_fit_pts = [p for p in mup2_pts if p["name"] in fit_mup]
    popt2, _ = fit_pl(mup_fit_pts)
    if popt2 is not None:
        Ns2 = [p["n_params"] for p in mup2_pts]
        N_fit2 = np.logspace(np.log10(min(Ns2) * 0.5), np.log10(max(Ns2) * 2), 300)
        a2, alpha2, c2 = popt2
        ax.plot(N_fit2, power_law(N_fit2, *popt2), color="crimson", linewidth=0.8, alpha=0.5,
                linestyle="--")
        print(f"  µP(3e-3) fit ({out_name}): α={alpha2:.3f}")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (non-embedding, log scale)", fontsize=11)
    ax.set_ylabel("Validation Loss after 1 epoch", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / out_name, dpi=150)
    plt.close()
    print(f"  saved plots/{out_name}")


def plot_compare_wide():
    sp   = load(SP_DIR,  WIDE)
    m1   = load(MUP_1E2, WIDE)
    m2   = load(MUP_3E3, WIDE)
    _scaling_compare(
        WIDE, sp, m1, m2,
        fit_sp=["tiny", "small_wide", "medium_wide", "large_wide"],
        fit_mup=["tiny", "small_wide", "medium_wide", "large_wide"],
        title="SVG Scaling — Wide Series: SP vs µP (n_layer=4 fixed)",
        out_name="compare_wide.png",
    )


def plot_compare_original():
    sp   = load(SP_DIR,  ORIGINAL)
    m1   = load(MUP_1E2, ORIGINAL)
    m2   = load(MUP_3E3, ORIGINAL)
    _scaling_compare(
        ORIGINAL, sp, m1, m2,
        fit_sp=["tiny", "small", "medium"],
        fit_mup=["tiny", "small", "medium"],
        title="SVG Scaling — Original Series: SP vs µP",
        out_name="compare_original.png",
    )


# ── Plot 4: µP training curves (out2, wide series) ───────────────────────────

def plot_training_curves_mup():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for name in WIDE:
        log = load_log(MUP_3E3, name)
        if log is None:
            continue
        c   = cmap(name)
        lbl = LABELS[name]
        axes[0].plot(log["steps"], log["train"],
                     color=c, linewidth=1.2, alpha=0.85, label=lbl)
        axes[1].plot(log["val_steps"], log["val"],
                     "o-", color=c, linewidth=1.5, markersize=3, label=lbl)

    for ax, t in zip(axes, ["Training Loss", "Validation Loss"]):
        ax.set_xlabel("Optimizer Step", fontsize=10)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=10)
        ax.set_title(t, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("µP Training Curves — Wide Series (nominal LR = 3×10⁻³)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "training_curves_mup.png", dpi=150)
    plt.close()
    print("  saved plots/training_curves_mup.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    PLOT_DIR.mkdir(exist_ok=True)
    print("Generating µP comparison plots...")
    plot_lr_sweep()
    plot_compare_wide()
    plot_compare_original()
    plot_training_curves_mup()
    print("Done.")


if __name__ == "__main__":
    main()
