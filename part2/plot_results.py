"""
Generate all Part 2 plots from saved results.

Plots produced (all in plots/):
    lr_sweep.png              — val loss vs LR for Tiny SP sweep
    scaling.png               — original 5-config scaling law (SP)
    scaling_wide.png          — wide 5-config scaling law (SP, n_layer=4 fixed)
    training_curves.png       — train + val loss over steps, original series
    training_curves_wide.png  — train + val loss over steps, wide series
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

OUT_DIR  = Path("out")
PLOT_DIR = Path("plots")

ORIGINAL = ["tiny", "small", "medium", "large", "xl"]
WIDE     = ["tiny", "small_wide", "medium_wide", "large_wide", "xl_wide"]

# Colors keyed by d_model value (shared between series)
_D_COLORS = {128: "#555555", 192: "#4C72B0", 384: "#55A868", 512: "#DD8452", 768: "#C44E52"}
_D_MODEL  = {
    "tiny": 128, "small": 192, "medium": 384, "large": 512, "xl": 768,
    "small_wide": 192, "medium_wide": 384, "large_wide": 512, "xl_wide": 768,
}
LABELS = {
    "tiny": "Tiny", "small": "Small", "medium": "Medium",
    "large": "Large", "xl": "XL",
    "small_wide": "Small-W", "medium_wide": "Medium-W",
    "large_wide": "Large-W", "xl_wide": "XL-W",
}


def color(name):
    return _D_COLORS[_D_MODEL[name]]


def power_law(N, a, alpha, c):
    return a * N ** (-alpha) + c


def fit_power_law(names, results):
    N = np.array([results[n]["n_params"]      for n in names], dtype=float)
    L = np.array([results[n]["best_val_loss"] for n in names], dtype=float)
    try:
        popt, pcov = curve_fit(
            power_law, N, L,
            p0=[5.0, 0.1, max(L.min() - 0.5, 0.01)],
            bounds=([0, 1e-4, 0], [1e8, 5.0, L.min() - 1e-4]),
            maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except Exception as e:
        print(f"  curve_fit failed: {e}")
        return None, None


def load_results(configs):
    data = {}
    for name in configs:
        p = OUT_DIR / name / "results.json"
        if p.exists():
            with open(p) as f:
                data[name] = json.load(f)
        else:
            print(f"  Missing: {p}")
    return data


def load_logs(configs):
    logs = {}
    for name in configs:
        p = OUT_DIR / name / "log.csv"
        if not p.exists():
            continue
        steps, train_loss, val_steps, val_loss = [], [], [], []
        with open(p) as f:
            for row in csv.DictReader(f):
                s = int(row["step"])
                steps.append(s)
                train_loss.append(float(row["train_loss"]))
                if row["val_loss"]:
                    val_steps.append(s)
                    val_loss.append(float(row["val_loss"]))
        logs[name] = {"steps": steps, "train": train_loss,
                      "val_steps": val_steps, "val": val_loss}
    return logs


# ── Plot 1: LR sweep ──────────────────────────────────────────────────────────

def plot_lr_sweep():
    summary_path = OUT_DIR / "sweep" / "sweep_summary.json"
    if not summary_path.exists():
        print("  Skipping lr_sweep.png — sweep_summary.json not found")
        return

    with open(summary_path) as f:
        sweep = json.load(f)

    results = sorted(sweep["results"], key=lambda x: x["lr"])
    lrs     = [r["lr"]       for r in results]
    vals    = [r["val_loss"] for r in results]
    best_lr = sweep["best_lr"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lrs, vals, "o-", color="#4C72B0", linewidth=2, markersize=8, zorder=5)
    ax.axvline(best_lr, color="crimson", linestyle="--", alpha=0.8,
               label=f"Best LR = {best_lr:.0e}")
    for lr, val in zip(lrs, vals):
        ax.annotate(f"{val:.3f}", (lr, val),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate (log scale)", fontsize=11)
    ax.set_ylabel("Best Validation Loss", fontsize=11)
    ax.set_title("LR Sweep — Tiny Model (Standard Parameterization)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "lr_sweep.png", dpi=150)
    plt.close()
    print("  saved plots/lr_sweep.png")


# ── Shared: scaling plot ──────────────────────────────────────────────────────

def _scaling_plot(configs, results, fit_names, title, out_name, marker="o", ls="-"):
    fig, ax = plt.subplots(figsize=(7, 5))

    for name in configs:
        if name not in results:
            continue
        N = results[name]["n_params"]
        L = results[name]["best_val_loss"]
        ax.scatter(N, L, color=color(name), s=90, zorder=5, marker=marker)
        ax.annotate(LABELS[name], (N, L),
                    textcoords="offset points", xytext=(7, 4), fontsize=9)

    fit_avail = [n for n in fit_names if n in results]
    popt = perr = None
    if len(fit_avail) >= 3:
        popt, perr = fit_power_law(fit_avail, results)

    if popt is not None:
        a, alpha, c = popt
        all_N = [results[n]["n_params"] for n in configs if n in results]
        N_fit = np.logspace(np.log10(min(all_N) * 0.5),
                            np.log10(max(all_N) * 3), 300)
        ax.plot(N_fit, power_law(N_fit, *popt), "k--", linewidth=1.5,
                label=rf"Power law (fit subset): $L={a:.2f}\,N^{{-{alpha:.3f}}}+{c:.2f}$")
        print(f"  [{out_name}] fit: a={a:.4f}, α={alpha:.4f}, c={c:.4f}  (±{perr[1]:.4f})")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (non-embedding, log scale)", fontsize=11)
    ax.set_ylabel("Validation Loss after 1 epoch", fontsize=11)
    ax.set_title(title, fontsize=12)
    if popt is not None:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / out_name, dpi=150)
    plt.close()
    print(f"  saved plots/{out_name}")
    return popt, perr


def plot_scaling_original(results):
    return _scaling_plot(
        configs=ORIGINAL, results=results,
        fit_names=["tiny", "small", "medium"],
        title="SVG Transformer Scaling — Standard Parameterization (Original Series)",
        out_name="scaling.png",
    )


def plot_scaling_wide(results):
    return _scaling_plot(
        configs=WIDE, results=results,
        fit_names=["tiny", "small_wide", "medium_wide", "large_wide"],
        title="SVG Transformer Scaling — Standard Parameterization (Wide Series, n_layer=4)",
        out_name="scaling_wide.png",
        marker="D",
    )


# ── Shared: training curves ───────────────────────────────────────────────────

def _training_curves(configs, logs, title, out_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for name in configs:
        if name not in logs:
            continue
        log = logs[name]
        c = color(name)
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

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / out_name, dpi=150)
    plt.close()
    print(f"  saved plots/{out_name}")


def plot_training_curves_original(logs):
    _training_curves(
        configs=ORIGINAL, logs=logs,
        title="Training Curves — Original Series (SP, LR = 3×10⁻³)",
        out_name="training_curves.png",
    )


def plot_training_curves_wide(logs):
    _training_curves(
        configs=WIDE, logs=logs,
        title="Training Curves — Wide Series, n_layer=4 (SP, LR = 3×10⁻³)",
        out_name="training_curves_wide.png",
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    PLOT_DIR.mkdir(exist_ok=True)

    all_configs = list(dict.fromkeys(ORIGINAL + WIDE))  # deduplicated, tiny once
    results = load_results(all_configs)
    logs    = load_logs(all_configs)

    print("Generating plots...")
    plot_lr_sweep()
    plot_scaling_original(results)
    plot_scaling_wide(results)
    plot_training_curves_original(logs)
    plot_training_curves_wide(logs)
    print("Done.")


if __name__ == "__main__":
    main()
