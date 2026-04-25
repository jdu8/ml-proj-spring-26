"""
Generate all Part 2 plots from saved results.

Outputs:
    plots/lr_sweep.png       — val loss vs learning rate
    plots/scaling.png        — scaling law (params vs val loss)
    plots/training_curves.png — train + val loss over steps for all models
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

OUT_DIR   = Path("out")
PLOT_DIR  = Path("plots")
CONFIGS   = ["tiny", "small", "medium", "large", "xl"]
COLORS    = {"tiny": "#4C72B0", "small": "#55A868", "medium": "#C44E52",
             "large": "#8172B2", "xl":   "#CCB974"}


def power_law(N, a, alpha, c):
    return a * N ** (-alpha) + c


def load_results():
    data = {}
    for name in CONFIGS:
        p = OUT_DIR / name / "results.json"
        if p.exists():
            with open(p) as f:
                data[name] = json.load(f)
    return data


def load_logs():
    logs = {}
    for name in CONFIGS:
        p = OUT_DIR / name / "log.csv"
        if not p.exists():
            continue
        steps, train_loss, val_loss = [], [], []
        with open(p) as f:
            next(f)  # header
            for line in f:
                parts = line.strip().split(",")
                steps.append(int(parts[0]))
                train_loss.append(float(parts[1]))
                val_loss.append(float(parts[2]) if parts[2] else None)
        logs[name] = {"steps": steps, "train": train_loss, "val": val_loss}
    return logs


def load_sweep():
    p = OUT_DIR / "sweep" / "sweep_summary.json"
    with open(p) as f:
        return json.load(f)


# ── Plot 1: LR sweep ───────────────────────────────────────────────────────────

def plot_lr_sweep(sweep):
    results = sorted(sweep["results"], key=lambda x: x["lr"])
    lrs  = [r["lr"]       for r in results]
    vals = [r["val_loss"] for r in results]
    best_lr = sweep["best_lr"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lrs, vals, "o-", color="#4C72B0", linewidth=2, markersize=8, zorder=5)
    ax.axvline(best_lr, color="crimson", linestyle="--", alpha=0.7,
               label=f"Best LR = {best_lr:.0e}")
    for lr, val in zip(lrs, vals):
        ax.annotate(f"{val:.3f}", (lr, val),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate (log scale)", fontsize=11)
    ax.set_ylabel("Best Validation Loss", fontsize=11)
    ax.set_title("LR Sweep — Tiny Model (SP)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "lr_sweep.png", dpi=150)
    plt.close()
    print("  saved plots/lr_sweep.png")


# ── Plot 2: Scaling law ────────────────────────────────────────────────────────

def plot_scaling(results):
    names  = list(results.keys())
    params = [results[n]["n_params"]      for n in names]
    losses = [results[n]["best_val_loss"] for n in names]

    # fit power law on first 3 models (monotone subset under SP)
    fit_names  = ["tiny", "small", "medium"]
    fit_params = [results[n]["n_params"]      for n in fit_names if n in results]
    fit_losses = [results[n]["best_val_loss"] for n in fit_names if n in results]
    popt = perr = None
    if len(fit_params) >= 3:
        try:
            popt, pcov = curve_fit(
                power_law, fit_params, fit_losses,
                p0=[5.0, 0.1, 0.5],
                bounds=([0, 1e-4, 0], [1e8, 5.0, min(fit_losses) - 1e-4]),
                maxfev=20000,
            )
            perr = np.sqrt(np.diag(pcov))
        except Exception as e:
            print(f"  curve_fit failed: {e}")

    fig, ax = plt.subplots(figsize=(7, 5))

    for name, N, L in zip(names, params, losses):
        ax.scatter(N, L, color=COLORS.get(name, "gray"), s=90, zorder=5)
        ax.annotate(name, (N, L), textcoords="offset points", xytext=(7, 4), fontsize=9)

    if popt is not None:
        a, alpha, c = popt
        N_fit = np.logspace(np.log10(min(fit_params) * 0.5),
                            np.log10(max(fit_params) * 3), 300)
        ax.plot(N_fit, power_law(N_fit, *popt), "k--", linewidth=1.5,
                label=rf"Fit (tiny–medium): $L={a:.2f}\,N^{{-{alpha:.3f}}}+{c:.2f}$")
        print(f"  Power law (tiny–medium): a={a:.4f}, α={alpha:.4f}, c={c:.4f}")
        if perr is not None:
            print(f"  Uncertainty: α ± {perr[1]:.4f}")

    # mark SP failure region
    if "large" in results and "medium" in results:
        ax.annotate("SP LR\nbreaks down", xy=(results["large"]["n_params"],
                    results["large"]["best_val_loss"]),
                    xytext=(results["medium"]["n_params"] * 1.5,
                            results["large"]["best_val_loss"] + 0.15),
                    fontsize=8, color="gray",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (non-embedding, log scale)", fontsize=11)
    ax.set_ylabel("Validation Loss after 1 epoch", fontsize=11)
    ax.set_title("SVG Transformer Scaling — Standard Parameterization", fontsize=12)
    if popt is not None:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "scaling.png", dpi=150)
    plt.close()
    print("  saved plots/scaling.png")

    return popt, perr


# ── Plot 3: Training curves ────────────────────────────────────────────────────

def plot_training_curves(logs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for name, log in logs.items():
        color = COLORS.get(name, "gray")
        steps = log["steps"]

        # train loss (every step)
        axes[0].plot(steps, log["train"], color=color, linewidth=1.2,
                     alpha=0.85, label=name)

        # val loss (only at eval steps)
        val_steps = [s for s, v in zip(steps, log["val"]) if v is not None]
        val_vals  = [v for v in log["val"] if v is not None]
        axes[1].plot(val_steps, val_vals, "o-", color=color, linewidth=1.5,
                     markersize=3, label=name)

    for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
        ax.set_xlabel("Optimizer Step", fontsize=10)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves — All Model Sizes (SP, LR = 3×10⁻³)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "training_curves.png", dpi=150)
    plt.close()
    print("  saved plots/training_curves.png")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    PLOT_DIR.mkdir(exist_ok=True)
    results = load_results()
    logs    = load_logs()
    sweep   = load_sweep()

    print("Generating plots...")
    plot_lr_sweep(sweep)
    popt, perr = plot_scaling(results)
    plot_training_curves(logs)
    print("Done.")

    return popt, perr


if __name__ == "__main__":
    main()
