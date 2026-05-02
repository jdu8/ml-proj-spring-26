"""
Plot SP (Part 2) vs µP (Part 3_3) scaling curves on the same graph,
fit power laws to both, and report the 10× extrapolation.

Usage:
    python compare_plot.py --series wide    --plot_path plots/compare_wide.png
    python compare_plot.py --series original --plot_path plots/compare_original.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SERIES = {
    "original": ["tiny", "small", "medium", "large", "xl"],
    "wide":     ["tiny", "small_wide", "medium_wide", "large_wide", "xl_wide"],
}


def power_law(N, a, alpha, c):
    return a * N ** (-alpha) + c


def load_results(results_dir: Path, label: str, configs: list) -> list[dict]:
    points = []
    for name in configs:
        p = results_dir / name / "results.json"
        if p.exists():
            with open(p) as f:
                r = json.load(f)
            points.append({"name": name, "n_params": r["n_params"], "val_loss": r["best_val_loss"]})
        else:
            print(f"  [{label}] missing: {p}")
    return points


def monotone_prefix(points: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split points (sorted by n_params) into the longest strictly-decreasing
    loss prefix and the remaining 'broken' tail.
    """
    pts = sorted(points, key=lambda p: p["n_params"])
    split = len(pts)
    for i in range(1, len(pts)):
        if pts[i]["val_loss"] >= pts[i - 1]["val_loss"]:
            split = i
            break
    return pts[:split], pts[split:]


def fit_power_law(points: list[dict]):
    if len(points) < 3:
        return None, None
    N = np.array([p["n_params"] for p in points], dtype=float)
    L = np.array([p["val_loss"]  for p in points], dtype=float)
    try:
        popt, pcov = curve_fit(
            power_law, N, L,
            p0=[5.0, 0.3, max(L.min() - 0.3, 0.01)],
            bounds=([0, 1e-4, 0], [1e9, 5.0, L.min() - 1e-4]),
            maxfev=30000,
        )
        return popt, np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"  curve_fit failed: {e}")
        return None, None


def plot_series(ax, points, fit_points, color, label, marker="o"):
    if not points:
        return None, None

    N_all = [p["n_params"] for p in points]
    L_all = [p["val_loss"]  for p in points]

    fit_set = {p["name"] for p in fit_points}

    # solid markers = included in fit, hollow = excluded (SP breakdown)
    N_fit  = [p["n_params"] for p in points if p["name"] in fit_set]
    L_fit  = [p["val_loss"]  for p in points if p["name"] in fit_set]
    N_brok = [p["n_params"] for p in points if p["name"] not in fit_set]
    L_brok = [p["val_loss"]  for p in points if p["name"] not in fit_set]

    ax.scatter(N_fit,  L_fit,  color=color, s=80, marker=marker, zorder=5,
               label=f"{label} (fit points)")
    if N_brok:
        ax.scatter(N_brok, L_brok, color=color, s=80, marker=marker, zorder=5,
                   facecolors="none", linewidths=1.5,
                   label=f"{label} (SP breakdown — excluded from fit)")

    for p in points:
        ax.annotate(p["name"], (p["n_params"], p["val_loss"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=color)

    popt, perr = fit_power_law(fit_points)
    if popt is not None:
        a, alpha, c = popt
        N_range = np.logspace(np.log10(min(N_all) * 0.5), np.log10(max(N_all) * 5), 300)
        ax.plot(N_range, power_law(N_range, *popt), "--", color=color,
                label=rf"{label}: $L={a:.2f}·N^{{-{alpha:.3f}}}+{c:.2f}$  (α={alpha:.3f})")

        N_extrap = max(N_all) * 10
        L_extrap = power_law(N_extrap, *popt)
        dL = np.sqrt((N_extrap**(-alpha) * perr[0])**2 +
                     (-a * N_extrap**(-alpha) * np.log(N_extrap) * perr[1])**2 +
                     perr[2]**2)
        ax.scatter([N_extrap], [L_extrap], marker="*", s=250, color=color, zorder=6,
                   label=f"{label} 10× extrap: {L_extrap:.3f} ± {dL:.3f}")

        print(f"\n{label} power law:  L = {a:.4f} · N^(-{alpha:.4f}) + {c:.4f}")
        print(f"  α = {alpha:.4f} ± {perr[1]:.4f}")
        print(f"  10× extrap ({N_extrap:.2e} params): {L_extrap:.4f} ± {dL:.4f}")
        return popt, perr

    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sp_dir",    default="../part2/out")
    p.add_argument("--mup_dir",   default="out")
    p.add_argument("--plot_path", default=None)
    p.add_argument("--series",    default="wide", choices=["original", "wide"])
    args = p.parse_args()

    if args.plot_path is None:
        args.plot_path = f"plots/compare_{args.series}.png"

    configs    = SERIES[args.series]
    sp_points  = load_results(Path(args.sp_dir),  "SP",  configs)
    mup_points = load_results(Path(args.mup_dir), "µP",  configs)

    if not sp_points and not mup_points:
        print("No results found. Train models first.")
        return

    # For SP: only fit on the monotone-decreasing prefix; broken tail is still shown
    sp_fit_points, sp_broken = monotone_prefix(sorted(sp_points, key=lambda p: p["n_params"]))

    # For µP: fit on all points (µP shouldn't have a broken tail)
    mup_fit_points = sorted(mup_points, key=lambda p: p["n_params"])

    if sp_broken:
        broken_names = [p["name"] for p in sp_broken]
        print(f"\nSP broken tail (excluded from fit): {broken_names}")

    Path(args.plot_path).parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    plot_series(ax, sorted(sp_points,  key=lambda p: p["n_params"]),
                sp_fit_points,  color="steelblue", label="SP")
    plot_series(ax, sorted(mup_points, key=lambda p: p["n_params"]),
                mup_fit_points, color="crimson",   label="µP", marker="s")

    series_label = "Original Series" if args.series == "original" else "Wide Series (n_layer=4)"
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (non-embedding, log scale)", fontsize=12)
    ax.set_ylabel("Validation Loss after 1 epoch", fontsize=12)
    ax.set_title(f"SVG Transformer: SP vs µP Scaling Laws\n({series_label})", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=150)
    print(f"\nPlot → {args.plot_path}")

    # save fit data
    fit_path = Path(args.mup_dir) / f"scaling_fit_{args.series}.json"
    out = {
        "series":      args.series,
        "sp_points":   sp_points,
        "sp_broken":   sp_broken,
        "mup_points":  mup_points,
    }
    with open(fit_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Fit data → {fit_path}")


if __name__ == "__main__":
    main()
