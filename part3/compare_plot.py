"""
Plot SP (Part 2) vs µP (Part 3) scaling curves on the same graph,
fit power laws to both, and report the 10× extrapolation.

Usage:
    python compare_plot.py
    python compare_plot.py --sp_dir ../part2/out --mup_dir out --plot_path plots/compare.png
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


def fit(points):
    if len(points) < 3:
        return None, None
    N = np.array([p["n_params"] for p in points], dtype=float)
    L = np.array([p["val_loss"] for p in points], dtype=float)
    try:
        popt, pcov = curve_fit(
            power_law, N, L,
            p0=[5.0, 0.1, max(L.min() - 0.5, 0.01)],
            bounds=([0, 1e-4, 0], [1e8, 5.0, L.min() - 1e-4]),
            maxfev=20000,
        )
        return popt, np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"  curve_fit failed: {e}")
        return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sp_dir",    default="../part2/out")
    p.add_argument("--mup_dir",   default="out")
    p.add_argument("--plot_path", default="plots/compare.png")
    p.add_argument("--series",    default="wide", choices=["original", "wide"],
                   help="Model series to compare (default: wide — the µP-clean series)")
    args = p.parse_args()

    configs    = SERIES[args.series]
    sp_points  = load_results(Path(args.sp_dir),  "SP",  configs)
    mup_points = load_results(Path(args.mup_dir), "µP",  configs)

    if not sp_points and not mup_points:
        print("No results found. Train models first.")
        return

    Path(args.plot_path).parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"SP": "steelblue", "µP": "crimson"}

    fit_results = {}
    for label, points, color in [("SP", sp_points, colors["SP"]), ("µP", mup_points, colors["µP"])]:
        if not points:
            continue
        N_pts = [pt["n_params"] for pt in points]
        L_pts = [pt["val_loss"]  for pt in points]
        names = [pt["name"]      for pt in points]

        ax.scatter(N_pts, L_pts, color=color, s=80, zorder=5, label=f"{label} (trained)")
        for x, y, name in zip(N_pts, L_pts, names):
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=color)

        popt, perr = fit(points)
        if popt is not None:
            a, alpha, c = popt
            N_fit = np.logspace(np.log10(min(N_pts) * 0.5), np.log10(max(N_pts) * 3), 300)
            ax.plot(N_fit, power_law(N_fit, *popt), "--", color=color,
                    label=rf"{label}: $L={a:.2f}N^{{-{alpha:.3f}}}+{c:.2f}$")

            N_extrap = max(N_pts) * 10
            L_extrap = power_law(N_extrap, *popt)
            dL = np.sqrt((N_extrap**(-alpha) * perr[0])**2 +
                         (-a * N_extrap**(-alpha) * np.log(N_extrap) * perr[1])**2 +
                         perr[2]**2)
            ax.scatter([N_extrap], [L_extrap], marker="*", s=200, color=color, zorder=5,
                       label=f"{label} 10× extrap: {L_extrap:.3f}±{dL:.3f}")

            fit_results[label] = {
                "a": float(a), "alpha": float(alpha), "c": float(c),
                "extrap_N": N_extrap, "extrap_L": float(L_extrap), "extrap_err": float(dL),
            }
            print(f"\n{label} power law:  L = {a:.4f} · N^(-{alpha:.4f}) + {c:.4f}")
            print(f"  α = {alpha:.4f} ± {perr[1]:.4f}")
            print(f"  10× extrap ({N_extrap:.2e} params): {L_extrap:.4f} ± {dL:.4f}")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (non-embedding, log scale)", fontsize=12)
    ax.set_ylabel("Validation Loss after 1 epoch", fontsize=12)
    ax.set_title("SVG Transformer: SP vs µP Scaling Laws", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=150)
    print(f"\nPlot → {args.plot_path}")

    out = {"SP": fit_results.get("SP"), "muP": fit_results.get("µP"),
           "sp_points": sp_points, "mup_points": mup_points}
    fit_path = Path(args.mup_dir) / "scaling_fit_comparison.json"
    with open(fit_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Fit data → {fit_path}")


if __name__ == "__main__":
    main()
