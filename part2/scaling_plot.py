"""
Fit the power law  L = a * N^{-alpha} + c  and produce the scaling plot.

Run after all 5 model sizes have been trained:
    python scaling_plot.py
    python scaling_plot.py --results_dir out --plot_path plots/scaling.png

Also prints the 10× extrapolation prediction required for Part 3.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


CONFIGS_ORDER = ["tiny", "small", "medium", "large", "xl"]


def power_law(N, a, alpha, c):
    return a * N ** (-alpha) + c


def load_results(results_dir: Path) -> list[dict]:
    points = []
    for name in CONFIGS_ORDER:
        p = results_dir / name / "results.json"
        if p.exists():
            with open(p) as f:
                r = json.load(f)
            points.append({
                "name":      name,
                "n_params":  r["n_params"],
                "val_loss":  r["best_val_loss"],
                "wall_time": r.get("wall_time_s"),
            })
        else:
            print(f"  Missing: {p}")
    return points


def fit_power_law(n_params, val_losses):
    N = np.array(n_params, dtype=float)
    L = np.array(val_losses, dtype=float)
    p0     = [5.0, 0.1, max(L.min() - 0.5, 0.01)]
    bounds = ([0, 1e-4, 0], [1e8, 5.0, L.min() - 1e-4])
    try:
        popt, pcov = curve_fit(power_law, N, L, p0=p0, bounds=bounds, maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except Exception as e:
        print(f"curve_fit failed: {e}")
        return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="out")
    p.add_argument("--plot_path",   default="plots/scaling.png")
    args = p.parse_args()

    points = load_results(Path(args.results_dir))
    if len(points) < 2:
        print(f"Need ≥2 trained models. Found: {[pt['name'] for pt in points]}")
        return

    n_params   = [pt["n_params"]  for pt in points]
    val_losses = [pt["val_loss"]  for pt in points]
    names      = [pt["name"]      for pt in points]

    popt, perr = fit_power_law(n_params, val_losses)

    Path(args.plot_path).parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(n_params, val_losses, color="steelblue", zorder=5, s=80, label="Trained models")
    for x, y, name in zip(n_params, val_losses, names):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)

    extrap_result = None
    if popt is not None:
        a, alpha, c = popt
        N_fit = np.logspace(np.log10(min(n_params) * 0.5), np.log10(max(n_params) * 3), 300)
        ax.plot(N_fit, power_law(N_fit, *popt), "r--",
                label=rf"$L = {a:.2f}\,N^{{-{alpha:.3f}}} + {c:.2f}$")

        # 10× extrapolation
        N_extrap  = max(n_params) * 10
        L_extrap  = power_law(N_extrap, *popt)
        # propagate uncertainty from perr
        dL_da     = N_extrap ** (-alpha)
        dL_dalpha = -a * N_extrap ** (-alpha) * np.log(N_extrap)
        dL_dc     = 1.0
        L_err = np.sqrt((dL_da * perr[0])**2 + (dL_dalpha * perr[1])**2 + (dL_dc * perr[2])**2)

        ax.axvline(N_extrap, color="orange", linestyle=":", alpha=0.6)
        ax.scatter([N_extrap], [L_extrap], color="orange", marker="*", s=200, zorder=5,
                   label=f"10× extrap: {L_extrap:.3f} ± {L_err:.3f}")

        extrap_result = {"N": N_extrap, "L": float(L_extrap), "L_err": float(L_err)}

        print(f"\nPower law fit: L = {a:.4f} * N^(-{alpha:.4f}) + {c:.4f}")
        print(f"Uncertainty:   a ± {perr[0]:.4f},  α ± {perr[1]:.4f},  c ± {perr[2]:.4f}")
        print(f"\n10× extrapolation ({N_extrap:.2e} params): L = {L_extrap:.4f} ± {L_err:.4f}")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (non-embedding, log scale)", fontsize=12)
    ax.set_ylabel("Validation Loss after 1 epoch", fontsize=12)
    ax.set_title("SVG Transformer Scaling Law", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=150)
    print(f"\nPlot → {args.plot_path}")

    # save fit data for Part 3 comparison
    out = {
        "parameterization": "SP",
        "points": points,
        "fit":    {"a": float(popt[0]), "alpha": float(popt[1]), "c": float(popt[2])} if popt is not None else None,
        "fit_uncertainty": {"a": float(perr[0]), "alpha": float(perr[1]), "c": float(perr[2])} if perr is not None else None,
        "extrapolation_10x": extrap_result,
    }
    fit_path = Path(args.results_dir) / "scaling_fit_SP.json"
    with open(fit_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Fit data → {fit_path}")


if __name__ == "__main__":
    main()
