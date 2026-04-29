"""
Compare SP (Part 2 wide series) vs µP (Part 3) scaling curves.

Loads results from both parameterizations, fits power laws L = a·N^{-α} + c,
plots them on the same graph, and produces a 10× extrapolation prediction.

Usage:
    python scaling_plot_mup.py
    python scaling_plot_mup.py --sp_dir ../part2/out --mup_dir out --plot_path plots/comparison.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


WIDE_CONFIGS = ["tiny", "small_wide", "medium_wide", "large_wide", "xl_wide"]


def power_law(N, a, alpha, c):
    return a * N ** (-alpha) + c


def load_results(results_dir: Path, configs: list) -> list:
    points = []
    for name in configs:
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
        print(f"  curve_fit failed: {e}")
        return None, None


def extrapolate_10x(n_params_list, popt, perr, label=""):
    """Predict loss at 10× the largest model size and print uncertainty."""
    N_extrap = max(n_params_list) * 10
    L_extrap = power_law(N_extrap, *popt)
    a, alpha, c = popt
    # Error propagation
    dL_da     = N_extrap ** (-alpha)
    dL_dalpha = -a * N_extrap ** (-alpha) * np.log(N_extrap)
    dL_dc     = 1.0
    L_err = np.sqrt((dL_da * perr[0])**2 + (dL_dalpha * perr[1])**2 + (dL_dc * perr[2])**2)
    print(f"\n10× extrapolation ({label}):")
    print(f"  N = {N_extrap:.3e} params")
    print(f"  Predicted loss = {L_extrap:.4f} ± {L_err:.4f}")
    return float(N_extrap), float(L_extrap), float(L_err)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sp_dir",     default="../part2/out",  help="Part 2 SP results directory")
    p.add_argument("--mup_dir",    default="out",           help="Part 3 µP results directory")
    p.add_argument("--plot_path",  default="plots/comparison.png")
    p.add_argument("--configs",    nargs="+", default=WIDE_CONFIGS)
    # Optionally exclude bad SP configs (those that diverged due to LR mismatch)
    p.add_argument("--sp_fit_configs",  nargs="+", default=None,
                   help="Subset of configs to use for SP power-law fit (default: all that loaded)")
    p.add_argument("--mup_fit_configs", nargs="+", default=None,
                   help="Subset of configs to use for µP power-law fit (default: all that loaded)")
    args = p.parse_args()

    # ── load ───────────────────────────────────────────────────────────────────
    print("Loading SP results from:", args.sp_dir)
    sp_points = load_results(Path(args.sp_dir), args.configs)
    print(f"  Found {len(sp_points)} SP models: {[pt['name'] for pt in sp_points]}")

    print("\nLoading µP results from:", args.mup_dir)
    mup_points = load_results(Path(args.mup_dir), args.configs)
    print(f"  Found {len(mup_points)} µP models: {[pt['name'] for pt in mup_points]}")

    if not sp_points and not mup_points:
        print("No results found. Train models first.")
        return

    # ── filter for power-law fit ────────────────────────────────────────────────
    # Only include monotonically improving points (exclude diverged large models)
    def filter_monotone(points):
        """Keep a prefix where loss strictly decreases."""
        filtered = []
        best = float("inf")
        for pt in sorted(points, key=lambda x: x["n_params"]):
            if pt["val_loss"] < best:
                best = pt["val_loss"]
                filtered.append(pt)
        return filtered

    sp_fit  = [pt for pt in sp_points  if not args.sp_fit_configs  or pt["name"] in args.sp_fit_configs]
    mup_fit = [pt for pt in mup_points if not args.mup_fit_configs or pt["name"] in args.mup_fit_configs]

    if not args.sp_fit_configs:
        sp_fit = filter_monotone(sp_fit)
        print(f"\nSP fit subset (monotone): {[pt['name'] for pt in sp_fit]}")
    if not args.mup_fit_configs:
        mup_fit = filter_monotone(mup_fit)
        print(f"µP fit subset (monotone): {[pt['name'] for pt in mup_fit]}")

    # ── fit ────────────────────────────────────────────────────────────────────
    sp_popt = sp_perr = mup_popt = mup_perr = None

    if len(sp_fit) >= 2:
        sp_N = [pt["n_params"] for pt in sp_fit]
        sp_L = [pt["val_loss"] for pt in sp_fit]
        sp_popt, sp_perr = fit_power_law(sp_N, sp_L)
        if sp_popt is not None:
            a, alpha, c = sp_popt
            print(f"\nSP  fit: L = {a:.4f} · N^(-{alpha:.4f}) + {c:.4f}")
            print(f"         uncertainty: α ± {sp_perr[1]:.4f}")

    if len(mup_fit) >= 2:
        mup_N = [pt["n_params"] for pt in mup_fit]
        mup_L = [pt["val_loss"] for pt in mup_fit]
        mup_popt, mup_perr = fit_power_law(mup_N, mup_L)
        if mup_popt is not None:
            a, alpha, c = mup_popt
            print(f"\nµP  fit: L = {a:.4f} · N^(-{alpha:.4f}) + {c:.4f}")
            print(f"         uncertainty: α ± {mup_perr[1]:.4f}")

    # ── 10× extrapolation ──────────────────────────────────────────────────────
    extrap_results = {}
    all_n = ([pt["n_params"] for pt in sp_points] + [pt["n_params"] for pt in mup_points])
    if all_n:
        largest_N = max(all_n)
        if sp_popt is not None:
            n_e, l_e, l_err = extrapolate_10x([largest_N], sp_popt, sp_perr, label="SP")
            extrap_results["SP"] = {"N": n_e, "L": l_e, "L_err": l_err}
        if mup_popt is not None:
            n_e, l_e, l_err = extrapolate_10x([largest_N], mup_popt, mup_perr, label="µP")
            extrap_results["muP"] = {"N": n_e, "L": l_e, "L_err": l_err}

    # ── plot ───────────────────────────────────────────────────────────────────
    Path(args.plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    # SP scatter + fit
    if sp_points:
        sp_ns = [pt["n_params"] for pt in sp_points]
        sp_ls = [pt["val_loss"] for pt in sp_points]
        sp_names = [pt["name"] for pt in sp_points]
        ax.scatter(sp_ns, sp_ls, color="steelblue", marker="o", s=80, zorder=5,
                   label="SP (Part 2 wide series)", alpha=0.9)
        for x, y, name in zip(sp_ns, sp_ls, sp_names):
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7,
                        color="steelblue")
        if sp_popt is not None:
            a, alpha, c = sp_popt
            N_range = np.logspace(np.log10(min(sp_ns) * 0.5), np.log10(max(all_n) * 12), 300)
            ax.plot(N_range, power_law(N_range, *sp_popt), "--",
                    color="steelblue", alpha=0.7,
                    label=rf"SP: $L = {a:.2f}N^{{-{alpha:.3f}}} + {c:.2f}$")

    # µP scatter + fit
    if mup_points:
        mup_ns = [pt["n_params"] for pt in mup_points]
        mup_ls = [pt["val_loss"] for pt in mup_points]
        mup_names = [pt["name"] for pt in mup_points]
        ax.scatter(mup_ns, mup_ls, color="crimson", marker="s", s=80, zorder=5,
                   label="µP (Part 3)", alpha=0.9)
        for x, y, name in zip(mup_ns, mup_ls, mup_names):
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, -10), fontsize=7,
                        color="crimson")
        if mup_popt is not None:
            a, alpha, c = mup_popt
            N_range = np.logspace(np.log10(min(mup_ns) * 0.5), np.log10(max(all_n) * 12), 300)
            ax.plot(N_range, power_law(N_range, *mup_popt), "-",
                    color="crimson", alpha=0.7,
                    label=rf"µP: $L = {a:.2f}N^{{-{alpha:.3f}}} + {c:.2f}$")

    # 10× extrapolation markers
    for tag, er in extrap_results.items():
        color = "steelblue" if tag == "SP" else "crimson"
        ax.scatter([er["N"]], [er["L"]], color=color, marker="*", s=220, zorder=6,
                   label=f"10× {tag}: {er['L']:.3f} ± {er['L_err']:.3f}")
        ax.errorbar([er["N"]], [er["L"]], yerr=er["L_err"], fmt="none", color=color, capsize=4)
        ax.axvline(er["N"], color=color, linestyle=":", alpha=0.3)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (non-embedding, log scale)", fontsize=12)
    ax.set_ylabel("Validation Loss after 1 epoch", fontsize=12)
    ax.set_title("SP vs µP Scaling: SVG Transformer (Wide Series)", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=150)
    print(f"\nPlot → {args.plot_path}")

    # ── save fit data ──────────────────────────────────────────────────────────
    def _to_dict(popt, perr):
        if popt is None:
            return None
        return {"a": float(popt[0]), "alpha": float(popt[1]), "c": float(popt[2]),
                "a_err": float(perr[0]), "alpha_err": float(perr[1]), "c_err": float(perr[2])}

    out = {
        "SP":  {"points": sp_points,  "fit": _to_dict(sp_popt, sp_perr),  "fit_configs": [pt["name"] for pt in sp_fit]},
        "muP": {"points": mup_points, "fit": _to_dict(mup_popt, mup_perr), "fit_configs": [pt["name"] for pt in mup_fit]},
        "extrapolation_10x": extrap_results,
    }
    fit_path = Path(args.mup_dir) / "scaling_comparison.json"
    with open(fit_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Fit data → {fit_path}")


if __name__ == "__main__":
    main()
