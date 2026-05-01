"""
Small hyperparameter sweep for Part 4 (large_wide, SP, 1 epoch per run).

Sweeps LR x dropout for enough runs to report meaningful results.
After the sweep, train the best config for multiple epochs with train.py.

Usage:
    python hp_sweep.py                  # default grid (6 runs)
    python hp_sweep.py --lrs 1e-3 3e-3  # custom LR list
    python hp_sweep.py --no_dropout     # LR sweep only (3 runs)

Default grid:
    LRs:     [1e-3, 2e-3, 3e-3]
    Dropout: [0.0, 0.1]
    Total:   6 runs × ~3 min each on an A100 ≈ 18 min
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_LRS     = [1e-3, 2e-3, 3e-3]
DEFAULT_DROPOUTS = [0.0, 0.1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="large_wide")
    p.add_argument("--out_dir",      default="out/sweep")
    p.add_argument("--data_dir",     default="../part2/data")
    p.add_argument("--lrs",          nargs="+", type=float, default=DEFAULT_LRS)
    p.add_argument("--dropouts",     nargs="+", type=float, default=DEFAULT_DROPOUTS)
    p.add_argument("--no_dropout",   action="store_true", help="Only sweep LR, fix dropout=0.0")
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--grad_accum",   type=int,   default=4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    return p.parse_args()


def run_label(lr, dropout):
    lr_s = f"{lr:.0e}".replace("e-0", "e-")
    return f"lr{lr_s}_drop{dropout}"


def main():
    args = parse_args()
    dropouts = [0.0] if args.no_dropout else args.dropouts
    sweep_dir = Path(args.out_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    grid = [(lr, d) for lr in args.lrs for d in dropouts]
    total = len(grid)
    print(f"HP sweep: {total} runs on config='{args.config}'")
    print(f"  LRs:     {[f'{lr:.1e}' for lr in args.lrs]}")
    print(f"  Dropout: {dropouts}\n")

    results = []
    for i, (lr, dropout) in enumerate(grid, 1):
        run_dir = sweep_dir / run_label(lr, dropout)
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  Run {i}/{total}: lr={lr:.2e}  dropout={dropout}  →  {run_dir}")
        print(sep)

        cmd = [
            sys.executable, "train.py",
            "--config",       args.config,
            "--lr",           str(lr),
            "--dropout",      str(dropout),
            "--weight_decay", str(args.weight_decay),
            "--num_epochs",   "1",
            "--out_dir",      str(run_dir),
            "--data_dir",     args.data_dir,
            "--batch_size",   str(args.batch_size),
            "--grad_accum",   str(args.grad_accum),
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"  WARNING: run failed (exit {ret.returncode}), skipping.")
            continue

        result_path = run_dir / "results.json"
        if result_path.exists():
            with open(result_path) as f:
                r = json.load(f)
            entry = {
                "lr":       lr,
                "dropout":  dropout,
                "val_loss": r["best_val_loss"],
                "run_dir":  str(run_dir),
            }
            results.append(entry)
            print(f"  → best val loss: {r['best_val_loss']:.4f}")

    if not results:
        print("No successful runs.")
        return

    results.sort(key=lambda x: x["val_loss"])
    best = results[0]

    print(f"\n{'=' * 60}")
    print("Sweep Summary (sorted by best val loss):")
    print(f"  {'LR':>8}  {'Dropout':>8}  {'Val Loss':>10}")
    for r in results:
        marker = "  ← best" if r is best else ""
        print(f"  {r['lr']:>8.2e}  {r['dropout']:>8.1f}  {r['val_loss']:>10.4f}{marker}")

    summary = {
        "config":   args.config,
        "results":  results,
        "best_lr":      best["lr"],
        "best_dropout": best["dropout"],
        "best_val_loss": best["val_loss"],
    }
    summary_path = sweep_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest config: lr={best['lr']:.2e}  dropout={best['dropout']}")
    print(f"  val loss: {best['val_loss']:.4f}")
    print(f"\nSummary → {summary_path}")
    print(f"\nTo train best config for full run:")
    print(f"  python train.py --config {args.config} --lr {best['lr']:.2e} "
          f"--dropout {best['dropout']} --num_epochs 3 --out_dir out/best")


if __name__ == "__main__":
    main()
