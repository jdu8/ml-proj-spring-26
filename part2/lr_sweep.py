"""
Learning rate sweep on the Tiny model (or any config).

Runs train.py once per LR value, collects best val loss for each,
and saves a summary so you can pick the best LR for Part 2 scaling runs.

Usage:
    python lr_sweep.py                        # sweep Tiny with default LRs
    python lr_sweep.py --config small         # sweep a different size
    python lr_sweep.py --lrs 1e-4 3e-4 1e-3  # custom LR list
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

# 6 values evenly spaced on log scale from 1e-4 to 3e-2
DEFAULT_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="tiny")
    p.add_argument("--out_dir",    default="out/sweep")
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lrs",        nargs="+", type=float, default=DEFAULT_LRS)
    return p.parse_args()


def lr_label(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def main():
    args = parse_args()
    sweep_dir = Path(args.out_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweeping {len(args.lrs)} learning rates on config='{args.config}'")
    print(f"LRs: {[f'{lr:.1e}' for lr in args.lrs]}\n")

    results = []
    for lr in args.lrs:
        run_dir = sweep_dir / f"lr_{lr_label(lr)}"
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  LR = {lr:.2e}  →  {run_dir}")
        print(sep)

        cmd = [
            sys.executable, "train.py",
            "--config",     args.config,
            "--lr",         str(lr),
            "--out_dir",    str(run_dir),
            "--data_dir",   args.data_dir,
            "--batch_size", str(args.batch_size),
            "--grad_accum", str(args.grad_accum),
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"  WARNING: run failed (exit code {ret.returncode}), skipping.")
            continue

        result_path = run_dir / "results.json"
        if result_path.exists():
            with open(result_path) as f:
                r = json.load(f)
            entry = {"lr": lr, "val_loss": r["best_val_loss"], "run_dir": str(run_dir)}
            results.append(entry)
            print(f"  → best val loss: {r['best_val_loss']:.4f}")

    if not results:
        print("No successful runs. Check errors above.")
        return

    results.sort(key=lambda x: x["val_loss"])
    best = results[0]

    print(f"\n{'=' * 60}")
    print("LR Sweep Summary (sorted by best val loss):")
    print(f"  {'LR':>10}  {'Val Loss':>10}")
    for r in results:
        marker = "  ← best" if r is best else ""
        print(f"  {r['lr']:>10.2e}  {r['val_loss']:>10.4f}{marker}")

    summary = {"config": args.config, "results": results, "best_lr": best["lr"]}
    summary_path = sweep_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBest LR: {best['lr']:.2e}  (val loss {best['val_loss']:.4f})")
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
