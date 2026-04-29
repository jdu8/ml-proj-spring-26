"""
LR sweep for the µP Tiny model.

Under µP, the optimal LR found here transfers to all larger models — no retuning needed.

Usage:
    python lr_sweep_mup.py
    python lr_sweep_mup.py --lrs 1e-3 3e-3 1e-2 3e-2 1e-1
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="tiny")
    p.add_argument("--out_dir",    default="out/sweep")
    p.add_argument("--data_dir",   default="../part2/data")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lrs",        nargs="+", type=float, default=DEFAULT_LRS)
    return p.parse_args()


def lr_label(lr):
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def main():
    args = parse_args()
    sweep_dir = Path(args.out_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"µP LR sweep on config='{args.config}'")
    print(f"LRs: {[f'{lr:.1e}' for lr in args.lrs]}\n")

    results = []
    for lr in args.lrs:
        run_dir = sweep_dir / f"lr_{lr_label(lr)}"
        print(f"\n{'='*60}")
        print(f"  µP LR = {lr:.2e}  →  {run_dir}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "train_mup.py",
            "--config",     args.config,
            "--lr",         str(lr),
            "--out_dir",    str(run_dir),
            "--data_dir",   args.data_dir,
            "--batch_size", str(args.batch_size),
            "--grad_accum", str(args.grad_accum),
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"  WARNING: run failed (exit {ret.returncode}), skipping.")
            continue

        rp = run_dir / "results.json"
        if rp.exists():
            with open(rp) as f:
                r = json.load(f)
            entry = {"lr": lr, "val_loss": r["best_val_loss"], "run_dir": str(run_dir)}
            results.append(entry)
            print(f"  → best val loss: {r['best_val_loss']:.4f}")

    if not results:
        print("No successful runs.")
        return

    results.sort(key=lambda x: x["val_loss"])
    best = results[0]

    print(f"\n{'='*60}")
    print("µP LR Sweep Summary:")
    print(f"  {'LR':>10}  {'Val Loss':>10}")
    for r in results:
        marker = "  ← best" if r is best else ""
        print(f"  {r['lr']:>10.2e}  {r['val_loss']:>10.4f}{marker}")

    summary = {"config": args.config, "parameterization": "muP",
                "results": results, "best_lr": best["lr"]}
    sp = sweep_dir / "sweep_summary_mup.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBest µP LR: {best['lr']:.2e}  (val loss {best['val_loss']:.4f})")
    print(f"Summary → {sp}")


if __name__ == "__main__":
    main()
