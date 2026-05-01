"""
Train the best model (large_wide, SP) for multiple epochs.

Usage:
    # from part4/
    python train.py                             # default: large_wide, 3 epochs, lr=3e-3
    python train.py --lr 2e-3 --dropout 0.1    # custom hyperparams
    python train.py --resume out/best/ckpt.pt  # resume from checkpoint

Output (in --out_dir):
    ckpt.pt       — best checkpoint by val loss
    log.csv       — per-step metrics
    results.json  — final summary
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "part2"))
from model import GPT, GPTConfig
from configs import get_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",        default="large_wide")
    p.add_argument("--lr",            type=float, default=3e-3)
    p.add_argument("--min_lr_frac",   type=float, default=0.1)
    p.add_argument("--num_epochs",    type=int,   default=3)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--grad_accum",    type=int,   default=4)
    p.add_argument("--weight_decay",  type=float, default=0.1)
    p.add_argument("--beta1",         type=float, default=0.9)
    p.add_argument("--beta2",         type=float, default=0.95)
    p.add_argument("--dropout",       type=float, default=0.0)
    p.add_argument("--warmup_ratio",  type=float, default=0.05)
    p.add_argument("--eval_interval", type=int,   default=50)
    p.add_argument("--eval_batches",  type=int,   default=50)
    p.add_argument("--out_dir",       default="out/best")
    p.add_argument("--data_dir",      default="../part2/data")
    p.add_argument("--resume",        default=None, help="Path to ckpt.pt to resume from")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--compile",       action="store_true")
    return p.parse_args()


def get_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    t = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def load_bin(path):
    return np.memmap(path, dtype="uint16", mode="r")


def get_batch(data, pos, batch_size, block_size, device):
    n = batch_size * block_size
    # wrap around if needed
    if pos + n + 1 > len(data):
        pos = 0
    x = torch.from_numpy(np.array(data[pos:pos + n],     dtype=np.int64).reshape(batch_size, block_size)).to(device)
    y = torch.from_numpy(np.array(data[pos + 1:pos + n + 1], dtype=np.int64).reshape(batch_size, block_size)).to(device)
    return x, y, pos + n


@torch.no_grad()
def evaluate(model, val_data, batch_size, block_size, eval_batches, device):
    model.eval()
    losses = []
    pos = 0
    for _ in range(eval_batches):
        if pos + batch_size * block_size + 1 > len(val_data):
            pos = 0
        x, y, pos = get_batch(val_data, pos, batch_size, block_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"Device: {device}")

    cfg_dict = get_config(args.config)
    gpt_cfg = GPTConfig(
        vocab_size=4096,
        block_size=1024,
        n_layer=cfg_dict["n_layer"],
        n_head=cfg_dict["n_head"],
        n_embd=cfg_dict["n_embd"],
        n_ff=cfg_dict["n_ff"],
        dropout=args.dropout,
        bias=False,
    )
    model = GPT(gpt_cfg).to(device)
    n_params = model.count_params()

    print(f"\n{'─'*55}")
    print(f"  Model : {args.config}  ({n_params/1e6:.3f}M non-emb params)")
    print(f"  d_model={gpt_cfg.n_embd}  n_layers={gpt_cfg.n_layer}  n_heads={gpt_cfg.n_head}")
    print(f"  lr={args.lr:.2e}  dropout={args.dropout}  wd={args.weight_decay}")
    print(f"  epochs={args.num_epochs}")
    print(f"{'─'*55}\n")

    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed ({e}), skipping.")

    data_dir = Path(args.data_dir)
    train_data = load_bin(data_dir / "train.bin")
    val_data   = load_bin(data_dir / "val.bin")

    block_size           = gpt_cfg.block_size
    tokens_per_microstep = args.batch_size * block_size
    effective_batch      = tokens_per_microstep * args.grad_accum
    steps_per_epoch      = (len(train_data) - 1) // effective_batch
    total_steps          = steps_per_epoch * args.num_epochs
    warmup_steps         = max(1, int(total_steps * args.warmup_ratio))

    print(f"Train tokens    : {len(train_data):,}")
    print(f"Steps per epoch : {steps_per_epoch}")
    print(f"Total steps     : {total_steps}  ({args.num_epochs} epochs)")
    print(f"Effective batch : {effective_batch:,} tokens/step\n")

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type=device_type,
    )

    start_step = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from {args.resume} at step {start_step}, val_loss={best_val_loss:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_mode = "a" if args.resume else "w"
    log_file = open(out_dir / "log.csv", log_mode, newline="")
    log_writer = csv.writer(log_file)
    if not args.resume:
        log_writer.writerow(["step", "epoch", "train_loss", "val_loss", "lr", "tokens_per_sec", "gpu_mem_gb"])

    use_amp = device_type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_pos     = 0
    t_start       = time.time()

    for step in range(start_step, total_steps):
        epoch = step // steps_per_epoch + 1

        lr = get_lr(step, warmup_steps, total_steps, args.lr, args.lr * args.min_lr_frac)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        t0 = time.time()

        for _ in range(args.grad_accum):
            x, y, train_pos = get_batch(train_data, train_pos, args.batch_size, block_size, device)
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                _, loss = model(x, y)
            loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            step_loss += loss.item()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        dt = time.time() - t0
        tok_per_sec = effective_batch / dt
        gpu_mem_gb  = torch.cuda.memory_reserved() / 1e9 if device_type == "cuda" else 0.0

        val_loss = None
        if step % args.eval_interval == 0 or step == total_steps - 1:
            val_loss = evaluate(model, val_data, args.batch_size, block_size, args.eval_batches, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model": model.state_dict(), "config": gpt_cfg, "step": step,
                     "val_loss": val_loss, "args": vars(args)},
                    out_dir / "ckpt.pt",
                )
            print(
                f"[ep {epoch}/{args.num_epochs}] step {step % steps_per_epoch:4d}/{steps_per_epoch} | "
                f"loss {step_loss:.4f} | val {val_loss:.4f} | lr {lr:.2e} | "
                f"{tok_per_sec:,.0f} tok/s | {gpu_mem_gb:.2f} GB"
            )

        log_writer.writerow([
            step, epoch, f"{step_loss:.6f}",
            f"{val_loss:.6f}" if val_loss is not None else "",
            f"{lr:.6e}", f"{tok_per_sec:.1f}", f"{gpu_mem_gb:.3f}",
        ])
        log_file.flush()

    wall_time_s = time.time() - t_start
    log_file.close()

    results = {
        "config":                args.config,
        "n_params":              n_params,
        "lr":                    args.lr,
        "dropout":               args.dropout,
        "weight_decay":          args.weight_decay,
        "num_epochs":            args.num_epochs,
        "best_val_loss":         best_val_loss,
        "final_train_loss":      step_loss,
        "total_steps":           total_steps,
        "effective_batch_tokens": effective_batch,
        "wall_time_s":           round(wall_time_s, 1),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone in {wall_time_s/60:.1f} min  |  best val loss: {best_val_loss:.4f}")
    print(f"Results → {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
