"""
Train a µP GPT model for exactly 1 epoch.

Run make_base_shapes.py once first, then:
    python train_mup.py --config tiny  --lr 3e-3 --out_dir out/tiny
    python train_mup.py --config xl    --lr 3e-3 --out_dir out/xl

The SAME --lr transfers across all model sizes — that is the µP guarantee.
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
from mup import set_base_shapes

sys.path.insert(0, str(Path(__file__).parent.parent / "part2"))
from configs import get_config

from model_mup import GPTMuP, GPTConfig

BASE_SHAPES = Path(__file__).parent / "base_shapes.bsh"


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="tiny")
    p.add_argument("--lr",           type=float, default=3e-3)
    p.add_argument("--min_lr_frac",  type=float, default=0.1)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--grad_accum",   type=int,   default=4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1",        type=float, default=0.9)
    p.add_argument("--beta2",        type=float, default=0.95)
    p.add_argument("--dropout",      type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--eval_interval",type=int,   default=50)
    p.add_argument("--eval_batches", type=int,   default=50)
    p.add_argument("--out_dir",      default="out/tiny")
    p.add_argument("--data_dir",     default="../part2/data")
    p.add_argument("--base_shapes",  default=str(BASE_SHAPES))
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--compile",      action="store_true")
    return p.parse_args()


# ── LR SCHEDULE ────────────────────────────────────────────────────────────────

def get_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    t = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * t))


# ── DATA ───────────────────────────────────────────────────────────────────────

def load_bin(path):
    return np.memmap(path, dtype="uint16", mode="r")


def get_batch(data, pos, batch_size, block_size, device):
    n = batch_size * block_size
    x = torch.from_numpy(np.array(data[pos     : pos + n],     dtype=np.int64).reshape(batch_size, block_size)).to(device)
    y = torch.from_numpy(np.array(data[pos + 1 : pos + n + 1], dtype=np.int64).reshape(batch_size, block_size)).to(device)
    return x, y


# ── EVALUATION ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_data, batch_size, block_size, eval_batches, device):
    model.eval()
    losses = []
    pos = 0
    tpb = batch_size * block_size
    for _ in range(eval_batches):
        if pos + tpb + 1 > len(val_data):
            pos = 0
        x, y = get_batch(val_data, pos, batch_size, block_size, device)
        pos += tpb
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    # ── build model ────────────────────────────────────────────────────────────
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
    model = GPTMuP(gpt_cfg)

    # µP: annotate params with infshape BEFORE weight init
    if not Path(args.base_shapes).exists():
        raise FileNotFoundError(
            f"base_shapes.bsh not found at {args.base_shapes}. "
            "Run: python make_base_shapes.py"
        )
    set_base_shapes(model, args.base_shapes)
    model.init_weights()          # mup-aware init (after set_base_shapes)
    model = model.to(device)

    n_params = model.count_params()
    n_params_total = model.count_params(non_embedding=False)
    print(f"\n{'─'*55}")
    print(f"  Model (µP) : {args.config}")
    print(f"  d_model={gpt_cfg.n_embd}  n_layers={gpt_cfg.n_layer}  "
          f"n_heads={gpt_cfg.n_head}  d_ff={gpt_cfg.n_ff}")
    print(f"  Params (non-emb) : {n_params/1e6:.3f}M")
    print(f"  Params (total)   : {n_params_total/1e6:.3f}M")
    if torch.cuda.is_available():
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"{'─'*55}\n")

    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed: {e}")

    # ── data ───────────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    train_data = load_bin(data_dir / "train.bin")
    val_data   = load_bin(data_dir / "val.bin")

    block_size           = gpt_cfg.block_size
    tokens_per_microstep = args.batch_size * block_size
    effective_batch      = tokens_per_microstep * args.grad_accum
    total_steps          = (len(train_data) - 1) // effective_batch
    warmup_steps         = max(1, int(total_steps * args.warmup_ratio))

    print(f"Train tokens : {len(train_data):,}")
    print(f"Total steps  : {total_steps}  (1 epoch, µP)")
    print(f"Warmup steps : {warmup_steps}")
    print(f"Eff. batch   : {effective_batch:,} tokens/step")

    # ── optimizer (MuAdamW applies µP LR multipliers per param) ───────────────
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type=device_type,
    )

    # ── output ─────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file   = open(out_dir / "log.csv", "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_sec", "gpu_mem_gb"])

    # ── AMP ────────────────────────────────────────────────────────────────────
    use_amp = device_type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── training loop ──────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    train_pos     = 0
    t_start       = time.time()

    for step in range(total_steps):
        t0 = time.time()
        lr = get_lr(step, warmup_steps, total_steps, args.lr, args.lr * args.min_lr_frac)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, train_pos, args.batch_size, block_size, device)
            train_pos += tokens_per_microstep
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
        gpu_mem_gb  = torch.cuda.memory_allocated() / 1e9 if device_type == "cuda" else 0.0

        val_loss = None
        if step % args.eval_interval == 0 or step == total_steps - 1:
            val_loss = evaluate(model, val_data, args.batch_size, block_size, args.eval_batches, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model": model.state_dict(), "config": gpt_cfg, "step": step, "val_loss": val_loss},
                    out_dir / "ckpt.pt",
                )
            print(
                f"step {step:4d}/{total_steps} | "
                f"loss {step_loss:.4f} | val {val_loss:.4f} | "
                f"lr {lr:.2e} | {tok_per_sec:,.0f} tok/s | {gpu_mem_gb:.2f} GB"
            )

        log_writer.writerow([
            step, f"{step_loss:.6f}",
            f"{val_loss:.6f}" if val_loss is not None else "",
            f"{lr:.6e}", f"{tok_per_sec:.1f}", f"{gpu_mem_gb:.3f}",
        ])
        log_file.flush()

    wall_time_s = time.time() - t_start
    log_file.close()

    results = {
        "config":                 args.config,
        "parameterization":       "muP",
        "n_params":               n_params,
        "lr":                     args.lr,
        "best_val_loss":          best_val_loss,
        "final_train_loss":       step_loss,
        "total_steps":            total_steps,
        "effective_batch_tokens": effective_batch,
        "wall_time_s":            round(wall_time_s, 1),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone in {wall_time_s/60:.1f} min  |  best val loss: {best_val_loss:.4f}")
    print(f"Results → {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
