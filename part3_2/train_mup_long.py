"""
Extended µP training with early stopping.

Trains beyond 1 epoch using patience-based early stopping: stops when val loss
has not improved for --patience consecutive evaluation intervals.

Can resume from existing checkpoints (e.g., Part 3 1-epoch runs) if the
--config and --lr match what was used to produce the checkpoint.

LR schedule: warmup + cosine decay over a max_steps budget
(max_epochs × epoch_steps). Training stops early via patience, so max_epochs
just defines the LR decay horizon — it does not force that many epochs.

Usage:
    # fresh run
    python train_mup_long.py --config xl_wide --lr 3e-3 --out_dir out/xl_wide_long

    # resume from the 1-epoch checkpoint (only if config + lr match)
    python train_mup_long.py --config xl_wide --lr 3e-3 --out_dir out/xl_wide
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model_mup import GPTMuP, GPTConfig, make_mup_model
from configs import get_config


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="tiny")
    p.add_argument("--lr",           type=float, default=3e-3)
    p.add_argument("--min_lr_frac",  type=float, default=0.1, help="min_lr = lr * min_lr_frac")
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--grad_accum",   type=int,   default=4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1",        type=float, default=0.9)
    p.add_argument("--beta2",        type=float, default=0.95)
    p.add_argument("--dropout",      type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05,
                   help="Warmup as fraction of one epoch")
    p.add_argument("--max_epochs",   type=int,   default=5,
                   help="LR cosine decay horizon (actual training stops earlier via patience)")
    p.add_argument("--patience",     type=int,   default=100,
                   help="Stop after this many evals with no val-loss improvement")
    p.add_argument("--eval_interval",type=int,   default=50)
    p.add_argument("--eval_batches", type=int,   default=50)
    p.add_argument("--out_dir",      default="out/tiny_long")
    p.add_argument("--data_dir",     default="../part2/data")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--compile",      action="store_true")
    return p.parse_args()


# ── LR SCHEDULE ────────────────────────────────────────────────────────────────

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / max_steps
    if step >= max_steps:
        return min_lr
    t = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * t))


# ── DATA ───────────────────────────────────────────────────────────────────────

def load_bin(path) -> np.ndarray:
    return np.memmap(path, dtype="uint16", mode="r")


def get_batch(data: np.ndarray, pos: int, batch_size: int, block_size: int, device):
    n = batch_size * block_size
    # wrap around so training can continue past one epoch
    if pos + n + 1 > len(data):
        pos = 0
    x = torch.from_numpy(np.array(data[pos     : pos + n],     dtype=np.int64).reshape(batch_size, block_size)).to(device)
    y = torch.from_numpy(np.array(data[pos + 1 : pos + n + 1], dtype=np.int64).reshape(batch_size, block_size)).to(device)
    return x, y, pos + n


# ── EVALUATION ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_data, batch_size, block_size, eval_batches, device):
    model.eval()
    losses = []
    pos = 0
    tokens_per_batch = batch_size * block_size
    for _ in range(eval_batches):
        if pos + tokens_per_batch + 1 > len(val_data):
            pos = 0
        x = torch.from_numpy(np.array(val_data[pos     : pos + tokens_per_batch],     dtype=np.int64).reshape(batch_size, block_size)).to(device)
        y = torch.from_numpy(np.array(val_data[pos + 1 : pos + tokens_per_batch + 1], dtype=np.int64).reshape(batch_size, block_size)).to(device)
        pos += tokens_per_batch
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


# ── CHECKPOINT RESUME ──────────────────────────────────────────────────────────

def try_load_checkpoint(out_dir: Path, config_name: str, lr: float, device):
    """
    Load checkpoint and resume state if config and LR match the current run.
    Returns (model_state_dict, start_step, best_val_loss, train_pos) or None.
    """
    ckpt_path    = out_dir / "ckpt.pt"
    results_path = out_dir / "results.json"

    if not ckpt_path.exists():
        return None

    # Check LR match from results.json (avoid loading the full model just to reject)
    if results_path.exists():
        with open(results_path) as f:
            prev = json.load(f)
        if prev.get("config") != config_name:
            print(f"  Checkpoint config '{prev.get('config')}' ≠ requested '{config_name}' — starting fresh.")
            return None
        prev_lr = prev.get("lr", None)
        if prev_lr is not None and not math.isclose(float(prev_lr), lr, rel_tol=1e-6):
            print(f"  Checkpoint LR {prev_lr} ≠ requested {lr} — starting fresh.")
            return None

    print(f"  Loading checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Sanity check config inside the checkpoint
    ckpt_cfg = ckpt.get("config")
    if ckpt_cfg is not None and hasattr(ckpt_cfg, "n_embd"):
        pass  # looks like a GPTConfig dataclass, trust it

    start_step    = ckpt["step"] + 1
    best_val_loss = ckpt["val_loss"]

    # Reconstruct train_pos: after `step+1` optimizer steps with effective_batch tokens each.
    # train_mup.py saves the checkpoint at the eval step, so train_pos ≈ (step+1)*effective_batch.
    # We don't know effective_batch here yet, so we return None for train_pos
    # and let the caller compute it after setting up the data.
    return ckpt["model"], start_step, best_val_loss


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"Device: {device}")

    # ── model ──────────────────────────────────────────────────────────────────
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
    model = make_mup_model(gpt_cfg).to(device)
    n_params = model.count_params()

    width_mult = gpt_cfg.n_embd / 128
    print(f"\n{'─'*55}")
    print(f"  Model : {args.config}  [µP, extended training]")
    print(f"  d_model={gpt_cfg.n_embd}  n_layers={gpt_cfg.n_layer}  n_heads={gpt_cfg.n_head}")
    print(f"  Params (non-emb) : {n_params/1e6:.3f}M")
    print(f"  Width multiplier : {width_mult:.2f}x  (effective base LR ≈ {args.lr/width_mult:.2e})")
    if torch.cuda.is_available():
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"{'─'*55}\n")

    # ── data ───────────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    train_data = load_bin(data_dir / "train.bin")
    val_data   = load_bin(data_dir / "val.bin")

    block_size           = gpt_cfg.block_size
    tokens_per_microstep = args.batch_size * block_size
    effective_batch      = tokens_per_microstep * args.grad_accum
    epoch_steps          = (len(train_data) - 1) // effective_batch
    warmup_steps         = max(1, int(epoch_steps * args.warmup_ratio))
    max_steps            = args.max_epochs * epoch_steps  # LR decay horizon

    print(f"Train tokens : {len(train_data):,}")
    print(f"Steps/epoch  : {epoch_steps}")
    print(f"Max steps    : {max_steps}  ({args.max_epochs} epochs, LR horizon)")
    print(f"Warmup steps : {warmup_steps}")
    print(f"Eff. batch   : {effective_batch:,} tokens/step")
    print(f"Patience     : {args.patience} evals  ({args.patience * args.eval_interval} steps)")

    # ── try resuming from checkpoint ───────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_step    = 0
    best_val_loss = float("inf")
    resume_result = try_load_checkpoint(out_dir, args.config, args.lr, device)

    if resume_result is not None:
        state_dict, start_step, best_val_loss = resume_result
        model.load_state_dict(state_dict)
        print(f"  Resumed from step {start_step - 1}  (best val loss so far: {best_val_loss:.4f})")
    else:
        print("  Starting fresh (no matching checkpoint found).")

    # Reconstruct train_pos from start_step so data continues where training left off
    train_pos = (start_step * effective_batch) % max(1, len(train_data) - 1)

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile: OK")
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without.")

    # ── optimizer ──────────────────────────────────────────────────────────────
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type=device_type,
    )

    # ── output ─────────────────────────────────────────────────────────────────
    log_mode = "a" if start_step > 0 else "w"
    log_file   = open(out_dir / "log_long.csv", log_mode, newline="")
    log_writer = csv.writer(log_file)
    if start_step == 0:
        log_writer.writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_sec", "gpu_mem_gb"])

    # ── AMP ────────────────────────────────────────────────────────────────────
    use_amp = device_type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── training loop with early stopping ──────────────────────────────────────
    patience_counter = 0
    t_start = time.time()
    step_loss = 0.0

    for step in range(start_step, max_steps):
        t0 = time.time()

        lr = get_lr(step, warmup_steps, max_steps, args.lr, args.lr * args.min_lr_frac)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
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
        if step % args.eval_interval == 0 or step == max_steps - 1:
            val_loss = evaluate(model, val_data, args.batch_size, block_size, args.eval_batches, device)

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "model":     model.state_dict(),
                        "config":    gpt_cfg,
                        "step":      step,
                        "val_loss":  val_loss,
                        "train_pos": train_pos,
                    },
                    out_dir / "ckpt.pt",
                )
            else:
                patience_counter += 1

            epoch_frac = step / epoch_steps
            print(
                f"step {step:5d} (ep {epoch_frac:.2f}) | "
                f"loss {step_loss:.4f} | val {val_loss:.4f} "
                f"{'↓' if improved else f'(no improv {patience_counter}/{args.patience})'} | "
                f"lr {lr:.2e} | {tok_per_sec:,.0f} tok/s"
            )

            if patience_counter >= args.patience:
                print(f"\nEarly stopping: no improvement for {args.patience} evals.")
                break

        log_writer.writerow([
            step, f"{step_loss:.6f}",
            f"{val_loss:.6f}" if val_loss is not None else "",
            f"{lr:.6e}", f"{tok_per_sec:.1f}", f"{gpu_mem_gb:.3f}",
        ])
        log_file.flush()

    wall_time_s = time.time() - t_start
    log_file.close()

    total_steps_run = step - start_step + 1
    print(f"\nDone in {wall_time_s/60:.1f} min  ({total_steps_run} steps, "
          f"{total_steps_run/epoch_steps:.2f} additional epochs)")
    print(f"Best val loss: {best_val_loss:.4f}")

    results = {
        "config":                 args.config,
        "parameterization":       "muP",
        "n_params":               n_params,
        "lr":                     args.lr,
        "best_val_loss":          best_val_loss,
        "final_train_loss":       step_loss,
        "total_steps":            step + 1,
        "steps_this_run":         total_steps_run,
        "effective_batch_tokens": effective_batch,
        "wall_time_s":            round(wall_time_s, 1),
        "width_mult":             width_mult,
        "early_stopped":          patience_counter >= args.patience,
    }
    with open(out_dir / "results_long.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results → {out_dir / 'results_long.json'}")


if __name__ == "__main__":
    main()
