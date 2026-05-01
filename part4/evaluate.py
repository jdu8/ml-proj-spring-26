"""
Quantitative evaluation of the trained model.

Computes:
  - Perplexity on test set
  - XML validity rate (lxml.etree)
  - SVG render rate (CairoSVG)
  - Structural validity (svg root, viewBox, closed tags)

Also generates N fresh samples to measure validity rates, so this script
can be run independently of generate.py.

Usage (Colab):
    python evaluate.py --ckpt /path/to/ckpt.pt
    python evaluate.py --ckpt /path/to/ckpt.pt --n_eval_samples 100
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "part2"))
from model import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",            required=True)
    p.add_argument("--tokenizer",       default=str(ROOT / "part1/tokenizer/tokenizer_4096.json"))
    p.add_argument("--data_dir",        default="../part2/data")
    p.add_argument("--out_dir",         default="out/eval")
    p.add_argument("--n_eval_samples",  type=int,   default=50,
                   help="Fresh samples to generate for validity metrics")
    p.add_argument("--temperature",     type=float, default=0.8)
    p.add_argument("--top_k",           type=int,   default=50)
    p.add_argument("--top_p",           type=float, default=0.95)
    p.add_argument("--max_new_tokens",  type=int,   default=512)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--eval_batches",    type=int,   default=200,
                   help="Test batches for perplexity (increase for tighter estimate)")
    p.add_argument("--seed",            type=int,   default=0)
    return p.parse_args()


# ── model / tokenizer ─────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    if not isinstance(cfg, GPTConfig):
        cfg = GPTConfig(**vars(cfg))
    cfg.dropout = 0.0
    model = GPT(cfg).to(device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, cfg, ckpt.get("val_loss")


def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


# ── perplexity ────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, test_bin, batch_size, block_size, n_batches, device):
    data = np.memmap(test_bin, dtype="uint16", mode="r")
    losses = []
    pos = 0
    tok_per_batch = batch_size * block_size

    use_amp = device.type == "cuda"

    for _ in range(n_batches):
        if pos + tok_per_batch + 1 > len(data):
            pos = 0
        x = torch.from_numpy(np.array(data[pos:pos + tok_per_batch],     dtype=np.int64)
                              .reshape(batch_size, block_size)).to(device)
        y = torch.from_numpy(np.array(data[pos + 1:pos + tok_per_batch + 1], dtype=np.int64)
                              .reshape(batch_size, block_size)).to(device)
        pos += tok_per_batch

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            _, loss = model(x, y)
        losses.append(loss.item())

    avg_loss = float(np.mean(losses))
    return avg_loss, math.exp(avg_loss)


# ── SVG validity ──────────────────────────────────────────────────────────────

def check_xml_valid(svg_text):
    try:
        from lxml import etree
        etree.fromstring(svg_text.encode())
        return True
    except Exception:
        return False


def check_svg_renders(svg_text):
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_text.encode(), output_width=64, output_height=64)
        return True
    except Exception:
        return False


def check_structural(svg_text):
    text = svg_text.strip()
    has_svg_root   = text.startswith("<svg") or "<svg" in text[:50]
    has_svg_close  = "</svg>" in text
    has_viewbox    = 'viewBox' in text or 'viewbox' in text
    has_xmlns      = 'xmlns' in text
    return {
        "has_svg_root":  has_svg_root,
        "has_svg_close": has_svg_close,
        "has_viewbox":   has_viewbox,
        "has_xmlns":     has_xmlns,
        "structural_ok": has_svg_root and has_svg_close,
    }


def truncate_at_svg_close(text):
    marker = "</svg>"
    pos = text.find(marker)
    return text[:pos + len(marker)] if pos != -1 else text


# ── generation for eval ───────────────────────────────────────────────────────

@torch.no_grad()
def generate_for_eval(model, tok, n, max_new_tokens, temperature, top_k, top_p, device):
    samples = []
    ids = tok.encode("<svg").ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for i in range(n):
        out = model.generate(idx, max_new_tokens, temperature=temperature,
                             top_k=top_k, top_p=top_p)
        text = tok.decode(out[0].tolist())
        text = truncate_at_svg_close(text)
        samples.append(text)
        if (i + 1) % 10 == 0:
            print(f"    generated {i+1}/{n}", flush=True)
    return samples


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model, gpt_cfg, saved_val_loss = load_model(args.ckpt, device)
    print(f"Model loaded  (saved val_loss={saved_val_loss:.4f})\n")

    tok = load_tokenizer(args.tokenizer)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── 1. perplexity on test set ─────────────────────────────────────────────
    print("=" * 60)
    print("PERPLEXITY (test set)")
    print("=" * 60)
    test_bin = Path(args.data_dir) / "test.bin"
    if not test_bin.exists():
        print(f"  WARNING: {test_bin} not found, skipping perplexity.")
        results["perplexity"] = None
        results["test_loss"]  = None
    else:
        t0 = time.time()
        test_loss, ppl = compute_perplexity(
            model, test_bin, args.batch_size, gpt_cfg.block_size,
            args.eval_batches, device
        )
        elapsed = time.time() - t0
        print(f"  Test loss   : {test_loss:.4f}")
        print(f"  Perplexity  : {ppl:.2f}")
        print(f"  ({args.eval_batches} batches, {elapsed:.1f}s)")
        results["test_loss"]  = round(test_loss, 6)
        results["perplexity"] = round(ppl, 4)

    # ── 2. validity metrics on fresh samples ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"VALIDITY METRICS  ({args.n_eval_samples} fresh samples)")
    print("=" * 60)
    print(f"  Generating {args.n_eval_samples} samples (temp={args.temperature}, "
          f"top_k={args.top_k}, top_p={args.top_p})...")

    samples = generate_for_eval(
        model, tok, args.n_eval_samples,
        args.max_new_tokens, args.temperature, args.top_k, args.top_p, device
    )

    xml_valid_count   = 0
    render_ok_count   = 0
    struct_ok_count   = 0
    has_viewbox_count = 0
    has_xmlns_count   = 0
    lengths           = []

    sample_details = []
    for i, svg in enumerate(samples):
        xml_ok    = check_xml_valid(svg)
        render_ok = check_svg_renders(svg)
        struct    = check_structural(svg)

        xml_valid_count   += int(xml_ok)
        render_ok_count   += int(render_ok)
        struct_ok_count   += int(struct["structural_ok"])
        has_viewbox_count += int(struct["has_viewbox"])
        has_xmlns_count   += int(struct["has_xmlns"])
        lengths.append(len(svg))

        sample_details.append({
            "index":       i,
            "xml_valid":   xml_ok,
            "renders":     render_ok,
            "structural":  struct,
            "length":      len(svg),
        })

    n = len(samples)
    xml_rate    = xml_valid_count / n
    render_rate = render_ok_count / n
    struct_rate = struct_ok_count / n

    print(f"\n  XML validity rate   : {xml_valid_count}/{n}  = {xml_rate:.1%}")
    print(f"  SVG render rate     : {render_ok_count}/{n}  = {render_rate:.1%}")
    print(f"  Structural validity : {struct_ok_count}/{n}  = {struct_rate:.1%}")
    print(f"  Has viewBox         : {has_viewbox_count}/{n}  = {has_viewbox_count/n:.1%}")
    print(f"  Has xmlns           : {has_xmlns_count}/{n}  = {has_xmlns_count/n:.1%}")
    print(f"  Avg length          : {np.mean(lengths):.0f} chars")
    print(f"  Median length       : {np.median(lengths):.0f} chars")

    results["validity"] = {
        "n_samples":          n,
        "temperature":        args.temperature,
        "top_k":              args.top_k,
        "top_p":              args.top_p,
        "xml_valid_count":    xml_valid_count,
        "xml_valid_rate":     round(xml_rate, 4),
        "render_ok_count":    render_ok_count,
        "render_rate":        round(render_rate, 4),
        "structural_ok_count": struct_ok_count,
        "structural_rate":    round(struct_rate, 4),
        "has_viewbox_rate":   round(has_viewbox_count / n, 4),
        "has_xmlns_rate":     round(has_xmlns_count / n, 4),
        "avg_length":         round(float(np.mean(lengths)), 1),
        "median_length":      round(float(np.median(lengths)), 1),
    }

    # ── 3. length distribution plot ───────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(lengths, bins=20, color="steelblue", edgecolor="white")
        ax.set_xlabel("Generated SVG length (chars)")
        ax.set_ylabel("Count")
        ax.set_title(f"Generated SVG length distribution (n={n})")
        plt.tight_layout()
        plot_path = out_dir / "length_distribution.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\n  Length distribution → {plot_path}")
    except Exception as e:
        print(f"\n  Length plot failed: {e}")

    # ── 4. save results ───────────────────────────────────────────────────────
    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # save sample details separately (can be large)
    details_path = out_dir / "sample_details.json"
    with open(details_path, "w") as f:
        json.dump(sample_details, f, indent=2)

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    if results.get("perplexity"):
        print(f"  Perplexity        : {results['perplexity']:.2f}")
    print(f"  XML valid rate    : {xml_rate:.1%}")
    print(f"  Render rate       : {render_rate:.1%}")
    print(f"  Structural rate   : {struct_rate:.1%}")
    print(f"  Results           : {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
