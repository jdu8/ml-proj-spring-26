"""
Generate SVG samples from a trained model.

Produces:
  - 10 unconditional samples at temperatures 0.5, 0.8, 1.0
  - 5 prefix-conditioned completions
  - PNGs for every sample (via CairoSVG)
  - A rendered grid image for the report

Usage (Colab):
    python generate.py --ckpt /path/to/ckpt.pt
    python generate.py --ckpt /path/to/ckpt.pt --out_dir samples --n_unconditional 15
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

# ── resolve sibling imports ────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "part2"))
from model import GPT, GPTConfig


# ── prefixes for conditioned generation ───────────────────────────────────────
PREFIX_SAMPLES = [
    {
        "name": "partial_face",
        "desc": "Circle + one eye — does the model add the other eye and mouth?",
        "svg":  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                '<circle cx="50" cy="50" r="40" fill="none" stroke="black" stroke-width="2"/>'
                '<circle cx="35" cy="40" r="5" fill="black"/>',
    },
    {
        "name": "open_path",
        "desc": "Open path — does the model close it?",
        "svg":  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                '<path d="M10 50 Q30 10 50 50 Q70 90 90 50" fill="none" stroke="black" stroke-width="2"/>',
    },
    {
        "name": "group_one_shape",
        "desc": "Group with one rect — does the model add related shapes?",
        "svg":  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                '<g fill="blue"><rect x="10" y="10" width="30" height="30"/>',
    },
    {
        "name": "star_partial",
        "desc": "Partial star path — does the model complete the polygon?",
        "svg":  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                '<polygon points="50,5 61,35 95,35 68,57" fill="gold" stroke="orange" stroke-width="1"/>',
    },
    {
        "name": "arrow_start",
        "desc": "Arrow shaft — does the model add the arrowhead?",
        "svg":  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                '<line x1="10" y1="50" x2="70" y2="50" stroke="black" stroke-width="3"/>',
    },
]

UNCONDITIONAL_PREFIX = "<svg"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",             required=True,  help="Path to ckpt.pt")
    p.add_argument("--tokenizer",        default=str(ROOT / "part1/tokenizer/tokenizer_4096.json"))
    p.add_argument("--out_dir",          default="out/samples")
    p.add_argument("--n_unconditional",  type=int,   default=10)
    p.add_argument("--temperatures",     nargs="+",  type=float, default=[0.5, 0.8, 1.0])
    p.add_argument("--top_k",            type=int,   default=50)
    p.add_argument("--top_p",            type=float, default=0.95)
    p.add_argument("--max_new_tokens",   type=int,   default=512)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--no_render",        action="store_true", help="Skip CairoSVG rendering")
    p.add_argument("--grid_cols",        type=int,   default=5)
    return p.parse_args()


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]
    if isinstance(cfg, GPTConfig):
        gpt_cfg = cfg
    else:
        gpt_cfg = GPTConfig(**{k: v for k, v in vars(cfg).items()})
    gpt_cfg.dropout = 0.0
    model = GPT(gpt_cfg).to(device)
    # strip compile prefix if present
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {ckpt_path}  (val_loss={ckpt.get('val_loss', '?'):.4f})")
    return model, gpt_cfg


# ── tokenizer ─────────────────────────────────────────────────────────────────

def load_tokenizer(path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    return tok


def encode(tok, text):
    return tok.encode(text).ids


def decode(tok, ids):
    return tok.decode(ids)


# ── generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_sample(model, tok, prompt_text, max_new_tokens, temperature, top_k, top_p, device):
    ids = encode(tok, prompt_text)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    return decode(tok, out[0].tolist())


def truncate_at_svg_close(text):
    """Trim to the first </svg> for cleaner outputs."""
    marker = "</svg>"
    pos = text.find(marker)
    if pos != -1:
        return text[:pos + len(marker)]
    return text


# ── rendering ────────────────────────────────────────────────────────────────

def try_render(svg_text, png_path):
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_text.encode(), write_to=str(png_path), output_width=128, output_height=128)
        return True
    except Exception:
        return False


def try_render_fallback(svg_text, png_path):
    """Return a blank placeholder PNG when rendering fails."""
    try:
        from PIL import Image
        img = Image.new("RGB", (128, 128), color=(240, 240, 240))
        img.save(str(png_path))
    except Exception:
        pass
    return False


# ── grid plot ─────────────────────────────────────────────────────────────────

def make_grid(image_paths, titles, out_path, cols=5, cell_size=128, font_size=8):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        valid = [(p, t) for p, t in zip(image_paths, titles) if Path(p).exists()]
        if not valid:
            return
        rows = (len(valid) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2 + 0.4))
        axes = np.array(axes).reshape(-1)
        for ax, (path, title) in zip(axes, valid):
            try:
                img = Image.open(path)
                ax.imshow(img)
            except Exception:
                ax.set_facecolor("#f0f0f0")
            ax.set_title(title, fontsize=font_size, wrap=True)
            ax.axis("off")
        for ax in axes[len(valid):]:
            ax.axis("off")
        plt.tight_layout(pad=0.5)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Grid saved → {out_path}")
    except Exception as e:
        print(f"  Grid generation failed: {e}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model, gpt_cfg = load_model(args.ckpt, device)
    tok = load_tokenizer(args.tokenizer)

    out_dir = Path(args.out_dir)
    unconditional_dir = out_dir / "unconditional"
    prefix_dir        = out_dir / "prefix_conditioned"
    for d in [unconditional_dir, prefix_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_results = {"unconditional": [], "prefix_conditioned": []}

    # ── 1. unconditional generation ───────────────────────────────────────────
    print("=" * 60)
    print("UNCONDITIONAL GENERATION")
    print("=" * 60)

    unc_png_paths, unc_titles = [], []
    sample_idx = 0

    for temp in args.temperatures:
        n_this_temp = max(1, args.n_unconditional // len(args.temperatures))
        for i in range(n_this_temp):
            label = f"sample_{sample_idx:02d}_t{temp}"
            print(f"  [{label}] generating...", end=" ", flush=True)

            svg_text = generate_sample(
                model, tok, UNCONDITIONAL_PREFIX,
                args.max_new_tokens, temp, args.top_k, args.top_p, device
            )
            svg_text = truncate_at_svg_close(svg_text)

            svg_path = unconditional_dir / f"{label}.svg"
            png_path = unconditional_dir / f"{label}.png"
            svg_path.write_text(svg_text, encoding="utf-8")

            rendered = False
            if not args.no_render:
                rendered = try_render(svg_text, png_path)
                if not rendered:
                    try_render_fallback(svg_text, png_path)

            print(f"{'rendered' if rendered else 'render failed'}  ({len(svg_text)} chars)")

            entry = {
                "label":      label,
                "temperature": temp,
                "top_k":      args.top_k,
                "top_p":      args.top_p,
                "svg_path":   str(svg_path),
                "png_path":   str(png_path) if not args.no_render else None,
                "rendered":   rendered,
                "length":     len(svg_text),
            }
            all_results["unconditional"].append(entry)
            unc_png_paths.append(str(png_path))
            unc_titles.append(f"T={temp} #{i+1}")
            sample_idx += 1

    # ── 2. prefix-conditioned generation ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("PREFIX-CONDITIONED GENERATION")
    print("=" * 60)

    pfx_png_paths, pfx_titles = [], []
    for pfx in PREFIX_SAMPLES:
        for temp in [0.8]:  # one temperature per prefix to keep it focused
            label = f"{pfx['name']}_t{temp}"
            print(f"  [{label}]")
            print(f"    {pfx['desc']}")

            svg_text = generate_sample(
                model, tok, pfx["svg"],
                args.max_new_tokens, temp, args.top_k, args.top_p, device
            )
            svg_text = truncate_at_svg_close(svg_text)

            svg_path = prefix_dir / f"{label}.svg"
            png_path = prefix_dir / f"{label}.png"
            svg_path.write_text(svg_text, encoding="utf-8")

            # also save the prefix alone for side-by-side comparison
            prefix_svg_path = prefix_dir / f"{pfx['name']}_prefix_only.svg"
            prefix_svg_path.write_text(pfx["svg"] + "</svg>", encoding="utf-8")

            rendered = False
            if not args.no_render:
                rendered = try_render(svg_text, png_path)
                if not rendered:
                    try_render_fallback(svg_text, png_path)
            print(f"    → {'rendered' if rendered else 'render failed'}  ({len(svg_text)} chars)\n")

            entry = {
                "label":       label,
                "name":        pfx["name"],
                "description": pfx["desc"],
                "prefix":      pfx["svg"],
                "temperature": temp,
                "svg_path":    str(svg_path),
                "png_path":    str(png_path) if not args.no_render else None,
                "rendered":    rendered,
            }
            all_results["prefix_conditioned"].append(entry)
            pfx_png_paths.append(str(png_path))
            pfx_titles.append(pfx["name"].replace("_", "\n"))

    # ── 3. grids ──────────────────────────────────────────────────────────────
    if not args.no_render:
        print("\nBuilding grids...")
        make_grid(unc_png_paths, unc_titles,
                  out_dir / "grid_unconditional.png", cols=args.grid_cols)
        make_grid(pfx_png_paths, pfx_titles,
                  out_dir / "grid_prefix.png", cols=len(PREFIX_SAMPLES))

    # ── 4. save manifest ─────────────────────────────────────────────────────
    manifest_path = out_dir / "generation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_results, f, indent=2)

    total_unc = len(all_results["unconditional"])
    total_pfx = len(all_results["prefix_conditioned"])
    unc_rendered = sum(e["rendered"] for e in all_results["unconditional"])
    pfx_rendered = sum(e["rendered"] for e in all_results["prefix_conditioned"])

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print(f"  Unconditional : {total_unc} samples  ({unc_rendered} rendered)")
    print(f"  Prefix-cond.  : {total_pfx} samples  ({pfx_rendered} rendered)")
    print(f"  Manifest      : {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
