"""
Generate SVG samples from a trained model.

Produces:
  - At least 10 unconditional samples (retries until 10 render, stops on 10 consecutive failures)
  - At least 5 prefix-conditioned completions (one per prefix, same retry logic)
  - PNGs for every successful sample (via CairoSVG)
  - A rendered grid image for the report

Usage:
    python generate.py --ckpt /path/to/ckpt.pt
    python generate.py --ckpt /path/to/ckpt.pt --out_dir samples --n_unconditional 15
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "part2"))
from model import GPT, GPTConfig

MAX_CONSECUTIVE_FAILURES = 10

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
TEMPERATURES = [0.5, 0.8, 1.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",             required=True,  help="Path to ckpt.pt")
    p.add_argument("--tokenizer",        default=str(ROOT / "part1/tokenizer/tokenizer_4096.json"))
    p.add_argument("--out_dir",          default="out/samples")
    p.add_argument("--n_unconditional",  type=int,   default=10)
    p.add_argument("--n_prefix",         type=int,   default=1,
                   help="Successful samples to collect per prefix")
    p.add_argument("--temperatures",     nargs="+",  type=float, default=TEMPERATURES)
    p.add_argument("--top_k",            type=int,   default=50)
    p.add_argument("--top_p",            type=float, default=0.95)
    p.add_argument("--max_new_tokens",   type=int,   default=900)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--no_render",        action="store_true", help="Skip CairoSVG rendering")
    p.add_argument("--grid_cols",        type=int,   default=5)
    return p.parse_args()


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    if isinstance(cfg, GPTConfig):
        gpt_cfg = cfg
    else:
        gpt_cfg = GPTConfig(**{k: v for k, v in vars(cfg).items()})
    gpt_cfg.dropout = 0.0
    model = GPT(gpt_cfg).to(device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {ckpt_path}  (val_loss={ckpt.get('val_loss', '?'):.4f})")
    return model, gpt_cfg


# ── tokenizer ─────────────────────────────────────────────────────────────────

def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


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
    marker = "</svg>"
    pos = text.find(marker)
    return text[:pos + len(marker)] if pos != -1 else text


# ── rendering ────────────────────────────────────────────────────────────────

def try_render(svg_text, png_path):
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_text.encode(), write_to=str(png_path),
                         output_width=128, output_height=128)
        return True
    except Exception:
        return False


def try_render_fallback(svg_text, png_path):
    try:
        from PIL import Image
        img = Image.new("RGB", (128, 128), color=(240, 240, 240))
        img.save(str(png_path))
    except Exception:
        pass
    return False


# ── retry-based generation ────────────────────────────────────────────────────

def generate_until_renders(
    model, tok, prompt_text, n_target,
    max_new_tokens, temperatures, top_k, top_p, device,
    save_dir, label_prefix, no_render=False,
):
    """
    Generate samples until n_target render successfully.
    Cycles through temperatures for variety.
    Stops early if MAX_CONSECUTIVE_FAILURES renders fail in a row.
    Returns list of result dicts.
    """
    results = []
    consecutive_failures = 0
    attempts = 0
    temp_cycle = list(temperatures)

    while len(results) < n_target:
        temp = temp_cycle[attempts % len(temp_cycle)]
        attempts += 1

        svg_text = generate_sample(
            model, tok, prompt_text,
            max_new_tokens, temp, top_k, top_p, device
        )
        svg_text = truncate_at_svg_close(svg_text)

        idx = len(results)
        label = f"{label_prefix}_{idx:02d}_t{temp}"
        svg_path = save_dir / f"{label}.svg"
        png_path = save_dir / f"{label}.png"
        svg_path.write_text(svg_text, encoding="utf-8")

        rendered = False
        if not no_render:
            rendered = try_render(svg_text, png_path)
            if not rendered:
                try_render_fallback(svg_text, png_path)

        status = "rendered" if rendered else "render failed"
        print(f"  attempt {attempts:3d} | T={temp} | {status} | {len(svg_text)} chars")

        if rendered or no_render:
            results.append({
                "label":       label,
                "temperature": temp,
                "top_k":       top_k,
                "top_p":       top_p,
                "svg_path":    str(svg_path),
                "png_path":    str(png_path),
                "rendered":    rendered,
                "length":      len(svg_text),
                "attempt":     attempts,
            })
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  Stopped: {MAX_CONSECUTIVE_FAILURES} consecutive render failures.")
                break

    return results, attempts


# ── grid plot ─────────────────────────────────────────────────────────────────

def make_grid(image_paths, titles, out_path, cols=5, font_size=8):
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
                from PIL import Image as PILImage
                img = PILImage.open(path)
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

    out_dir           = Path(args.out_dir)
    unconditional_dir = out_dir / "unconditional"
    prefix_dir        = out_dir / "prefix_conditioned"
    for d in [unconditional_dir, prefix_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_results = {"unconditional": [], "prefix_conditioned": []}

    # ── 1. unconditional generation ───────────────────────────────────────────
    print("=" * 60)
    print(f"UNCONDITIONAL GENERATION  (target: {args.n_unconditional} rendered)")
    print(f"Temperatures: {args.temperatures}  |  stop after {MAX_CONSECUTIVE_FAILURES} consecutive failures")
    print("=" * 60)

    unc_results, unc_attempts = generate_until_renders(
        model, tok, UNCONDITIONAL_PREFIX,
        n_target      = args.n_unconditional,
        max_new_tokens= args.max_new_tokens,
        temperatures  = args.temperatures,
        top_k         = args.top_k,
        top_p         = args.top_p,
        device        = device,
        save_dir      = unconditional_dir,
        label_prefix  = "sample",
        no_render     = args.no_render,
    )
    all_results["unconditional"] = unc_results
    print(f"\n  Got {len(unc_results)}/{args.n_unconditional} in {unc_attempts} attempts.\n")

    # ── 2. prefix-conditioned generation ─────────────────────────────────────
    print("=" * 60)
    print(f"PREFIX-CONDITIONED GENERATION  (target: {args.n_prefix} rendered per prefix)")
    print(f"Stop after {MAX_CONSECUTIVE_FAILURES} consecutive failures per prefix")
    print("=" * 60)

    pfx_png_paths, pfx_titles = [], []
    for pfx in PREFIX_SAMPLES:
        print(f"\n  Prefix: {pfx['name']}")
        print(f"  {pfx['desc']}")

        # save the prefix alone for side-by-side comparison
        prefix_svg_path = prefix_dir / f"{pfx['name']}_prefix_only.svg"
        prefix_svg_path.write_text(pfx["svg"] + "</svg>", encoding="utf-8")

        pfx_results, pfx_attempts = generate_until_renders(
            model, tok, pfx["svg"],
            n_target      = args.n_prefix,
            max_new_tokens= args.max_new_tokens,
            temperatures  = [0.8],
            top_k         = args.top_k,
            top_p         = args.top_p,
            device        = device,
            save_dir      = prefix_dir,
            label_prefix  = pfx["name"],
            no_render     = args.no_render,
        )

        for entry in pfx_results:
            entry["name"]        = pfx["name"]
            entry["description"] = pfx["desc"]
            entry["prefix"]      = pfx["svg"]
            all_results["prefix_conditioned"].append(entry)
            pfx_png_paths.append(entry["png_path"])
            pfx_titles.append(pfx["name"].replace("_", "\n"))

        got = len(pfx_results)
        print(f"  → {got}/{args.n_prefix} rendered in {pfx_attempts} attempts")

    # ── 3. grids ──────────────────────────────────────────────────────────────
    if not args.no_render:
        print("\nBuilding grids...")
        unc_png  = [e["png_path"] for e in unc_results]
        unc_lbls = [f"T={e['temperature']}" for e in unc_results]
        make_grid(unc_png, unc_lbls, out_dir / "grid_unconditional.png", cols=args.grid_cols)
        make_grid(pfx_png_paths, pfx_titles, out_dir / "grid_prefix.png", cols=len(PREFIX_SAMPLES))

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
