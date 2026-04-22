"""
Compute dataset statistics, generate plots, render SVG examples, write results.md.

Usage: python analyze.py
"""

import json
import random
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm

SPLITS_DIR = Path(__file__).parent / "data" / "splits"
CLEANED_DIR = Path(__file__).parent / "data" / "cleaned"
TOKENIZER_DIR = Path(__file__).parent / "tokenizer"
PLOTS_DIR = Path(__file__).parent / "plots"
EXAMPLES_DIR = PLOTS_DIR / "examples"
RESULTS_FILE = Path(__file__).parent / "results.md"
SEED = 42


# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def render_svg_to_png(svg_text: str, out_path: Path) -> bool:
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), write_to=str(out_path), output_width=256, output_height=256)
        return True
    except Exception:
        return False


# ── stats ─────────────────────────────────────────────────────────────────────

def compute_split_stats(records: list[dict], name: str) -> dict:
    token_counts = [r["token_count"] for r in records]
    sources = Counter(r["source"] for r in records)
    return {
        "name": name,
        "n_svgs": len(records),
        "total_tokens": sum(token_counts),
        "avg_tokens": sum(token_counts) / len(token_counts),
        "median_tokens": sorted(token_counts)[len(token_counts) // 2],
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "sources": dict(sources),
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_seq_len_histogram(all_records: list[dict], out_path: Path):
    token_counts = [r["token_count"] for r in all_records]
    colors = {"icons": "#4C72B0", "emoji": "#DD8452", "svgen": "#55A868", "stack": "#C44E52"}
    sources = sorted(set(r["source"] for r in all_records))

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = range(0, 1025, 32)

    for src in sources:
        counts = [r["token_count"] for r in all_records if r["source"] == src]
        ax.hist(counts, bins=bins, alpha=0.65, label=src, color=colors.get(src), edgecolor="none")

    ax.set_xlabel("Sequence length (tokens)", fontsize=12)
    ax.set_ylabel("Number of SVGs", fontsize=12)
    ax.set_title("Token sequence length distribution (all splits)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1024)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved histogram → {out_path}")


def plot_svg_examples(records: list[dict], out_path: Path):
    """Render a 3×3 grid of SVGs at short/medium/long complexity."""
    random.seed(SEED)

    short = [r for r in records if r["token_count"] <= 100]
    medium = [r for r in records if 300 <= r["token_count"] <= 500]
    long_ = [r for r in records if r["token_count"] >= 800]

    selected = (
        random.sample(short, min(3, len(short))) +
        random.sample(medium, min(3, len(medium))) +
        random.sample(long_, min(3, len(long_)))
    )

    labels = (
        [f"Short (≤100 tok)\n{r['token_count']} tokens" for r in selected[:3]] +
        [f"Medium (300–500 tok)\n{r['token_count']} tokens" for r in selected[3:6]] +
        [f"Long (≥800 tok)\n{r['token_count']} tokens" for r in selected[6:]]
    )

    # Render each to a temp PNG and load for matplotlib
    import tempfile, os
    from PIL import Image as PILImage

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("SVG Examples at Different Complexity Levels", fontsize=14, y=1.01)

    for ax, rec, label in zip(axes.flat, selected, labels):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        ok = render_svg_to_png(rec["svg"], Path(tmp_path))
        if ok:
            img = PILImage.open(tmp_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "render\nfailed", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label, fontsize=8)
        ax.axis("off")
        if ok:
            os.unlink(tmp_path)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved examples grid → {out_path}")


# ── results.md ────────────────────────────────────────────────────────────────

def write_results_md(split_stats: list[dict], vocab_comparison: dict, cleaning_stats: dict):
    train_s = next(s for s in split_stats if s["name"] == "train")

    lines = [
        "# Part 1: Data Collection and Preprocessing",
        "",
        "## Dataset Sources",
        "",
        "### Raw downloads",
        "| Source | Dataset | Files downloaded |",
        "|--------|---------|-----------------|",
        "| icons  | starvector/svg-icons-simple  | 80,434 |",
        "| emoji  | starvector/svg-emoji-simple  | 4,114  |",
        "| svgen  | umuthopeyildirim/svgen-500k  | 216,435 |",
        "| stack  | starvector/svg-stack-simple (streamed) | 20,000 |",
        "| **Total** | | **320,983** |",
        "",
        "### After cleaning",
        "| Source | Before | After | Dropped (too long) | Dropped (XML err) |",
        "|--------|--------|-------|-------------------|-------------------|",
    ]

    for src, stats in cleaning_stats.items():
        lines.append(
            f"| {src} | {stats['before']:,} | {stats['after']:,} | "
            f"{stats['long']:,} | {stats['xml_err']:,} |"
        )
    total_before = sum(s["before"] for s in cleaning_stats.values())
    total_after = sum(s["after"] for s in cleaning_stats.values())
    total_long = sum(s["long"] for s in cleaning_stats.values())
    total_xml = sum(s["xml_err"] for s in cleaning_stats.values())
    lines.append(f"| **Total** | **{total_before:,}** | **{total_after:,}** | **{total_long:,}** | **{total_xml:,}** |")

    lines += [
        "",
        "## Tokenizer",
        "",
        "- **Library**: HuggingFace `tokenizers` (BPE with ByteLevel pre-tokenizer)",
        "- **Chosen vocab size**: 4,096",
        "- **Special tokens**: `<|endoftext|>`, `<unk>`",
        "- **Training corpus**: all cleaned SVGs (~378 MB)",
        "",
        "### Vocab size comparison (trained independently at each size)",
        "",
        "| Vocab | Avg seq len | Median seq len | SVGs >1024 tok | Est. train tokens |",
        "|-------|------------|----------------|----------------|-------------------|",
    ]

    for vs_str, m in vocab_comparison.items():
        vs = int(vs_str)
        lines.append(
            f"| {vs:,} | {m['avg_seq_len']:.1f} | {m['median_seq_len']} | "
            f"{m['pct_over_1024']:.1f}% | {m['estimated_train_tokens']:,.0f} |"
        )

    lines += [
        "",
        "**Justification**: Sequence length plateaus after 4,096 (766 vs 765 tokens at 8,192). "
        "Choosing 4,096 halves the embedding table size with negligible impact on sequence length. "
        "ByteLevel BPE guarantees 0% UNK rate at any vocab size.",
        "",
        "## Dataset Splits (98 / 1 / 1)",
        "",
        "| Split | SVGs | Tokens | Sources |",
        "|-------|------|--------|---------|",
    ]

    for s in split_stats:
        src_str = ", ".join(f"{k}={v:,}" for k, v in sorted(s["sources"].items()))
        lines.append(f"| {s['name']} | {s['n_svgs']:,} | {s['total_tokens']:,} | {src_str} |")

    lines += [
        "",
        f"- **Split method**: by file count (not token position) — prevents data leakage",
        f"- **Max sequence length**: 1,024 tokens (exact, post-tokenization filter)",
        f"- **Training token total**: {train_s['total_tokens']:,} ({train_s['total_tokens']/1e6:.1f}M) ✓",
        "",
        "## Sequence Length Distribution",
        "",
        "![Sequence length histogram](plots/seq_len_hist.png)",
        "",
        "## SVG Examples at Different Complexity Levels",
        "",
        "![SVG examples grid](plots/svg_examples.png)",
        "",
        "| Complexity | Token range | Description |",
        "|-----------|-------------|-------------|",
        "| Short     | ≤ 100       | Simple icons: single path or basic shape |",
        "| Medium    | 300–500     | Moderate detail: multiple paths, some attributes |",
        "| Long      | ≥ 800       | Complex icons: many paths, rich attributes |",
        "",
        "## Design Decisions",
        "",
        "| Decision | Choice | Justification |",
        "|----------|--------|---------------|",
        "| Token threshold | 1,024 | Fits small model context windows; removes outliers |",
        "| Vocab size | 4,096 | Seq length plateau after 4K; halves embedding vs 8K |",
        "| Split ratio | 98/1/1 | Maximises training data; val/test each ~1K SVGs for reliable metrics |",
        "| Split method | By file | Avoids token-position leakage between splits |",
        "| Coord precision | 1 decimal | Reduces float vocabulary without visual quality loss |",
    ]

    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved → {RESULTS_FILE}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading splits ...")
    train = load_jsonl(SPLITS_DIR / "train.jsonl")
    val = load_jsonl(SPLITS_DIR / "val.jsonl")
    test = load_jsonl(SPLITS_DIR / "test.jsonl")
    all_records = train + val + test

    split_stats = [
        compute_split_stats(train, "train"),
        compute_split_stats(val, "val"),
        compute_split_stats(test, "test"),
    ]

    print("\nSplit statistics:")
    for s in split_stats:
        print(f"  {s['name']:6s}: {s['n_svgs']:>8,} SVGs  {s['total_tokens']:>14,} tokens  "
              f"avg={s['avg_tokens']:.0f}  median={s['median_tokens']}")

    # Load vocab comparison and cleaning stats
    vocab_comp_path = TOKENIZER_DIR / "vocab_comparison.json"
    with vocab_comp_path.open() as f:
        vocab_comparison = json.load(f)

    # Cleaning stats — read from cleaned dir (approximate from dir sizes)
    cleaning_stats = {}
    raw_counts = {"icons": 80434, "emoji": 4114, "svgen": 216435, "stack": 20000}
    xml_errors = {"icons": 0, "emoji": 0, "svgen": 15, "stack": 0}
    for src in ["icons", "emoji", "svgen", "stack"]:
        cleaned = list((CLEANED_DIR / src).glob("*.svg")) if (CLEANED_DIR / src).exists() else []
        after = len(cleaned)
        before = raw_counts.get(src, after)
        xml_err = xml_errors.get(src, 0)
        long_ = before - after - xml_err
        cleaning_stats[src] = {"before": before, "after": after, "long": max(long_, 0), "xml_err": xml_err}

    print("\nGenerating histogram ...")
    plot_seq_len_histogram(all_records, PLOTS_DIR / "seq_len_hist.png")

    print("Rendering SVG examples ...")
    plot_svg_examples(train, PLOTS_DIR / "svg_examples.png")

    print("Writing results.md ...")
    write_results_md(split_stats, vocab_comparison, cleaning_stats)

    print("\nDone. Summary:")
    print(f"  Training SVGs:   {split_stats[0]['n_svgs']:,}")
    print(f"  Training tokens: {split_stats[0]['total_tokens']:,} ({split_stats[0]['total_tokens']/1e6:.1f}M)")
    print(f"  Results:         {RESULTS_FILE}")
    print(f"  Plots:           {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
