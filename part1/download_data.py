"""
Download all SVG datasets from HuggingFace.
Just pulls everything — char counting happens after cleaning.

Usage: python download_data.py
"""

from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data" / "raw"

SOURCES = [
    {
        "name": "icons",
        "dataset": "starvector/svg-icons-simple",
        "split": "train",
    },
    {
        "name": "emoji",
        "dataset": "starvector/svg-emoji-simple",
        "split": "train",
    },
    {
        "name": "svgen",
        "dataset": "umuthopeyildirim/svgen-500k",
        "split": "train",
        "svg_field": "output",  # columns: input, output, description, source, license
    },
    {
        "name": "stack",
        "dataset": "starvector/svg-stack-simple",
        "split": "train",
        "max_samples": 500_000,
    },
]


def find_svg_field(column_names: list[str]) -> str:
    for name in column_names:
        if "svg" in name.lower():
            return name
    raise ValueError(f"No SVG field found in columns: {column_names}")


def download_source(source: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("*.svg"))
    if existing:
        print(f"[{source['name']}] already downloaded ({len(existing):,} files) — skipping")
        return

    split = source["split"]
    if "max_samples" in source:
        split = f"{split}[:{source['max_samples']}]"

    print(f"[{source['name']}] downloading {source['dataset']} ...")
    ds = load_dataset(source["dataset"], split=split)

    field = source.get("svg_field") or find_svg_field(ds.column_names)
    print(f"[{source['name']}] using field '{field}', {len(ds):,} rows")

    saved = 0
    for i, row in enumerate(tqdm(ds, desc=source["name"])):
        svg = row[field]
        if svg and isinstance(svg, str):
            (out_dir / f"{i:08d}.svg").write_text(svg, encoding="utf-8")
            saved += 1

    print(f"[{source['name']}] saved {saved:,} files")


def main():
    for source in SOURCES:
        download_source(source, DATA_DIR / source["name"])

    print("\nDownload complete. Per-source file counts:")
    for source in SOURCES:
        src_dir = DATA_DIR / source["name"]
        if src_dir.exists():
            n = len(list(src_dir.glob("*.svg")))
            print(f"  {source['name']:10s}: {n:>8,} files")


if __name__ == "__main__":
    main()
