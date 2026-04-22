"""
Normalize and filter SVG files from data/raw/ → data/cleaned/.

Cleaning steps:
1. Parse with lxml — skip invalid XML
2. Strip XML comments and processing instructions
3. Normalize coordinate numbers: round floats to 1 decimal place
4. Re-serialize to compact string
5. Filter: remove if len < 50 chars or len > 5000 chars

Usage: python clean_svg.py
"""

import re
import shutil
from pathlib import Path

from lxml import etree
from tqdm import tqdm

RAW_DIR = Path(__file__).parent / "data" / "raw"
CLEANED_DIR = Path(__file__).parent / "data" / "cleaned"
MIN_CHARS = 50
MAX_CHARS = 5000

# Regex to normalize floats with 2+ decimal places to 1 decimal
FLOAT_RE = re.compile(r"-?\d+\.\d{2,}")


def round_floats(text: str) -> str:
    def _round(m: re.Match) -> str:
        return f"{float(m.group()):.1f}"
    return FLOAT_RE.sub(_round, text)


def clean_svg(svg_text: str) -> str | None:
    """Return cleaned SVG string, or None if it should be discarded."""
    try:
        root = etree.fromstring(svg_text.encode("utf-8"))
    except etree.XMLSyntaxError:
        return None

    # Remove comments and processing instructions
    for node in root.iter():
        for child in list(node):
            if isinstance(child, (etree._Comment, etree._ProcessingInstruction)):
                node.remove(child)

    # Re-serialize (compact, no XML declaration)
    cleaned = etree.tostring(root, encoding="unicode")

    # Normalize coordinate precision
    cleaned = round_floats(cleaned)

    # Length filters
    if len(cleaned) < MIN_CHARS or len(cleaned) > MAX_CHARS:
        return None

    return cleaned


def main():
    sources = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    if not sources:
        print(f"No source directories found in {RAW_DIR}")
        return

    total_before = 0
    total_after = 0

    for src_dir in sorted(sources):
        source_name = src_dir.name
        out_dir = CLEANED_DIR / source_name
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(src_dir.glob("*.svg"))
        before = len(files)
        total_before += before

        kept = 0
        skipped_xml = 0
        skipped_short = 0
        skipped_long = 0

        for f in tqdm(files, desc=f"Cleaning [{source_name}]", unit="svg"):
            raw = f.read_text(encoding="utf-8", errors="replace")
            result = None

            try:
                root = etree.fromstring(raw.encode("utf-8"))
            except etree.XMLSyntaxError:
                skipped_xml += 1
                continue

            for node in root.iter():
                for child in list(node):
                    if isinstance(child, (etree._Comment, etree._ProcessingInstruction)):
                        node.remove(child)

            cleaned = etree.tostring(root, encoding="unicode")
            cleaned = round_floats(cleaned)

            if len(cleaned) < MIN_CHARS:
                skipped_short += 1
                continue
            if len(cleaned) > MAX_CHARS:
                skipped_long += 1
                continue

            result = cleaned

            (out_dir / f.name).write_text(result, encoding="utf-8")
            kept += 1

        total_after += kept
        print(
            f"[{source_name}] {before:>8,} → {kept:>8,} kept "
            f"(xml_err={skipped_xml:,}  short={skipped_short:,}  long={skipped_long:,})"
        )

    print(f"\nTotal: {total_before:,} → {total_after:,} kept "
          f"({total_after/total_before*100:.1f}% retention)")

    # Compute char stats on cleaned data
    total_chars = sum(f.stat().st_size for f in CLEANED_DIR.rglob("*.svg"))
    print(f"Cleaned corpus: {total_chars:,} chars ({total_chars/1e6:.1f} MB)")
    print(f"Avg chars/SVG:  {total_chars/total_after:,.0f}")


if __name__ == "__main__":
    main()
