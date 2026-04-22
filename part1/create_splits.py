"""
Create 98/1/1 train/val/test splits from the cleaned SVG corpus.
Tokenizes each SVG with the chosen tokenizer, drops SVGs exceeding 1024 tokens,
then splits by file count (not token position) to avoid leakage.

Saves each split as JSONL: {"svg": "...", "token_count": N, "source": "..."}

Usage: python create_splits.py [--vocab-size 4096]
"""

import json
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer
from tqdm import tqdm

CLEANED_DIR = Path(__file__).parent / "data" / "cleaned"
SPLITS_DIR = Path(__file__).parent / "data" / "splits"
TOKENIZER_DIR = Path(__file__).parent / "tokenizer"

MAX_TOKENS = 1024
SEED = 42
TRAIN_RATIO = 0.98


def get_vocab_size() -> int:
    args = sys.argv[1:]
    if "--vocab-size" in args:
        return int(args[args.index("--vocab-size") + 1])
    return 4096  # default chosen vocab size


def load_tokenizer(vocab_size: int) -> Tokenizer:
    path = TOKENIZER_DIR / f"tokenizer_{vocab_size}.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {path}. Run train_tokenizer.py first.")
    return Tokenizer.from_file(str(path))


def collect_and_tokenize(tokenizer: Tokenizer) -> list[dict]:
    """Tokenize all SVGs, drop those exceeding MAX_TOKENS, return records."""
    paths = sorted(CLEANED_DIR.rglob("*.svg"))
    print(f"Found {len(paths):,} cleaned SVGs")

    records = []
    dropped = 0

    for p in tqdm(paths, desc="Tokenizing"):
        source = p.parent.name
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        enc = tokenizer.encode(text)
        n_tokens = len(enc.ids)

        if n_tokens > MAX_TOKENS:
            dropped += 1
            continue

        records.append({"svg": text, "token_count": n_tokens, "source": source})

    print(f"Dropped {dropped:,} SVGs exceeding {MAX_TOKENS} tokens")
    print(f"Kept {len(records):,} SVGs")
    return records


def split_records(records: list[dict]) -> tuple[list, list, list]:
    random.seed(SEED)
    random.shuffle(records)

    n = len(records)
    n_train = int(n * TRAIN_RATIO)
    n_val = (n - n_train) // 2

    train = records[:n_train]
    val = records[n_train : n_train + n_val]
    test = records[n_train + n_val :]
    return train, val, test


def save_split(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    size_mb = path.stat().st_size / 1e6
    print(f"  Saved {len(records):,} records → {path.name} ({size_mb:.1f} MB)")


def print_split_stats(name: str, records: list[dict]):
    total_tokens = sum(r["token_count"] for r in records)
    sources = {}
    for r in records:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    src_str = ", ".join(f"{s}={n:,}" for s, n in sorted(sources.items()))
    print(f"  {name:6s}: {len(records):>8,} SVGs  {total_tokens:>14,} tokens  [{src_str}]")


def main():
    vocab_size = get_vocab_size()
    print(f"Using vocab size: {vocab_size} (loading tokenizer_{vocab_size}.json, capping at {MAX_TOKENS} tokens)")

    tokenizer = load_tokenizer(vocab_size)
    records = collect_and_tokenize(tokenizer)

    train, val, test = split_records(records)

    print("\nSplit statistics:")
    print_split_stats("train", train)
    print_split_stats("val", val)
    print_split_stats("test", test)

    total_train_tokens = sum(r["token_count"] for r in train)
    if total_train_tokens < 100_000_000:
        print(f"\nWARNING: Training tokens ({total_train_tokens:,}) < 100M target!")
    else:
        print(f"\nTraining token target met: {total_train_tokens:,} tokens ({total_train_tokens/1e6:.1f}M)")

    print("\nSaving splits ...")
    save_split(train, SPLITS_DIR / "train.jsonl")
    save_split(val, SPLITS_DIR / "val.jsonl")
    save_split(test, SPLITS_DIR / "test.jsonl")

    # Save metadata
    meta = {
        "vocab_size": vocab_size,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "train_tokens": total_train_tokens,
        "val_tokens": sum(r["token_count"] for r in val),
        "test_tokens": sum(r["token_count"] for r in test),
    }
    with (SPLITS_DIR / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print("Metadata saved to splits/metadata.json")


if __name__ == "__main__":
    main()
