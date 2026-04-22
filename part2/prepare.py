"""
Tokenize the Part 1 JSONL splits and pack them into flat binary files for training.

Reads:
    ../part1/data/splits/{train,val,test}.jsonl
    ../part1/tokenizer/tokenizer_4096.json

Writes:
    data/train.bin  (~200 MB, uint16)
    data/val.bin
    data/test.bin
    data/stats.json

Each .bin file is a flat uint16 numpy array of token IDs.
SVGs are separated by the EOS token <|endoftext|>.
uint16 is safe since vocab_size=4096 < 65536.
"""
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

PART1_ROOT = Path(__file__).resolve().parent.parent / "part1"
SPLITS_DIR = PART1_ROOT / "data" / "splits"
TOKENIZER_PATH = PART1_ROOT / "tokenizer" / "tokenizer_4096.json"
OUT_DIR = Path(__file__).resolve().parent / "data"


def tokenize_split(split_path: Path, tokenizer: Tokenizer, eos_id: int) -> np.ndarray:
    svgs = []
    with open(split_path) as f:
        for line in f:
            svgs.append(json.loads(line)["svg"])

    all_ids: list[int] = []
    for svg in tqdm(svgs, desc=split_path.stem, unit="svg"):
        all_ids.extend(tokenizer.encode(svg).ids)
        all_ids.append(eos_id)

    return np.array(all_ids, dtype=np.uint16)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    assert eos_id is not None, "EOS token <|endoftext|> not found in tokenizer"
    print(f"EOS token id: {eos_id}")

    stats = {}
    for split in ("train", "val", "test"):
        split_path = SPLITS_DIR / f"{split}.jsonl"
        if not split_path.exists():
            print(f"Skipping {split} — {split_path} not found")
            continue

        arr = tokenize_split(split_path, tokenizer, eos_id)
        out_path = OUT_DIR / f"{split}.bin"
        arr.tofile(out_path)
        mb = out_path.stat().st_size / 1e6
        stats[split] = {"tokens": int(len(arr)), "size_mb": round(mb, 1)}
        print(f"  {split}: {len(arr):,} tokens  →  {out_path}  ({mb:.1f} MB)")

    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nDone. Stats → {OUT_DIR / 'stats.json'}")


if __name__ == "__main__":
    main()
