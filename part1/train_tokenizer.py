"""
Train BPE tokenizers at each vocab size [1024, 2048, 4096, 8192].
Reports real avg sequence length and coverage for each — no simulation.

Usage: python train_tokenizer.py
"""

import json
import random
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tqdm import tqdm

CLEANED_DIR = Path(__file__).parent / "data" / "cleaned"
TOKENIZER_DIR = Path(__file__).parent / "tokenizer"
CORPUS_FILE = TOKENIZER_DIR / "corpus.txt"

VOCAB_SIZES = [1024, 2048, 4096, 8192]
SPECIAL_TOKENS = ["<|endoftext|>", "<unk>"]
EVAL_SAMPLE = 5000
SEED = 42


def collect_svg_paths() -> list[Path]:
    paths = sorted(CLEANED_DIR.rglob("*.svg"))
    print(f"Found {len(paths):,} cleaned SVG files")
    return paths


def build_corpus_file(paths: list[Path]) -> Path:
    if CORPUS_FILE.exists():
        print(f"Corpus file exists ({CORPUS_FILE.stat().st_size / 1e6:.1f} MB) — skipping")
        return CORPUS_FILE
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    print("Writing corpus file ...")
    with CORPUS_FILE.open("w", encoding="utf-8") as f:
        for p in tqdm(paths, desc="Building corpus"):
            f.write(p.read_text(encoding="utf-8", errors="replace").strip() + "\n")
    print(f"Corpus: {CORPUS_FILE.stat().st_size / 1e6:.1f} MB")
    return CORPUS_FILE


def train_one(vocab_size: int, corpus_file: Path) -> Tokenizer:
    out_path = TOKENIZER_DIR / f"tokenizer_{vocab_size}.json"
    if out_path.exists():
        print(f"  [vocab={vocab_size}] already trained — loading")
        return Tokenizer.from_file(str(out_path))

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train(files=[str(corpus_file)], trainer=trainer)
    tokenizer.save(str(out_path))
    print(f"  [vocab={vocab_size}] saved → {out_path.name}")
    return tokenizer


def evaluate(tokenizer: Tokenizer, sample_paths: list[Path], vocab_size: int) -> dict:
    unk_id = tokenizer.token_to_id("<unk>")
    lengths = []
    n_unk = 0
    n_over_1024 = 0

    for p in sample_paths:
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        enc = tokenizer.encode(text)
        ids = enc.ids
        lengths.append(len(ids))
        if unk_id in ids:
            n_unk += 1
        if len(ids) > 1024:
            n_over_1024 += 1

    n = len(sample_paths)
    return {
        "vocab_size": vocab_size,
        "avg_seq_len": sum(lengths) / n,
        "median_seq_len": sorted(lengths)[n // 2],
        "pct_no_unk": (n - n_unk) / n * 100,
        "pct_over_1024": n_over_1024 / n * 100,
        "estimated_total_tokens": sum(lengths) / n * len(sample_paths),  # rough, corrected later
    }


def main():
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    paths = collect_svg_paths()
    corpus_file = build_corpus_file(paths)

    random.seed(SEED)
    sample = random.sample(paths, min(EVAL_SAMPLE, len(paths)))

    all_results = {}

    print(f"\nTraining {len(VOCAB_SIZES)} tokenizers and evaluating on {len(sample):,} SVGs ...\n")
    for vs in VOCAB_SIZES:
        print(f"── vocab={vs} ──")
        tok = train_one(vs, corpus_file)
        metrics = evaluate(tok, sample, vs)

        # Correct token estimate using full path count
        metrics["estimated_total_tokens"] = metrics["avg_seq_len"] * len(paths)
        metrics["estimated_train_tokens"] = metrics["estimated_total_tokens"] * 0.98
        all_results[vs] = metrics

    # Print comparison table
    print(f"\n{'Vocab':>8}  {'Avg len':>8}  {'Median':>8}  {'No-UNK':>8}  {'>1024':>8}  {'Est. train tokens':>20}")
    print("-" * 75)
    for vs, m in all_results.items():
        print(
            f"{vs:>8,}  {m['avg_seq_len']:>8.1f}  {m['median_seq_len']:>8}  "
            f"{m['pct_no_unk']:>7.1f}%  {m['pct_over_1024']:>7.1f}%  "
            f"{m['estimated_train_tokens']:>18,.0f}"
        )

    with (TOKENIZER_DIR / "vocab_comparison.json").open("w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to tokenizer/vocab_comparison.json")


if __name__ == "__main__":
    main()
