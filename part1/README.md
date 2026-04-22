# Part 1: Data Collection and Preprocessing

Data pipeline for the SVG scaling law project. Downloads SVG datasets from HuggingFace, cleans and normalizes them, trains BPE tokenizers, and produces train/val/test splits.

## Scripts

Run these in order. Each is idempotent — re-running skips already-completed work.

### 1. `download_data.py`
Downloads SVG datasets from HuggingFace into `data/raw/`.

```bash
python download_data.py
```

Sources pulled (in order):
- `starvector/svg-icons-simple` → `data/raw/icons/`
- `starvector/svg-emoji-simple` → `data/raw/emoji/`
- `umuthopeyildirim/svgen-500k` → `data/raw/svgen/` (SVG field is `output`, not `svg`)
- `starvector/svg-stack-simple` → streamed directly into `data/cleaned/stack/` (bypasses raw, avoids caching 3.87 GB)

After running, delete `data/raw/` to free ~1.6 GB — the cleaned versions are what matter.

---

### 2. `clean_svg.py`
Normalizes and filters all SVGs from `data/raw/` into `data/cleaned/`.

```bash
python clean_svg.py
```

What it does per SVG:
- Parses with `lxml.etree` — drops invalid XML
- Strips XML comments and processing instructions
- Rounds floats to 1 decimal place (`M10.396875` → `M10.4`) via regex
- Filters: drops if `len(cleaned) < 50` or `len(cleaned) > 5000` chars

Outputs one `.svg` file per input, preserving source subdirectory structure.

> **Note**: stack SVGs are cleaned inline during streaming (see `download_data.py`) and land directly in `data/cleaned/stack/` — `clean_svg.py` does not touch them.

---

### 3. `train_tokenizer.py`
Trains a BPE tokenizer at each vocab size and reports real stats for comparison.

```bash
python train_tokenizer.py
```

- Builds `tokenizer/corpus.txt` (all cleaned SVGs concatenated, one per line)
- Trains at vocab sizes: **1024, 2048, 4096, 8192**
- Uses HuggingFace `tokenizers` with `ByteLevel` pre-tokenizer (guarantees 0% UNK)
- Saves each as `tokenizer/tokenizer_{vocab_size}.json`
- Saves comparison table to `tokenizer/vocab_comparison.json`

**Chosen vocab size: 4096** — sequence length plateaus here (766 vs 765 tokens at 8192); embedding table is half the size.

---

### 4. `create_splits.py`
Tokenizes all cleaned SVGs, drops those exceeding 1024 tokens, and creates 98/1/1 splits.

```bash
python create_splits.py [--vocab-size 4096]
```

- Loads `tokenizer/tokenizer_{vocab_size}.json`
- Drops SVGs with `token_count > 1024` (exact, post-tokenization)
- Shuffles with `seed=42`, splits **by file count** (not token position) to prevent leakage
- Saves to `data/splits/train.jsonl`, `val.jsonl`, `test.jsonl`

Each JSONL line:
```json
{"svg": "<svg ...>...</svg>", "token_count": 312, "source": "icons"}
```

Token IDs are not stored — re-tokenize at training time from `svg`.

---

### 5. `analyze.py`
Generates statistics, plots, and writes `results.md`.

```bash
python analyze.py
```

Outputs:
- `plots/seq_len_hist.png` — token length distribution per source
- `plots/svg_examples.png` — 3×3 grid of rendered SVGs (short/medium/long)
- `results.md` — full stats table for the report

---

## Directory Layout

```
part1/
├── data/
│   ├── raw/          # Raw downloads (delete after cleaning to save ~1.6 GB)
│   │   ├── icons/
│   │   ├── emoji/
│   │   └── svgen/
│   ├── cleaned/      # Normalized SVGs (source of truth)
│   │   ├── icons/    # 76,292 files
│   │   ├── emoji/    # 3,563 files
│   │   ├── svgen/    # 206,660 files
│   │   └── stack/    # 20,000 files (streamed + cleaned inline)
│   └── splits/
│       ├── train.jsonl   # 224,150 SVGs, 104.8M tokens
│       ├── val.jsonl     # 2,287 SVGs
│       ├── test.jsonl    # 2,288 SVGs
│       └── metadata.json
├── tokenizer/
│   ├── corpus.txt           # Full training corpus (378 MB, built once)
│   ├── tokenizer_1024.json
│   ├── tokenizer_2048.json
│   ├── tokenizer_4096.json  # ← used by default
│   ├── tokenizer_8192.json
│   └── vocab_comparison.json
├── plots/
├── download_data.py
├── clean_svg.py
├── train_tokenizer.py
├── create_splits.py
├── analyze.py
└── results.md           # Report-facing output (stats, plots, decisions)
```

## Loading a Split for Training

```python
import json
from pathlib import Path
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("part1/tokenizer/tokenizer_4096.json")

def load_split(path):
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            ids = tokenizer.encode(rec["svg"]).ids  # list of ints, len <= 1024
            yield ids

for token_ids in load_split("part1/data/splits/train.jsonl"):
    ...
```

## Key Numbers

| | |
|---|---|
| Total cleaned SVGs | 306,515 |
| Training SVGs | 224,150 |
| Training tokens | 104.8M |
| Avg sequence length | 468 tokens |
| Median sequence length | 424 tokens |
| Tokenizer | BPE, vocab=4096, ByteLevel |
| Context window | 1,024 tokens |

## Dependencies

```
datasets
tokenizers
lxml
cairosvg
matplotlib
numpy
tqdm
Pillow
```

Install: `pip install datasets tokenizers lxml cairosvg matplotlib numpy tqdm Pillow`
