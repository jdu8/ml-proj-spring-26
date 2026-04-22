# Part 2: Transformer Scaling Study

Decoder-only GPT models at 5 scales, trained on SVG data from Part 1.
Architecture adapted from [nanoGPT](https://github.com/karpathy/nanoGPT).

## Setup

```bash
pip install -r requirements.txt
```

## Colab setup (one-time)

```python
# Cell 1 — mount Drive and unzip data
from google.colab import drive
drive.mount('/content/drive')

import subprocess
subprocess.run([
    "unzip", "-q",
    "/content/drive/MyDrive/svg_data.zip",
    "-d", "/content/ml-proj"
])
```

```bash
# Cell 2 — clone repo and install deps
!git clone https://github.com/<your-username>/<your-repo>.git /content/ml-proj
%cd /content/ml-proj/part2
!pip install -r requirements.txt -q
```

After this, `part2/data/train.bin` and friends are in place and you can run any
`python train.py ...` command directly in a `!` cell or the Colab terminal.

---

## Step-by-step

### 1. Prepare binary data (run once)

```bash
python prepare.py
```

Reads Part 1's JSONL splits, tokenizes, and writes `data/train.bin`, `data/val.bin`, `data/test.bin`.
Train set: ~104.8M tokens (~210 MB as uint16).

---

### 2. LR sweep on Tiny (~10–60 min on T4)

```bash
python lr_sweep.py
```

Trains the Tiny model for 1 epoch at 6 learning rates (`1e-4` → `3e-2`).
Saves per-LR results under `out/sweep/` and writes `out/sweep/sweep_summary.json`.

Check the best LR:
```bash
cat out/sweep/sweep_summary.json | python -c "import sys,json; d=json.load(sys.stdin); print('best LR:', d['best_lr'])"
```

---

### 3. Train all 5 model sizes (use the best LR from step 2)

```bash
BEST_LR=3e-4   # replace with your best LR

python train.py --config tiny   --lr $BEST_LR --out_dir out/tiny
python train.py --config small  --lr $BEST_LR --out_dir out/small
python train.py --config medium --lr $BEST_LR --out_dir out/medium
python train.py --config large  --lr $BEST_LR --out_dir out/large
python train.py --config xl     --lr $BEST_LR --out_dir out/xl
```

Each run trains for exactly 1 epoch (~800 optimizer steps with the default batch size).
Output per run: `ckpt.pt`, `log.csv`, `results.json`.

---

### 4. Fit power law and produce scaling plot

```bash
python scaling_plot.py
```

Reads `out/*/results.json`, fits `L = a * N^{-α} + c`, and writes:
- `plots/scaling.png`
- `out/scaling_fit_SP.json`  (used by Part 3 for SP vs µP comparison)

---

## Model configs

| Name   | ~Params | d_model | n_layers | n_heads | d_ff |
|--------|---------|---------|----------|---------|------|
| Tiny   | ~1.4M   | 128     | 4        | 4       | 512  |
| Small  | ~3.6M   | 192     | 6        | 6       | 768  |
| Medium | ~12.6M  | 384     | 6        | 6       | 1536 |
| Large  | ~34M    | 512     | 10       | 8       | 2048 |
| XL     | ~88M    | 768     | 12       | 12      | 3072 |

Parameter counts exclude positional embeddings (Kaplan et al. convention).

---

## Key hyperparameters

| Setting | Default | Notes |
|---------|---------|-------|
| batch_size | 8 sequences | per micro-step |
| grad_accum | 16 | effective batch = 8 × 16 × 1024 ≈ 131K tokens |
| block_size | 1024 tokens | fixed (matches Part 1 max seq len) |
| optimizer | AdamW | β₁=0.9, β₂=0.95, wd=0.1 |
| schedule | cosine + warmup | warmup_ratio=0.05 |
| AMP | fp16 on CUDA | auto-disabled on CPU |

Adjust `--batch_size` / `--grad_accum` if you hit OOM on the XL model.

---

## Output layout

```
part2/
├── data/
│   ├── train.bin       # ~210 MB, uint16
│   ├── val.bin
│   ├── test.bin
│   └── stats.json
├── out/
│   ├── sweep/          # LR sweep runs
│   │   ├── lr_1e-4/
│   │   │   ├── log.csv
│   │   │   └── results.json
│   │   └── sweep_summary.json
│   ├── tiny/
│   ├── small/
│   ├── medium/
│   ├── large/
│   ├── xl/
│   └── scaling_fit_SP.json
└── plots/
    └── scaling.png
```

Each `out/<config>/results.json` has:
```json
{
  "config": "tiny",
  "n_params": 1310720,
  "lr": 3e-4,
  "best_val_loss": 1.23,
  "wall_time_s": 600
}
```
