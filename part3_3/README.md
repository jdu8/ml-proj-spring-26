# Part 3 (v3): µP Scaling Study — Manual Width Scaling

Maximal Update Parameterization (µP) for the SVG transformer, implemented without the `mup` package.
The optimal LR found on Tiny transfers to all larger models via explicit width scaling.

## Key differences from Part 2 (SP) and Part 3 v1/v2

| | SP (Part 2) | µP v1/v2 (part3) | µP v3 (this) |
|--|-------------|-------------------|---------------|
| Attention scale | `1/sqrt(d_head)` | `1/d_head` | `1/d_head` |
| Output layer | `nn.Linear` | `MuReadout` | `nn.Linear` |
| Optimizer | `AdamW` | `MuAdamW` | `AdamW` |
| LR scaling | fixed | mup package (per-param) | `lr × BASE_WIDTH/n_embd` |
| Steps/epoch | ~800 | ~800 | ~800 |
| Model series | original 5 | wide (n_layer=4) | original 5 |

The core µP change is just two things: **1/d_head attention** and **LR ∝ 1/width**.
Everything else is standard — no `mup` package dependency.

## Setup

```bash
pip install -r requirements.txt   # same as part2 — no mup package needed
```

## Step-by-step

### 1. LR sweep on µP Tiny

```bash
cd part3_3/
python lr_sweep_mup.py
# Output: out/sweep/sweep_summary_mup.json  — pick best_lr from here
```

Sweeps 6 LRs on Tiny. The best nominal LR transfers to all larger models because
`train_mup.py` automatically scales it: `mup_lr = lr × 128 / n_embd`.

---

### 2. Train all 5 µP models

```bash
BEST_LR=3e-3   # replace with best_lr from sweep_summary_mup.json

python train_mup.py --config tiny   --lr $BEST_LR --out_dir out/tiny
python train_mup.py --config small  --lr $BEST_LR --out_dir out/small
python train_mup.py --config medium --lr $BEST_LR --out_dir out/medium
python train_mup.py --config large  --lr $BEST_LR --out_dir out/large
python train_mup.py --config xl     --lr $BEST_LR --out_dir out/xl
```

Same nominal LR for all sizes — effective LR is width-scaled automatically per model.

---

### 3. Compare SP vs µP

```bash
python compare_plot.py --series original --sp_dir ../part2/out --mup_dir out
```

Output: `plots/compare.png`

---

## How LR scaling works

`train_mup.py` computes `mup_lr = lr × (BASE_WIDTH / n_embd)` where `BASE_WIDTH = 128` (Tiny's width).

| Model | d_model | Multiplier | Effective LR @ nominal 3×10⁻³ |
|-------|---------|------------|--------------------------------|
| Tiny   | 128 | 128/128 = 1.000 | 3.00×10⁻³ |
| Small  | 192 | 128/192 = 0.667 | 2.00×10⁻³ |
| Medium | 384 | 128/384 = 0.333 | 1.00×10⁻³ |
| Large  | 512 | 128/512 = 0.250 | 7.50×10⁻⁴ |
| XL     | 768 | 128/768 = 0.167 | 5.00×10⁻⁴ |

XL gets a 6× smaller LR than Tiny — exactly the correction SP was missing.
