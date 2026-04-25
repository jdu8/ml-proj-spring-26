# Part 3: µP Scaling Study

Maximal Update Parameterization (µP) for the SVG transformer.
Under µP, the optimal LR found on Tiny transfers to all larger models without retuning.

## Key differences from Part 2 (SP)

| | SP (Part 2) | µP (Part 3) |
|--|-------------|-------------|
| Attention scale | `1/sqrt(d_head)` | `1/d_head` |
| Output layer | `nn.Linear` | `MuReadout` |
| Optimizer | `AdamW` | `MuAdamW` |
| Weight tying | yes | no (breaks µP LR scaling) |
| LR transfer | breaks for large models | transfers exactly |

## Setup

```bash
pip install -r requirements.txt   # adds mup on top of part2 deps
```

## Step-by-step

### 1. Generate base shapes (run once)

```bash
python make_base_shapes.py
```

Creates `base_shapes.bsh` — tells mup which dimensions are "width" dimensions.

---

### 2. LR sweep on µP Tiny (~30 min on T4)

```bash
python lr_sweep_mup.py
```

Sweeps 6 LRs on the µP Tiny model. The best LR transfers to ALL larger models.

---

### 3. Train all 5 µP model sizes

```bash
BEST_LR=3e-3   # replace with best from sweep

python train_mup.py --config tiny   --lr $BEST_LR --out_dir out/tiny
python train_mup.py --config small  --lr $BEST_LR --out_dir out/small
python train_mup.py --config medium --lr $BEST_LR --out_dir out/medium
python train_mup.py --config large  --lr $BEST_LR --out_dir out/large
python train_mup.py --config xl     --lr $BEST_LR --out_dir out/xl
```

Same LR for all sizes — that is the µP guarantee.

For Large/XL, increase batch size for better GPU utilization (same effective batch):
```bash
python train_mup.py --config large --lr $BEST_LR --out_dir out/large --batch_size 128 --grad_accum 1
python train_mup.py --config xl    --lr $BEST_LR --out_dir out/xl    --batch_size 64  --grad_accum 2
```

---

### 4. Compare SP vs µP

```bash
python compare_plot.py
```

Reads `../part2/out/` (SP) and `out/` (µP), fits power laws to both, plots on one graph.
Output: `plots/compare.png`, `out/scaling_fit_comparison.json`

---

## What to expect

- **µP Tiny**: similar or slightly worse than SP Tiny (same model, minor init differences)
- **µP Large/XL**: should be significantly better than SP Large/XL — the LR transfers
- **Scaling exponent α**: µP should give a steeper (more negative) α, meaning better scaling
