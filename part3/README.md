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
| LR transfer | breaks at large width | transfers exactly |

## Why the wide series (not the original 5)?

µP's LR multipliers correct for **width** (1/d_model) only — depth is not accounted for.
The original series grows both d_model and n_layer simultaneously, which muddies µP's guarantees.
The wide series fixes n_layer=4 and only varies d_model, giving µP a clean controlled experiment.

## Setup

```bash
pip install -r requirements.txt   # adds mup on top of part2 deps
```

## Step-by-step

> **Note:** `make_base_shapes.py` is not needed — `train_mup.py` builds base shapes
> inline for each model size, which is more robust. Skip directly to step 1.

### 1. LR sweep on µP Tiny (~5 min on T4)

```bash
cd part3/
python lr_sweep_mup.py
# Output: out/sweep/sweep_summary_mup.json  — pick best_lr from here
```

Sweeps 6 LRs on the µP Tiny model. The best nominal LR transfers to all wider models
because MuAdamW automatically scales effective LR as `base_width / d_model`.

---

### 2. Train all µP wide-series models

```bash
BEST_LR=3e-3   # replace with best_lr from sweep_summary_mup.json

python train_mup.py --config tiny        --lr $BEST_LR --out_dir out/tiny
python train_mup.py --config small_wide  --lr $BEST_LR --out_dir out/small_wide
python train_mup.py --config medium_wide --lr $BEST_LR --out_dir out/medium_wide
python train_mup.py --config large_wide  --lr $BEST_LR --out_dir out/large_wide
python train_mup.py --config xl_wide     --lr $BEST_LR --out_dir out/xl_wide
```

Same nominal LR for all sizes — MuAdamW applies the correct per-width multiplier automatically.

---

### 3. Compare SP vs µP (wide series)

```bash
python compare_plot.py --series wide --sp_dir ../part2/out --mup_dir out
```

Reads SP results from `../part2/out/` and µP results from `out/`, fits power laws to both,
plots on one graph.
Output: `plots/compare.png`, `out/scaling_fit_comparison.json`

---

## What to expect

- **µP Tiny**: similar to SP Tiny (same architecture, minor init/scaling differences)
- **µP Small-Wide / Medium-Wide / Large-Wide**: should improve over SP counterparts — LR transfers
- **µP XL-Wide**: biggest gain — SP XL-Wide fails (LR 6× too large), µP corrects it automatically
- **Scaling exponent α**: µP should give a steeper α (monotone curve through all 5 points)

## How LR multipliers work (inline base shapes)

`train_mup.py` builds base shapes inline in `make_mup_base_shapes()`:

| Model | d_model | LR multiplier (base=96) | Effective LR @ nominal 3×10⁻³ |
|-------|---------|-------------------------|---------------------------------|
| Tiny | 128 | 96/128 = 0.750 | 2.25×10⁻³ |
| Small-Wide | 192 | 96/192 = 0.500 | 1.50×10⁻³ |
| Medium-Wide | 384 | 96/384 = 0.250 | 7.50×10⁻⁴ |
| Large-Wide | 512 | 96/512 = 0.188 | 5.63×10⁻⁴ |
| XL-Wide | 768 | 96/768 = 0.125 | 3.75×10⁻⁴ |

The LR sweep is run on Tiny, finding the best nominal LR. When that nominal LR is passed to
XL-Wide, the effective LR is automatically 6× smaller — exactly the µP transfer guarantee.
