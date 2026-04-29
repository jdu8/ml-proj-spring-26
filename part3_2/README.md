# Part 3: µP Scaling and Extrapolation

Investigates whether Maximal Update Parameterization (µP) improves scaling behavior
by enabling principled learning rate transfer across model widths.

Architecture: same decoder-only GPT as Part 2 (based on nanoGPT), but using
`MuLinear`/`MuReadout` from the [`mup`](https://github.com/microsoft/mup) package,
with attention scaled by `1/d_head` instead of `1/sqrt(d_head)`.

---

## Setup

```bash
pip install -r requirements.txt
```

Data is read from Part 2 (no re-download needed):
```
part2/data/train.bin   ~105M tokens
part2/data/val.bin
part2/data/test.bin
```

---

## Step-by-step

### 1. µP LR sweep on Tiny (same sweep protocol as Part 2)

```bash
python lr_sweep_mup.py
# outputs: out/sweep_mup/sweep_summary.json
```

Tests 6 learning rates (1e-4 → 3e-2) on the Tiny model under µP.
The best LR found here transfers to **all larger models** via MuAdamW — no retuning.

Check the best LR:
```bash
python -c "import json; d=json.load(open('out/sweep_mup/sweep_summary.json')); print('best µP LR:', d['best_lr'])"
```

---

### 2. Train all 5 wide-series models with µP (use the best LR from step 1)

```bash
BEST_LR=3e-3   # replace with your best µP LR

python train_mup.py --config tiny         --lr $BEST_LR --out_dir out/tiny
python train_mup.py --config small_wide   --lr $BEST_LR --out_dir out/small_wide
python train_mup.py --config medium_wide  --lr $BEST_LR --out_dir out/medium_wide
python train_mup.py --config large_wide   --lr $BEST_LR --out_dir out/large_wide
python train_mup.py --config xl_wide      --lr $BEST_LR --out_dir out/xl_wide
```

MuAdamW automatically scales the effective per-layer LR by `base_width / layer_width`,
so you pass the same `--lr` to all models.

---

### 3. SP vs µP comparison plot + 10× extrapolation

```bash
python scaling_plot_mup.py
# outputs: plots/comparison.png
#          out/scaling_comparison.json
```

Loads Part 2 SP wide-series results from `../part2/out/` and Part 3 µP results from
`out/`, fits power laws to both, and plots them on the same graph with a 10× extrapolation.

---

## Model configs (wide series — n_layer=4 fixed)

| Name        | ~Params | d_model | n_layers | n_heads | d_ff | Width mult |
|-------------|---------|---------|----------|---------|------|-----------|
| Tiny        | ~1.31M  | 128     | 4        | 4       | 512  | 1×        |
| Small-Wide  | ~2.56M  | 192     | 4        | 6       | 768  | 1.5×      |
| Medium-Wide | ~8.65M  | 384     | 4        | 6       | 1536 | 3×        |
| Large-Wide  | ~14.68M | 512     | 4        | 8       | 2048 | 4×        |
| XL-Wide     | ~31.46M | 768     | 4        | 12      | 3072 | 6×        |

Fixing n_layer=4 isolates the width dimension so µP's 1/width LR multiplier has the
cleanest possible signal. Tiny is the µP base model (width_mult=1).

---

## µP changes vs Part 2 SP model

| Component            | SP (Part 2)                | µP (Part 3)                          |
|----------------------|----------------------------|--------------------------------------|
| Hidden linear layers | `nn.Linear`                | `mup.MuLinear`                       |
| LM head              | `nn.Linear` (weight-tied)  | `mup.MuReadout` (no weight tying)    |
| Attention scale      | `1/sqrt(d_head)`           | `1/d_head`                           |
| Optimizer            | `torch.optim.AdamW`        | `mup.MuAdamW`                        |
| LR per layer         | uniform (same for all)     | scaled by `base_width / layer_width` |
| Param init           | N(0, 0.02)                 | N(0, 0.02) × sqrt(base/target width) |

---

## Key hyperparameters (same as Part 2)

| Setting      | Default     | Notes                                    |
|--------------|-------------|------------------------------------------|
| batch_size   | 32 seqs     | per micro-step                           |
| grad_accum   | 4           | effective batch = 32 × 4 × 1024 = 131K tokens |
| block_size   | 1024 tokens | fixed                                    |
| optimizer    | MuAdamW     | β₁=0.9, β₂=0.95, wd=0.1                |
| schedule     | cosine + warmup | warmup_ratio=0.05                   |
| data_dir     | ../part2/data | reuses Part 2 prepared data           |

---

## Output layout

```
part3_2/
├── out/
│   ├── sweep_mup/            # µP LR sweep
│   │   ├── lr_3e-3/
│   │   │   ├── log.csv
│   │   │   └── results.json
│   │   └── sweep_summary.json
│   ├── tiny/
│   ├── small_wide/
│   ├── medium_wide/
│   ├── large_wide/
│   ├── xl_wide/
│   └── scaling_comparison.json
└── plots/
    └── comparison.png
```

Each `out/<config>/results.json` contains:
```json
{
  "config": "xl_wide",
  "parameterization": "muP",
  "n_params": 31464192,
  "lr": 0.003,
  "best_val_loss": 0.97,
  "wall_time_s": 280,
  "width_mult": 6.0
}
```

---

## Background

µP (Maximal Update Parameterization) ensures that every parameter in the network
receives a gradient update of O(1) magnitude regardless of model width, by:

1. **Attention scaling**: `QK^T / d_head` instead of `QK^T / sqrt(d_head)` — prevents
   the attention logits from growing with width.

2. **Per-layer LR multipliers**: `MuAdamW` scales the effective LR of each weight matrix
   by `base_fan_in / layer_fan_in`, compensating for the larger weight matrices in wider
   models that would otherwise receive updates too large for stable training.

3. **Init rescaling**: `set_base_shapes(..., rescale_params=True)` scales the initialization
   std by `sqrt(base_width / target_width)`, keeping pre-activation variance ≈ O(1).

Under SP, the optimal LR scales roughly as `1/d_model`. Using the Tiny LR on XL-Wide
(6× wider) is equivalent to training with 6× too large a LR — leading to instability.
µP corrects this automatically through `MuAdamW`'s per-layer scaling.
