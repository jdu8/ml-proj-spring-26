"""
Generate base_shapes.bsh — run this ONCE before any µP training.

The file tells mup which parameter dimensions are "width" dimensions
(i.e. scale with d_model) so it can apply the correct LR multipliers.

Usage:
    python make_base_shapes.py
"""
import sys
from pathlib import Path

from mup import make_base_shapes

sys.path.insert(0, str(Path(__file__).parent.parent / "part2"))
from configs import get_config
from model_mup import GPTMuP, GPTConfig

OUT = Path(__file__).parent / "base_shapes.bsh"


def build_model(n_embd, cfg_dict):
    cfg = GPTConfig(
        vocab_size=4096,
        block_size=1024,
        n_layer=cfg_dict["n_layer"],
        n_head=cfg_dict["n_head"],
        n_embd=n_embd,
        n_ff=n_embd * 4,   # keep ratio fixed; mup uses ratio not absolute size
        bias=False,
    )
    return GPTMuP(cfg)


def main():
    # base  = Tiny width (128)
    # delta = double the width (256) — just needs to differ to define scaling dims
    tiny_cfg = get_config("tiny")
    base  = build_model(tiny_cfg["n_embd"],         tiny_cfg)
    delta = build_model(tiny_cfg["n_embd"] * 2,     tiny_cfg)  # wider, same depth

    make_base_shapes(base, delta, savefile=str(OUT))
    print(f"Base shapes saved → {OUT}")


if __name__ == "__main__":
    main()
