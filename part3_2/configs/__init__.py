from configs.tiny import config as _tiny
from configs.small_wide import config as _small_wide
from configs.medium_wide import config as _medium_wide
from configs.large_wide import config as _large_wide
from configs.xl_wide import config as _xl_wide

CONFIGS = {
    "tiny":         _tiny,
    # Pure-width series (n_layer=4 fixed) — used for µP LR transfer study
    "small_wide":   _small_wide,
    "medium_wide":  _medium_wide,
    "large_wide":   _large_wide,
    "xl_wide":      _xl_wide,
}


def get_config(name: str) -> dict:
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Choose from: {list(CONFIGS.keys())}")
    return CONFIGS[name]
