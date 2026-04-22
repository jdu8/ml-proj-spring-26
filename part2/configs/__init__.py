from configs.tiny import config as _tiny
from configs.small import config as _small
from configs.medium import config as _medium
from configs.large import config as _large
from configs.xl import config as _xl

CONFIGS = {
    "tiny":   _tiny,
    "small":  _small,
    "medium": _medium,
    "large":  _large,
    "xl":     _xl,
}


def get_config(name: str) -> dict:
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Choose from: {list(CONFIGS.keys())}")
    return CONFIGS[name]
