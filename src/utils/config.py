"""Configuration loading and repository paths."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def load_config(name: str = "data_config.yaml") -> dict:
    with open(ROOT / "configs" / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def path(key: str) -> Path:
    """Resolve a configured data path (raw, interim, processed, warehouse)."""
    p = ROOT / load_config()["paths"][key]
    if p.suffix == "":
        p.mkdir(parents=True, exist_ok=True)
    return p


def regions() -> dict[str, str]:
    """ITL1 code -> region name for the modelled regions."""
    return dict(load_config()["regions"])
