"""YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a YAML mapping.")
    return config


def require(config: dict[str, Any], *keys: str) -> Any:
    current: Any = config
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        if not isinstance(current, dict) or key not in current:
            raise KeyError(
                f"Missing required configuration key: {'.'.join(traversed)}"
            )
        current = current[key]
    return current
