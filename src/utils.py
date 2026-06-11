"""Shared utilities for configuration, paths, and reproducibility."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration from disk."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(relative_path: str | Path, config: dict[str, Any] | None = None) -> Path:
    """Resolve a project-relative path."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return the path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Persist a dictionary as formatted JSON."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def get_label_names(config: dict[str, Any]) -> dict[int, str]:
    """Return human-readable label names."""
    data_cfg = config["data"]
    return {
        int(data_cfg["fake_label"]): "Fake",
        int(data_cfg["real_label"]): "Real",
    }
