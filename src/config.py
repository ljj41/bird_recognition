"""Configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_paths(cfg: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    """Resolve relative paths against project root."""
    root = Path(project_root).resolve()
    paths = cfg.setdefault("paths", {})
    for key, value in list(paths.items()):
        if key == "data_root":
            paths[key] = str((root / value).resolve()) if value != "." else str(root)
        elif isinstance(value, str) and not Path(value).is_absolute():
            paths[key] = str((root / value).resolve())
    # 确保缓存/图表目录存在
    for sub in ("cache_dir", "figures_dir", "features_dir", "models_dir", "logs_dir"):
        if sub in paths:
            Path(paths[sub]).mkdir(parents=True, exist_ok=True)
    return cfg


def get_device(cfg: dict[str, Any]) -> str:
    import torch

    requested = cfg.get("project", {}).get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
