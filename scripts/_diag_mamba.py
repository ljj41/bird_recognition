#!/usr/bin/env python
# -*- coding: utf-8__
"""Mamba 快速训练诊断：跑 3 轮看 val acc 是否脱离 10% 随机基线。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import setup_console  # noqa: F401

import torch
from src.config import get_device, load_config, resolve_paths
from src.utils.experiment import load_experiment_data
from compare_all import run_dl_model


def main():
    project_root = ROOT.parent
    cfg = resolve_paths(load_config(ROOT / "configs" / "compare_fast.yaml"), project_root)
    cfg["deep_learning"]["model_epochs"]["mamba"] = 5
    cfg["deep_learning"]["model_early_stop"]["mamba"] = 5
    device = get_device(cfg)
    exp = load_experiment_data(cfg)
    print("Mamba 诊断训练 (3 epoch)...")
    metrics, _ = run_dl_model("mamba", exp, cfg, device)
    print(
        f"结果: acc={metrics['accuracy']*100:.1f}% "
        f"F1={metrics['macro_f1']*100:.1f}%"
    )
    if metrics["accuracy"] < 0.25:
        print("FAIL: 仍接近随机猜测")
        sys.exit(1)
    print("OK: 已开始学习")


if __name__ == "__main__":
    main()
