#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据集统计分析。"""

import setup_console  # noqa: F401
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_paths
from src.utils.experiment import load_experiment_data, print_data_summary

project_root = ROOT.parent
cfg = resolve_paths(load_config(ROOT / "configs" / "compare.yaml"), project_root)

exp = load_experiment_data(cfg)
print("=" * 50)
print("  BirdCLEF2026 鸟类数据集分析")
print("=" * 50)
print_data_summary(exp["train_df"], exp["val_df"], len(exp["label2id"]))
print("\n样本最多的10个鸟种:")
print(exp["train_df"]["primary_label"].value_counts().head(10))
