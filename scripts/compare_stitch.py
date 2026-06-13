#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块缝合模型 (StitchedFusionNet) 训练脚本。

融合 CNN 全局池化 + CRNN 时序注意力 + Mamba 时序建模 + 门控融合。
用于课程报告中的「原创改进 / 模块缝合」实验。

用法（Anaconda Prompt）:
  cd /d D:\\语音识别大作业
  conda activate niumayolo

  python bird_recognition\\scripts\\compare_stitch.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import setup_console  # noqa: F401

from src.config import get_device, load_config, resolve_paths
from src.utils.experiment import get_best_result, load_experiment_data, print_data_summary, save_result
from compare_all import run_dl_model


def parse_args():
    p = argparse.ArgumentParser(description="训练模块缝合模型 StitchedFusionNet")
    p.add_argument("--config", default=str(ROOT / "configs" / "compare_fast.yaml"))
    p.add_argument("--no-viz", action="store_true")
    return p.parse_args()


def merge_results(out_path: Path, metrics: dict) -> list[dict]:
    prev: list[dict] = []
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            prev = []
    keep = [r for r in prev if not (r.get("type") == "dl" and r.get("model") == "stitch")]
    return keep + [metrics]


def main():
    args = parse_args()
    project_root = ROOT.parent
    cfg = resolve_paths(load_config(args.config), project_root)
    device = get_device(cfg)
    paths = cfg["paths"]

    print("=" * 60)
    print("  模块缝合模型 — StitchedFusionNet")
    print("  CNN全局 + CRNN注意力 + Mamba时序 + 门控融合")
    print("=" * 60)

    exp = load_experiment_data(cfg)
    n_cls = len(exp["label2id"])
    print("\n[数据集概况]")
    print_data_summary(exp["train_df"], exp["val_df"], n_cls)
    print(f"  计算设备: {device}")

    metrics, _ = run_dl_model("stitch", exp, cfg, device)

    out_path = Path(paths["output_dir"]) / "comparison_results.json"
    merged = merge_results(out_path, metrics)
    save_result(merged, out_path)
    print(f"\n结果已合并: {out_path}")
    print(
        f"  Stitch 验证集: 准确率 {metrics['accuracy']*100:.1f}% | "
        f"Macro-F1 {metrics['macro_f1']*100:.1f}% | "
        f"Top-5 {metrics.get('top_5_acc', 0)*100:.1f}%"
    )

    if not args.no_viz:
        viz = ROOT / "scripts" / "visualize_results.py"
        if viz.exists():
            import subprocess
            print("\n[可视化] 更新对比图表...")
            subprocess.run([
                sys.executable, str(viz),
                "--config", args.config,
                "--results", str(out_path),
            ], check=False)


if __name__ == "__main__":
    main()
