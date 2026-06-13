#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer + Mamba 对比实验脚本。

与 compare_all.py 使用相同的数据划分、特征缓存和训练流程，
专门用于补全课程要求的 Transformer / Mamba 模型对比。

用法（Anaconda Prompt）:
  cd /d D:\\语音识别大作业
  conda activate niumayolo

  # 1. 先做前向测试（约 10 秒）
  python bird_recognition\\scripts\\test_transformer_mamba.py

  # 2. 训练 Transformer + Mamba（约 15~25 分钟，GPU）
  python bird_recognition\\scripts\\compare_transformer_mamba.py

  # 3. 只训其中一个
  python bird_recognition\\scripts\\compare_transformer_mamba.py --only transformer
  python bird_recognition\\scripts\\compare_transformer_mamba.py --only mamba

  # 4. 重画对比图
  python bird_recognition\\scripts\\visualize_results.py --config bird_recognition\\configs\\compare_fast.yaml
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


DEFAULT_MODELS = ["transformer", "mamba"]


def parse_args():
    p = argparse.ArgumentParser(description="Transformer / Mamba 对比实验")
    p.add_argument(
        "--config",
        default=str(ROOT / "configs" / "compare_fast.yaml"),
        help="配置文件（需含 transformer/mamba 的 batch 与 epoch 设置）",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        choices=["transformer", "mamba"],
        help="只训练指定模型",
    )
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="训练完成后不自动生成图表",
    )
    return p.parse_args()


def merge_results(out_path: Path, new_results: list[dict], model_names: list[str]) -> list[dict]:
    """合并到 comparison_results.json，替换同名 DL 条目，保留 ML 与其他 DL。"""
    prev: list[dict] = []
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            prev = []

    replace_models = set(model_names)
    keep = [
        r for r in prev
        if not (r.get("type") == "dl" and r.get("model") in replace_models)
    ]
    return keep + new_results


def main():
    args = parse_args()
    project_root = ROOT.parent
    cfg = resolve_paths(load_config(args.config), project_root)
    device = get_device(cfg)
    paths = cfg["paths"]
    models = [args.only] if args.only else DEFAULT_MODELS

    print("=" * 60)
    print("  Transformer / Mamba 对比实验")
    print("=" * 60)

    exp = load_experiment_data(cfg)
    n_cls = len(exp["label2id"])
    print("\n[数据集概况]")
    print_data_summary(exp["train_df"], exp["val_df"], n_cls)
    print(f"  计算设备: {device}")
    print(f"  待训练模型: {', '.join(m.upper() for m in models)}")

    all_results: list[dict] = []
    for model_name in models:
        print("\n" + "-" * 40)
        print(f"[深度学习] 训练 {model_name.upper()} 模型...")
        print("-" * 40)
        metrics, _ = run_dl_model(model_name, exp, cfg, device)
        all_results.append(metrics)

    out_path = Path(paths["output_dir"]) / "comparison_results.json"
    merged = merge_results(out_path, all_results, models)
    save_result(merged, out_path)
    print(f"\n对比结果已合并保存: {out_path}")

    dl_rows = [r for r in merged if r.get("type") == "dl"]
    if dl_rows:
        best = get_best_result(dl_rows, "macro_f1")
        print("\n" + "=" * 60)
        print(f"  本次 DL 最优: {best.get('model', 'N/A').upper()}")
        print(f"  Macro-F1: {best.get('macro_f1', 0)*100:.1f}%")
        print(f"  准确率:   {best.get('accuracy', 0)*100:.1f}%")
        print("=" * 60)

    if not args.no_viz:
        viz_script = ROOT / "scripts" / "visualize_results.py"
        if viz_script.exists():
            import subprocess
            print("\n[可视化] 正在更新对比图表...")
            subprocess.run([
                sys.executable, str(viz_script),
                "--config", args.config,
                "--results", str(out_path),
            ], check=False)


if __name__ == "__main__":
    main()
