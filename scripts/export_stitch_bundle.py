#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出 Stitch 37 类推理部署包，供 GUI 使用。

用法（Anaconda Prompt）:
  cd /d D:\\语音识别大作业
  conda activate niumayolo
  python bird_recognition\\scripts\\export_stitch_bundle.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import setup_console  # noqa: F401

from src.config import load_config, resolve_paths
from src.utils.experiment import load_experiment_data


def load_taxonomy_map(taxonomy_csv: Path) -> dict[str, dict]:
    df = pd.read_csv(taxonomy_csv)
    aves = df[df["class_name"] == "Aves"]
    return {
        row["primary_label"]: {
            "common_name": str(row.get("common_name", row["primary_label"])),
            "scientific_name": str(row.get("scientific_name", "")),
        }
        for _, row in aves.iterrows()
    }


def parse_args():
    p = argparse.ArgumentParser(description="导出 Stitch 推理部署包")
    p.add_argument("--config", default=str(ROOT / "configs" / "compare.yaml"))
    p.add_argument(
        "--checkpoint",
        default=None,
        help="权重路径，默认 outputs/models/deep/stitch/compare/best_model.pt",
    )
    p.add_argument("--bundle-name", default="stitch_top37", help="部署目录名")
    return p.parse_args()


def main():
    args = parse_args()
    project_root = ROOT.parent
    cfg = resolve_paths(load_config(args.config), project_root)
    paths = cfg["paths"]

    ckpt_src = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(paths["models_dir"]) / "deep" / "stitch" / "compare" / "best_model.pt"
    )
    if not ckpt_src.exists():
        raise FileNotFoundError(
            f"未找到 Stitch 权重: {ckpt_src}\n"
            "请先完成训练:\n"
            "  python bird_recognition\\scripts\\compare_all.py "
            "--config bird_recognition\\configs\\compare.yaml --skip-ml --dl-only stitch"
        )

    ckpt = torch.load(ckpt_src, map_location="cpu", weights_only=False)
    n_cls_ckpt = None
    best_idx = -1
    for key, tensor in ckpt["model_state"].items():
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] == "head" and parts[-1] == "weight":
            try:
                layer_idx = int(parts[1])
            except ValueError:
                continue
            if layer_idx > best_idx:
                best_idx = layer_idx
                n_cls_ckpt = int(tensor.shape[0])
    if n_cls_ckpt is None:
        raise RuntimeError(f"无法从 checkpoint 读取类别数: {ckpt_src}")

    exp = load_experiment_data(cfg)
    n_cls = len(exp["label2id"])
    if n_cls != n_cls_ckpt:
        raise ValueError(
            f"配置类别数 {n_cls} 与权重类别数 {n_cls_ckpt} 不一致，请使用与训练相同的 compare.yaml"
        )

    deploy_dir = Path(paths["models_dir"]) / "deploy" / args.bundle_name
    deploy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_src, deploy_dir / "stitch.pt")

    metrics_path = ckpt_src.parent / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

    tax_map = load_taxonomy_map(Path(paths["taxonomy_csv"]))
    id2label = {int(v): k for k, v in exp["label2id"].items()}
    species = []
    for i in range(n_cls):
        code = id2label[i]
        meta = tax_map.get(code, {})
        species.append({
            "id": i,
            "code": code,
            "common_name": meta.get("common_name", code),
            "scientific_name": meta.get("scientific_name", ""),
        })

    manifest = {
        "model_type": "dl",
        "model_name": "stitch",
        "weight_file": "stitch.pt",
        "metrics": {
            "accuracy": float(metrics.get("accuracy", ckpt.get("val_metrics", {}).get("accuracy", 0))),
            "macro_f1": float(metrics.get("macro_f1", ckpt.get("val_metrics", {}).get("macro_f1", 0))),
            "top_5_acc": float(metrics.get("top_5_acc", ckpt.get("val_metrics", {}).get("top_5_acc", 0))),
        },
        "best_epoch": int(ckpt.get("epoch", 0)),
        "n_classes": n_cls,
        "sample_rate": cfg["data"]["sample_rate"],
        "duration": cfg["data"]["duration"],
        "handcrafted_dim": exp["handcrafted_dim"],
        "n_mels": exp["n_mels"],
        "feature_config": cfg["features"],
        "species": species,
        "config": str(Path("bird_recognition/configs/compare.yaml")),
        "checkpoint_source": str(ckpt_src.resolve()),
    }
    with open(deploy_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("  Stitch 部署包已导出")
    print("=" * 60)
    print(f"  目录: {deploy_dir}")
    print(f"  类别: {n_cls}  |  最佳轮次: {manifest['best_epoch']}")
    print(f"  准确率: {manifest['metrics']['accuracy']*100:.1f}%")
    print(f"  Macro-F1: {manifest['metrics']['macro_f1']*100:.1f}%")
    print(f"  Top-5: {manifest['metrics']['top_5_acc']*100:.1f}%")
    print("\n启动 GUI:")
    print("  python bird_recognition\\scripts\\run_gui.py --bundle stitch_top37")
    print("=" * 60)


if __name__ == "__main__":
    main()
