#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出 Stacking 推理部署包 (DL + SVM + Meta-Learner)。

用法（Anaconda Prompt）:
  cd /d D:\\语音识别大作业
  conda activate niumayolo

  # 自动读取 comparison_results.json 中的最优 DL/ML 组件（compare.yaml 37类）
  python bird_recognition\\scripts\\export_stacking_bundle.py --config bird_recognition\\configs\\compare.yaml

  # 10 类快速配置
  python bird_recognition\\scripts\\export_stacking_bundle.py --config bird_recognition\\configs\\compare_fast.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import setup_console  # noqa: F401

from src.config import get_device, load_config, resolve_paths
from src.data.cache_dataset import CachedBirdDataset, build_feature_cache
from src.models.architectures import build_model
from src.training.ensemble import StackingEnsemble
from src.training.metrics import compute_metrics
from src.training.ml_trainer import build_ml_model, extract_features_from_df, ml_predict_proba
from src.training.trainer import predict_proba
from src.utils.experiment import get_best_result, load_experiment_data


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


def load_components_from_results(results_path: Path) -> tuple[str, str]:
    if not results_path.exists():
        return "cnn", "svm"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    stack = next(
        (r for r in results if r.get("model") == "stacking_ensemble" and r.get("phase") == "holdout"),
        None,
    )
    if stack:
        return stack.get("dl_component", "cnn"), stack.get("ml_component", "svm")
    best_dl = get_best_result(
        [r for r in results if r.get("type") == "dl" and r.get("phase") == "holdout"], "macro_f1"
    )
    best_ml = get_best_result(
        [r for r in results if r.get("type") == "ml" and r.get("phase") == "holdout"], "macro_f1"
    )
    return best_dl.get("model", "cnn"), best_ml.get("model", "svm")


def parse_args():
    p = argparse.ArgumentParser(description="导出 Stacking 推理部署包")
    p.add_argument("--config", default=str(ROOT / "configs" / "compare.yaml"))
    p.add_argument("--dl", default=None, help="DL 组件名，默认从 comparison_results.json 读取")
    p.add_argument("--ml", default=None, help="ML 组件名，默认从 comparison_results.json 读取")
    p.add_argument("--bundle-name", default=None, help="部署目录名，默认 stacking_top{n_classes}")
    p.add_argument("--results", default=None, help="comparison_results.json 路径")
    return p.parse_args()


def export_bundle(
    cfg: dict,
    project_root: Path,
    dl_name: str,
    ml_name: str,
    bundle_name: str | None = None,
) -> Path:
    device = get_device(cfg)
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    ml_cfg = cfg["ml"]

    print("=" * 60)
    print("  导出 Stacking 推理部署包")
    print("=" * 60)

    exp = load_experiment_data(cfg)
    n_cls = len(exp["label2id"])
    deploy_dir = Path(paths["models_dir"]) / "deploy" / (bundle_name or f"stacking_top{n_cls}")
    deploy_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(paths.get("cache_dir", Path(paths["output_dir"]) / "cache"))
    train_cache = cache_dir / f"train_top{n_cls}.npz"
    val_cache = cache_dir / f"val_top{n_cls}.npz"
    build_feature_cache(
        exp["train_df"], exp["label2id"], train_cache,
        data_cfg["sample_rate"], data_cfg["duration"], exp["feat_cfg"],
    )
    build_feature_cache(
        exp["val_df"], exp["label2id"], val_cache,
        data_cfg["sample_rate"], data_cfg["duration"], exp["feat_cfg"],
    )

    # --- ML ---
    print(f"\n[1/4] 训练/导出 {ml_name.upper()} (probability=True，供 Stacking 使用)...")
    ml_cfg_deploy = {**ml_cfg, "svm_probability": True}
    feat_dir = Path(paths["features_dir"])
    tag = f"top{n_cls}_train{len(exp['train_df'])}"
    cache_train = feat_dir / f"X_train_{tag}.npy"
    if cache_train.exists():
        X_train = np.load(cache_train)
        y_train = np.load(feat_dir / f"y_train_{tag}.npy")
        X_val = np.load(feat_dir / f"X_val_{tag}.npy")
        y_val = np.load(feat_dir / f"y_val_{tag}.npy")
    else:
        X_train, y_train = extract_features_from_df(
            exp["train_df"], data_cfg["sample_rate"], data_cfg["duration"], exp["feat_cfg"]
        )
        X_val, y_val = extract_features_from_df(
            exp["val_df"], data_cfg["sample_rate"], data_cfg["duration"], exp["feat_cfg"]
        )
    ml_model = build_ml_model(ml_name, ml_cfg_deploy)
    ml_model.fit(X_train, y_train)
    joblib.dump(ml_model, deploy_dir / f"{ml_name}.joblib")
    ml_probs = ml_predict_proba(ml_model, X_val)

    # --- DL ---
    dl_dir = Path(paths["models_dir"]) / "deep" / dl_name / "compare"
    ckpt_src = dl_dir / "best_model.pt"
    if not ckpt_src.exists():
        raise FileNotFoundError(
            f"未找到 {dl_name.upper()} 权重: {ckpt_src}，请先完成训练。"
        )
    weight_file = f"{dl_name}.pt"
    shutil.copy2(ckpt_src, deploy_dir / weight_file)
    print(f"[2/4] 复制 {dl_name.upper()} 权重: {ckpt_src.name}")

    model = build_model(dl_name, n_cls, exp["handcrafted_dim"], n_mels=exp["n_mels"])
    ckpt = torch.load(deploy_dir / weight_file, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    val_ds = CachedBirdDataset(val_cache, device="cpu", preload=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    dl_probs, _, labels = predict_proba(model, val_loader, device, data_on_gpu=False)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Stacking ---
    print("[3/4] 拟合并导出 Stacking 元学习器...")
    stacking = StackingEnsemble()
    stacking.fit(dl_probs, ml_probs, labels)
    stacking.save(deploy_dir / "stacking.joblib")
    stack_probs = stacking.predict_proba(dl_probs, ml_probs)
    stack_metrics = compute_metrics(labels, stack_probs.argmax(1), stack_probs)

    # --- Manifest ---
    print("[4/4] 写入 manifest.json ...")
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

    config_rel = Path("bird_recognition/configs") / Path(cfg.get("_config_path", "compare.yaml")).name
    manifest = {
        "model_type": "stacking",
        "dl_component": dl_name,
        "ml_component": ml_name,
        "weight_file": weight_file,
        "ml_file": f"{ml_name}.joblib",
        "metrics": {
            "accuracy": float(stack_metrics["accuracy"]),
            "macro_f1": float(stack_metrics["macro_f1"]),
            "top_5_acc": float(stack_metrics.get("top_5_acc", 0)),
        },
        "n_classes": n_cls,
        "sample_rate": data_cfg["sample_rate"],
        "duration": data_cfg["duration"],
        "handcrafted_dim": exp["handcrafted_dim"],
        "n_mels": exp["n_mels"],
        "feature_config": cfg["features"],
        "species": species,
        "config": str(config_rel),
    }
    with open(deploy_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"  部署包已保存: {deploy_dir}")
    print(f"  组件: {dl_name.upper()} + {ml_name.upper()}  |  {n_cls} 类")
    print(f"  Stacking 验证 Macro-F1: {stack_metrics['macro_f1']*100:.1f}%")
    print(f"  准确率: {stack_metrics['accuracy']*100:.1f}%")
    print(f"  Top-5: {stack_metrics.get('top_5_acc', 0)*100:.1f}%")
    print("\n启动 GUI:")
    print(f"  python bird_recognition\\scripts\\run_gui.py --bundle {deploy_dir.name}")
    print("=" * 60)
    return deploy_dir


def main():
    args = parse_args()
    project_root = ROOT.parent
    cfg = resolve_paths(load_config(args.config), project_root)
    cfg["_config_path"] = Path(args.config).name

    results_path = (
        Path(args.results)
        if args.results
        else Path(cfg["paths"]["output_dir"]) / "comparison_results.json"
    )
    dl_name = args.dl
    ml_name = args.ml
    if dl_name is None or ml_name is None:
        auto_dl, auto_ml = load_components_from_results(results_path)
        dl_name = dl_name or auto_dl
        ml_name = ml_name or auto_ml
    print(f"  使用组件: DL={dl_name} | ML={ml_name}")

    export_bundle(cfg, project_root, dl_name, ml_name, args.bundle_name)


if __name__ == "__main__":
    main()
