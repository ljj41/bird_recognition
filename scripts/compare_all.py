#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模型对比实验主脚本。
对比: KNN / SVM / 随机森林 / CNN / CRNN / Transformer / Mamba / 融合模型 / Stacking集成
"""

from __future__ import annotations

import setup_console  # noqa: F401  Windows中文输出
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_device, load_config, resolve_paths
from src.data.cache_dataset import CachedBirdDataset, build_feature_cache
from src.data.dataset import BirdAudioDataset
from src.models.architectures import build_model
from src.training.ensemble import StackingEnsemble
from src.training.metrics import compute_metrics
from src.training.ml_trainer import (
    build_ml_model, cross_validate_ml, extract_features_from_df, ml_predict_proba,
)
from src.training.trainer import DeepTrainer, predict_proba
from src.utils.experiment import get_best_result, load_experiment_data, print_data_summary, save_result


def parse_args():
    p = argparse.ArgumentParser(description="鸟类声音识别 - 多模型对比实验")
    p.add_argument("--config", default=str(ROOT / "configs" / "compare_fast.yaml"),
                   help="配置文件，推荐 compare_fast.yaml（快）或 compare.yaml（全）")
    p.add_argument("--skip-dl", action="store_true", help="跳过深度学习模型")
    p.add_argument("--skip-ml", action="store_true", help="跳过传统机器学习模型")
    p.add_argument("--skip-cv", action="store_true", help="跳过交叉验证，直接训练（更快）")
    p.add_argument("--dl-only", type=str, default=None,
                   help="只训练指定DL模型: cnn/crnn/hybrid/stitch/transformer/mamba")
    p.add_argument("--stacking-only", action="store_true", help="仅运行 Stacking 融合（需 ML/DL 模型已训练）")
    return p.parse_args()


def run_ml_models(exp: dict, cfg: dict, skip_cv: bool = False) -> tuple[list[dict], dict[str, np.ndarray]]:
    data_cfg = cfg["data"]
    ml_cfg = cfg["ml"]
    paths = cfg["paths"]
    feat_dir = Path(paths["features_dir"])
    out_dir = Path(paths["models_dir"]) / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cls = len(exp["label2id"])
    tag = f"top{n_cls}_train{len(exp['train_df'])}"
    cache_train = feat_dir / f"X_train_{tag}.npy"
    cache_val = feat_dir / f"X_val_{tag}.npy"

    if cache_train.exists() and cache_val.exists():
        X_train = np.load(cache_train)
        y_train = np.load(feat_dir / f"y_train_{tag}.npy")
        X_val = np.load(cache_val)
        y_val = np.load(feat_dir / f"y_val_{tag}.npy")
        print(f"[传统ML] 加载缓存特征: 训练 {X_train.shape}, 验证 {X_val.shape}")
    else:
        print("[传统ML] 正在提取声学特征...")
        X_train, y_train = extract_features_from_df(
            exp["train_df"], data_cfg["sample_rate"], data_cfg["duration"], exp["feat_cfg"]
        )
        X_val, y_val = extract_features_from_df(
            exp["val_df"], data_cfg["sample_rate"], data_cfg["duration"], exp["feat_cfg"]
        )
        np.save(cache_train, X_train)
        np.save(feat_dir / f"y_train_{tag}.npy", y_train)
        np.save(cache_val, X_val)
        np.save(feat_dir / f"y_val_{tag}.npy", y_val)
        print(f"[传统ML] 特征已缓存: 训练 {X_train.shape}, 验证 {X_val.shape}", flush=True)

    results = []
    ml_probs: dict[str, np.ndarray] = {}

    for name in ml_cfg["models"]:
        t0 = time.time()
        print(f"\n[传统ML] 训练 {name.upper()} 模型...", flush=True)
        if not skip_cv:
            cv = cross_validate_ml(X_train, y_train, name, ml_cfg, cfg["data"]["n_folds"])
            cv["type"] = "ml"
            cv["time_sec"] = time.time() - t0
            results.append(cv)
            print(f"  交叉验证 Macro-F1: {cv['cv_macro_f1_mean']*100:.1f}% (±{cv['cv_macro_f1_std']*100:.1f}%)", flush=True)
        else:
            print(f"  已跳过交叉验证，直接在训练集上拟合...", flush=True)

        pipe = build_ml_model(name, ml_cfg)
        print(f"  → 正在拟合 {name.upper()}（训练集 {len(X_train)} 条）...", flush=True)
        pipe.fit(X_train, y_train)
        print(f"  → 拟合完成，正在验证集预测（{len(X_val)} 条）...", flush=True)
        probs = ml_predict_proba(pipe, X_val)
        preds = probs.argmax(axis=1)
        print(f"  → 正在保存模型...", flush=True)
        joblib.dump(pipe, out_dir / f"{name}_model.joblib")
        metrics = compute_metrics(y_val, preds, probs)
        metrics.update({
            "model": name, "type": "ml", "phase": "holdout",
            "time_sec": time.time() - t0,
        })
        results.append(metrics)
        ml_probs[name] = probs
        print(
            f"  验证集: 准确率 {metrics['accuracy']*100:.1f}% | "
            f"Macro-F1 {metrics['macro_f1']*100:.1f}% | "
            f"Top-5 {metrics.get('top_5_acc',0)*100:.1f}%",
            flush=True,
        )

    return results, ml_probs


def run_dl_model(
    model_name: str, exp: dict, cfg: dict, device: str
) -> tuple[dict, np.ndarray]:
    data_cfg = cfg["data"]
    dl_cfg = cfg["deep_learning"]
    aug_cfg = cfg["augmentation"]
    paths = cfg["paths"]
    cache_dir = Path(paths.get("cache_dir", Path(paths["output_dir"]) / "cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(paths["models_dir"]) / "deep" / model_name / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cls = len(exp["label2id"])
    train_cache = cache_dir / f"train_top{n_cls}.npz"
    val_cache = cache_dir / f"val_top{n_cls}.npz"

    print(f"  预计算/加载特征缓存...")
    build_feature_cache(
        exp["train_df"], exp["label2id"], train_cache,
        data_cfg["sample_rate"], data_cfg["duration"], exp["feat_cfg"],
    )
    build_feature_cache(
        exp["val_df"], exp["label2id"], val_cache,
        data_cfg["sample_rate"], data_cfg["duration"], exp["feat_cfg"],
    )

    train_ds = CachedBirdDataset(train_cache, device="cpu", preload=True)
    val_ds = CachedBirdDataset(val_cache, device="cpu", preload=True)

    batch_size = dl_cfg.get("model_batch_size", {}).get(model_name, dl_cfg["batch_size"])
    val_batch = dl_cfg.get("model_val_batch_size", {}).get(model_name, batch_size)
    data_on_gpu = dl_cfg.get("data_on_gpu", False) and device == "cuda"
    if data_on_gpu:
        train_ds = CachedBirdDataset(train_cache, device=device, preload=True)
        val_ds = CachedBirdDataset(val_cache, device=device, preload=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda" and not data_on_gpu),
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda" and not data_on_gpu),
    )

    model = build_model(
        model_name, n_cls, exp["handcrafted_dim"], n_mels=exp["n_mels"]
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  模型参数量: {n_params:.2f}M")

    counts = exp["train_df"]["label_id"].value_counts()
    weights = np.ones(n_cls, dtype=np.float32)
    for lid, cnt in counts.items():
        weights[int(lid)] = len(exp["train_df"]) / (n_cls * cnt)

    t0 = time.time()
    model_lr = dl_cfg.get("model_lr", {}).get(model_name, dl_cfg["lr"])
    use_amp = dl_cfg.get("use_amp", True) and model_name not in dl_cfg.get("no_amp_models", [])
    if data_on_gpu:
        print(f"  特征已预加载到 GPU，batch={batch_size}，混合精度: {use_amp}，lr={model_lr}", flush=True)
    else:
        print(f"  低显存模式: 特征在 CPU，batch={batch_size}，混合精度: {use_amp}，lr={model_lr}", flush=True)
    trainer = DeepTrainer(
        model, device,
        lr=model_lr, weight_decay=dl_cfg["weight_decay"],
        focal_gamma=dl_cfg["focal_gamma"],
        label_smoothing=dl_cfg["label_smoothing"],
        class_weights=None,
        aug_cfg=aug_cfg,
        use_amp=use_amp,
        data_on_gpu=data_on_gpu,
    )
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=dl_cfg.get("model_epochs", {}).get(model_name, dl_cfg["epochs"]),
    )

    fit_result = trainer.fit(
        train_loader, val_loader,
        dl_cfg.get("model_epochs", {}).get(model_name, dl_cfg["epochs"]),
        out_dir,
        early_stop_patience=dl_cfg.get("model_early_stop", {}).get(
            model_name, dl_cfg["early_stop_patience"]
        ),
    )

    ckpt = torch.load(fit_result["best_model"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    probs, preds, labels = predict_proba(model, val_loader, device, data_on_gpu=data_on_gpu)
    metrics = compute_metrics(labels, preds, probs)
    metrics.update({
        "model": model_name, "type": "dl", "phase": "holdout",
        "best_f1": fit_result["best_f1"],
        "time_sec": time.time() - t0,
        "checkpoint": fit_result["best_model"],
    })
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(
        f"  验证集: 准确率 {metrics['accuracy']*100:.1f}% | "
        f"Macro-F1 {metrics['macro_f1']*100:.1f}% | Top-5 {metrics.get('top_5_acc',0)*100:.1f}%"
    )
    del train_ds, val_ds, train_loader, val_loader, model, trainer
    if device == "cuda":
        torch.cuda.empty_cache()
    return metrics, probs


def run_stacking(
    best_dl_probs: np.ndarray,
    best_ml_probs: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    print("\n[模型融合] 训练 Stacking 二级集成...")
    t0 = time.time()
    ensemble = StackingEnsemble()
    ensemble.fit(best_dl_probs, best_ml_probs, y_val)
    probs = ensemble.predict_proba(best_dl_probs, best_ml_probs)
    preds = probs.argmax(axis=1)
    metrics = compute_metrics(y_val, preds, probs)
    metrics.update({
        "model": "stacking_ensemble", "type": "ensemble", "phase": "holdout",
        "time_sec": time.time() - t0,
    })
    print(
        f"  验证集: 准确率 {metrics['accuracy']*100:.1f}% | "
        f"Macro-F1 {metrics['macro_f1']*100:.1f}% | Top-5 {metrics.get('top_5_acc',0)*100:.1f}%"
    )
    return metrics


def main():
    args = parse_args()
    project_root = ROOT.parent
    cfg = resolve_paths(load_config(args.config), project_root)
    device = get_device(cfg)
    paths = cfg["paths"]

    print("=" * 60)
    print("  鸟类声音识别系统 — 多模型对比实验")
    print("=" * 60)

    exp = load_experiment_data(cfg)
    n_cls = len(exp["label2id"])
    print("\n[数据集概况]")
    print_data_summary(exp["train_df"], exp["val_df"], n_cls)
    print(f"  计算设备: {device}")

    all_results: list[dict] = []
    dl_probs_map: dict[str, np.ndarray] = {}
    ml_probs_map: dict[str, np.ndarray] = {}
    y_val = exp["val_df"]["label_id"].values

    if not args.skip_ml:
        print("\n" + "-" * 40)
        print("第一阶段: 传统机器学习模型")
        print("-" * 40)
        ml_results, ml_probs_map = run_ml_models(exp, cfg, skip_cv=args.skip_cv)
        all_results.extend(ml_results)

    if not args.skip_dl:
        print("\n" + "-" * 40)
        print("第二阶段: 深度学习模型")
        print("-" * 40)
        dl_models = [args.dl_only] if args.dl_only else cfg["deep_learning"]["models"]
        for model_name in dl_models:
            print(f"\n[深度学习] 训练 {model_name.upper()} 模型...")
            metrics, probs = run_dl_model(model_name, exp, cfg, device)
            all_results.append(metrics)
            dl_probs_map[model_name] = probs

    if dl_probs_map and ml_probs_map:
        print("\n" + "-" * 40)
        print("第三阶段: 模型融合 (Stacking)")
        print("-" * 40)
        best_dl_name = get_best_result(
            [r for r in all_results if r.get("type") == "dl"], "macro_f1"
        ).get("model", list(dl_probs_map.keys())[0])
        best_ml_name = get_best_result(
            [r for r in all_results if r.get("type") == "ml" and r.get("phase") == "holdout"],
            "macro_f1",
        ).get("model", list(ml_probs_map.keys())[0])
        print(f"  最优DL组件: {best_dl_name} | 最优ML组件: {best_ml_name}")
        stack_metrics = run_stacking(
            dl_probs_map[best_dl_name], ml_probs_map[best_ml_name], y_val
        )
        stack_metrics["dl_component"] = best_dl_name
        stack_metrics["ml_component"] = best_ml_name
        all_results.append(stack_metrics)

    out_path = Path(paths["output_dir"]) / "comparison_results.json"
    if out_path.exists() and (args.dl_only or args.skip_ml):
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            keep = [
                r for r in prev
                if not (args.dl_only and r.get("model") == args.dl_only and r.get("type") == "dl")
            ]
            all_results = keep + all_results
        except (json.JSONDecodeError, OSError):
            pass
    save_result(all_results, out_path)
    print(f"\n对比结果已保存: {out_path}")

    best = get_best_result(
        [r for r in all_results if r.get("phase") == "holdout"], "macro_f1"
    )
    print("\n" + "=" * 60)
    print(f"  最优模型: {best.get('model', 'N/A')}")
    print(f"  准确率: {best.get('accuracy', 0)*100:.1f}%")
    print(f"  Macro-F1: {best.get('macro_f1', 0)*100:.1f}%")
    print(f"  Top-5准确率: {best.get('top_5_acc', 0)*100:.1f}%")
    print("=" * 60)

    viz_script = ROOT / "scripts" / "visualize_results.py"
    if viz_script.exists():
        import subprocess
        print("\n[可视化] 正在生成对比图表...")
        subprocess.run([
            sys.executable, str(viz_script),
            "--config", args.config,
            "--results", str(out_path),
        ], check=False)


if __name__ == "__main__":
    main()
