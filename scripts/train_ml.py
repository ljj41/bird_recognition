#!/usr/bin/env python
"""训练传统机器学习模型 (KNN / SVM / Random Forest)。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_paths
from src.data.dataset import build_label_mapping, prepare_bird_dataset, split_dataset
from src.features.extractor import FeatureConfig
from src.training.ml_trainer import (
    cross_validate_ml,
    extract_features_from_df,
    train_ml_model,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train ML bird classifier")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--model", default="all", choices=["all", "knn", "svm", "rf"])
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    project_root = ROOT.parent
    cfg = resolve_paths(load_config(args.config), project_root)

    if args.quick:
        cfg["deep_learning"]["max_train_samples"] = 300
        cfg["data"]["min_samples_per_class"] = 3

    paths = cfg["paths"]
    data_cfg = cfg["data"]
    feat_cfg = FeatureConfig(**cfg["features"])
    ml_cfg = cfg["ml"]

    out_dir = Path(paths["models_dir"]) / "ml"
    feat_dir = Path(paths["features_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_bird_dataset(
        paths["data_root"],
        paths["train_csv"],
        paths["taxonomy_csv"],
        paths["audio_dir"],
        data_cfg["target_class"],
        data_cfg["min_samples_per_class"],
    )
    label2id, id2label = build_label_mapping(df["primary_label"].tolist())
    df["label_id"] = df["primary_label"].map(label2id)

    train_df, val_df, test_df = split_dataset(
        df, data_cfg["val_ratio"], data_cfg["test_ratio"], cfg["project"]["seed"]
    )

    max_n = cfg["deep_learning"].get("max_train_samples")
    if max_n:
        train_df = train_df.head(max_n)
        val_df = val_df.head(max_n // 5)

    cache_train = feat_dir / "X_train.npy"
    cache_val = feat_dir / "X_val.npy"
    if cache_train.exists() and cache_val.exists():
        X_train = np.load(cache_train)
        y_train = np.load(feat_dir / "y_train.npy")
        X_val = np.load(cache_val)
        y_val = np.load(feat_dir / "y_val.npy")
        print("Loaded cached features")
    else:
        print("Extracting train features...")
        X_train, y_train = extract_features_from_df(
            train_df, data_cfg["sample_rate"], data_cfg["duration"], feat_cfg
        )
        print("Extracting val features...")
        X_val, y_val = extract_features_from_df(
            val_df, data_cfg["sample_rate"], data_cfg["duration"], feat_cfg
        )
        np.save(cache_train, X_train)
        np.save(feat_dir / "y_train.npy", y_train)
        np.save(cache_val, X_val)
        np.save(feat_dir / "y_val.npy", y_val)

    print(f"Features: train {X_train.shape}, val {X_val.shape}")

    models = ml_cfg["models"] if args.model == "all" else [args.model]
    all_results = []

    for name in models:
        print(f"\n=== Cross-validation: {name} ===")
        cv_result = cross_validate_ml(X_train, y_train, name, ml_cfg, data_cfg["n_folds"])
        print(json.dumps(cv_result, indent=2))
        all_results.append(cv_result)

        print(f"=== Training {name} on full train set ===")
        metrics = train_ml_model(
            X_train, y_train, X_val, y_val, name, ml_cfg,
            save_path=out_dir / f"{name}_model.joblib",
        )
        print(json.dumps(metrics, indent=2))
        all_results.append({"phase": "holdout", **metrics})

    with open(out_dir / "ml_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    mapping = {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}
    with open(out_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
