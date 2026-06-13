#!/usr/bin/env python
"""训练深度学习模型 (CNN / CRNN / Transformer / Mamba)。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_device, load_config, resolve_paths
from src.data.dataset import (
    BirdAudioDataset,
    build_label_mapping,
    get_stratified_folds,
    prepare_bird_dataset,
    split_dataset,
)
from src.features.extractor import FeatureConfig
from src.models.architectures import build_model
from src.training.trainer import DeepTrainer, predict_proba
from src.training.metrics import compute_metrics, format_classification_report


def parse_args():
    p = argparse.ArgumentParser(description="Train deep learning bird classifier")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--model", default="cnn", choices=["cnn", "crnn", "lstm", "transformer", "mamba", "hybrid", "stitch"])
    p.add_argument("--fold", type=int, default=-1, help="-1: hold-out split; 0..k-1: CV fold")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="快速调试: 少量样本+少 epoch")
    return p.parse_args()


def main():
    args = parse_args()
    project_root = ROOT.parent
    cfg = resolve_paths(load_config(args.config), project_root)

    if args.quick:
        cfg["deep_learning"]["epochs"] = 3
        cfg["deep_learning"]["max_train_samples"] = 500
        cfg["deep_learning"]["batch_size"] = 16
        cfg["data"]["min_samples_per_class"] = 3

    paths = cfg["paths"]
    data_cfg = cfg["data"]
    feat_cfg = FeatureConfig(**cfg["features"])
    dl_cfg = cfg["deep_learning"]
    aug_cfg = cfg["augmentation"]

    out_dir = Path(paths["models_dir"]) / "deep" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

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

    with open(out_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)

    print(f"Dataset: {len(df)} samples, {len(label2id)} bird classes")

    # 推断手工特征维度
    sample_ds = BirdAudioDataset(df.head(1), label2id, feature_cfg=feat_cfg)
    sample = sample_ds[0]
    handcrafted_dim = sample["handcrafted"].shape[0]
    n_mels = sample["mel"].shape[1]

    device = get_device(cfg)
    print(f"Device: {device}")

    if args.fold >= 0:
        folds = get_stratified_folds(df, data_cfg["n_folds"], cfg["project"]["seed"])
        train_idx, val_idx = folds[args.fold]
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        fold_tag = f"fold{args.fold}"
    else:
        train_df, val_df, test_df = split_dataset(
            df, data_cfg["val_ratio"], data_cfg["test_ratio"], cfg["project"]["seed"]
        )
        fold_tag = "holdout"

    max_samples = dl_cfg.get("max_train_samples")
    if max_samples:
        train_df = train_df.head(max_samples)

    train_ds = BirdAudioDataset(
        train_df, label2id,
        sr=data_cfg["sample_rate"],
        duration=data_cfg["duration"],
        feature_cfg=feat_cfg,
        augment=True,
        aug_cfg=aug_cfg,
    )
    val_ds = BirdAudioDataset(
        val_df, label2id,
        sr=data_cfg["sample_rate"],
        duration=data_cfg["duration"],
        feature_cfg=feat_cfg,
        augment=False,
    )

    batch_size = args.batch_size or dl_cfg["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=dl_cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=dl_cfg["num_workers"], pin_memory=True,
    )

    model = build_model(args.model, len(label2id), handcrafted_dim, n_mels=n_mels)
    print(f"Model {args.model}: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    # 类别权重 (处理长尾分布)
    num_classes = len(label2id)
    counts = train_df["label_id"].value_counts()
    weights = np.ones(num_classes, dtype=np.float32)
    for lid, cnt in counts.items():
        weights[int(lid)] = len(train_df) / (num_classes * cnt)
    class_weights = torch.tensor(weights)

    trainer = DeepTrainer(
        model, device,
        lr=dl_cfg["lr"],
        weight_decay=dl_cfg["weight_decay"],
        focal_gamma=dl_cfg["focal_gamma"],
        label_smoothing=dl_cfg["label_smoothing"],
        class_weights=class_weights,
        aug_cfg=aug_cfg,
    )

    epochs = args.epochs or dl_cfg["epochs"]
    result = trainer.fit(
        train_loader, val_loader, epochs,
        save_dir=out_dir / fold_tag,
        early_stop_patience=dl_cfg["early_stop_patience"],
    )
    print(f"Training done. Best macro-F1: {result['best_f1']:.4f}")

    # 加载最佳模型评估
    ckpt = torch.load(result["best_model"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    probs, preds, labels = predict_proba(model, val_loader, device)
    report = format_classification_report(
        labels, preds, target_names=[id2label[i] for i in range(len(id2label))]
    )
    with open(out_dir / fold_tag / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report[:2000])


if __name__ == "__main__":
    main()
