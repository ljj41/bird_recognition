"""实验通用工具: 数据划分、子采样、结果记录。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.dataset import (
    BirdAudioDataset,
    build_label_mapping,
    prepare_bird_dataset,
    split_dataset,
)
from ..features.extractor import FeatureConfig


def filter_top_species(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """保留样本数最多的 top_n 个鸟种，保证每类有足够训练数据。"""
    counts = df["primary_label"].value_counts()
    top_labels = counts.head(top_n).index.tolist()
    out = df[df["primary_label"].isin(top_labels)].copy().reset_index(drop=True)
    return out


def stratified_subsample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if n >= len(df):
        return df.reset_index(drop=True)
    parts = []
    rng = np.random.RandomState(seed)
    groups = df.groupby("primary_label")
    per_class = max(1, n // len(groups))
    for _, g in groups:
        k = min(per_class, len(g))
        parts.append(g.sample(n=k, random_state=rng))
    out = pd.concat(parts).reset_index(drop=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    return out


def print_data_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, n_classes: int) -> None:
    """打印数据集概况（中文）。"""
    train_per_class = train_df.groupby("primary_label").size()
    print(f"  训练集: {len(train_df)} 条 | 验证集: {len(val_df)} 条 | 类别数: {n_classes}")
    print(f"  训练集每类样本: 最少 {train_per_class.min()} | 最多 {train_per_class.max()} | 均值 {train_per_class.mean():.1f}")
    random_loss = math.log(n_classes)
    print(f"  随机猜测基线: 准确率 ≈ {100/n_classes:.1f}% | 交叉熵损失 ≈ {random_loss:.2f}")
    if train_per_class.min() < 10:
        print("  ⚠ 警告: 部分类别训练样本不足10条，模型难以收敛！")


def load_experiment_data(cfg: dict) -> dict:
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    feat_cfg = FeatureConfig(**cfg["features"])

    df = prepare_bird_dataset(
        paths["data_root"],
        paths["train_csv"],
        paths["taxonomy_csv"],
        paths["audio_dir"],
        data_cfg["target_class"],
        data_cfg["min_samples_per_class"],
    )

    top_n = data_cfg.get("top_n_species")
    if top_n:
        before = df["primary_label"].nunique()
        df = filter_top_species(df, top_n)
        print(f"  筛选鸟种: 从 {before} 类 → Top-{top_n} 类 ({len(df)} 条样本)")

    label2id, id2label = build_label_mapping(df["primary_label"].tolist())
    df["label_id"] = df["primary_label"].map(label2id)

    train_df, val_df, test_df = split_dataset(
        df, data_cfg["val_ratio"], data_cfg["test_ratio"], cfg["project"]["seed"]
    )

    max_train = data_cfg.get("max_train_samples")
    max_val = data_cfg.get("max_val_samples")
    if max_train:
        train_df = stratified_subsample(train_df, max_train, cfg["project"]["seed"])
    if max_val:
        val_df = stratified_subsample(val_df, max_val, cfg["project"]["seed"] + 1)

    sample_ds = BirdAudioDataset(train_df.head(1), label2id, feature_cfg=feat_cfg)
    sample = sample_ds[0]

    return {
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "label2id": label2id,
        "id2label": id2label,
        "feat_cfg": feat_cfg,
        "handcrafted_dim": int(sample["handcrafted"].shape[0]),
        "n_mels": int(sample["mel"].shape[1]),
    }


def save_result(results: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def get_best_result(results: list[dict], metric: str = "macro_f1") -> dict:
    valid = [r for r in results if metric in r]
    if not valid:
        return {}
    return max(valid, key=lambda x: x[metric])
