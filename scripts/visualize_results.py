#!/usr/bin/env python
"""实验结果可视化: 模型对比、训练曲线、混淆矩阵、频谱样例。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_paths
from src.utils.experiment import get_best_result, load_experiment_data

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODEL_NAME_ZH = {
    "knn": "K近邻(KNN)", "svm": "支持向量机(SVM)", "rf": "随机森林(RF)",
    "cnn": "卷积神经网络(CNN)", "crnn": "CNN+LSTM",
    "transformer": "Transformer", "mamba": "Mamba",
    "hybrid": "融合模型(Hybrid)", "stitch": "模块缝合(Stitch)",
    "stacking_ensemble": "Stacking集成",
}


def zh_name(name: str) -> str:
    return MODEL_NAME_ZH.get(name, name)


def parse_args():
    p = argparse.ArgumentParser(description="可视化鸟类识别实验结果")
    p.add_argument("--config", default=str(ROOT / "configs" / "compare.yaml"))
    p.add_argument("--results", default=None)
    return p.parse_args()


def plot_model_comparison(results: list[dict], out_dir: Path) -> dict:
    holdout = [r for r in results if r.get("phase") == "holdout"]
    if not holdout:
        return {}

    names = [zh_name(r["model"]) for r in holdout]
    acc = [r.get("accuracy", 0) * 100 for r in holdout]
    f1 = [r.get("macro_f1", 0) * 100 for r in holdout]
    top5 = [r.get("top_5_acc", 0) * 100 for r in holdout]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, acc, width, label="准确率(%)", color="#4C72B0")
    ax.bar(x, f1, width, label="Macro-F1(%)", color="#55A868")
    ax.bar(x + width, top5, width, label="Top-5准确率(%)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("百分比 (%)")
    ax.set_title("鸟类识别模型性能对比")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close(fig)
    return get_best_result(holdout, "macro_f1")


def plot_cv_comparison(results: list[dict], out_dir: Path) -> None:
    cv_results = [r for r in results if "cv_macro_f1_mean" in r]
    if not cv_results:
        return
    names = [zh_name(r["model"]) for r in cv_results]
    means = [r["cv_macro_f1_mean"] * 100 for r in cv_results]
    stds = [r["cv_macro_f1_std"] * 100 for r in cv_results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, means, yerr=stds, capsize=5, color="#8172B2", alpha=0.85)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("传统机器学习交叉验证结果")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "ml_cv_comparison.png", dpi=150)
    plt.close(fig)


def plot_training_curves(cfg: dict, best_dl: dict, out_dir: Path) -> None:
    if not best_dl or best_dl.get("type") != "dl":
        return
    model_name = best_dl["model"]
    hist_path = (
        Path(cfg["paths"]["models_dir"]) / "deep" / model_name / "compare" / "history.json"
    )
    if not hist_path.exists():
        return
    with open(hist_path, encoding="utf-8") as f:
        history = json.load(f)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]
    train_acc = [h["train"]["accuracy"] * 100 for h in history]
    val_acc = [h["val"]["accuracy"] * 100 for h in history]
    val_f1 = [h["val"]["macro_f1"] * 100 for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(epochs, train_loss, "o-", label="训练损失")
    axes[0].plot(epochs, val_loss, "s-", label="验证损失")
    axes[0].set_xlabel("训练轮次")
    axes[0].set_ylabel("损失值")
    axes[0].set_title(f"{zh_name(model_name)} — 损失曲线")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_acc, "o-", label="训练准确率")
    axes[1].plot(epochs, val_acc, "s-", label="验证准确率")
    axes[1].set_xlabel("训练轮次")
    axes[1].set_ylabel("准确率 (%)")
    axes[1].set_title("准确率变化")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, val_f1, "D-", color="#C44E52", label="验证 Macro-F1")
    axes[2].set_xlabel("训练轮次")
    axes[2].set_ylabel("Macro-F1 (%)")
    axes[2].set_title("F1分数变化")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"training_curves_{model_name}.png", dpi=150)
    plt.close(fig)


def plot_confusion_subset(
    y_true: np.ndarray, y_pred: np.ndarray, id2label: dict, out_dir: Path, top_n: int = 12
) -> None:
    from sklearn.metrics import confusion_matrix
    from collections import Counter
    freq = Counter(y_true.tolist())
    labels_present = [l for l, _ in freq.most_common(top_n)]
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    names = [id2label.get(int(i), str(i)) for i in labels_present]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵 (主要鸟种, 行归一化)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_mel_samples(cfg: dict, exp: dict, out_dir: Path, n: int = 6) -> None:
    import librosa.display
    from src.utils.audio import load_audio, normalize_audio, pad_or_trim
    from src.features.extractor import AudioFeatureExtractor

    sample_df = exp["val_df"].sample(n=min(n, len(exp["val_df"])), random_state=42)
    extractor = AudioFeatureExtractor(exp["feat_cfg"])
    sr = cfg["data"]["sample_rate"]
    duration = cfg["data"]["duration"]
    target_len = int(sr * duration)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, (_, row) in enumerate(sample_df.iterrows()):
        if i >= len(axes):
            break
        y, _ = load_audio(row["audio_path"], sr=sr, duration=duration)
        y = normalize_audio(pad_or_trim(y, target_len))
        mel = extractor.extract_mel_image(y, sr)
        librosa.display.specshow(mel, sr=sr, hop_length=cfg["features"]["hop_length"],
                                 x_axis="time", y_axis="mel", ax=axes[i], cmap="magma")
        axes[i].set_title(f"{row['primary_label']}", fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("验证集鸟鸣 Mel 频谱样例", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "mel_spectrogram_samples.png", dpi=150)
    plt.close(fig)


def plot_best_summary(best: dict, out_dir: Path) -> None:
    if not best:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = {
        "准确率": best.get("accuracy", 0) * 100,
        "Macro-F1": best.get("macro_f1", 0) * 100,
        "加权F1": best.get("weighted_f1", 0) * 100,
        "Top-3准确率": best.get("top_3_acc", 0) * 100,
        "Top-5准确率": best.get("top_5_acc", 0) * 100,
    }
    keys = list(metrics.keys())
    vals = list(metrics.values())
    colors = sns.color_palette("viridis", len(keys))
    bars = ax.barh(keys, vals, color=colors)
    ax.set_xlim(0, 105)
    ax.set_xlabel("百分比 (%)")
    ax.set_title(f"最优模型: {zh_name(best.get('model', 'N/A'))}")
    for bar, v in zip(bars, vals):
        ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center")
    plt.tight_layout()
    fig.savefig(out_dir / "best_model_summary.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    project_root = ROOT.parent
    cfg = resolve_paths(load_config(args.config), project_root)
    fig_dir = Path(cfg["paths"].get("figures_dir", Path(cfg["paths"]["output_dir"]) / "figures"))
    fig_dir.mkdir(parents=True, exist_ok=True)

    results_path = Path(args.results) if args.results else Path(cfg["paths"]["output_dir"]) / "comparison_results.json"
    if not results_path.exists():
        print(f"未找到结果文件: {results_path}")
        return

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    print("[可视化] 正在生成对比图表...")
    best = plot_model_comparison(results, fig_dir)
    plot_cv_comparison(results, fig_dir)
    plot_best_summary(best, fig_dir)

    if best and best.get("type") == "dl":
        plot_training_curves(cfg, best, fig_dir)

    exp = load_experiment_data(cfg)
    plot_mel_samples(cfg, exp, fig_dir)

    dl_holdout = [r for r in results if r.get("type") == "dl" and r.get("phase") == "holdout"]
    best_dl = get_best_result(dl_holdout, "macro_f1")
    if best_dl and "checkpoint" in best_dl:
        try:
            import torch
            from torch.utils.data import DataLoader
            from src.data.cache_dataset import CachedBirdDataset
            from src.models.architectures import build_model
            from src.training.trainer import predict_proba
            from src.config import get_device

            device = get_device(cfg)
            n_cls = len(exp["label2id"])
            cache_dir = Path(cfg["paths"].get("cache_dir", "outputs/cache"))
            val_cache = cache_dir / f"val_top{n_cls}.npz"

            def _checkpoint_num_classes(state: dict) -> int | None:
                best_idx = -1
                n_out = None
                for key, tensor in state.items():
                    parts = key.split(".")
                    if len(parts) >= 3 and parts[0] == "head" and parts[-1] == "weight":
                        try:
                            layer_idx = int(parts[1])
                        except ValueError:
                            continue
                        if layer_idx > best_idx:
                            best_idx = layer_idx
                            n_out = int(tensor.shape[0])
                return n_out

            def _pick_dl_for_confusion() -> dict | None:
                matching = []
                for row in dl_holdout:
                    ckpt_path = row.get("checkpoint")
                    if not ckpt_path or not Path(ckpt_path).exists():
                        continue
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    ckpt_n = _checkpoint_num_classes(ckpt["model_state"])
                    if ckpt_n == n_cls:
                        matching.append(row)
                return get_best_result(matching, "macro_f1") if matching else None

            dl_for_cm = _pick_dl_for_confusion()
            if dl_for_cm is None:
                ckpt_n = None
                if Path(best_dl["checkpoint"]).exists():
                    ckpt = torch.load(best_dl["checkpoint"], map_location="cpu", weights_only=False)
                    ckpt_n = _checkpoint_num_classes(ckpt["model_state"])
                print(
                    f"混淆矩阵生成跳过: 无与当前 {n_cls} 类匹配的 checkpoint"
                    + (f"（最优 DL 为 {ckpt_n} 类）" if ckpt_n is not None else "")
                )
            else:
                model = build_model(
                    dl_for_cm["model"], n_cls, exp["handcrafted_dim"], n_mels=exp["n_mels"]
                )
                ckpt = torch.load(dl_for_cm["checkpoint"], map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                val_ds = CachedBirdDataset(val_cache)
                val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
                _, preds, labels = predict_proba(model, val_loader, device)
                plot_confusion_subset(labels, preds, exp["id2label"], fig_dir)
        except Exception as e:
            print(f"混淆矩阵生成跳过: {e}")

    summary = {
        "best_model": best,
        "figures": [str(p.name) for p in sorted(fig_dir.glob("*.png"))],
    }
    with open(fig_dir / "visualization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"图表已保存至: {fig_dir}")
    for p in sorted(fig_dir.glob("*.png")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
