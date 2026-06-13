"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    top_k_accuracy_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    labels: list[int] | None = None,
    top_k: tuple[int, ...] = (1, 3, 5),
) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_prob is not None and len(top_k) > 0:
        max_k = min(max(top_k), y_prob.shape[1])
        for k in top_k:
            kk = min(k, max_k)
            metrics[f"top_{k}_acc"] = float(
                top_k_accuracy_score(y_true, y_prob, k=kk, labels=labels)
            )
    return metrics


def format_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str] | None = None
) -> str:
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
