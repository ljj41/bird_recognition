"""Traditional ML training pipeline."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from ..features.extractor import AudioFeatureExtractor, FeatureConfig
from ..utils.audio import load_audio, normalize_audio, pad_or_trim


def extract_features_from_df(
    df,
    sr: int = 32000,
    duration: float = 5.0,
    feature_cfg: FeatureConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    extractor = AudioFeatureExtractor(feature_cfg)
    target_len = int(sr * duration)
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="提取声学特征"):
        try:
            y_audio, _ = load_audio(row["audio_path"], sr=sr, duration=duration)
            y_audio = normalize_audio(pad_or_trim(y_audio, target_len))
            feat = extractor.extract_waveform_features(y_audio, sr)
            X.append(feat)
            y.append(row["label_id"])
        except Exception as e:
            print(f"跳过 {row['audio_path']}: {e}")
    return np.stack(X), np.array(y)


def _safe_n_jobs() -> int:
    """Windows 多进程易导致 KNN/RF 预测卡死，统一用单进程。"""
    import sys
    return 1 if sys.platform == "win32" else -1


def build_ml_model(name: str, cfg: dict) -> Pipeline:
    name = name.lower()
    n_jobs = _safe_n_jobs()
    if name == "knn":
        clf = KNeighborsClassifier(
            n_neighbors=cfg.get("knn_k", 5),
            weights="distance",
            algorithm="brute",
            n_jobs=1,
        )
    elif name == "svm":
        # probability=True 会在 fit 时做额外交叉验证，3469 样本下极慢
        use_prob = cfg.get("svm_probability", False)
        clf = SVC(
            C=cfg.get("svm_c", 10.0),
            kernel="rbf",
            gamma="scale",
            probability=use_prob,
            class_weight="balanced",
        )
    elif name == "rf":
        clf = RandomForestClassifier(
            n_estimators=cfg.get("rf_n_estimators", 300),
            max_depth=None,
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown ML model: {name}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def cross_validate_ml(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    ml_cfg: dict,
    n_folds: int = 5,
) -> dict:
    import sys
    # Windows 下 n_jobs=-1 容易导致交叉验证卡住无输出，强制单进程
    n_jobs = 1 if sys.platform == "win32" else -1
    pipe = build_ml_model(model_name, ml_cfg)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    print(f"  正在进行 {n_folds} 折交叉验证（{model_name}，请耐心等待）...", flush=True)
    acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=n_jobs)
    print(f"  交叉验证准确率完成，正在计算 F1...", flush=True)
    f1 = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro", n_jobs=n_jobs)
    return {
        "model": model_name,
        "cv_accuracy_mean": float(acc.mean()),
        "cv_accuracy_std": float(acc.std()),
        "cv_macro_f1_mean": float(f1.mean()),
        "cv_macro_f1_std": float(f1.std()),
    }


def ml_predict_proba(pipe: Pipeline, X: np.ndarray, chunk_size: int = 128) -> np.ndarray:
    """统一获取概率；大验证集分块预测并打印进度，避免 Windows 上看似卡死。"""
    if len(X) <= chunk_size:
        return _ml_predict_proba_batch(pipe, X)

    chunks: list[np.ndarray] = []
    for start in range(0, len(X), chunk_size):
        end = min(start + chunk_size, len(X))
        print(f"    预测进度 {end}/{len(X)}...", flush=True)
        chunks.append(_ml_predict_proba_batch(pipe, X[start:end]))
    return np.concatenate(chunks)


def _ml_predict_proba_batch(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    clf = pipe.named_steps["clf"]
    X_scaled = pipe.named_steps["scaler"].transform(X)
    try:
        return pipe.predict_proba(X)
    except AttributeError:
        pass
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_scaled)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
    preds = pipe.predict(X)
    n_cls = len(clf.classes_)
    probs = np.zeros((len(X), n_cls), dtype=np.float32)
    probs[np.arange(len(X)), preds] = 1.0
    return probs


def train_ml_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    ml_cfg: dict,
    save_path: str | Path,
) -> dict:
    pipe = build_ml_model(model_name, ml_cfg)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    y_prob = ml_predict_proba(pipe, X_val)
    metrics = {
        "model": model_name,
        "val_accuracy": float(accuracy_score(y_val, y_pred)),
        "val_macro_f1": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
        "val_weighted_f1": float(f1_score(y_val, y_pred, average="weighted", zero_division=0)),
    }
    joblib.dump(pipe, save_path)
    return metrics
