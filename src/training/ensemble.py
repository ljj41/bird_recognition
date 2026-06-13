"""模型融合: 深度学习 + 传统机器学习 Stacking Ensemble。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class StackingEnsemble:
    """
    将 DL 概率输出与 ML 手工特征概率进行二级融合。
    创新点: 用验证集学习最优融合权重 (Logistic Regression meta-learner)。
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.meta = LogisticRegression(
            max_iter=1000, multi_class="multinomial", solver="lbfgs", C=1.0
        )
        self.num_classes: int = 0

    def fit(
        self,
        dl_probs: np.ndarray,
        ml_probs: np.ndarray,
        y_true: np.ndarray,
    ) -> "StackingEnsemble":
        self.num_classes = dl_probs.shape[1]
        meta_X = np.hstack([dl_probs, ml_probs])
        meta_X = self.scaler.fit_transform(meta_X)
        self.meta.fit(meta_X, y_true)
        return self

    def predict_proba(self, dl_probs: np.ndarray, ml_probs: np.ndarray) -> np.ndarray:
        meta_X = np.hstack([dl_probs, ml_probs])
        meta_X = self.scaler.transform(meta_X)
        return self.meta.predict_proba(meta_X)

    def predict(self, dl_probs: np.ndarray, ml_probs: np.ndarray) -> np.ndarray:
        return self.predict_proba(dl_probs, ml_probs).argmax(axis=1)

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "meta": self.meta,
                "num_classes": self.num_classes,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "StackingEnsemble":
        import joblib

        data = joblib.load(path)
        obj = cls()
        obj.scaler = data["scaler"]
        obj.meta = data["meta"]
        obj.num_classes = int(data["num_classes"])
        return obj
