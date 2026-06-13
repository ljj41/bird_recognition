"""Stacking 集成推理模块。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from ..features.extractor import AudioFeatureExtractor, FeatureConfig
from ..models.architectures import build_model
from ..training.ensemble import StackingEnsemble
from ..utils.audio import load_audio, normalize_audio, pad_or_trim


@dataclass
class SpeciesInfo:
    id: int
    code: str
    common_name: str
    scientific_name: str


@dataclass
class PredictionItem:
    rank: int
    species: SpeciesInfo
    probability: float


@dataclass
class PredictionResult:
    items: list[PredictionItem]
    waveform: np.ndarray
    sample_rate: int
    mel: np.ndarray
    dl_probs: np.ndarray
    ml_probs: np.ndarray
    final_probs: np.ndarray


class StackingBirdPredictor:
    """加载 CNN + SVM + Stacking 部署包，对单条音频推理。"""

    def __init__(self, bundle_dir: str | Path, device: str | None = None):
        self.bundle_dir = Path(bundle_dir)
        manifest_path = self.bundle_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"未找到部署包 {manifest_path}，请先运行:\n"
                "  python bird_recognition/scripts/export_stacking_bundle.py"
            )
        with open(manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        feat_cfg = FeatureConfig(**self.manifest["feature_config"])
        self.extractor = AudioFeatureExtractor(feat_cfg)
        self.sample_rate = int(self.manifest["sample_rate"])
        self.duration = float(self.manifest["duration"])
        self.target_len = int(self.sample_rate * self.duration)
        self.n_mels = int(self.manifest["n_mels"])
        self.handcrafted_dim = int(self.manifest["handcrafted_dim"])
        self.n_classes = int(self.manifest["n_classes"])

        self.species = [
            SpeciesInfo(
                id=s["id"],
                code=s["code"],
                common_name=s.get("common_name") or s["code"],
                scientific_name=s.get("scientific_name") or "",
            )
            for s in self.manifest["species"]
        ]

        ml_file = self.manifest.get("ml_file", "svm.joblib")
        self.ml_model = joblib.load(self.bundle_dir / ml_file)
        self.stacking = StackingEnsemble.load(self.bundle_dir / "stacking.joblib")
        self._load_dl()

    def _load_dl(self) -> None:
        dl_name = self.manifest.get("dl_component", "cnn")
        weight_file = self.manifest.get("weight_file", f"{dl_name}.pt")
        if not (self.bundle_dir / weight_file).exists() and (self.bundle_dir / "cnn.pt").exists():
            weight_file = "cnn.pt"
        ckpt_path = self.bundle_dir / weight_file
        self.dl_model = build_model(
            dl_name, self.n_classes, self.handcrafted_dim, n_mels=self.n_mels
        )
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        self.dl_model.load_state_dict(state)
        self.dl_model.to(self.device)
        self.dl_model.eval()

    @classmethod
    def from_project(
        cls,
        project_root: str | Path,
        bundle_name: str = "stacking_top37",
        device: str | None = None,
    ) -> "StackingBirdPredictor":
        root = Path(project_root).resolve()
        bundle = root / "outputs" / "models" / "deploy" / bundle_name
        return cls(bundle, device=device)

    def _extract(self, audio_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y, sr = load_audio(str(audio_path), sr=self.sample_rate, duration=self.duration)
        y = normalize_audio(pad_or_trim(y, self.target_len))
        mel = self.extractor.extract_mel_image(y, sr)
        hand = self.extractor.extract_waveform_features(y, sr)
        return y, mel, hand

    @torch.no_grad()
    def predict(
        self,
        audio_path: str | Path,
        top_k: int = 5,
    ) -> PredictionResult:
        waveform, mel, hand = self._extract(audio_path)

        mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(self.device)
        hand_t = torch.from_numpy(hand).unsqueeze(0).to(self.device)
        logits = self.dl_model(mel_t, hand_t)
        dl_probs = F.softmax(logits, dim=1).cpu().numpy()

        try:
            ml_probs = self.ml_model.predict_proba(hand.reshape(1, -1))
        except AttributeError as e:
            raise RuntimeError(
                "SVM 模型不支持 predict_proba（训练时 probability=False）。\n"
                "请重新导出部署包:\n"
                "  python bird_recognition\\scripts\\export_stacking_bundle.py"
            ) from e
        final_probs = self.stacking.predict_proba(dl_probs, ml_probs)[0]

        k = min(top_k, self.n_classes)
        top_idx = np.argsort(final_probs)[::-1][:k]
        items = [
            PredictionItem(
                rank=i + 1,
                species=self.species[int(idx)],
                probability=float(final_probs[int(idx)]),
            )
            for i, idx in enumerate(top_idx)
        ]
        return PredictionResult(
            items=items,
            waveform=waveform,
            sample_rate=self.sample_rate,
            mel=mel,
            dl_probs=dl_probs[0],
            ml_probs=ml_probs[0],
            final_probs=final_probs,
        )

    @property
    def metrics(self) -> dict:
        return self.manifest.get("metrics", {})

    @property
    def model_label(self) -> str:
        dl = self.manifest.get("dl_component", "cnn").upper()
        ml = self.manifest.get("ml_component", "svm").upper()
        acc = self.metrics.get("accuracy", 0) * 100
        return f"Stacking ({dl} + {ml}) · {acc:.1f}%"
