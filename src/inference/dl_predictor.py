"""单深度学习模型推理（如 Stitch / CNN），供 GUI 部署包使用。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..features.extractor import AudioFeatureExtractor, FeatureConfig
from ..models.architectures import build_model
from ..utils.audio import load_audio, normalize_audio, pad_or_trim
from .stacking_predictor import PredictionItem, PredictionResult, SpeciesInfo


class DLBirdPredictor:
    """加载 manifest + 单个 DL 权重，对单条音频推理。"""

    def __init__(self, bundle_dir: str | Path, device: str | None = None):
        self.bundle_dir = Path(bundle_dir)
        manifest_path = self.bundle_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"未找到部署包 {manifest_path}，请先运行导出脚本，例如:\n"
                "  python bird_recognition\\scripts\\export_stitch_bundle.py"
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
        self.model_name = self.manifest.get("model_name", "cnn")

        self.species = [
            SpeciesInfo(
                id=s["id"],
                code=s["code"],
                common_name=s.get("common_name") or s["code"],
                scientific_name=s.get("scientific_name") or "",
            )
            for s in self.manifest["species"]
        ]

        self._load_model()

    def _load_model(self) -> None:
        weight_name = self.manifest.get("weight_file", f"{self.model_name}.pt")
        ckpt_path = self.bundle_dir / weight_name
        self.model = build_model(
            self.model_name, self.n_classes, self.handcrafted_dim, n_mels=self.n_mels
        )
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_project(
        cls,
        project_root: str | Path,
        bundle_name: str = "stitch_top37",
        device: str | None = None,
    ) -> "DLBirdPredictor":
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
    def predict(self, audio_path: str | Path, top_k: int = 5) -> PredictionResult:
        waveform, mel, hand = self._extract(audio_path)

        mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(self.device)
        hand_t = torch.from_numpy(hand).unsqueeze(0).to(self.device)
        logits = self.model(mel_t, hand_t)
        final_probs = F.softmax(logits, dim=1).cpu().numpy()[0]

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
        zeros = np.zeros(self.n_classes, dtype=np.float32)
        return PredictionResult(
            items=items,
            waveform=waveform,
            sample_rate=self.sample_rate,
            mel=mel,
            dl_probs=final_probs,
            ml_probs=zeros,
            final_probs=final_probs,
        )

    @property
    def metrics(self) -> dict:
        return self.manifest.get("metrics", {})

    @property
    def model_label(self) -> str:
        acc = self.metrics.get("accuracy", 0) * 100
        return f"{self.model_name.upper()} · {acc:.1f}% · {self.n_classes} 类"
