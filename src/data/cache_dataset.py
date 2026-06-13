"""预计算特征缓存，避免训练时重复提取导致极慢。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..features.extractor import AudioFeatureExtractor, FeatureConfig
from ..utils.audio import load_audio, normalize_audio, pad_or_trim


class CachedBirdDataset(Dataset):
    """
    从 .npz 缓存加载。
    preload=True 时一次性转为 contiguous Tensor，训练时避免反复 from_numpy + CPU→GPU 拷贝。
    """

    def __init__(
        self,
        cache_path: str | Path,
        device: str = "cpu",
        preload: bool = True,
    ):
        cache_path = Path(cache_path)
        data = np.load(cache_path)
        mels_np = np.ascontiguousarray(data["mels"], dtype=np.float32)
        hand_np = np.ascontiguousarray(data["handcrafted"], dtype=np.float32)
        labels_np = data["labels"].astype(np.int64)

        if preload:
            # mel: (N, 1, n_mels, time)
            self.mels = torch.from_numpy(mels_np).unsqueeze(1)
            self.handcrafted = torch.from_numpy(hand_np)
            self.labels = torch.from_numpy(labels_np)
            self._preloaded = True
            if device == "cuda" and torch.cuda.is_available():
                self.mels = self.mels.to(device, non_blocking=True)
                self.handcrafted = self.handcrafted.to(device, non_blocking=True)
                self.labels = self.labels.to(device, non_blocking=True)
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.mels_np = mels_np
            self.hand_np = hand_np
            self.labels_np = labels_np
            self._preloaded = False
            self.device = "cpu"

    def __len__(self) -> int:
        if self._preloaded:
            return len(self.labels)
        return len(self.labels_np)

    def __getitem__(self, idx: int) -> dict:
        if self._preloaded:
            return {
                "mel": self.mels[idx],
                "handcrafted": self.handcrafted[idx],
                "label": self.labels[idx],
            }
        return {
            "mel": torch.from_numpy(self.mels_np[idx]).unsqueeze(0),
            "handcrafted": torch.from_numpy(self.hand_np[idx]),
            "label": torch.tensor(int(self.labels_np[idx]), dtype=torch.long),
        }


def build_feature_cache(
    df,
    label2id: dict,
    cache_path: str | Path,
    sr: int = 32000,
    duration: float = 5.0,
    feature_cfg: FeatureConfig | None = None,
) -> Path:
    """将 Mel 频谱 + 手工特征预计算并保存。"""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    extractor = AudioFeatureExtractor(feature_cfg)
    target_len = int(sr * duration)

    mels, handcrafted, labels = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="预计算特征"):
        y, _ = load_audio(row["audio_path"], sr=sr, duration=duration)
        y = normalize_audio(pad_or_trim(y, target_len))
        mels.append(extractor.extract_mel_image(y, sr))
        handcrafted.append(extractor.extract_waveform_features(y, sr))
        labels.append(label2id[row["primary_label"]])

    np.savez_compressed(
        cache_path,
        mels=np.stack(mels).astype(np.float32),
        handcrafted=np.stack(handcrafted).astype(np.float32),
        labels=np.array(labels, dtype=np.int64),
    )
    return cache_path
