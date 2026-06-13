"""Data loading, splitting, and augmentation."""

from __future__ import annotations

import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

from ..features.extractor import AudioFeatureExtractor, FeatureConfig
from ..utils.audio import load_audio, normalize_audio, pad_or_trim


def prepare_bird_dataset(
    data_root: str | Path,
    train_csv: str = "train.csv",
    taxonomy_csv: str = "taxonomy.csv",
    audio_dir: str = "train_audio",
    target_class: str = "Aves",
    min_samples_per_class: int = 5,
) -> pd.DataFrame:
    """加载并过滤鸟类样本，返回含 audio_path 和 label 的 DataFrame。"""
    root = Path(data_root)
    train = pd.read_csv(root / train_csv)
    taxonomy = pd.read_csv(root / taxonomy_csv)

    bird_labels = set(taxonomy.loc[taxonomy["class_name"] == target_class, "primary_label"])
    bird_df = train[train["primary_label"].isin(bird_labels)].copy()

    counts = bird_df["primary_label"].value_counts()
    valid_labels = counts[counts >= min_samples_per_class].index
    bird_df = bird_df[bird_df["primary_label"].isin(valid_labels)].copy()

    bird_df["audio_path"] = bird_df["filename"].apply(
        lambda f: str(root / audio_dir / f)
    )
    bird_df = bird_df[bird_df["audio_path"].apply(lambda p: Path(p).exists())].copy()
    bird_df.reset_index(drop=True, inplace=True)
    return bird_df


def build_label_mapping(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    unique = sorted(set(labels))
    label2id = {lb: i for i, lb in enumerate(unique)}
    id2label = {i: lb for lb, i in label2id.items()}
    return label2id, id2label


class BirdAudioDataset(Dataset):
    """PyTorch Dataset：返回 Mel 频谱 + 手工特征 + 标签。"""

    def __init__(
        self,
        df: pd.DataFrame,
        label2id: dict[str, int],
        sr: int = 32000,
        duration: float = 5.0,
        feature_cfg: FeatureConfig | None = None,
        augment: bool = False,
        aug_cfg: dict | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.sr = sr
        self.duration = duration
        self.target_len = int(sr * duration)
        self.extractor = AudioFeatureExtractor(feature_cfg)
        self.augment = augment
        self.aug_cfg = aug_cfg or {}

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, y: np.ndarray) -> np.ndarray:
        if not self.augment:
            return y
        cfg = self.aug_cfg
        if random.random() < 0.5:
            lo, hi = cfg.get("time_stretch", [0.85, 1.15])
            rate = random.uniform(lo, hi)
            y = librosa.effects.time_stretch(y, rate=rate)
        if random.random() < 0.5:
            lo, hi = cfg.get("pitch_shift", [-2, 2])
            steps = random.uniform(lo, hi)
            y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=steps)
        if random.random() < 0.3:
            snr_lo, snr_hi = cfg.get("noise_snr", [15, 30])
            noise = np.random.randn(len(y)).astype(np.float32)
            snr = random.uniform(snr_lo, snr_hi)
            sig_power = np.mean(y**2) + 1e-8
            noise_power = sig_power / (10 ** (snr / 10))
            noise = noise * np.sqrt(noise_power / (np.mean(noise**2) + 1e-8))
            y = y + noise
        return y.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        y, _ = load_audio(row["audio_path"], sr=self.sr, duration=self.duration)
        y = normalize_audio(y)
        y = self._augment(y)
        y = pad_or_trim(y, self.target_len)

        mel = self.extractor.extract_mel_image(y, self.sr)
        handcrafted = self.extractor.extract_waveform_features(y, self.sr)
        label = self.label2id[row["primary_label"]]

        return {
            "mel": torch.from_numpy(mel).unsqueeze(0),  # (1, n_mels, time)
            "handcrafted": torch.from_numpy(handcrafted),
            "label": torch.tensor(label, dtype=torch.long),
            "path": row["audio_path"],
        }


def spec_augment(mel: torch.Tensor, freq_mask: int = 16, time_mask: int = 32) -> torch.Tensor:
    """SpecAugment on mel tensor (1, n_mels, time)."""
    x = mel.clone()
    _, n_mels, n_time = x.shape
    if freq_mask > 0:
        f = random.randint(0, min(freq_mask, n_mels - 1))
        f0 = random.randint(0, n_mels - f)
        x[:, f0 : f0 + f, :] = 0
    if time_mask > 0:
        t = random.randint(0, min(time_mask, n_time - 1))
        t0 = random.randint(0, n_time - t)
        x[:, :, t0 : t0 + t] = 0
    return x


def split_dataset(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = df["primary_label"].values
    train_val, test = train_test_split(
        df, test_size=test_ratio, stratify=labels, random_state=seed
    )
    val_size = val_ratio / (1 - test_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val["primary_label"],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def get_stratified_folds(
    df: pd.DataFrame, n_folds: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    labels = df["primary_label"].values
    return [(tr, va) for tr, va in skf.split(df, labels)]
