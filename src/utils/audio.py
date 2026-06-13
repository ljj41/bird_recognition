"""Audio I/O and preprocessing utilities."""

from __future__ import annotations

import numpy as np
import librosa


def load_audio(
    path: str,
    sr: int = 32000,
    duration: float | None = 5.0,
    offset: float = 0.0,
) -> tuple[np.ndarray, int]:
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration, offset=offset)
    return y.astype(np.float32), sr


def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) == target_len:
        return y
    if len(y) > target_len:
        return y[:target_len]
    return np.pad(y, (0, target_len - len(y)), mode="constant")


def normalize_audio(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak < eps:
        return y
    return (y / peak).astype(np.float32)


def compute_mel(
    y: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 20,
    fmax: float | None = 16000,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
