"""
多维度声学特征提取模块。

涵盖项目要求的 MFCC、音高(pitch)、音量(volume)、音色(timbre)、语速(rate) 等特征，
并额外加入 chroma、spectral contrast 等增强特征以提升识别效果。
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


def _stats(x: np.ndarray) -> np.ndarray:
    """对时序特征计算 mean/std/min/max，压缩为固定长度向量。"""
    return np.concatenate(
        [
            np.mean(x, axis=1),
            np.std(x, axis=1),
            np.min(x, axis=1),
            np.max(x, axis=1),
        ]
    ).astype(np.float32)


@dataclass
class FeatureConfig:
    n_mfcc: int = 40
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    fmin: float = 20.0
    fmax: float = 16000.0
    include_pitch: bool = True
    include_timbre: bool = True
    include_rate: bool = True
    fast_mode: bool = False


class AudioFeatureExtractor:
    """从原始波形提取手工特征向量。"""

    def __init__(self, cfg: FeatureConfig | None = None):
        self.cfg = cfg or FeatureConfig()

    def extract_waveform_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        cfg = self.cfg
        feats: list[np.ndarray] = []

        # --- MFCC (含 delta) ---
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=cfg.n_mfcc,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        feats.extend([_stats(mfcc), _stats(mfcc_delta), _stats(mfcc_delta2)])

        # --- 音量 (RMS energy) ---
        rms = librosa.feature.rms(y=y, frame_length=cfg.n_fft, hop_length=cfg.hop_length)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=cfg.n_fft, hop_length=cfg.hop_length)
        feats.extend([_stats(rms), _stats(zcr)])

        # --- 音色 (spectral centroid, bandwidth, rolloff, flatness, contrast) ---
        if cfg.include_timbre:
            centroid = librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length
            )
            bandwidth = librosa.feature.spectral_bandwidth(
                y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length
            )
            rolloff = librosa.feature.spectral_rolloff(
                y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length
            )
            flatness = librosa.feature.spectral_flatness(
                y=y, n_fft=cfg.n_fft, hop_length=cfg.hop_length
            )
            contrast = librosa.feature.spectral_contrast(
                y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length, fmin=cfg.fmin
            )
            chroma = librosa.feature.chroma_stft(
                y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length
            )
            feats.extend(
                [
                    _stats(centroid),
                    _stats(bandwidth),
                    _stats(rolloff),
                    _stats(flatness),
                    _stats(contrast),
                    _stats(chroma),
                ]
            )

        # --- 音高 (pitch / F0) ---
        if cfg.include_pitch:
            if cfg.fast_mode:
                f0 = librosa.yin(
                    y, fmin=80, fmax=2000, sr=sr,
                    frame_length=cfg.n_fft, hop_length=cfg.hop_length,
                )
                f0_clean = np.nan_to_num(f0, nan=0.0)
                voiced = f0_clean > 0
                pitch_feats = np.array(
                    [
                        np.mean(f0_clean[voiced]) if np.any(voiced) else 0.0,
                        np.std(f0_clean[voiced]) if np.any(voiced) else 0.0,
                        np.mean(voiced.astype(np.float32)),
                        0.0,
                    ],
                    dtype=np.float32,
                )
            else:
                f0, voiced_flag, voiced_prob = librosa.pyin(
                    y,
                    fmin=librosa.note_to_hz("C2"),
                    fmax=librosa.note_to_hz("C7"),
                    sr=sr,
                    frame_length=cfg.n_fft,
                    hop_length=cfg.hop_length,
                )
                f0_clean = np.nan_to_num(f0, nan=0.0)
                pitch_feats = np.array(
                    [
                        np.mean(f0_clean[f0_clean > 0]) if np.any(f0_clean > 0) else 0.0,
                        np.std(f0_clean[f0_clean > 0]) if np.any(f0_clean > 0) else 0.0,
                        np.mean(voiced_flag.astype(np.float32)),
                        np.mean(voiced_prob),
                    ],
                    dtype=np.float32,
                )
            feats.append(pitch_feats)

        # --- 语速/节奏 (tempo, onset rate) ---
        if cfg.include_rate:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=cfg.hop_length)
            tempo = librosa.feature.tempo(
                onset_envelope=onset_env, sr=sr, hop_length=cfg.hop_length
            )
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=cfg.hop_length)
            duration = len(y) / sr
            rate_feats = np.array(
                [
                    float(tempo[0]) if len(tempo) else 0.0,
                    len(onsets) / max(duration, 1e-6),
                    np.mean(onset_env),
                    np.std(onset_env),
                ],
                dtype=np.float32,
            )
            feats.append(rate_feats)

        return np.concatenate(feats).astype(np.float32)

    def extract_mel_image(self, y: np.ndarray, sr: int) -> np.ndarray:
        """提取 Mel 频谱图 (用于 CNN/Transformer 输入)。"""
        cfg = self.cfg
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=cfg.n_mels,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.astype(np.float32)

    @property
    def feature_dim_hint(self) -> int:
        """估算手工特征维度 (用于调试)。"""
        dummy = np.random.randn(32000 * 5).astype(np.float32) * 0.01
        return len(self.extract_waveform_features(dummy, 32000))
