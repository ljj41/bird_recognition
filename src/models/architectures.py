"""Deep learning model architectures."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BirdCNN(nn.Module):
    """
    EfficientNet-style CNN on Mel spectrogram.
    创新点: 多尺度卷积 stem + 手工特征 late fusion。
    """

    def __init__(self, num_classes: int, handcrafted_dim: int, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(1, 32, pool=False),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.handcrafted_proj = nn.Sequential(
            nn.Linear(handcrafted_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, mel: torch.Tensor, handcrafted: torch.Tensor) -> torch.Tensor:
        x = self.stem(mel).flatten(1)
        h = self.handcrafted_proj(handcrafted)
        return self.head(torch.cat([x, h], dim=1))


class TemporalAttentionPool(nn.Module):
    """沿时间维注意力池化，输出固定维度上下文向量。"""

    def __init__(self, channels: int, attn_dim: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(channels, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(w * x, dim=1)


class BirdCRNN(nn.Module):
    """
    与 BirdCNN 同深度卷积骨干 + 时间注意力。
    创新点: 在 CNN 特征图上做时序建模，而非全局平均池化。
    """

    def __init__(
        self,
        num_classes: int,
        handcrafted_dim: int,
        n_mels: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, 32, pool=False),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )
        self.temporal_pool = TemporalAttentionPool(512)
        self.handcrafted_proj = nn.Sequential(
            nn.Linear(handcrafted_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, mel: torch.Tensor, handcrafted: torch.Tensor) -> torch.Tensor:
        x = self.backbone(mel)
        x = x.mean(dim=2).permute(0, 2, 1).contiguous()
        ctx = self.temporal_pool(x)
        h = self.handcrafted_proj(handcrafted)
        return self.head(torch.cat([ctx, h], dim=1))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BirdTransformer(nn.Module):
    """Audio Spectrogram Transformer 风格模型。"""

    def __init__(
        self,
        num_classes: int,
        handcrafted_dim: int,
        n_mels: int = 128,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.patch = nn.Conv2d(1, d_model, kernel_size=(16, 16), stride=(8, 8))
        self.pos = PositionalEncoding(d_model, max_len=2048, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.handcrafted_proj = nn.Linear(handcrafted_dim, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, mel: torch.Tensor, handcrafted: torch.Tensor) -> torch.Tensor:
        b = mel.size(0)
        x = self.patch(mel)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos(x)
        x = self.encoder(x)
        cls_out = x[:, 0]
        h = self.handcrafted_proj(handcrafted)
        return self.head(torch.cat([cls_out, h], dim=1))


class MambaBlockLite(nn.Module):
    """
    轻量 Mamba 风格块：深度可分离卷积 + GLU 门控 + 残差。
    不依赖 mamba-ssm 包，数值上比纯 gate*conv 更稳定。
    """

    def __init__(self, dim: int, kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.conv = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        value = self.conv(value.transpose(1, 2)).transpose(1, 2)
        value = F.silu(value)
        gate = torch.sigmoid(gate)
        x = self.dropout(self.out_proj(gate * value))
        return residual + x


class BirdMamba(nn.Module):
    """
    与 BirdCRNN 相同的 512 维 CNN 骨干，在时序维插入 Mamba 块 + 注意力池化。
    """

    def __init__(
        self,
        num_classes: int,
        handcrafted_dim: int,
        n_mels: int = 128,
        d_model: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, 32, pool=False),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )
        self.blocks = nn.ModuleList(
            [MambaBlockLite(d_model, dropout=dropout) for _ in range(num_layers)]
        )
        self.temporal_pool = TemporalAttentionPool(d_model)
        self.handcrafted_proj = nn.Sequential(
            nn.Linear(handcrafted_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model + 256, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, mel: torch.Tensor, handcrafted: torch.Tensor) -> torch.Tensor:
        x = self.backbone(mel)
        x = x.mean(dim=2).permute(0, 2, 1).contiguous()
        for block in self.blocks:
            x = block(x)
        ctx = self.temporal_pool(x)
        h = self.handcrafted_proj(handcrafted)
        return self.head(torch.cat([ctx, h], dim=1))


class GatedFusion(nn.Module):
    """可学习门控融合多分支特征 (模块缝合核心组件)。"""

    def __init__(self, dims: list[int], out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, out_dim) for d in dims])
        self.gate = nn.Sequential(
            nn.Linear(sum(dims), len(dims)),
            nn.Softmax(dim=-1),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, branches: list[torch.Tensor]) -> torch.Tensor:
        gates = self.gate(torch.cat(branches, dim=-1))
        fused = sum(
            g.unsqueeze(-1) * self.projs[i](b)
            for i, (g, b) in enumerate(zip(gates.unbind(-1), branches))
        )
        return self.drop(self.norm(fused))


class StitchedFusionNet(nn.Module):
    """
    创新模块缝合模型（本项目改进）:
    共享 512 维 CNN 骨干，并行融合三条互补支路:
      - CNN 全局池化: 整体频谱模式 (来自 BirdCNN)
      - CRNN 时序注意力: 关键发声时刻 (来自 BirdCRNN)
      - Mamba 时序建模: 长程依赖 (来自 BirdMamba)
    门控融合 (GatedFusion) 自适应加权，再与手工声学特征拼接分类。
    """

    def __init__(
        self,
        num_classes: int,
        handcrafted_dim: int,
        n_mels: int = 128,
        d_model: int = 512,
        mamba_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, 32, pool=False),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, d_model),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.temporal_pool = TemporalAttentionPool(d_model)
        self.mamba_blocks = nn.ModuleList(
            [MambaBlockLite(d_model, dropout=dropout) for _ in range(mamba_layers)]
        )
        self.branch_fusion = GatedFusion([d_model, d_model, d_model], d_model, dropout)
        self.hand_branch = nn.Sequential(
            nn.Linear(handcrafted_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model + 256, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, mel: torch.Tensor, handcrafted: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(mel)
        global_feat = self.global_pool(feat).flatten(1)

        seq = feat.mean(dim=2).permute(0, 2, 1).contiguous()
        attn_feat = self.temporal_pool(seq)

        mamba_seq = seq
        for block in self.mamba_blocks:
            mamba_seq = block(mamba_seq)
        mamba_feat = self.temporal_pool(mamba_seq)

        fused = self.branch_fusion([global_feat, attn_feat, mamba_feat])
        hand_feat = self.hand_branch(handcrafted)
        return self.head(torch.cat([fused, hand_feat], dim=1))


class HybridFusionNet(nn.Module):
    """
    三域融合: CNN全局特征 + 时序卷积分支 + 手工声学特征。
    分类头与 BirdCNN 一致，训练更稳定。
    """

    def __init__(
        self,
        num_classes: int,
        handcrafted_dim: int,
        n_mels: int = 128,
        d_model: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cnn_stem = nn.Sequential(
            ConvBlock(1, 32, pool=False),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.temporal_cnn = nn.Sequential(
            ConvBlock(1, 32, pool=False),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.temporal_pool = TemporalAttentionPool(d_model)
        self.hand_branch = nn.Sequential(
            nn.Linear(handcrafted_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(512 + d_model + d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, mel: torch.Tensor, handcrafted: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn_stem(mel).flatten(1)
        t = self.temporal_cnn(mel).mean(dim=2)
        t = self.temporal_conv(t).permute(0, 2, 1).contiguous()
        seq_feat = self.temporal_pool(t)
        hand_feat = self.hand_branch(handcrafted)
        return self.head(torch.cat([cnn_feat, seq_feat, hand_feat], dim=1))


def build_model(
    name: str,
    num_classes: int,
    handcrafted_dim: int,
    n_mels: int = 128,
    dropout: float = 0.3,
) -> nn.Module:
    name = name.lower()
    if name == "cnn":
        return BirdCNN(num_classes, handcrafted_dim, dropout)
    if name in ("crnn", "lstm"):
        return BirdCRNN(num_classes, handcrafted_dim, n_mels=n_mels, dropout=dropout)
    if name == "transformer":
        return BirdTransformer(num_classes, handcrafted_dim, n_mels=n_mels, dropout=dropout)
    if name == "mamba":
        return BirdMamba(num_classes, handcrafted_dim, n_mels=n_mels, dropout=dropout)
    if name == "hybrid":
        return HybridFusionNet(num_classes, handcrafted_dim, n_mels=n_mels, dropout=dropout)
    if name in ("stitch", "stitched", "fusion"):
        return StitchedFusionNet(num_classes, handcrafted_dim, n_mels=n_mels, dropout=dropout)
    raise ValueError(f"Unknown model: {name}")
