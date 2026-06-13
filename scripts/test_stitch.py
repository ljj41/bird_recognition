#!/usr/bin/env python
# -*- coding: utf-8__
"""StitchedFusionNet 前向传播测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_paths
from src.models.architectures import build_model
from src.utils.experiment import load_experiment_data


def main():
    cfg = resolve_paths(load_config(ROOT / "configs" / "compare_fast.yaml"), ROOT.parent)
    exp = load_experiment_data(cfg)
    n_cls = len(exp["label2id"])
    hand_dim = exp["handcrafted_dim"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    steps = int(cfg["data"]["sample_rate"] * cfg["data"]["duration"] / cfg["features"]["hop_length"]) + 1

    model = build_model("stitch", n_cls, hand_dim, n_mels=exp["n_mels"]).to(device)
    mel = torch.randn(2, 1, exp["n_mels"], steps, device=device)
    hand = torch.randn(2, hand_dim, device=device)
    with torch.no_grad():
        out = model(mel, hand)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"StitchedFusionNet OK | params={params:.2f}M | out={tuple(out.shape)}")


if __name__ == "__main__":
    main()
