#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速验证 Transformer / Mamba 模型能否正常前向传播（不训练）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_paths
from src.models.architectures import build_model
from src.utils.experiment import load_experiment_data


def parse_args():
    p = argparse.ArgumentParser(description="Transformer/Mamba 前向传播测试")
    p.add_argument("--config", default=str(ROOT / "configs" / "compare_fast.yaml"))
    p.add_argument("--models", nargs="+", default=["transformer", "mamba"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    project_root = ROOT.parent
    cfg = resolve_paths(load_config(args.config), project_root)
    exp = load_experiment_data(cfg)
    n_cls = len(exp["label2id"])
    hand_dim = exp["handcrafted_dim"]
    n_mels = exp["n_mels"]
    device = args.device

    print("=" * 50)
    print("  Transformer / Mamba 前向传播测试")
    print("=" * 50)
    print(f"  类别数: {n_cls} | 手工特征维: {hand_dim} | Mel: {n_mels}")
    print(f"  设备: {device} | batch: {args.batch_size}")
    print("-" * 50)

    time_steps = int(cfg["data"]["sample_rate"] * cfg["data"]["duration"] / cfg["features"]["hop_length"]) + 1
    mel = torch.randn(args.batch_size, 1, n_mels, time_steps, device=device)
    hand = torch.randn(args.batch_size, hand_dim, device=device)

    ok = True
    for name in args.models:
        name = name.lower()
        try:
            model = build_model(name, n_cls, hand_dim, n_mels=n_mels).to(device)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            with torch.no_grad():
                logits = model(mel, hand)
            assert logits.shape == (args.batch_size, n_cls), f"输出形状异常: {logits.shape}"
            print(f"  [OK] {name:12s} | 参数量 {n_params:.2f}M | 输出 {tuple(logits.shape)}")
        except Exception as e:
            ok = False
            print(f"  [FAIL] {name}: {e}")

    print("-" * 50)
    if ok:
        print("  全部通过，可以开始训练。")
    else:
        print("  存在失败项，请先修复后再训练。")
        sys.exit(1)


if __name__ == "__main__":
    main()
