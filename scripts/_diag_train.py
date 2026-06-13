"""Train 3 epochs on GPU to verify learning."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.cache_dataset import CachedBirdDataset
from src.models.architectures import build_model
from src.training.metrics import compute_metrics
from src.training.trainer import DeepTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
train_cache = Path("outputs/cache/train_top10.npz")
val_cache = Path("outputs/cache/val_top10.npz")

train_ds = CachedBirdDataset(train_cache, device=device, preload=True)
val_ds = CachedBirdDataset(val_cache, device=device, preload=True)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

sample = train_ds[0]
n_cls = 10
hand_dim = sample["handcrafted"].shape[0]

for name in ["cnn", "crnn"]:
    print(f"\n=== {name} ===")
    model = build_model(name, n_cls, hand_dim, n_mels=128)
    trainer = DeepTrainer(
        model, device, lr=1e-3, focal_gamma=1.5, label_smoothing=0.05,
        use_amp=True, data_on_gpu=(device == "cuda"),
    )
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=3)
    for ep in range(1, 4):
        tm = trainer.train_epoch(train_loader)
        vm = trainer.evaluate(val_loader)
        print(f"ep{ep} train_loss={tm['loss']:.4f} val_loss={vm['loss']:.4f} acc={vm['accuracy']*100:.1f}% f1={vm['macro_f1']*100:.1f}%")
