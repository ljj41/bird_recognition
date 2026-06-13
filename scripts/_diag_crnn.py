"""Test fixed CRNN."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.cache_dataset import CachedBirdDataset
from src.models.architectures import BirdCRNN
from src.training.trainer import DeepTrainer

device = "cuda"
train_ds = CachedBirdDataset("outputs/cache/train_top10.npz", device=device, preload=True)
val_ds = CachedBirdDataset("outputs/cache/val_top10.npz", device=device, preload=True)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)
hand_dim = train_ds[0]["handcrafted"].shape[0]

m = BirdCRNN(10, hand_dim, n_mels=128)
t = DeepTrainer(
    m, device, lr=1e-4, focal_gamma=1.0, label_smoothing=0.05,
    use_amp=False, data_on_gpu=True,
)
t.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(t.optimizer, T_max=10)
for ep in range(1, 11):
    tm = t.train_epoch(train_loader)
    vm = t.evaluate(val_loader)
    print(
        f"ep{ep} train={tm['loss']:.4f} val={vm['loss']:.4f} "
        f"acc={vm['accuracy']*100:.1f}% f1={vm['macro_f1']*100:.1f}%"
    )
