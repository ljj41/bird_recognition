"""Quick DL diagnostic."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.cache_dataset import CachedBirdDataset
from src.models.architectures import build_model

cache = Path("outputs/cache/train_top10.npz")
d = np.load(cache)
print("mels", d["mels"].shape, "hand", d["handcrafted"].shape)
print("mel stats", float(d["mels"].mean()), float(d["mels"].std()))
print("label counts", np.bincount(d["labels"]))

ds = CachedBirdDataset(cache, device="cpu", preload=True)
loader = DataLoader(ds, batch_size=64, shuffle=True)
batch = next(iter(loader))
print("batch mel", batch["mel"].shape)

for name in ["cnn", "crnn"]:
    m = build_model(name, 10, batch["handcrafted"].shape[1], n_mels=128)
    m.train()
    logits = m(batch["mel"], batch["handcrafted"])
    loss = torch.nn.functional.cross_entropy(logits, batch["label"])
    loss.backward()
    gn = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
    print(name, "loss", float(loss), "grad", gn, "logits_std", float(logits.std()))
