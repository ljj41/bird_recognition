"""Deep learning training loop."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import spec_augment
from .losses import FocalLoss
from .metrics import compute_metrics

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


class DeepTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        class_weights: torch.Tensor | None = None,
        aug_cfg: dict | None = None,
        use_amp: bool = True,
        data_on_gpu: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.aug_cfg = aug_cfg or {}
        self.use_amp = use_amp and device == "cuda"
        self.data_on_gpu = data_on_gpu
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.criterion = FocalLoss(
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
            weight=class_weights.to(device) if class_weights is not None else None,
        )
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.history: list[dict] = []

    def _to_device(self, batch: dict) -> dict:
        if self.data_on_gpu:
            return batch
        return {
            "mel": batch["mel"].to(self.device, non_blocking=True),
            "handcrafted": batch["handcrafted"].to(self.device, non_blocking=True),
            "label": batch["label"].to(self.device, non_blocking=True),
        }

    def _step_batch(self, batch: dict, train: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self._to_device(batch)
        mel = batch["mel"]
        handcrafted = batch["handcrafted"]
        labels = batch["label"]
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        if train and self.aug_cfg.get("enabled", False):
            spec_cfg = self.aug_cfg.get("spec_augment", {})
            if mel.dim() == 3:
                mel = mel.unsqueeze(1)
            mel = torch.stack(
                [
                    spec_augment(
                        mel[i],
                        freq_mask=spec_cfg.get("freq_mask", 16),
                        time_mask=spec_cfg.get("time_mask", 32),
                    )
                    for i in range(mel.size(0))
                ]
            )

        with autocast("cuda", enabled=self.use_amp):
            logits = self.model(mel, handcrafted)
            loss = self.criterion(logits, labels)
        return loss, logits, labels

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0
        for batch in loader:
            loss, logits, labels = self._step_batch(batch, train=False)
            total_loss += loss.item() * labels.size(0)
            all_logits.append(logits.float().cpu())
            all_labels.append(labels.cpu())
        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        probs = torch.softmax(logits_cat, dim=1).numpy()
        preds = logits_cat.argmax(dim=1).numpy()
        metrics = compute_metrics(labels_cat.numpy(), preds, probs)
        metrics["loss"] = total_loss / len(loader.dataset)
        return metrics

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        for batch in tqdm(loader, desc="训练中", leave=False):
            self.optimizer.zero_grad(set_to_none=True)
            loss, _, labels = self._step_batch(batch, train=True)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            # 仅累计 loss，避免每 batch .cpu() 强制 GPU 同步
            total_loss += loss.detach().float().item() * labels.size(0)
            n_samples += labels.size(0)
        return {"loss": total_loss / max(n_samples, 1), "accuracy": 0.0, "macro_f1": 0.0}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_dir: str | Path,
        early_stop_patience: int = 10,
    ) -> dict:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        best_f1 = -1.0
        patience = 0
        best_path = save_dir / "best_model.pt"

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_m = self.train_epoch(train_loader)
            val_m = self.evaluate(val_loader)
            self.scheduler.step()
            record = {
                "epoch": epoch,
                "time_sec": time.time() - t0,
                "train": train_m,
                "val": val_m,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(record)
            print(
                f"第 {epoch:02d} 轮 ({record['time_sec']:.0f}秒) | "
                f"训练损失 {train_m['loss']:.4f} | "
                f"验证损失 {val_m['loss']:.4f} 准确率 {val_m['accuracy']*100:.1f}% "
                f"Macro-F1 {val_m['macro_f1']*100:.1f}% Top-5 {val_m.get('top_5_acc', 0)*100:.1f}%",
                flush=True,
            )

            if val_m["macro_f1"] > best_f1:
                best_f1 = val_m["macro_f1"]
                patience = 0
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "epoch": epoch,
                        "val_metrics": val_m,
                    },
                    best_path,
                )
            else:
                patience += 1
                if patience >= early_stop_patience:
                    print(f"验证指标连续 {early_stop_patience} 轮未提升，提前停止于第 {epoch} 轮", flush=True)
                    break

        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        return {"best_f1": best_f1, "best_model": str(best_path)}


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    data_on_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    all_probs, all_preds, all_labels = [], [], []
    for batch in loader:
        if data_on_gpu:
            mel, handcrafted, labels = batch["mel"], batch["handcrafted"], batch["label"]
        else:
            mel = batch["mel"].to(device)
            handcrafted = batch["handcrafted"].to(device)
            labels = batch["label"]
        with autocast("cuda", enabled=device == "cuda"):
            logits = model(mel, handcrafted)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(probs.argmax(axis=1))
        all_labels.append(labels.cpu().numpy() if data_on_gpu else labels.numpy())
    return (
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )
