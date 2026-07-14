# train.py
from __future__ import annotations

import os
import json
import random
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

from config import Config
from model import DynamicViTRNNRegressor


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------
# Dataset
# -----------------------------
class BloodSpotSequenceDataset(Dataset):
   
    def __init__(
        self,
        root_dir: str,
        labels_csv: str,
        time_points: List[int],
        transform: Optional[nn.Module] = None,
    ):
        self.root_dir = root_dir
        self.time_points = time_points
        self.transform = transform

        self.labels = self._load_labels(labels_csv)
        self.samples = self._index_samples()

    @staticmethod
    def _load_labels(labels_csv: str) -> Dict[str, Tuple[float, float]]:
        import csv
        labels: Dict[str, Tuple[float, float]] = {}
        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["sample_id"].strip()
                hb = float(row["Hb"])
                hct = float(row["Hct"])
                labels[sid] = (hb, hct)
        return labels

    def _index_samples(self) -> List[Dict]:
        """
        Build a minimal index:
          [
            {"sample_id": str, "frame_paths": [str...], "label": (hb,hct)}
          ]
        """
        sample_ids = sorted(self.labels.keys())
        samples: List[Dict] = []

        for sid in sample_ids:
            sample_dir = os.path.join(self.root_dir, sid)
            frame_paths = []
            for t in self.time_points:
                frame_paths.append(os.path.join(sample_dir, f"{t}s.jpg"))

            samples.append({
                "sample_id": sid,
                "frame_paths": frame_paths,
                "label": self.labels[sid],
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        hb, hct = s["label"]

        frames = []
        for fp in s["frame_paths"]:
            img = Image.open(fp).convert("RGB")
            img = self.transform(img) if self.transform else T.ToTensor()(img)
            frames.append(img)

        x = torch.stack(frames, dim=0)  # [T,C,H,W]
        y = torch.tensor([hb, hct], dtype=torch.float32)
        info = {"sample_id": s["sample_id"]}
        return x, y, info


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    out: Dict[str, float] = {}
    for i, name in enumerate(["Hb", "Hct"]):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mse = float(np.mean((yp - yt) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(yp - yt)))
        var = float(np.var(yt))
        r2 = 0.0 if var == 0 else float(1.0 - mse / (var + 1e-8))

        out[f"{name}_MSE"] = mse
        out[f"{name}_RMSE"] = rmse
        out[f"{name}_MAE"] = mae
        out[f"{name}_R2"] = r2

    return out


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    y_true, y_pred = [], []

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(total_loss / len(loader.dataset))
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    y_true, y_pred = [], []

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = criterion(pred, y)

        total_loss += float(loss.item()) * x.size(0)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(total_loss / len(loader.dataset))
    return metrics


# -----------------------------
# Builders
# -----------------------------
def build_transforms(cfg: Config):
    return T.Compose([
        T.Resize((cfg.image_size, cfg.image_size)),
        T.ToTensor(),
        T.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
    ])


def build_dataloaders(cfg: Config):
    time_points = list(range(cfg.time_start_sec, cfg.time_end_sec + 1, cfg.time_step_sec))
    transform = build_transforms(cfg)

    train_ds = BloodSpotSequenceDataset(
        root_dir=cfg.train_root,
        labels_csv=cfg.train_labels_csv,
        time_points=time_points,
        transform=transform,
    )
    val_ds = BloodSpotSequenceDataset(
        root_dir=cfg.val_root,
        labels_csv=cfg.val_labels_csv,
        time_points=time_points,
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(cfg: Config, pruning_ratio: float, device: torch.device) -> nn.Module:
    model = DynamicViTRNNRegressor(
        vit_backbone=cfg.vit_backbone,
        pruning_ratio=pruning_ratio,
        pruning_method=cfg.pruning_method,
        rnn_type=cfg.rnn_type,
        rnn_hidden_size=cfg.rnn_hidden_size,
        rnn_num_layers=cfg.rnn_num_layers,
        rnn_bidirectional=cfg.rnn_bidirectional,
        pretrained=True,
    ).to(device)

    if cfg.use_dataparallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=cfg.gpu_ids, output_device=cfg.gpu_ids[0])

    return model


# -----------------------------
# Training pipeline
# -----------------------------
def train_pipeline(cfg: Config, run_dir: str, pruning_ratio: float):
    set_seed(cfg.random_seed)

    device = torch.device(f"cuda:{cfg.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    os.makedirs(run_dir, exist_ok=True)

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg, pruning_ratio=pruning_ratio, device=device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and cfg.use_amp) else None

    pruning_dir = os.path.join(run_dir, f"pruning_{pruning_ratio:.2f}")
    os.makedirs(pruning_dir, exist_ok=True)

    best_model_path = os.path.join(pruning_dir, "best_model.pth")
    log_path = os.path.join(pruning_dir, "train_log.json")
    summary_path = os.path.join(pruning_dir, "summary.json")

    best_val_loss = float("inf")
    history: List[Dict] = []

    for epoch in range(1, cfg.num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history.append({"epoch": epoch, "split": "train", "pruning_ratio": pruning_ratio, **train_metrics})
        history.append({"epoch": epoch, "split": "val", "pruning_ratio": pruning_ratio, **val_metrics})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_model_path)

    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    with open(summary_path, "w") as f:
        json.dump({
            "config": asdict(cfg),
            "pruning_ratio": pruning_ratio,
            "best_val_loss": float(best_val_loss),
            "best_model_path": best_model_path,
        }, f, indent=2)

    return {
        "best_model_path": best_model_path,
        "device": str(device),
        "pruning_dir": pruning_dir,
    }
