# config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # -----------------------------
    # Paths
    # -----------------------------
    # Expected dataset layout:
    # train_root/
    #   sample_0001/ 0s.jpg 10s.jpg ...
    # train_labels_csv: CSV with columns: sample_id,Hb,Hct
    train_root: str = "/path/to/train_root"
    val_root: str = "/path/to/val_root"
    test_root: str = "/path/to/test_root"

    train_labels_csv: str = "/path/to/train_labels.csv"
    val_labels_csv: str = "/path/to/val_labels.csv"
    test_labels_csv: str = "/path/to/test_labels.csv"

    # Where to save runs (models/logs)
    output_dir: str = "./runs"

    # -----------------------------
    # Time-series sampling
    # -----------------------------
    time_start_sec: int = 0
    time_end_sec: int = 120
    time_step_sec: int = 10

    # -----------------------------
    # Image preprocessing
    # -----------------------------
    image_size: int = 224
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # -----------------------------
    # DataLoader
    # -----------------------------
    batch_size: int = 32
    num_workers: int = 8

    # -----------------------------
    # Training hyperparameters
    # -----------------------------
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    use_amp: bool = True
    random_seed: int = 42

    # -----------------------------
    # Model config (DynamicViT + RNN)
    # -----------------------------
    vit_backbone: str = "vit_b_16"          # ["vit_b_16", "vit_b_32"]
    pruning_method: str = "adaptive"        # ["adaptive", "uniform"]
    pruning_ratios: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    rnn_type: str = "lstm"                  # ["lstm", "gru"]
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1
    rnn_bidirectional: bool = False

    # -----------------------------
    # Device
    # -----------------------------
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    use_dataparallel: bool = False

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

        # Basic sanity checks (lightweight, no verbose errors)
        assert self.time_step_sec > 0
        assert self.time_end_sec >= self.time_start_sec
        assert self.batch_size > 0
        assert self.num_epochs > 0
        assert 0.0 <= min(self.pruning_ratios) and max(self.pruning_ratios) < 1.0
