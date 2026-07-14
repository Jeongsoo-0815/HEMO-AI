# test.py
from __future__ import annotations

import os
import csv
from typing import List

import torch
import torch.nn as nn

from config import Config
from model import DynamicViTRNNRegressor
from train import eval_one_epoch


def save_test_predictions_csv(dataset, test_indices: List[int], y_true, y_pred, csv_path: str):
    rows = []
    for local_idx, dataset_idx in enumerate(test_indices):
        info = dataset.samples[dataset_idx]
        yt_hb, yt_hct = y_true[local_idx]
        yp_hb, yp_hct = y_pred[local_idx]

        rows.append({
            "dataset_idx": dataset_idx,
            "batch": info.get("batch", ""),
            "hb_key": info.get("hb_key", ""),
            "phone_model": info.get("phone_model", ""),
            "sample_id": info.get("sample_id", ""),
            "Hb_true": float(yt_hb),
            "Hct_true": float(yt_hct),
            "Hb_pred": float(yp_hb),
            "Hct_pred": float(yp_hct),
        })

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_pipeline(
    cfg: Config,
    dataset,
    splits,
    best_model_path: str,
    pruning_dir: str,
    device,
    pruning_ratio: float,
):
    _, _, test_idx = splits

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, test_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

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

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    criterion = nn.MSELoss()
    metrics, y_true, y_pred = eval_one_epoch(model, test_loader, criterion, device)

    test_dir = os.path.join(pruning_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    save_test_predictions_csv(
        dataset, test_idx, y_true, y_pred,
        os.path.join(test_dir, "test_predictions.csv")
    )

    with open(os.path.join(test_dir, "test_summary.json"), "w") as f:
        import json
        json.dump(metrics, f, indent=2)

    return metrics
