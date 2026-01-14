# main.py
from __future__ import annotations

import os
import json
from datetime import datetime

from config import Config
from train import train_pipeline
from test import test_pipeline


def main():
    cfg = Config()

    aug_suffix = "_aug" if len(cfg.active_augmentations) > 0 else ""
    run_name = datetime.now().strftime("%Y%m%d_%H%M") + f"_DynamicViT_{cfg.vit_backbone}_{cfg.rnn_type}{aug_suffix}"
    run_dir = os.path.join(cfg.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    pruning_results = []
    baseline_time = None
    baseline_peak_mem = None

    for pruning_ratio in cfg.pruning_ratios:
        train_out = train_pipeline(cfg, run_dir, pruning_ratio=pruning_ratio)

        metrics = test_pipeline(
            cfg=cfg,
            dataset=train_out["dataset"],
            splits=train_out["splits"],
            best_model_path=train_out["best_model_path"],
            pruning_dir=train_out["run_dir"],
            device=train_out["device"],
            pruning_ratio=pruning_ratio,
        )

        if pruning_ratio == 0.0:
            baseline_time = metrics.get("avg_time_per_batch", None)
            baseline_peak_mem = metrics.get("peak_memory_mb", None)

        speedup = (baseline_time / metrics["avg_time_per_batch"]) if baseline_time and metrics["avg_time_per_batch"] > 0 else 1.0
        mem_reduction = (1.0 - metrics["peak_memory_mb"] / baseline_peak_mem) * 100.0 if baseline_peak_mem and metrics["peak_memory_mb"] > 0 else 0.0

        pruning_results.append({
            "pruning_ratio": pruning_ratio,
            "pruning_method": cfg.pruning_method,
            "Hb_R2": metrics.get("Hb_R2"),
            "Hb_RMSE": metrics.get("Hb_RMSE"),
            "Hb_MAE": metrics.get("Hb_MAE"),
            "Hb_MAPE": metrics.get("Hb_MAPE"),
            "Hct_R2": metrics.get("Hct_R2"),
            "Hct_RMSE": metrics.get("Hct_RMSE"),
            "Hct_MAE": metrics.get("Hct_MAE"),
            "Hct_MAPE": metrics.get("Hct_MAPE"),
            "avg_time_per_batch": metrics.get("avg_time_per_batch"),
            "peak_memory_mb": metrics.get("peak_memory_mb"),
            "avg_memory_mb": metrics.get("avg_memory_mb"),
            "speedup_vs_baseline": speedup,
            "memory_reduction_vs_baseline": mem_reduction,
        })

    with open(os.path.join(run_dir, "overall_pruning_results.json"), "w") as f:
        json.dump(pruning_results, f, indent=2)

    print(f"[DONE] Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
