from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from homebrain.brain.spatial_memory_v0 import load_checkpoint
from homebrain.data.spatial_dataset import write_json
from homebrain.train.spatial_dataset import SpatialTrainDataset
from homebrain.train.spatial_v0_common import evaluate_spatial_v0


def eval_spatial_v0(
    *,
    checkpoint: str | Path,
    dataset_dir: str | Path,
    out_path: str | Path,
    feature_dir: str | Path | None = None,
    device_name: str | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model.to(device)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    resolved_features = feature_dir or metadata.get("feature_artifacts")
    if not isinstance(resolved_features, (str, Path)):
        raise ValueError("DINO feature artifacts must be supplied with --features or checkpoint metadata")
    dataset = SpatialTrainDataset(dataset_dir, feature_dir=resolved_features)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    eval_metrics = evaluate_spatial_v0(model, loader, device=device)
    train_metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    metrics: dict[str, Any] = {
        "schema_version": "homebrain.spatial_v0_eval_metrics.v0",
        "checkpoint": Path(checkpoint).as_posix(),
        "dataset": Path(dataset_dir).as_posix(),
        "features": Path(resolved_features).as_posix(),
        "device": str(device),
        "example_count": len(dataset),
        "train_loss_start": train_metrics.get("train_loss_start"),
        "train_loss_end": train_metrics.get("train_loss_end"),
        "loss_reduction_ratio": train_metrics.get("loss_reduction_ratio"),
        "val_loss": float(eval_metrics["loss"]),
        "bev_loss": float(eval_metrics["bev_loss"]),
        "bev_iou_or_proxy": float(eval_metrics["bev_iou_or_proxy"]),
        "pose_delta_rmse": eval_metrics["pose_delta_rmse"],
        "uncertainty_calibration_proxy": float(eval_metrics["uncertainty_calibration_proxy"]),
        "inference_fps": float(eval_metrics["inference_fps"]),
        "representation_pretraining_only": True,
        "control_safe": False,
    }
    write_json(out_path, metrics, pretty=True)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a SpatialMemoryNet v0 checkpoint on a SpatialTrainPack.")
    parser.add_argument("--checkpoint", required=True, help="SpatialMemoryNet v0 checkpoint.")
    parser.add_argument("--dataset", required=True, help="SpatialTrainPack directory.")
    parser.add_argument("--features", default=None, help="DINO feature artifact directory; defaults to checkpoint metadata.")
    parser.add_argument("--out", required=True, help="Output eval JSON.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    metrics = eval_spatial_v0(
        checkpoint=args.checkpoint,
        dataset_dir=args.dataset,
        feature_dir=args.features,
        out_path=args.out,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
