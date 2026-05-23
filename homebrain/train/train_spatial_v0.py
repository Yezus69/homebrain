from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from homebrain.brain.spatial_memory_v0 import SpatialMemoryNetConfig, SpatialMemoryNetV0, save_checkpoint
from homebrain.data.spatial_dataset import write_json
from homebrain.train.spatial_dataset import SpatialTrainDataset
from homebrain.train.spatial_v0_common import batch_to_device, compute_spatial_v0_losses, evaluate_spatial_v0


def train_spatial_v0(
    *,
    dataset_dir: str | Path,
    feature_dir: str | Path,
    out_dir: str | Path,
    max_steps: int = 300,
    tiny_overfit: bool = False,
    device_name: str | None = None,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
) -> dict[str, Any]:
    torch.manual_seed(7)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = SpatialTrainDataset(
        dataset_dir,
        feature_dir=feature_dir,
        tiny_overfit=tiny_overfit,
        tiny_limit=8,
    )
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, generator=_generator())
    eval_loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)

    feature_dim = int(dataset.feature_shape[-1])
    config = SpatialMemoryNetConfig(feature_dim=feature_dim, bev_shape=dataset.grid_shape)
    model = SpatialMemoryNetV0(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    initial = evaluate_spatial_v0(model, eval_loader, device=device)
    iterator = cycle(loader)
    model.train()
    for _step in range(max_steps):
        batch = batch_to_device(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["features"], batch["timestamp_s"], batch["sensor_mask"])
        losses = compute_spatial_v0_losses(outputs, batch)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    final = evaluate_spatial_v0(model, eval_loader, device=device)
    start_loss = float(initial["loss"])
    end_loss = float(final["loss"])
    loss_reduction_ratio = (start_loss - end_loss) / start_loss if start_loss > 0.0 else 0.0
    metrics: dict[str, Any] = {
        "schema_version": "homebrain.spatial_v0_train_metrics.v0",
        "dataset": Path(dataset_dir).as_posix(),
        "features": Path(feature_dir).as_posix(),
        "out_dir": Path(out_dir).as_posix(),
        "max_steps": max_steps,
        "tiny_overfit": tiny_overfit,
        "device": str(device),
        "example_count": len(dataset),
        "feature_shape": list(dataset.feature_shape),
        "bev_shape": list(dataset.grid_shape),
        "train_loss_start": start_loss,
        "train_loss_end": end_loss,
        "loss_reduction_ratio": float(loss_reduction_ratio),
        "bev_loss": float(final["bev_loss"]),
        "bev_iou_or_proxy": float(final["bev_iou_or_proxy"]),
        "pose_delta_rmse": final["pose_delta_rmse"],
        "uncertainty_calibration_proxy": float(final["uncertainty_calibration_proxy"]),
        "inference_fps": float(final["inference_fps"]),
        "representation_pretraining_only": True,
        "control_safe": False,
    }

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "train_metrics.json", metrics, pretty=True)
    save_checkpoint(
        output / "checkpoint.pt",
        model.cpu(),
        metadata={
            "dataset": Path(dataset_dir).as_posix(),
            "feature_artifacts": Path(feature_dir).as_posix(),
            "tiny_overfit": tiny_overfit,
            "representation_pretraining_only": True,
            "control_safe": False,
            "robot_supervision_grade": dataset.robot_supervision_grade,
        },
        metrics=metrics,
    )
    return metrics


def _generator() -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(7)
    return generator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train tiny SpatialMemoryNet v0 on DINO-feature SpatialTrainPacks.")
    parser.add_argument("--dataset", required=True, help="SpatialTrainPack directory.")
    parser.add_argument("--features", required=True, help="DINO teacher artifact directory.")
    parser.add_argument("--out", required=True, help="Output training run directory.")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args(argv)
    metrics = train_spatial_v0(
        dataset_dir=args.dataset,
        feature_dir=args.features,
        out_dir=args.out,
        max_steps=args.max_steps,
        tiny_overfit=args.tiny_overfit,
        device_name=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
