from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

from homebrain.brain.spatial_memory_v1 import SpatialMemoryNetV1, SpatialMemoryNetV1Config, save_checkpoint
from homebrain.data.spatial_dataset import read_json, write_json
from homebrain.train.spatial_dataset import (
    SpatialPackSpec,
    dataset_manifest_hashes,
    load_spatial_dataset_manifest,
)
from homebrain.train.spatial_temporal_dataset import (
    SpatialTemporalMultiPackDataset,
    SpatialTemporalTrainDataset,
)
from homebrain.train.spatial_v0_common import batch_to_device
from homebrain.train.spatial_v1_common import compute_spatial_v1_losses, evaluate_spatial_v1


def train_spatial_v1(
    *,
    dataset_dir: str | Path | None = None,
    feature_dir: str | Path | None = None,
    dataset_manifest: str | Path | None = None,
    out_dir: str | Path,
    max_steps: int = 300,
    window_length: int = 4,
    tiny_overfit: bool = False,
    device_name: str | None = None,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    seed: int = 12,
    sensor_context_mode: str = "masks",
    missing_pose_behavior: str = "reset",
    command: str | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    pack_specs = _pack_specs(dataset_dir=dataset_dir, feature_dir=feature_dir, dataset_manifest=dataset_manifest)
    train_dataset = _build_dataset(
        pack_specs,
        split="train",
        window_length=window_length,
        tiny_overfit=tiny_overfit,
        sensor_context_mode=sensor_context_mode,
        missing_pose_behavior=missing_pose_behavior,
    )
    val_dataset = train_dataset if tiny_overfit else _build_dataset(
        pack_specs,
        split="val",
        window_length=window_length,
        tiny_overfit=False,
        sensor_context_mode=sensor_context_mode,
        missing_pose_behavior=missing_pose_behavior,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        generator=_generator(seed),
    )
    train_eval_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)

    config = SpatialMemoryNetV1Config(
        feature_dim=int(train_dataset.feature_shape[-1]),
        bev_shape=train_dataset.grid_shape,
        sensor_dim=int(train_dataset[0]["sensor_mask"].shape[1]) + 1,
        meters_per_cell=_meters_per_cell(train_dataset),
        missing_pose_behavior=missing_pose_behavior,  # type: ignore[arg-type]
    )
    model = SpatialMemoryNetV1(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    initial_train = evaluate_spatial_v1(model, train_eval_loader, device=device)
    initial_val = evaluate_spatial_v1(model, val_loader, device=device)
    iterator = cycle(train_loader)
    model.train()
    for _step in range(max_steps):
        batch = batch_to_device(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model.forward_sequence(
            batch["features"],
            batch["timestamp_s"],
            batch["sensor_mask"],
            pose_delta_to_current=batch["pose_delta_to_current"],
            pose_delta_to_current_mask=batch["pose_delta_to_current_mask"],
            observation_mask=batch["observation_mask"],
        )
        losses = compute_spatial_v1_losses(outputs, batch)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    final_train = evaluate_spatial_v1(model, train_eval_loader, device=device)
    final_val = evaluate_spatial_v1(model, val_loader, device=device)
    start_loss = float(initial_train["loss"])
    end_loss = float(final_train["loss"])
    val_start_loss = float(initial_val["loss"])
    val_end_loss = float(final_val["loss"])
    loss_reduction_ratio = (start_loss - end_loss) / start_loss if start_loss > 0.0 else 0.0
    val_loss_reduction_ratio = (val_start_loss - val_end_loss) / val_start_loss if val_start_loss > 0.0 else 0.0
    failure_flags: list[str] = []
    if loss_reduction_ratio <= 0.0:
        failure_flags.append("train_loss_not_reduced")
    if not tiny_overfit and val_loss_reduction_ratio < 0.01:
        failure_flags.append("val_loss_not_improved_meaningfully")

    data_hashes = dataset_manifest_hashes(dataset_manifest=dataset_manifest, pack_specs=pack_specs)
    train_sources = _source_metrics(model, train_dataset, device=device, batch_size=batch_size, split="train")
    val_sources = _source_metrics(model, val_dataset, device=device, batch_size=batch_size, split="val")
    metrics: dict[str, Any] = {
        "schema_version": "homebrain.spatial_v1_train_metrics.v0",
        "dataset": Path(dataset_dir).as_posix() if dataset_dir is not None else None,
        "features": Path(feature_dir).as_posix() if feature_dir is not None else None,
        "dataset_manifest": Path(dataset_manifest).as_posix() if dataset_manifest is not None else None,
        "out_dir": Path(out_dir).as_posix(),
        "train_command": command,
        "max_steps": max_steps,
        "window_length": window_length,
        "tiny_overfit": tiny_overfit,
        "device": str(device),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": int(seed),
        "sensor_context_mode": sensor_context_mode,
        "missing_pose_behavior": missing_pose_behavior,
        "window_count": len(train_dataset),
        "val_window_count": len(val_dataset),
        "frame_count": int(len(train_dataset) * window_length),
        "feature_shape": list(train_dataset.feature_shape),
        "bev_shape": list(train_dataset.grid_shape),
        "source_name": train_dataset.source_name,
        "robot_supervision_grade": train_dataset.robot_supervision_grade,
        "train_loss_start": start_loss,
        "train_loss_end": end_loss,
        "loss_reduction_ratio": float(loss_reduction_ratio),
        "val_loss_start": val_start_loss,
        "val_loss": val_end_loss,
        "val_loss_reduction_ratio": float(val_loss_reduction_ratio),
        "current_bev_loss": float(final_val["current_bev_loss"]),
        "fused_memory_bev_loss": float(final_val["fused_memory_bev_loss"]),
        "current_bev_iou_or_proxy": float(final_val["current_bev_iou_or_proxy"]),
        "fused_memory_bev_iou_or_proxy": float(final_val["fused_memory_bev_iou_or_proxy"]),
        "unknown_reduction_vs_current": float(final_val["unknown_reduction_vs_current"]),
        "temporal_reprojection_consistency_iou": float(final_val["temporal_reprojection_consistency_iou"]),
        "pose_delta_rmse": final_val["pose_delta_rmse"],
        "uncertainty_calibration_proxy": float(final_val["uncertainty_calibration_proxy"]),
        "inference_fps": float(final_val["inference_fps"]),
        "memory_warp_valid_fraction": float(final_val["memory_warp_valid_fraction"]),
        "memory_reset_fraction": float(final_val["memory_reset_fraction"]),
        "failure_flags": failure_flags,
        "feature_alignment": train_dataset.feature_alignment,
        "data_manifest_hashes": data_hashes,
        "source_metrics": {
            "train": train_sources,
            "val": val_sources,
        },
        "temporal_supervision": getattr(train_dataset, "target_metadata", {}),
        "representation_pretraining_only": True,
        "control_safe": False,
        "replay_only": True,
        "not_executed": True,
        "product_training_approved": False,
    }

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "train_metrics.json", metrics, pretty=True)
    write_json(
        output / "config.json",
        {
            "schema_version": "homebrain.spatial_v1_train_config.v0",
            "dataset_manifest": Path(dataset_manifest).as_posix() if dataset_manifest is not None else None,
            "packs": [_pack_spec_record(spec) for spec in pack_specs],
            "max_steps": max_steps,
            "window_length": window_length,
            "tiny_overfit": tiny_overfit,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": int(seed),
            "sensor_context_mode": sensor_context_mode,
            "missing_pose_behavior": missing_pose_behavior,
            "device": str(device),
            "representation_pretraining_only": True,
            "control_safe": False,
        },
        pretty=True,
    )
    save_checkpoint(
        output / "checkpoint.pt",
        model.cpu(),
        metadata={
            "dataset": Path(dataset_dir).as_posix() if dataset_dir is not None else None,
            "feature_artifacts": Path(feature_dir).as_posix() if feature_dir is not None else None,
            "dataset_manifest": Path(dataset_manifest).as_posix() if dataset_manifest is not None else None,
            "packs": [_pack_spec_record(spec) for spec in pack_specs],
            "window_length": window_length,
            "tiny_overfit": tiny_overfit,
            "representation_pretraining_only": True,
            "control_safe": False,
            "replay_only": True,
            "not_executed": True,
            "product_training_approved": False,
            "robot_supervision_grade": train_dataset.robot_supervision_grade,
            "data_manifest_hashes": data_hashes,
            "sensor_context_mode": sensor_context_mode,
            "missing_pose_behavior": missing_pose_behavior,
            "train_command": command,
        },
        metrics=metrics,
    )
    return metrics


def _pack_specs(
    *,
    dataset_dir: str | Path | None,
    feature_dir: str | Path | None,
    dataset_manifest: str | Path | None,
) -> list[SpatialPackSpec]:
    if dataset_manifest is not None:
        return load_spatial_dataset_manifest(dataset_manifest)
    if dataset_dir is None or feature_dir is None:
        raise ValueError("supply either --dataset-manifest or both --dataset and --features")
    dataset_path = Path(dataset_dir)
    return [
        SpatialPackSpec(
            source_name=dataset_path.name,
            dataset_dir=dataset_path,
            feature_dir=Path(feature_dir),
        )
    ]


def _build_dataset(
    pack_specs: list[SpatialPackSpec],
    *,
    split: str,
    window_length: int,
    tiny_overfit: bool,
    sensor_context_mode: str,
    missing_pose_behavior: str,
) -> SpatialTemporalTrainDataset | SpatialTemporalMultiPackDataset:
    if len(pack_specs) == 1:
        spec = pack_specs[0]
        return SpatialTemporalTrainDataset(
            spec.dataset_dir,
            feature_dir=spec.feature_dir,
            split=split,
            source_name=spec.source_name,
            window_length=window_length,
            tiny_overfit=tiny_overfit,
            sensor_context_mode=sensor_context_mode,
            missing_pose_behavior=missing_pose_behavior,  # type: ignore[arg-type]
        )
    return SpatialTemporalMultiPackDataset(
        pack_specs,
        split=split,
        window_length=window_length,
        tiny_overfit=tiny_overfit,
        sensor_context_mode=sensor_context_mode,
        missing_pose_behavior=missing_pose_behavior,  # type: ignore[arg-type]
    )


def _source_metrics(
    model: torch.nn.Module,
    dataset: SpatialTemporalTrainDataset | SpatialTemporalMultiPackDataset,
    *,
    device: torch.device,
    batch_size: int,
    split: str,
) -> list[dict[str, Any]]:
    source_datasets = dataset.source_datasets() if isinstance(dataset, SpatialTemporalMultiPackDataset) else [dataset]
    records: list[dict[str, Any]] = []
    for source_dataset in source_datasets:
        loader = DataLoader(source_dataset, batch_size=min(batch_size, len(source_dataset)), shuffle=False)
        values = evaluate_spatial_v1(model, loader, device=device)
        records.append(
            {
                "source_name": source_dataset.source_name,
                "robot_supervision_grade": source_dataset.robot_supervision_grade,
                "split": split,
                "window_count": len(source_dataset),
                "frame_count": int(len(source_dataset) * source_dataset.window_length),
                "loss": float(values["loss"]),
                "current_bev_iou_or_proxy": float(values["current_bev_iou_or_proxy"]),
                "fused_memory_bev_iou_or_proxy": float(values["fused_memory_bev_iou_or_proxy"]),
                "temporal_reprojection_consistency_iou": float(values["temporal_reprojection_consistency_iou"]),
                "pose_delta_rmse": values["pose_delta_rmse"],
                "memory_warp_valid_fraction": float(values["memory_warp_valid_fraction"]),
                "memory_reset_fraction": float(values["memory_reset_fraction"]),
                "control_safe": False,
            }
        )
    return records


def _meters_per_cell(dataset: SpatialTemporalTrainDataset | SpatialTemporalMultiPackDataset) -> float:
    first = dataset.source_datasets()[0] if isinstance(dataset, SpatialTemporalMultiPackDataset) else dataset
    camera_config = first.base.manifest.get("camera_config")
    if isinstance(camera_config, dict):
        value = camera_config.get("meters_per_cell")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0.0:
            return float(value)
    return 0.05


def _pack_spec_record(spec: SpatialPackSpec) -> dict[str, Any]:
    return {
        "source_name": spec.source_name,
        "dataset": spec.dataset_dir.as_posix(),
        "features": spec.feature_dir.as_posix(),
    }


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train SpatialMemoryNet v1 on temporal SpatialTrainPack windows.")
    parser.add_argument("--dataset", default=None, help="SpatialTrainPack directory.")
    parser.add_argument("--features", default=None, help="DINO teacher artifact directory.")
    parser.add_argument("--dataset-manifest", default=None, help="JSON manifest listing SpatialTrainPacks and DINO features.")
    parser.add_argument("--out", required=True, help="Output training run directory.")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--window-length", type=int, default=4)
    parser.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--sensor-context-mode", choices=("masks", "odom"), default="masks")
    parser.add_argument("--missing-pose-behavior", choices=("reset", "no_warp", "masked_update"), default="reset")
    args = parser.parse_args(argv)
    command = "python -m homebrain.train.train_spatial_v1 " + " ".join(sys.argv[1:])
    metrics = train_spatial_v1(
        dataset_dir=args.dataset,
        feature_dir=args.features,
        dataset_manifest=args.dataset_manifest,
        out_dir=args.out,
        max_steps=args.max_steps,
        window_length=args.window_length,
        tiny_overfit=args.tiny_overfit,
        device_name=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        sensor_context_mode=args.sensor_context_mode,
        missing_pose_behavior=args.missing_pose_behavior,
        command=command,
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
