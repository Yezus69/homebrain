from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from homebrain.brain.spatial_memory_v0 import load_checkpoint
from homebrain.data.spatial_dataset import write_json
from homebrain.train.spatial_dataset import (
    SpatialMultiPackDataset,
    SpatialPackSpec,
    SpatialTrainDataset,
    dataset_manifest_hashes,
    load_spatial_dataset_manifest,
)
from homebrain.train.spatial_v0_common import evaluate_spatial_v0


def eval_spatial_v0(
    *,
    checkpoint: str | Path,
    dataset_dir: str | Path | None = None,
    out_path: str | Path,
    feature_dir: str | Path | None = None,
    dataset_manifest: str | Path | None = None,
    split: str | None = None,
    device_name: str | None = None,
    batch_size: int = 8,
    sensor_context_mode: str | None = None,
) -> dict[str, Any]:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model.to(device)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    pack_specs = _pack_specs(
        dataset_dir=dataset_dir,
        feature_dir=feature_dir,
        dataset_manifest=dataset_manifest,
        checkpoint_metadata=metadata,
    )
    resolved_sensor_context_mode = sensor_context_mode or str(metadata.get("sensor_context_mode", "masks"))
    dataset = _build_dataset(pack_specs, split=split, sensor_context_mode=resolved_sensor_context_mode)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    eval_metrics = evaluate_spatial_v0(model, loader, device=device)
    train_metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    train_loss_end = _as_float_or_none(train_metrics.get("train_loss_end"))
    val_loss = float(eval_metrics["loss"])
    sources = _source_metrics(model, dataset, device=device, batch_size=batch_size, split=split)
    failure_flags = list(train_metrics.get("failure_flags", [])) if isinstance(train_metrics.get("failure_flags"), list) else []
    if train_loss_end is not None and val_loss > train_loss_end * 1.50:
        failure_flags.append("eval_loss_much_higher_than_train_loss")
    metrics: dict[str, Any] = {
        "schema_version": "homebrain.spatial_v0_eval_metrics.v1",
        "checkpoint": Path(checkpoint).as_posix(),
        "dataset": Path(dataset_dir).as_posix() if dataset_dir is not None else None,
        "features": Path(feature_dir).as_posix() if feature_dir is not None else None,
        "dataset_manifest": Path(dataset_manifest).as_posix() if dataset_manifest is not None else None,
        "split": split,
        "device": str(device),
        "sensor_context_mode": resolved_sensor_context_mode,
        "example_count": len(dataset),
        "source_name": dataset.source_name,
        "robot_supervision_grade": dataset.robot_supervision_grade,
        "train_loss_start": train_metrics.get("train_loss_start"),
        "train_loss_end": train_metrics.get("train_loss_end"),
        "loss_reduction_ratio": train_metrics.get("loss_reduction_ratio"),
        "val_loss": val_loss,
        "bev_loss": float(eval_metrics["bev_loss"]),
        "bev_iou_or_proxy": float(eval_metrics["bev_iou_or_proxy"]),
        "pose_delta_rmse": eval_metrics["pose_delta_rmse"],
        "uncertainty_calibration_proxy": float(eval_metrics["uncertainty_calibration_proxy"]),
        "inference_fps": float(eval_metrics["inference_fps"]),
        "overfit_gap": (train_loss_end - val_loss) if train_loss_end is not None else None,
        "overfit_gap_loss": (val_loss - train_loss_end) if train_loss_end is not None else None,
        "failure_flags": failure_flags,
        "feature_alignment": dataset.feature_alignment,
        "data_manifest_hashes": dataset_manifest_hashes(dataset_manifest=dataset_manifest, pack_specs=pack_specs),
        "sources": sources,
        "representation_pretraining_only": True,
        "control_safe": False,
    }
    write_json(out_path, metrics, pretty=True)
    return metrics


def _pack_specs(
    *,
    dataset_dir: str | Path | None,
    feature_dir: str | Path | None,
    dataset_manifest: str | Path | None,
    checkpoint_metadata: dict[str, Any],
) -> list[SpatialPackSpec]:
    if dataset_manifest is not None:
        return load_spatial_dataset_manifest(dataset_manifest)
    resolved_dataset = dataset_dir or checkpoint_metadata.get("dataset")
    resolved_features = feature_dir or checkpoint_metadata.get("feature_artifacts")
    if not isinstance(resolved_dataset, (str, Path)) or not isinstance(resolved_features, (str, Path)):
        raise ValueError("supply --dataset-manifest or both --dataset and --features")
    dataset_path = Path(resolved_dataset)
    return [
        SpatialPackSpec(
            source_name=dataset_path.name,
            dataset_dir=dataset_path,
            feature_dir=Path(resolved_features),
        )
    ]


def _build_dataset(
    pack_specs: list[SpatialPackSpec],
    *,
    split: str | None,
    sensor_context_mode: str,
) -> SpatialTrainDataset | SpatialMultiPackDataset:
    if len(pack_specs) == 1:
        spec = pack_specs[0]
        return SpatialTrainDataset(
            spec.dataset_dir,
            feature_dir=spec.feature_dir,
            split=split,
            source_name=spec.source_name,
            sensor_context_mode=sensor_context_mode,
        )
    return SpatialMultiPackDataset(pack_specs, split=split, sensor_context_mode=sensor_context_mode)


def _source_metrics(
    model: torch.nn.Module,
    dataset: SpatialTrainDataset | SpatialMultiPackDataset,
    *,
    device: torch.device,
    batch_size: int,
    split: str | None,
) -> list[dict[str, Any]]:
    source_datasets = dataset.source_datasets() if isinstance(dataset, SpatialMultiPackDataset) else [dataset]
    records: list[dict[str, Any]] = []
    for source_dataset in source_datasets:
        loader = DataLoader(source_dataset, batch_size=min(batch_size, len(source_dataset)), shuffle=False)
        values = evaluate_spatial_v0(model, loader, device=device)
        records.append(
            {
                "source_name": source_dataset.source_name,
                "robot_supervision_grade": source_dataset.robot_supervision_grade,
                "split": split,
                "example_count": len(source_dataset),
                "val_loss": float(values["loss"]),
                "bev_loss": float(values["bev_loss"]),
                "bev_iou_or_proxy": float(values["bev_iou_or_proxy"]),
                "pose_delta_rmse": values["pose_delta_rmse"],
                "uncertainty_calibration_proxy": float(values["uncertainty_calibration_proxy"]),
                "inference_fps": float(values["inference_fps"]),
                "control_safe": False,
            }
        )
    return records


def _as_float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a SpatialMemoryNet v0 checkpoint on SpatialTrainPacks.")
    parser.add_argument("--checkpoint", required=True, help="SpatialMemoryNet v0 checkpoint.")
    parser.add_argument("--dataset", default=None, help="SpatialTrainPack directory.")
    parser.add_argument("--features", default=None, help="DINO feature artifact directory; defaults to checkpoint metadata.")
    parser.add_argument("--dataset-manifest", default=None, help="JSON manifest listing SpatialTrainPacks and DINO features.")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate; use an empty string for all examples.")
    parser.add_argument("--out", required=True, help="Output eval JSON.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sensor-context-mode", choices=("masks", "odom"), default=None)
    args = parser.parse_args(argv)
    split = args.split if args.split else None
    metrics = eval_spatial_v0(
        checkpoint=args.checkpoint,
        dataset_dir=args.dataset,
        feature_dir=args.features,
        dataset_manifest=args.dataset_manifest,
        split=split,
        out_path=args.out,
        device_name=args.device,
        batch_size=args.batch_size,
        sensor_context_mode=args.sensor_context_mode,
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
