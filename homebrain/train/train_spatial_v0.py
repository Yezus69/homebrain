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
from homebrain.train.spatial_dataset import (
    SpatialMultiPackDataset,
    SpatialPackSpec,
    SpatialTrainDataset,
    dataset_manifest_hashes,
    load_spatial_dataset_manifest,
)
from homebrain.train.spatial_v0_common import batch_to_device, compute_spatial_v0_losses, evaluate_spatial_v0


def train_spatial_v0(
    *,
    dataset_dir: str | Path | None = None,
    feature_dir: str | Path | None = None,
    dataset_manifest: str | Path | None = None,
    out_dir: str | Path,
    max_steps: int = 300,
    tiny_overfit: bool = False,
    device_name: str | None = None,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
) -> dict[str, Any]:
    torch.manual_seed(7)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    pack_specs = _pack_specs(dataset_dir=dataset_dir, feature_dir=feature_dir, dataset_manifest=dataset_manifest)
    train_dataset = _build_dataset(pack_specs, split="train", tiny_overfit=tiny_overfit)
    val_dataset = train_dataset if tiny_overfit else _build_dataset(pack_specs, split="val", tiny_overfit=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        generator=_generator(),
    )
    train_eval_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)

    feature_dim = int(train_dataset.feature_shape[-1])
    config = SpatialMemoryNetConfig(feature_dim=feature_dim, bev_shape=train_dataset.grid_shape)
    model = SpatialMemoryNetV0(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    initial_train = evaluate_spatial_v0(model, train_eval_loader, device=device)
    initial_val = evaluate_spatial_v0(model, val_loader, device=device)
    iterator = cycle(train_loader)
    model.train()
    for _step in range(max_steps):
        batch = batch_to_device(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["features"], batch["timestamp_s"], batch["sensor_mask"])
        losses = compute_spatial_v0_losses(outputs, batch)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    final_train = evaluate_spatial_v0(model, train_eval_loader, device=device)
    final_val = evaluate_spatial_v0(model, val_loader, device=device)
    start_loss = float(initial_train["loss"])
    end_loss = float(final_train["loss"])
    val_start_loss = float(initial_val["loss"])
    val_end_loss = float(final_val["loss"])
    loss_reduction_ratio = (start_loss - end_loss) / start_loss if start_loss > 0.0 else 0.0
    val_loss_reduction_ratio = (val_start_loss - val_end_loss) / val_start_loss if val_start_loss > 0.0 else 0.0
    val_improved = val_loss_reduction_ratio >= 0.01
    failure_flags: list[str] = []
    if not val_improved:
        failure_flags.append("val_loss_not_improved_meaningfully")
    if loss_reduction_ratio <= 0.0:
        failure_flags.append("train_loss_not_reduced")

    data_hashes = dataset_manifest_hashes(dataset_manifest=dataset_manifest, pack_specs=pack_specs)
    train_sources = _source_metrics(model, train_dataset, device=device, batch_size=batch_size, split="train")
    val_sources = _source_metrics(model, val_dataset, device=device, batch_size=batch_size, split="val")
    metrics: dict[str, Any] = {
        "schema_version": "homebrain.spatial_v0_train_metrics.v1",
        "dataset": Path(dataset_dir).as_posix() if dataset_dir is not None else None,
        "features": Path(feature_dir).as_posix() if feature_dir is not None else None,
        "dataset_manifest": Path(dataset_manifest).as_posix() if dataset_manifest is not None else None,
        "out_dir": Path(out_dir).as_posix(),
        "max_steps": max_steps,
        "tiny_overfit": tiny_overfit,
        "device": str(device),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "example_count": len(train_dataset),
        "val_example_count": len(val_dataset),
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
        "bev_loss": float(final_val["bev_loss"]),
        "bev_iou_or_proxy": float(final_val["bev_iou_or_proxy"]),
        "pose_delta_rmse": final_val["pose_delta_rmse"],
        "uncertainty_calibration_proxy": float(final_val["uncertainty_calibration_proxy"]),
        "inference_fps": float(final_val["inference_fps"]),
        "overfit_gap": float(end_loss - val_end_loss),
        "overfit_gap_loss": float(val_end_loss - end_loss),
        "val_improved_meaningfully": val_improved,
        "failure_flags": failure_flags,
        "feature_alignment": train_dataset.feature_alignment,
        "data_manifest_hashes": data_hashes,
        "source_metrics": {
            "train": train_sources,
            "val": val_sources,
        },
        "representation_pretraining_only": True,
        "control_safe": False,
    }

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "train_metrics.json", metrics, pretty=True)
    write_json(
        output / "config.json",
        {
            "schema_version": "homebrain.spatial_v0_train_config.v1",
            "dataset_manifest": Path(dataset_manifest).as_posix() if dataset_manifest is not None else None,
            "packs": [_pack_spec_record(spec) for spec in pack_specs],
            "max_steps": max_steps,
            "tiny_overfit": tiny_overfit,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
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
            "tiny_overfit": tiny_overfit,
            "representation_pretraining_only": True,
            "control_safe": False,
            "robot_supervision_grade": train_dataset.robot_supervision_grade,
            "data_manifest_hashes": data_hashes,
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
    tiny_overfit: bool,
) -> SpatialTrainDataset | SpatialMultiPackDataset:
    if len(pack_specs) == 1:
        spec = pack_specs[0]
        return SpatialTrainDataset(
            spec.dataset_dir,
            feature_dir=spec.feature_dir,
            split=split,
            source_name=spec.source_name,
            tiny_overfit=tiny_overfit,
        )
    return SpatialMultiPackDataset(pack_specs, split=split, tiny_overfit=tiny_overfit)


def _source_metrics(
    model: torch.nn.Module,
    dataset: SpatialTrainDataset | SpatialMultiPackDataset,
    *,
    device: torch.device,
    batch_size: int,
    split: str,
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
                "val_loss" if split == "val" else "train_loss": float(values["loss"]),
                "bev_loss": float(values["bev_loss"]),
                "bev_iou_or_proxy": float(values["bev_iou_or_proxy"]),
                "pose_delta_rmse": values["pose_delta_rmse"],
                "uncertainty_calibration_proxy": float(values["uncertainty_calibration_proxy"]),
                "inference_fps": float(values["inference_fps"]),
                "control_safe": False,
            }
        )
    return records


def _pack_spec_record(spec: SpatialPackSpec) -> dict[str, Any]:
    return {
        "source_name": spec.source_name,
        "dataset": spec.dataset_dir.as_posix(),
        "features": spec.feature_dir.as_posix(),
    }


def _generator() -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(7)
    return generator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train SpatialMemoryNet v0 on DINO-feature SpatialTrainPacks.")
    parser.add_argument("--dataset", default=None, help="SpatialTrainPack directory.")
    parser.add_argument("--features", default=None, help="DINO teacher artifact directory.")
    parser.add_argument("--dataset-manifest", default=None, help="JSON manifest listing SpatialTrainPacks and DINO features.")
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
        dataset_manifest=args.dataset_manifest,
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
