from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

from homebrain.brain.spatial_memory_v0 import load_checkpoint as load_v0_checkpoint
from homebrain.brain.spatial_memory_v1 import load_checkpoint
from homebrain.data.spatial_dataset import read_json, write_json
from homebrain.train.spatial_dataset import (
    SpatialPackSpec,
    dataset_manifest_hashes,
    load_spatial_dataset_manifest,
)
from homebrain.train.spatial_temporal_dataset import SpatialTemporalMultiPackDataset, SpatialTemporalTrainDataset
from homebrain.train.spatial_v0_common import batch_to_device
from homebrain.train.spatial_v1_common import (
    current_bev_parity_result,
    evaluate_spatial_v1,
    memory_benefit_result,
    memory_disabled_baseline_metrics,
)


def eval_spatial_v1(
    *,
    checkpoint: str | Path,
    dataset_dir: str | Path | None = None,
    out_path: str | Path,
    feature_dir: str | Path | None = None,
    dataset_manifest: str | Path | None = None,
    split: str | None = "val",
    device_name: str | None = None,
    batch_size: int = 8,
    window_length: int | None = None,
    sensor_context_mode: str | None = None,
    baseline_checkpoint_v0: str | Path | None = None,
    report_json: str | Path | None = None,
    report_md: str | Path | None = None,
    command: str | None = None,
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
    resolved_window_length = int(window_length or metadata.get("window_length", 4))
    resolved_sensor_context_mode = sensor_context_mode or str(metadata.get("sensor_context_mode", "masks"))
    resolved_missing_pose_behavior = str(metadata.get("missing_pose_behavior", model.config.missing_pose_behavior))
    dataset = _build_dataset(
        pack_specs,
        split=split,
        window_length=resolved_window_length,
        sensor_context_mode=resolved_sensor_context_mode,
        missing_pose_behavior=resolved_missing_pose_behavior,
    )
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    eval_metrics = evaluate_spatial_v1(model, loader, device=device)
    train_metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    sources = _source_metrics(model, dataset, device=device, batch_size=batch_size, split=split)
    baseline = (
        _v0_baseline_metrics(
            baseline_checkpoint_v0,
            dataset,
            batch_size=batch_size,
            device=device,
        )
        if baseline_checkpoint_v0 is not None
        else memory_disabled_baseline_metrics(eval_metrics)
    )
    parity = current_bev_parity_result(v1_metrics=eval_metrics, v0_baseline_metrics=baseline)
    benefit = memory_benefit_result(
        v1_metrics=eval_metrics,
        baseline_metrics=baseline,
        current_bev_parity_pass=parity["current_bev_parity_pass"],
    )
    failure_flags = list(train_metrics.get("failure_flags", [])) if isinstance(train_metrics.get("failure_flags"), list) else []
    failure_flags.extend(benefit["failure_reasons"])
    if float(eval_metrics["memory_warp_valid_fraction"]) <= 0.0 and resolved_window_length > 1:
        failure_flags.append("no_valid_pose_warps_observed")

    metrics: dict[str, Any] = {
        "schema_version": "homebrain.spatial_v1_eval_metrics.v0",
        "checkpoint": Path(checkpoint).as_posix(),
        "dataset": Path(dataset_dir).as_posix() if dataset_dir is not None else None,
        "features": Path(feature_dir).as_posix() if feature_dir is not None else None,
        "dataset_manifest": Path(dataset_manifest).as_posix() if dataset_manifest is not None else None,
        "split": split,
        "device": str(device),
        "batch_size": batch_size,
        "window_length": resolved_window_length,
        "sensor_context_mode": resolved_sensor_context_mode,
        "missing_pose_behavior": resolved_missing_pose_behavior,
        "eval_command": command,
        "window_count": len(dataset),
        "source_name": dataset.source_name,
        "robot_supervision_grade": dataset.robot_supervision_grade,
        "train_loss_start": train_metrics.get("train_loss_start"),
        "train_loss_end": train_metrics.get("train_loss_end"),
        "loss_reduction_ratio": train_metrics.get("loss_reduction_ratio"),
        "val_loss": float(eval_metrics["loss"]),
        "current_bev_loss": float(eval_metrics["current_bev_loss"]),
        "fused_memory_bev_loss": float(eval_metrics["fused_memory_bev_loss"]),
        "current_bev_iou_or_proxy": float(eval_metrics["current_bev_iou_or_proxy"]),
        "fused_memory_bev_iou_or_proxy": float(eval_metrics["fused_memory_bev_iou_or_proxy"]),
        "unknown_reduction_vs_current": float(eval_metrics["unknown_reduction_vs_current"]),
        "temporal_reprojection_consistency_iou": float(eval_metrics["temporal_reprojection_consistency_iou"]),
        "pose_delta_rmse": eval_metrics["pose_delta_rmse"],
        "uncertainty_calibration_proxy": float(eval_metrics["uncertainty_calibration_proxy"]),
        "inference_fps": float(eval_metrics["inference_fps"]),
        "memory_warp_valid_fraction": float(eval_metrics["memory_warp_valid_fraction"]),
        "valid_warp_fraction": float(eval_metrics["valid_warp_fraction"]),
        "memory_reset_fraction": float(eval_metrics["memory_reset_fraction"]),
        "update_mask_coverage_mean": float(eval_metrics["update_mask_coverage_mean"]),
        "memory_overwrite_fraction": float(eval_metrics["memory_overwrite_fraction"]),
        "pose_warp_source": str(eval_metrics["pose_warp_source"]),
        "predicted_pose_warp_ablation": bool(eval_metrics["predicted_pose_warp_ablation"]),
        "baseline_comparison": {
            "baseline_checkpoint_v0": Path(baseline_checkpoint_v0).as_posix() if baseline_checkpoint_v0 is not None else None,
            "baseline_metrics": baseline,
            **parity,
            **benefit,
        },
        "failure_flags": sorted(set(failure_flags)),
        "feature_alignment": dataset.feature_alignment,
        "data_manifest_hashes": dataset_manifest_hashes(dataset_manifest=dataset_manifest, pack_specs=pack_specs),
        "sources": sources,
        "route_out_metrics": _route_out_proxy(sources),
        "representation_pretraining_only": True,
        "control_safe": False,
        "replay_only": True,
        "not_executed": True,
        "product_training_approved": False,
    }
    write_json(out_path, metrics, pretty=True)
    if report_json is not None or report_md is not None:
        report = _goal_report(
            metrics=metrics,
            dataset=dataset,
            pack_specs=pack_specs,
            train_metrics=train_metrics,
            baseline_checkpoint_v0=baseline_checkpoint_v0,
        )
        if report_json is not None:
            write_json(report_json, report, pretty=True)
        if report_md is not None:
            _write_report_md(report_md, report)
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
    window_length: int,
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
            sensor_context_mode=sensor_context_mode,
            missing_pose_behavior=missing_pose_behavior,  # type: ignore[arg-type]
        )
    return SpatialTemporalMultiPackDataset(
        pack_specs,
        split=split,
        window_length=window_length,
        sensor_context_mode=sensor_context_mode,
        missing_pose_behavior=missing_pose_behavior,  # type: ignore[arg-type]
    )


def _source_metrics(
    model: torch.nn.Module,
    dataset: SpatialTemporalTrainDataset | SpatialTemporalMultiPackDataset,
    *,
    device: torch.device,
    batch_size: int,
    split: str | None,
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
                "val_loss": float(values["loss"]),
                "current_bev_iou_or_proxy": float(values["current_bev_iou_or_proxy"]),
                "fused_memory_bev_iou_or_proxy": float(values["fused_memory_bev_iou_or_proxy"]),
                "unknown_reduction_vs_current": float(values["unknown_reduction_vs_current"]),
                "temporal_reprojection_consistency_iou": float(values["temporal_reprojection_consistency_iou"]),
                "pose_delta_rmse": values["pose_delta_rmse"],
                "memory_warp_valid_fraction": float(values["memory_warp_valid_fraction"]),
                "valid_warp_fraction": float(values["valid_warp_fraction"]),
                "memory_reset_fraction": float(values["memory_reset_fraction"]),
                "update_mask_coverage_mean": float(values["update_mask_coverage_mean"]),
                "memory_overwrite_fraction": float(values["memory_overwrite_fraction"]),
                "control_safe": False,
            }
        )
    return records


@torch.no_grad()
def _v0_baseline_metrics(
    checkpoint: str | Path,
    dataset: SpatialTemporalTrainDataset | SpatialTemporalMultiPackDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model, _payload = load_v0_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()
    ious: list[float] = []
    frame_count = 0
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    for batch in loader:
        batch = batch_to_device(batch, device)
        batch_size_value, steps = int(batch["features"].shape[0]), int(batch["features"].shape[1])
        features = batch["features"].reshape(batch_size_value * steps, *batch["features"].shape[2:])
        timestamp_s = batch["timestamp_s"].reshape(batch_size_value * steps, 1)
        sensor_mask = batch["sensor_mask"].reshape(batch_size_value * steps, batch["sensor_mask"].shape[-1])
        outputs = model(features, timestamp_s, sensor_mask)
        probs = torch.sigmoid(outputs["bev_logits"]).reshape(
            batch_size_value,
            steps,
            *outputs["bev_logits"].shape[1:],
        )
        labels = batch["bev_labels"] > 0.5
        mask = batch["bev_label_mask"] > 0.0
        for channel in range(int(probs.shape[2])):
            pred = probs[:, :, channel] > 0.5
            label = labels[:, :, channel] & mask[:, :, channel]
            union = torch.count_nonzero(pred | label)
            if int(union.item()) == 0:
                continue
            intersection = torch.count_nonzero(pred & label)
            ious.append(float((intersection.float() / union.float()).detach().cpu()))
        frame_count += batch_size_value * steps
    value = float(sum(ious) / len(ious)) if ious else 0.0
    return {
        "baseline_name": "SpatialMemoryNetV0_current_frame_only",
        "checkpoint": Path(checkpoint).as_posix(),
        "current_bev_iou_or_proxy": value,
        "fused_memory_bev_iou_or_proxy": value,
        "temporal_reprojection_consistency_iou": 0.0,
        "unknown_reduction_vs_current": 0.0,
        "frame_count": frame_count,
    }


def _route_out_proxy(sources: list[dict[str, Any]]) -> dict[str, Any]:
    if len(sources) <= 1:
        return {
            "available": False,
            "reason": "single_source_eval_no_route_out_split",
        }
    return {
        "available": True,
        "method": "per_source_eval_proxy_not_retrained_leave_one_route_out",
        "sources": sources,
    }


def _goal_report(
    *,
    metrics: dict[str, Any],
    dataset: SpatialTemporalTrainDataset | SpatialTemporalMultiPackDataset,
    pack_specs: list[SpatialPackSpec],
    train_metrics: dict[str, Any],
    baseline_checkpoint_v0: str | Path | None,
) -> dict[str, Any]:
    source_reports = []
    source_datasets = dataset.source_datasets() if isinstance(dataset, SpatialTemporalMultiPackDataset) else [dataset]
    for source_dataset in source_datasets:
        manifest = source_dataset.base.manifest
        source_reports.append(
            {
                "source_name": source_dataset.source_name,
                "dataset": source_dataset.base.root.as_posix(),
                "features": next(
                    (spec.feature_dir.as_posix() for spec in pack_specs if spec.source_name == source_dataset.source_name),
                    None,
                ),
                "sequence_count": len({str(record.get("sequence_id", "")) for record in source_dataset.base.records}),
                "frame_count": len(source_dataset.base.records),
                "window_count": len(source_dataset),
                "robot_supervision_grade": source_dataset.robot_supervision_grade,
                "dataset_frame_type": manifest.get("dataset_frame_type"),
                "robot_frame_truth": manifest.get("robot_frame_truth"),
                "pose_label_count": manifest.get("pose_label_count"),
                "pose_label_frame": manifest.get("pose_label_frame"),
                "source_depth_teacher_name": manifest.get("source_depth_teacher_name"),
                "license_review_status": manifest.get("license_review_status", "pending_human_review"),
                "control_safe": False,
            }
        )
    skipped = []
    if baseline_checkpoint_v0 is None:
        skipped.append(
            {
                "artifact": "SpatialMemoryNetV0 baseline checkpoint",
                "reason": "not supplied to eval_spatial_v1; used memory-disabled v1 proxy baseline",
            }
        )
    return {
        "schema_version": "homebrain.goal12a_spatial_memory_v1_report.v0",
        "goal": "12A SpatialMemoryNetV1 real temporal egocentric memory",
        "data_sources": source_reports,
        "route_frame_window_counts": {
            "source_count": len(source_reports),
            "frame_count": sum(int(item["frame_count"]) for item in source_reports),
            "window_count": sum(int(item["window_count"]) for item in source_reports),
        },
        "real_public_robot_frame_data": {
            "used": any(item.get("dataset_frame_type") == "public_robot_mounted" for item in source_reports),
            "sources": [
                item["source_name"]
                for item in source_reports
                if item.get("dataset_frame_type") == "public_robot_mounted"
            ],
            "license_review_status": "pending_human_review",
            "product_training_approved": False,
        },
        "temporal_supervision": {
            "merge_policy": "warp_previous_label_memory_then_current_observed_overwrite",
            "starts_unknown": True,
            "uses_model_predictions_as_labels": False,
            "missing_pose_behavior": metrics["missing_pose_behavior"],
            "control_safe": False,
        },
        "train_command": train_metrics.get("train_command"),
        "eval_command": metrics.get("eval_command"),
        "metrics": metrics,
        "baseline_comparison": metrics["baseline_comparison"],
        "current_bev_parity_pass": metrics["baseline_comparison"]["current_bev_parity_pass"],
        "memory_benefit_pass": metrics["baseline_comparison"]["memory_benefit_pass"],
        "failure_flags": metrics["failure_flags"],
        "skipped_artifacts": skipped,
        "blockers": [
            "current BEV parity gate did not pass"
            if not metrics["baseline_comparison"]["current_bev_parity_pass"]
            else (
                "memory benefit gate did not pass"
                if not metrics["baseline_comparison"]["memory_benefit_pass"]
                else "none"
            )
        ],
        "safety_flags": {
            "control_safe": False,
            "replay_only": True,
            "not_executed": True,
            "product_training_approved": False,
            "cmd_vel_emitted": False,
        },
        "next_goal": "Train/evaluate v1 on longer route-out windows with reviewed robot-frame data before any control-facing claim.",
    }


def _write_report_md(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Goal 12A SpatialMemoryNetV1 Report",
        "",
        f"- current_bev_parity_pass: `{str(report['current_bev_parity_pass']).lower()}`",
        f"- memory_benefit_pass: `{str(report['memory_benefit_pass']).lower()}`",
        f"- sources: `{report['route_frame_window_counts']['source_count']}`",
        f"- frames: `{report['route_frame_window_counts']['frame_count']}`",
        f"- windows: `{report['route_frame_window_counts']['window_count']}`",
        f"- current_bev_iou_or_proxy: `{report['metrics']['current_bev_iou_or_proxy']}`",
        f"- fused_memory_bev_iou_or_proxy: `{report['metrics']['fused_memory_bev_iou_or_proxy']}`",
        f"- temporal_reprojection_consistency_iou: `{report['metrics']['temporal_reprojection_consistency_iou']}`",
        f"- update_mask_coverage_mean: `{report['metrics']['update_mask_coverage_mean']}`",
        f"- memory_overwrite_fraction: `{report['metrics']['memory_overwrite_fraction']}`",
        f"- pose_warp_source: `{report['metrics']['pose_warp_source']}`",
        f"- pose_delta_rmse: `{report['metrics']['pose_delta_rmse']}`",
        f"- control_safe: `false`",
        f"- cmd_vel_emitted: `false`",
        "",
        "## Data Sources",
    ]
    for item in report["data_sources"]:
        lines.append(
            f"- `{item['source_name']}`: frames `{item['frame_count']}`, windows `{item['window_count']}`, "
            f"pose `{item['pose_label_frame']}`, robot_frame_truth `{item['robot_frame_truth']}`"
        )
    lines.extend(
        [
            "",
            "## Baseline",
            f"- baseline: `{report['baseline_comparison']['baseline_metrics']['baseline_name']}`",
            f"- failure_flags: `{', '.join(report['failure_flags'])}`",
            "",
            "## Safety",
            "- Replay/eval only. `control_safe=false`, `product_training_approved=false`, and no `cmd_vel` is emitted.",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a SpatialMemoryNet v1 checkpoint on temporal SpatialTrainPack windows.")
    parser.add_argument("--checkpoint", required=True, help="SpatialMemoryNet v1 checkpoint.")
    parser.add_argument("--dataset", default=None, help="SpatialTrainPack directory.")
    parser.add_argument("--features", default=None, help="DINO feature artifact directory; defaults to checkpoint metadata.")
    parser.add_argument("--dataset-manifest", default=None, help="JSON manifest listing SpatialTrainPacks and DINO features.")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate; use an empty string for all examples.")
    parser.add_argument("--out", required=True, help="Output eval JSON.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-length", type=int, default=None)
    parser.add_argument("--sensor-context-mode", choices=("masks", "odom"), default=None)
    parser.add_argument("--baseline-checkpoint-v0", default=None, help="Optional SpatialMemoryNet v0 checkpoint baseline.")
    parser.add_argument("--report-json", default=None, help="Optional Goal 12A report JSON path.")
    parser.add_argument("--report-md", default=None, help="Optional Goal 12A report Markdown path.")
    args = parser.parse_args(argv)
    split = args.split if args.split else None
    command = "python -m homebrain.train.eval_spatial_v1 " + " ".join(sys.argv[1:])
    metrics = eval_spatial_v1(
        checkpoint=args.checkpoint,
        dataset_dir=args.dataset,
        feature_dir=args.features,
        dataset_manifest=args.dataset_manifest,
        split=split,
        out_path=args.out,
        device_name=args.device,
        batch_size=args.batch_size,
        window_length=args.window_length,
        sensor_context_mode=args.sensor_context_mode,
        baseline_checkpoint_v0=args.baseline_checkpoint_v0,
        report_json=args.report_json,
        report_md=args.report_md,
        command=command,
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
