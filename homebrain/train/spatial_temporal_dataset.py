from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from homebrain.brain.spatial_memory_v1 import MissingPoseBehavior, warp_memory_se2
from homebrain.train.spatial_dataset import (
    BEV_OUTPUT_CHANNELS,
    SpatialPackSpec,
    SpatialTrainDataset,
)

TEMPORAL_TARGET_SCHEMA_VERSION = "homebrain.spatial_memory_v1_temporal_targets.v0"
TEMPORAL_MERGE_POLICY = "warp_previous_label_memory_then_current_observed_overwrite"


@dataclass(frozen=True)
class TemporalWindowRecord:
    source_name: str
    sequence_id: str
    camera_id: str
    split: str
    split_unit_id: str
    base_indices: tuple[int, ...]


class SpatialTemporalTrainDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        feature_dir: str | Path | None = None,
        split: str | None = None,
        source_name: str | None = None,
        window_length: int = 4,
        tiny_overfit: bool = False,
        tiny_limit: int = 8,
        allow_tiny_fixture_features: bool = False,
        sensor_context_mode: str = "masks",
        missing_pose_behavior: MissingPoseBehavior = "reset",
    ) -> None:
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        self.base = SpatialTrainDataset(
            dataset_dir,
            feature_dir=feature_dir,
            split=None,
            source_name=source_name,
            tiny_overfit=False,
            allow_tiny_fixture_features=allow_tiny_fixture_features,
            sensor_context_mode=sensor_context_mode,
        )
        self.window_length = int(window_length)
        self.missing_pose_behavior = missing_pose_behavior
        self.records = self._build_windows(split=split)
        if tiny_overfit:
            preferred = [record for record in self.records if record.split == "train"]
            self.records = (preferred or self.records)[:tiny_limit]
        if not self.records:
            raise ValueError(f"no temporal windows found in {dataset_dir} for split={split!r}")
        self.grid_shape = self.base.grid_shape
        self.feature_shape = self.base.feature_shape
        self.source_name = self.base.source_name
        self.robot_supervision_grade = self.base.robot_supervision_grade
        self.feature_alignment = self.base.feature_alignment
        self.target_metadata = {
            "schema_version": TEMPORAL_TARGET_SCHEMA_VERSION,
            "merge_policy": TEMPORAL_MERGE_POLICY,
            "starts_unknown": True,
            "uses_model_predictions_as_labels": False,
            "missing_pose_behavior": missing_pose_behavior,
            "control_safe": False,
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.records[index]
        samples = [self.base[base_index] for base_index in window.base_indices]
        features = torch.stack([sample["features"] for sample in samples], dim=0)
        cls_feature = torch.stack([sample["cls_feature"] for sample in samples], dim=0)
        feature_mask = torch.stack([sample["feature_mask"] for sample in samples], dim=0)
        timestamp_s = torch.stack([sample["timestamp_s"] for sample in samples], dim=0)
        sensor_mask = torch.stack([sample["sensor_mask"] for sample in samples], dim=0)
        bev_labels = torch.stack([sample["bev_labels"] for sample in samples], dim=0)
        bev_label_mask = torch.stack([sample["bev_label_mask"] for sample in samples], dim=0)
        pose_delta = torch.stack([sample["pose_delta"] for sample in samples], dim=0)
        pose_mask = torch.stack([sample["pose_mask"] for sample in samples], dim=0)
        frame_ids = torch.stack([sample["frame_id"] for sample in samples], dim=0)
        timestamp_ns = torch.stack([sample["timestamp_ns"] for sample in samples], dim=0)

        pose_to_current = torch.zeros_like(pose_delta)
        pose_to_current_mask = torch.zeros_like(pose_mask)
        records = [self.base.records[base_index] for base_index in window.base_indices]
        for local_index in range(1, len(records)):
            previous = records[local_index - 1]
            current = records[local_index]
            target_frame_id = int(previous.get("pose_label_target_frame_id", -1))
            if target_frame_id == int(current["frame_id"]) and float(previous.get("pose_delta_mask", 0.0)) > 0.0:
                pose_to_current[local_index] = pose_delta[local_index - 1]
                pose_to_current_mask[local_index] = pose_mask[local_index - 1]

        observation_mask = _observation_mask_from_labels(bev_labels)
        temporal_targets = build_temporal_memory_targets(
            bev_labels.unsqueeze(0),
            observation_mask.unsqueeze(0),
            pose_to_current.unsqueeze(0),
            pose_to_current_mask.unsqueeze(0),
            meters_per_cell=_meters_per_cell(self.base.manifest),
            missing_pose_behavior=self.missing_pose_behavior,
        )
        fused_targets = temporal_targets["fused_memory_targets"].squeeze(0)
        fused_observed = temporal_targets["observed_mask_targets"].squeeze(0)
        fused_mask = torch.maximum(bev_label_mask, fused_observed.repeat(1, len(BEV_OUTPUT_CHANNELS), 1, 1))

        pose_sources = sorted({str(record.get("pose_label_source", "missing")) for record in records})
        pose_frames = sorted({str(record.get("pose_label_frame", "none")) for record in records})
        return {
            "features": features,
            "cls_feature": cls_feature,
            "feature_mask": feature_mask,
            "timestamp_s": timestamp_s,
            "sensor_mask": sensor_mask,
            "bev_labels": bev_labels,
            "bev_label_mask": bev_label_mask,
            "fused_memory_targets": fused_targets,
            "fused_memory_target_mask": fused_mask,
            "observation_mask": observation_mask,
            "observed_mask_targets": fused_observed,
            "pose_delta": pose_delta,
            "pose_mask": pose_mask,
            "pose_delta_to_current": pose_to_current,
            "pose_delta_to_current_mask": pose_to_current_mask,
            "frame_id": frame_ids,
            "timestamp_ns": timestamp_ns,
            "window_index": torch.tensor(index, dtype=torch.int64),
            "source_name": window.source_name,
            "sequence_id": window.sequence_id,
            "camera_id": window.camera_id,
            "split": window.split,
            "split_unit_id": window.split_unit_id,
            "target_merge_policy": TEMPORAL_MERGE_POLICY,
            "target_pose_sources": ",".join(pose_sources),
            "target_pose_frames": ",".join(pose_frames),
            "target_uses_model_predictions": False,
            "control_safe": torch.tensor(False, dtype=torch.bool),
        }

    def _build_windows(self, *, split: str | None) -> list[TemporalWindowRecord]:
        groups: dict[tuple[str, str, str, str, str], list[int]] = {}
        for index, record in enumerate(self.base.records):
            key = (
                str(record.get("source_name") or self.base.source_name),
                str(record.get("sequence_id", "")),
                str(record.get("camera_id", "")),
                str(record.get("split", "")),
                str(record.get("split_unit_id", "")),
            )
            if split is not None and key[3] != split:
                continue
            groups.setdefault(key, []).append(index)

        windows: list[TemporalWindowRecord] = []
        for key, indices in sorted(groups.items()):
            source_name, sequence_id, camera_id, split_name, split_unit_id = key
            ordered = sorted(
                indices,
                key=lambda base_index: (
                    int(self.base.records[base_index]["timestamp_ns"]),
                    int(self.base.records[base_index]["frame_id"]),
                ),
            )
            contiguous: list[int] = []
            previous_frame_id: int | None = None
            for base_index in ordered:
                frame_id = int(self.base.records[base_index]["frame_id"])
                starts_new = previous_frame_id is not None and frame_id != previous_frame_id + 1
                if starts_new:
                    windows.extend(
                        _windows_from_contiguous(
                            contiguous,
                            source_name=source_name,
                            sequence_id=sequence_id,
                            camera_id=camera_id,
                            split_name=split_name,
                            split_unit_id=split_unit_id,
                            window_length=self.window_length,
                        )
                    )
                    contiguous = []
                contiguous.append(base_index)
                previous_frame_id = frame_id
            windows.extend(
                _windows_from_contiguous(
                    contiguous,
                    source_name=source_name,
                    sequence_id=sequence_id,
                    camera_id=camera_id,
                    split_name=split_name,
                    split_unit_id=split_unit_id,
                    window_length=self.window_length,
                )
            )
        return windows

    def source_datasets(self) -> list["SpatialTemporalTrainDataset"]:
        return [self]


class SpatialTemporalMultiPackDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        pack_specs: list[SpatialPackSpec],
        *,
        split: str | None = None,
        window_length: int = 4,
        tiny_overfit: bool = False,
        tiny_limit: int = 8,
        sensor_context_mode: str = "masks",
        missing_pose_behavior: MissingPoseBehavior = "reset",
    ) -> None:
        if not pack_specs:
            raise ValueError("dataset manifest must include at least one pack")
        self.datasets = [
            SpatialTemporalTrainDataset(
                spec.dataset_dir,
                feature_dir=spec.feature_dir,
                split=split,
                source_name=spec.source_name,
                window_length=window_length,
                tiny_overfit=tiny_overfit,
                tiny_limit=tiny_limit,
                sensor_context_mode=sensor_context_mode,
                missing_pose_behavior=missing_pose_behavior,
            )
            for spec in pack_specs
        ]
        self.datasets = [dataset for dataset in self.datasets if len(dataset) > 0]
        if not self.datasets:
            raise ValueError("no temporal windows found across dataset manifest")
        self.cumulative_sizes: list[int] = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)
        self.grid_shape = self.datasets[0].grid_shape
        self.feature_shape = self.datasets[0].feature_shape
        for dataset in self.datasets[1:]:
            if dataset.grid_shape != self.grid_shape:
                raise ValueError(f"mixed BEV grid shapes are not supported: {dataset.grid_shape} != {self.grid_shape}")
            if dataset.feature_shape[-1] != self.feature_shape[-1]:
                raise ValueError(
                    "mixed DINO feature dimensions are not supported: "
                    f"{dataset.feature_shape[-1]} != {self.feature_shape[-1]}"
                )
        grades = sorted({dataset.robot_supervision_grade for dataset in self.datasets})
        self.robot_supervision_grade = grades[0] if len(grades) == 1 else "mixed"
        names = sorted({dataset.source_name for dataset in self.datasets})
        self.source_name = names[0] if len(names) == 1 else "combined"
        self.feature_alignment = {
            "sources": [dataset.feature_alignment for dataset in self.datasets],
            "example_count": len(self),
            "aligned_example_count": sum(int(dataset.feature_alignment["aligned_example_count"]) for dataset in self.datasets),
            "missing_feature_count": sum(int(dataset.feature_alignment["missing_feature_count"]) for dataset in self.datasets),
        }

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_index = bisect_right(self.cumulative_sizes, index)
        previous = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        return self.datasets[dataset_index][index - previous]

    def source_datasets(self) -> list[SpatialTemporalTrainDataset]:
        return list(self.datasets)


def build_temporal_memory_targets(
    bev_labels: torch.Tensor,
    observation_mask: torch.Tensor,
    pose_delta_to_current: torch.Tensor,
    pose_delta_to_current_mask: torch.Tensor,
    *,
    meters_per_cell: float,
    missing_pose_behavior: MissingPoseBehavior = "reset",
) -> dict[str, torch.Tensor]:
    if bev_labels.ndim != 5:
        raise ValueError("bev_labels must be BTCHW")
    batch, steps, channels, height, width = bev_labels.shape
    if channels != len(BEV_OUTPUT_CHANNELS):
        raise ValueError(f"expected {len(BEV_OUTPUT_CHANNELS)} BEV channels")
    memory = _unknown_template(batch, channels, height, width, device=bev_labels.device, dtype=bev_labels.dtype)
    observed = torch.zeros((batch, 1, height, width), dtype=bev_labels.dtype, device=bev_labels.device)
    targets: list[torch.Tensor] = []
    observed_targets: list[torch.Tensor] = []
    warp_valid: list[torch.Tensor] = []
    resets: list[torch.Tensor] = []
    for step_index in range(steps):
        if step_index > 0:
            pose = pose_delta_to_current[:, step_index]
            pose_mask = pose_delta_to_current_mask[:, step_index]
            warp = warp_memory_se2(
                memory,
                pose,
                pose_mask,
                meters_per_cell=meters_per_cell,
                missing_pose_behavior=missing_pose_behavior,
            )
            observed_warp = warp_memory_se2(
                observed,
                pose,
                pose_mask,
                meters_per_cell=meters_per_cell,
                missing_pose_behavior=missing_pose_behavior,
                mode="nearest",
            )
            memory = warp.tensor
            observed = observed_warp.tensor.clamp(0.0, 1.0)
            unknown = _unknown_template(batch, channels, height, width, device=bev_labels.device, dtype=bev_labels.dtype)
            memory = torch.where(observed > 0.0, memory, unknown)
            warp_valid.append(warp.pose_warp_valid)
            resets.append(warp.memory_reset)
        else:
            warp_valid.append(torch.zeros((batch,), dtype=torch.bool, device=bev_labels.device))
            resets.append(torch.ones((batch,), dtype=torch.bool, device=bev_labels.device))

        current_observed = observation_mask[:, step_index].view(batch, 1, height, width).to(bev_labels.dtype).clamp(0.0, 1.0)
        current_labels = bev_labels[:, step_index]
        memory = torch.where(current_observed > 0.0, current_labels, memory)
        observed = torch.maximum(observed, current_observed)
        targets.append(memory)
        observed_targets.append(observed)

    return {
        "fused_memory_targets": torch.stack(targets, dim=1),
        "observed_mask_targets": torch.stack(observed_targets, dim=1),
        "pose_warp_valid_targets": torch.stack(warp_valid, dim=1),
        "memory_reset_targets": torch.stack(resets, dim=1),
        "schema_version": torch.tensor(0),
    }


def _windows_from_contiguous(
    indices: list[int],
    *,
    source_name: str,
    sequence_id: str,
    camera_id: str,
    split_name: str,
    split_unit_id: str,
    window_length: int,
) -> list[TemporalWindowRecord]:
    if len(indices) < window_length:
        return []
    return [
        TemporalWindowRecord(
            source_name=source_name,
            sequence_id=sequence_id,
            camera_id=camera_id,
            split=split_name,
            split_unit_id=split_unit_id,
            base_indices=tuple(indices[start : start + window_length]),
        )
        for start in range(0, len(indices) - window_length + 1)
    ]


def _observation_mask_from_labels(labels: torch.Tensor) -> torch.Tensor:
    observed = torch.clamp(labels[..., 0:1, :, :] + labels[..., 1:2, :, :], 0.0, 1.0)
    return observed.to(labels.dtype)


def _unknown_template(
    batch: int,
    channels: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    memory = torch.zeros((batch, channels, height, width), dtype=dtype, device=device)
    memory[:, BEV_OUTPUT_CHANNELS.index("unknown")] = 1.0
    return memory


def _meters_per_cell(manifest: dict[str, Any]) -> float:
    camera_config = manifest.get("camera_config")
    if isinstance(camera_config, dict):
        value = camera_config.get("meters_per_cell")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0.0:
            return float(value)
    return 0.05
