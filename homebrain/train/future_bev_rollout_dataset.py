from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from homebrain.brain.spatial_memory_v1 import warp_memory_se2
from homebrain.data.spatial_dataset import load_example_npz, read_json
from homebrain.datasets.openloris_scene import OPENLORIS_ROUTE_ASSOCIATIONS_FILE
from homebrain.datasets.tum_rgbd import TUM_RGBD_ROUTE_ASSOCIATIONS_FILE
from homebrain.messages.schema import JsonDict
from homebrain.policies.candidate_trajectories import CandidateTrajectory, generate_default_candidates
from homebrain.train.spatial_dataset import BEV_OUTPUT_CHANNELS, DINOFeatureStore

FUTURE_ROLLOUT_PACK_SCHEMA_VERSION = "homebrain.future_bev_rollout_pack.v1"
FUTURE_ROLLOUT_EXAMPLE_SCHEMA_VERSION = "homebrain.future_bev_rollout_example.v1"
FUTURE_BEV_CHANNELS: tuple[str, ...] = ("free", "occupied", "unknown")
FUTURE_DERIVED_CHANNELS: tuple[str, ...] = (
    "newly_observed",
    "persistent_obstacle",
    "cleared_or_changed",
)
DEFAULT_FUTURE_HORIZONS_S: tuple[float, ...] = (0.5, 1.0, 2.0)
ROLLOUT_PROVENANCE_FLAGS: JsonDict = {
    "replay_only": True,
    "not_executed": True,
    "control_safe": False,
    "weak_label": True,
    "product_training_approved": False,
    "raw_pwm_emitted": False,
}


@dataclass(frozen=True)
class RoutePoseSample:
    frame_id: int
    timestamp_ns: int
    pose: JsonDict
    pose_frame: str
    pose_source: str


@dataclass(frozen=True)
class SpatialRolloutFrame:
    root: Path
    record: JsonDict
    example_path: Path
    sequence_id: str
    camera_id: str
    frame_id: int
    timestamp_ns: int
    split: str
    split_unit_id: str
    source_name: str
    source_family: str
    supervision_grade: str
    dataset_frame_type: str
    robot_frame_truth: bool
    action_supervision_ok: bool
    pose_sample: RoutePoseSample | None


@dataclass(frozen=True)
class FutureRolloutTargets:
    arrays: dict[str, np.ndarray]
    metadata: JsonDict


class FutureBEVRolloutDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        pack_dir: str | Path,
        *,
        split: str | None = None,
        tiny_overfit: bool = False,
        tiny_limit: int = 8,
    ) -> None:
        self.root = Path(pack_dir)
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest.get("control_safe") is not False:
            raise ValueError("Future BEV rollout packs must remain control_safe=false")
        examples = self.manifest.get("examples")
        if not isinstance(examples, list):
            raise ValueError("Future BEV rollout manifest examples must be a list")
        self.records = [record for record in examples if isinstance(record, dict)]
        if split is not None:
            self.records = [record for record in self.records if str(record.get("split", "")) == split]
        if tiny_overfit:
            train_records = [record for record in self.records if str(record.get("split", "")) == "train"]
            self.records = (train_records or self.records)[:tiny_limit]
        if not self.records:
            raise ValueError(f"no Future BEV rollout examples found in {self.root} for split={split!r}")
        grid_shape = self.manifest.get("grid_shape")
        if not isinstance(grid_shape, list) or len(grid_shape) != 2:
            raise ValueError("Future BEV rollout manifest must include grid_shape")
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.horizons_s = tuple(float(value) for value in self.manifest.get("horizons_s", DEFAULT_FUTURE_HORIZONS_S))
        self.candidate_ids = tuple(str(value) for value in self.manifest.get("candidate_ids", []))
        self.feature_shape = self._first_feature_shape()
        self.sensor_dim = 4

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path_value = record.get("example_path")
        if not isinstance(path_value, str):
            raise ValueError(f"rollout example is missing example_path: {record}")
        example = load_example_npz(self.root / path_value)
        features, cls_feature, feature_mask = _feature_tensors_from_example(example)
        current_bev = np.stack(
            [
                np.asarray(example["current_bev_free"], dtype=np.float32),
                np.asarray(example["current_bev_occupied"], dtype=np.float32),
                np.asarray(example["current_bev_unknown"], dtype=np.float32),
                np.asarray(example["current_bev_traversable"], dtype=np.float32),
                np.asarray(example["current_bev_risky"], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        future_bev = np.stack(
            [
                np.asarray(example["future_free"], dtype=np.float32),
                np.asarray(example["future_occupied"], dtype=np.float32),
                np.asarray(example["future_unknown"], dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        derived = np.stack(
            [
                np.asarray(example["newly_observed"], dtype=np.float32),
                np.asarray(example["persistent_obstacle"], dtype=np.float32),
                np.asarray(example["cleared_or_changed"], dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        return {
            "current_bev": torch.from_numpy(current_bev),
            "features": torch.from_numpy(np.transpose(features, (2, 0, 1)).copy()),
            "cls_feature": torch.from_numpy(cls_feature.copy()),
            "feature_mask": torch.from_numpy(feature_mask),
            "sensor_mask": torch.from_numpy(np.asarray(example["sensor_mask"], dtype=np.float32)),
            "future_bev": torch.from_numpy(future_bev),
            "future_derived": torch.from_numpy(derived),
            "future_valid_mask": torch.from_numpy(np.asarray(example["future_valid_mask"], dtype=np.float32)[:, None, :, :]),
            "horizon_valid": torch.from_numpy(np.asarray(example["horizon_valid"], dtype=np.float32)),
            "pose_delta_to_future": torch.from_numpy(np.asarray(example["pose_delta_to_future"], dtype=np.float32)),
            "candidate_collision": torch.from_numpy(np.asarray(example["candidate_collision"], dtype=np.float32)),
            "candidate_future_collision": torch.from_numpy(
                np.asarray(example.get("candidate_future_collision", example["candidate_collision"]), dtype=np.float32)
            ),
            "candidate_unsafe_now": torch.from_numpy(
                np.asarray(example.get("candidate_unsafe_now", np.zeros_like(example["candidate_collision"])), dtype=np.float32)
            ),
            "candidate_unknown_exposure": torch.from_numpy(
                np.asarray(example["candidate_unknown_exposure"], dtype=np.float32)
            ),
            "candidate_new_area_gain": torch.from_numpy(np.asarray(example["candidate_new_area_gain"], dtype=np.float32)),
            "candidate_progress": torch.from_numpy(np.asarray(example["candidate_progress"], dtype=np.float32)),
            "candidate_oracle_cost": torch.from_numpy(
                _candidate_oracle_cost_from_example(example).astype(np.float32)
            ),
            "candidate_valid_mask": torch.from_numpy(np.asarray(example["candidate_valid_mask"], dtype=np.float32)),
            "frame_id": torch.tensor(int(np.asarray(example["frame_id"]).item()), dtype=torch.int64),
            "timestamp_ns": torch.tensor(int(np.asarray(example["timestamp_ns"]).item()), dtype=torch.int64),
            "split": str(record.get("split", "")),
            "split_unit_id": str(record.get("split_unit_id", "")),
            "source_name": str(record.get("source_name", "")),
            "sequence_id": str(record.get("sequence_id", "")),
            "control_safe": torch.tensor(False, dtype=torch.bool),
            "replay_only": torch.tensor(True, dtype=torch.bool),
            "not_executed": torch.tensor(True, dtype=torch.bool),
        }

    def source_datasets(self) -> list["FutureBEVRolloutDataset"]:
        return [self]

    def _first_feature_shape(self) -> tuple[int, int, int]:
        first = self.records[0]
        path_value = first.get("example_path")
        if not isinstance(path_value, str):
            return (1, 1, 1)
        example = load_example_npz(self.root / path_value)
        features, _cls, _mask = _feature_tensors_from_example(example)
        return tuple(int(value) for value in features.shape)


def load_spatial_rollout_frames(spatial_root: str | Path) -> tuple[list[SpatialRolloutFrame], JsonDict]:
    root = Path(spatial_root)
    manifest = read_json(root / "manifest.json")
    if manifest.get("control_safe") is not False:
        raise ValueError(f"SpatialTrainPack must remain control_safe=false: {root}")
    raw_records = manifest.get("examples") or manifest.get("frames")
    if not isinstance(raw_records, list):
        raise ValueError(f"SpatialTrainPack manifest examples must be a list: {root}")
    pose_samples = _route_pose_samples(manifest)
    default_source_name = str(manifest.get("source_name") or manifest.get("source_segment_id") or root.name)
    default_source_family = str(manifest.get("source_family") or "unknown")
    default_grade = str(manifest.get("supervision_grade") or manifest.get("robot_supervision_grade") or "unknown")
    frames: list[SpatialRolloutFrame] = []
    for record in raw_records:
        if not isinstance(record, dict) or not isinstance(record.get("example_path"), str):
            continue
        example_path = root / str(record["example_path"])
        frame_id = int(record["frame_id"])
        frames.append(
            SpatialRolloutFrame(
                root=root,
                record=record,
                example_path=example_path,
                sequence_id=str(record.get("sequence_id", manifest.get("source_segment_id", root.name))),
                camera_id=str(record.get("camera_id", "front_rgb")),
                frame_id=frame_id,
                timestamp_ns=int(record.get("timestamp_ns", 0)),
                split=str(record.get("split", "train")),
                split_unit_id=str(record.get("split_unit_id", record.get("sequence_id", root.name))),
                source_name=str(record.get("source_name", default_source_name)),
                source_family=str(record.get("source_family", default_source_family)),
                supervision_grade=str(record.get("supervision_grade", default_grade)),
                dataset_frame_type=str(record.get("dataset_frame_type", manifest.get("dataset_frame_type", "unknown"))),
                robot_frame_truth=bool(record.get("robot_frame_truth", manifest.get("robot_frame_truth", False))),
                action_supervision_ok=bool(
                    record.get("action_supervision_ok", manifest.get("action_supervision_ok", False))
                ),
                pose_sample=pose_samples.get(frame_id),
            )
        )
    frames.sort(key=lambda item: (item.source_name, item.sequence_id, item.camera_id, item.timestamp_ns, item.frame_id))
    return frames, manifest


def build_future_rollout_targets_for_frame(
    *,
    frame: SpatialRolloutFrame,
    sequence_frames: list[SpatialRolloutFrame],
    horizons_s: tuple[float, ...],
    candidates: list[CandidateTrajectory],
    meters_per_cell: float,
    feature_store: DINOFeatureStore | None = None,
) -> FutureRolloutTargets:
    current = load_example_npz(frame.example_path)
    current_bev = _bev_stack(current)
    current_free = current_bev[0]
    current_occupied = current_bev[1]
    current_unknown = current_bev[2]
    sensor_mask = np.asarray(
        [
            _scalar_float(current, "pose_delta_mask", 0.0),
            _scalar_float(current, "action_label_mask", 0.0),
            _scalar_float(current, "imu_label_mask", 0.0),
            _scalar_float(current, "wheel_label_mask", 0.0),
        ],
        dtype=np.float32,
    )
    height, width = current_free.shape
    target_frames = [_future_target_frame(sequence_frames, frame.timestamp_ns, horizon) for horizon in horizons_s]
    future_free: list[np.ndarray] = []
    future_occupied: list[np.ndarray] = []
    future_unknown: list[np.ndarray] = []
    future_valid: list[np.ndarray] = []
    newly_observed: list[np.ndarray] = []
    persistent_obstacle: list[np.ndarray] = []
    cleared_or_changed: list[np.ndarray] = []
    pose_delta_to_future: list[tuple[float, float, float]] = []
    pose_delta_future_to_current: list[tuple[float, float, float]] = []
    horizon_valid: list[float] = []
    target_frame_ids: list[int] = []
    invalid_reasons: list[str] = []

    for target in target_frames:
        if frame.pose_sample is None:
            future = _empty_future(height, width)
            reason = "current_pose_missing"
        elif target is None:
            future = _empty_future(height, width)
            reason = "future_frame_missing"
        elif target.pose_sample is None:
            future = _empty_future(height, width)
            reason = "future_pose_missing"
        else:
            target_example = load_example_npz(target.example_path)
            target_bev = _bev_stack(target_example)[:3]
            current_to_future = _relative_delta(frame.pose_sample.pose, target.pose_sample.pose, pose_frame=frame.pose_sample.pose_frame)
            future_to_current = _relative_delta(target.pose_sample.pose, frame.pose_sample.pose, pose_frame=frame.pose_sample.pose_frame)
            warped, valid_mask = warp_future_bev_to_current(
                target_bev,
                future_to_current,
                meters_per_cell=meters_per_cell,
            )
            future = {
                "free": warped[0],
                "occupied": warped[1],
                "unknown": warped[2],
                "valid": valid_mask,
                "pose_delta_to_future": current_to_future,
                "pose_delta_future_to_current": future_to_current,
                "target_frame_id": target.frame_id,
                "valid_scalar": 1.0,
            }
            reason = "valid"
        future_free.append(future["free"])
        future_occupied.append(future["occupied"])
        future_unknown.append(future["unknown"])
        future_valid.append(future["valid"])
        observed_future = np.clip(future["free"] + future["occupied"], 0.0, 1.0)
        new = ((current_unknown >= 0.5) & (observed_future >= 0.5) & (future["valid"] > 0.5)).astype(np.float32)
        persistent = ((current_occupied >= 0.5) & (future["occupied"] >= 0.5) & (future["valid"] > 0.5)).astype(np.float32)
        changed = (
            (np.abs(future["occupied"] - current_occupied) >= 0.5)
            & (future["valid"] > 0.5)
            & ((current_occupied >= 0.5) | (future["occupied"] >= 0.5))
        ).astype(np.float32)
        newly_observed.append(new)
        persistent_obstacle.append(persistent)
        cleared_or_changed.append(changed)
        pose_delta_to_future.append(tuple(float(value) for value in future["pose_delta_to_future"]))
        pose_delta_future_to_current.append(tuple(float(value) for value in future["pose_delta_future_to_current"]))
        horizon_valid.append(float(future["valid_scalar"]))
        target_frame_ids.append(int(future["target_frame_id"]))
        invalid_reasons.append(reason)

    stacked_free = np.stack(future_free, axis=0).astype(np.float32)
    stacked_occupied = np.stack(future_occupied, axis=0).astype(np.float32)
    stacked_unknown = np.stack(future_unknown, axis=0).astype(np.float32)
    stacked_valid = np.stack(future_valid, axis=0).astype(np.float32)
    stacked_new = np.stack(newly_observed, axis=0).astype(np.float32)
    stacked_persistent = np.stack(persistent_obstacle, axis=0).astype(np.float32)
    stacked_changed = np.stack(cleared_or_changed, axis=0).astype(np.float32)
    candidate_labels = candidate_outcome_labels(
        candidates=candidates,
        current_occupied=current_occupied,
        current_risky=current_bev[4],
        current_unknown=current_unknown,
        current_free=current_free,
        current_traversable=current_bev[3],
        future_occupied=stacked_occupied,
        future_unknown=stacked_unknown,
        newly_observed=stacked_new,
        future_free=stacked_free,
        horizon_valid=np.asarray(horizon_valid, dtype=np.float32),
    )
    feature_arrays = _feature_arrays(frame.record, feature_store)
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(FUTURE_ROLLOUT_EXAMPLE_SCHEMA_VERSION),
        "frame_id": np.asarray(frame.frame_id, dtype=np.int64),
        "timestamp_ns": np.asarray(frame.timestamp_ns, dtype=np.int64),
        "sequence_id": np.asarray(frame.sequence_id),
        "camera_id": np.asarray(frame.camera_id),
        "source_name": np.asarray(frame.source_name),
        "source_family": np.asarray(frame.source_family),
        "supervision_grade": np.asarray(frame.supervision_grade),
        "dataset_frame_type": np.asarray(frame.dataset_frame_type),
        "split": np.asarray(frame.split),
        "split_unit_id": np.asarray(frame.split_unit_id),
        "current_bev_free": current_bev[0].astype(np.float32),
        "current_bev_occupied": current_bev[1].astype(np.float32),
        "current_bev_unknown": current_bev[2].astype(np.float32),
        "current_bev_traversable": current_bev[3].astype(np.float32),
        "current_bev_risky": current_bev[4].astype(np.float32),
        "current_bev_confidence": np.asarray(current["bev_confidence"], dtype=np.float32),
        "sensor_mask": sensor_mask,
        "horizons_s": np.asarray(horizons_s, dtype=np.float32),
        "horizon_valid": np.asarray(horizon_valid, dtype=np.float32),
        "future_target_frame_id": np.asarray(target_frame_ids, dtype=np.int64),
        "pose_delta_to_future": np.asarray(pose_delta_to_future, dtype=np.float32),
        "pose_delta_future_to_current": np.asarray(pose_delta_future_to_current, dtype=np.float32),
        "future_free": stacked_free,
        "future_occupied": stacked_occupied,
        "future_unknown": stacked_unknown,
        "future_valid_mask": stacked_valid,
        "newly_observed": stacked_new,
        "persistent_obstacle": stacked_persistent,
        "cleared_or_changed": stacked_changed,
        "candidate_ids": np.asarray([candidate.id for candidate in candidates]),
        **candidate_labels,
        "replay_only": np.asarray(True, dtype=np.bool_),
        "not_executed": np.asarray(True, dtype=np.bool_),
        "control_safe": np.asarray(False, dtype=np.bool_),
        "weak_label": np.asarray(True, dtype=np.bool_),
        "product_training_approved": np.asarray(False, dtype=np.bool_),
    }
    arrays.update(feature_arrays)
    metadata: JsonDict = {
        "sequence_id": frame.sequence_id,
        "camera_id": frame.camera_id,
        "frame_id": frame.frame_id,
        "timestamp_ns": frame.timestamp_ns,
        "source_name": frame.source_name,
        "source_family": frame.source_family,
        "supervision_grade": frame.supervision_grade,
        "dataset_frame_type": frame.dataset_frame_type,
        "robot_frame_truth": bool(frame.robot_frame_truth),
        "action_supervision_ok": bool(frame.action_supervision_ok),
        "split": frame.split,
        "split_unit_id": frame.split_unit_id,
        "future_target_frame_id": target_frame_ids,
        "horizon_valid": horizon_valid,
        "invalid_reasons": invalid_reasons,
        "pose_source": frame.pose_sample.pose_source if frame.pose_sample is not None else "missing",
        "pose_frame": frame.pose_sample.pose_frame if frame.pose_sample is not None else "missing",
        **ROLLOUT_PROVENANCE_FLAGS,
    }
    return FutureRolloutTargets(arrays=arrays, metadata=metadata)


def warp_future_bev_to_current(
    future_bev: np.ndarray,
    future_to_current_delta: tuple[float, float, float],
    *,
    meters_per_cell: float,
) -> tuple[np.ndarray, np.ndarray]:
    future_t = torch.from_numpy(np.asarray(future_bev, dtype=np.float32)[None, ...])
    pose_t = torch.tensor([future_to_current_delta], dtype=torch.float32)
    pose_mask = torch.ones((1, 1), dtype=torch.float32)
    warped = warp_memory_se2(
        future_t,
        pose_t,
        pose_mask,
        meters_per_cell=meters_per_cell,
        mode="nearest",
    ).tensor[0].detach().cpu().numpy()
    ones = torch.ones((1, 1, future_bev.shape[-2], future_bev.shape[-1]), dtype=torch.float32)
    valid = warp_memory_se2(
        ones,
        pose_t,
        pose_mask,
        meters_per_cell=meters_per_cell,
        mode="nearest",
    ).tensor[0, 0].detach().cpu().numpy()
    return np.clip(warped, 0.0, 1.0).astype(np.float32), (valid > 0.5).astype(np.float32)


def candidate_outcome_labels(
    *,
    candidates: list[CandidateTrajectory],
    future_occupied: np.ndarray,
    future_unknown: np.ndarray,
    newly_observed: np.ndarray,
    future_free: np.ndarray,
    horizon_valid: np.ndarray,
    current_occupied: np.ndarray | None = None,
    current_risky: np.ndarray | None = None,
    current_unknown: np.ndarray | None = None,
    current_free: np.ndarray | None = None,
    current_traversable: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    horizons, height, width = future_occupied.shape
    valid_horizons = np.asarray(horizon_valid, dtype=np.float32) > 0.0
    any_valid = bool(np.any(valid_horizons))
    max_forward = max((max((pose.x_m for pose in candidate.poses), default=0.0) for candidate in candidates), default=1.0)
    max_forward = max(float(max_forward), 1.0e-6)
    max_turn = max(
        (max((abs(float(pose.yaw_rad)) for pose in candidate.poses), default=0.0) for candidate in candidates),
        default=1.0,
    )
    max_turn = max(float(max_turn), 1.0e-6)
    current_occupied = _optional_grid(current_occupied, height, width, fill=0.0)
    current_risky = _optional_grid(current_risky, height, width, fill=0.0)
    current_unknown = _optional_grid(current_unknown, height, width, fill=0.0)
    current_free = _optional_grid(current_free, height, width, fill=0.0)
    current_traversable = _optional_grid(current_traversable, height, width, fill=0.0)
    collision: list[float] = []
    future_collision: list[float] = []
    unsafe_now: list[float] = []
    unknown_exposure: list[float] = []
    new_area_gain: list[float] = []
    progress: list[float] = []
    oracle_cost: list[float] = []
    valid_mask: list[float] = []
    for candidate in candidates:
        cells = tuple(candidate.footprint_cells)
        if not cells:
            collision.append(0.0)
            future_collision.append(0.0)
            unsafe_now.append(0.0)
            unknown_exposure.append(0.0)
            new_area_gain.append(0.0)
            progress.append(0.0)
            oracle_cost.append(0.0)
            valid_mask.append(0.0)
            continue
        label_cells = _candidate_label_cells(candidate, height, width)
        occupied_values = _cells_over_horizons(future_occupied, cells, default=1.0)[valid_horizons]
        unknown_values = _cells_over_horizons(future_unknown, label_cells, default=1.0)[valid_horizons]
        new_values = _cells_over_horizons(newly_observed, label_cells, default=0.0)[valid_horizons]
        free_values = _cells_over_horizons(future_free, label_cells, default=0.0)[valid_horizons]
        current_occupied_values = _cells_on_grid(current_occupied, cells, default=1.0)
        current_risky_values = _cells_on_grid(current_risky, cells, default=1.0)
        current_unknown_values = _cells_on_grid(current_unknown, label_cells, default=1.0)
        current_free_values = _cells_on_grid(current_free, label_cells, default=0.0)
        current_traversable_values = _cells_on_grid(current_traversable, label_cells, default=0.0)
        terminal = candidate.poses[-1] if candidate.poses else None
        forward_norm = 0.0 if terminal is None else max(0.0, float(terminal.x_m)) / max_forward
        turn_norm = 0.0 if terminal is None else abs(float(terminal.yaw_rad)) / max_turn
        future_collision_value = _risk_value(occupied_values) if any_valid else 0.0
        current_occupied_risk = _risk_value(current_occupied_values)
        current_risky_risk = _risk_value(current_risky_values)
        current_unknown_mean = _mean_array(current_unknown_values)
        current_free_mean = _mean_array(np.maximum(current_free_values, current_traversable_values))
        low_free_risk = max(0.0, 1.0 - current_free_mean)
        physical_unsafe_value = max(
            current_occupied_risk,
            0.85 * current_risky_risk,
        )
        unsafe_value = max(
            physical_unsafe_value,
            0.25 * current_unknown_mean,
            0.15 * low_free_risk,
        )
        future_unknown_mean = _mean_array(unknown_values) if any_valid else 0.0
        unknown_value = max(current_unknown_mean, future_unknown_mean)
        new_value = _mean_array(new_values) if any_valid else 0.0
        future_free_mean = _mean_array(free_values) if any_valid else 0.0
        free_mean = max(current_free_mean, future_free_mean)
        collision_value = max(future_collision_value, physical_unsafe_value, 0.5 * unsafe_value)
        progress_value = (0.85 * forward_norm + 0.15 * turn_norm) * free_mean * (1.0 - min(1.0, collision_value))
        stop_penalty = 0.8 if candidate.id == "stop" else 0.0
        cost_value = (
            8.0 * collision_value
            + 2.0 * unsafe_value
            + 1.25 * unknown_value
            - 2.4 * new_value
            - 2.0 * progress_value
            + stop_penalty
        )
        collision.append(float(np.clip(collision_value, 0.0, 1.0)))
        future_collision.append(float(np.clip(future_collision_value, 0.0, 1.0)))
        unsafe_now.append(float(np.clip(unsafe_value, 0.0, 1.0)))
        unknown_exposure.append(float(np.clip(unknown_value, 0.0, 1.0)))
        new_area_gain.append(float(np.clip(new_value, 0.0, 1.0)))
        progress.append(float(np.clip(progress_value, 0.0, 1.0)))
        oracle_cost.append(float(cost_value))
        valid_mask.append(1.0)
    return {
        "candidate_collision": np.asarray(collision, dtype=np.float32),
        "candidate_future_collision": np.asarray(future_collision, dtype=np.float32),
        "candidate_unsafe_now": np.asarray(unsafe_now, dtype=np.float32),
        "candidate_unknown_exposure": np.asarray(unknown_exposure, dtype=np.float32),
        "candidate_new_area_gain": np.asarray(new_area_gain, dtype=np.float32),
        "candidate_progress": np.asarray(progress, dtype=np.float32),
        "candidate_oracle_cost": np.asarray(oracle_cost, dtype=np.float32),
        "candidate_valid_mask": np.asarray(valid_mask, dtype=np.float32),
    }


def meters_per_cell_from_manifest(manifest: JsonDict, default: float = 0.05) -> float:
    camera_config = manifest.get("camera_config")
    if isinstance(camera_config, dict):
        value = camera_config.get("meters_per_cell")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0.0:
            return float(value)
    value = manifest.get("meters_per_cell")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0.0:
        return float(value)
    return float(default)


def robot_radius_m_from_manifest(manifest: JsonDict, default: float = 0.18) -> float:
    camera_config = manifest.get("camera_config")
    if isinstance(camera_config, dict):
        value = camera_config.get("robot_radius_m")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) >= 0.0:
            return float(value)
    value = manifest.get("robot_radius_m")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) >= 0.0:
        return float(value)
    return float(default)


def default_candidates_for_manifest(manifest: JsonDict) -> list[CandidateTrajectory]:
    grid_shape_raw = manifest.get("grid_shape")
    if not isinstance(grid_shape_raw, list) or len(grid_shape_raw) != 2:
        raise ValueError("source manifest must include grid_shape")
    grid_shape = (int(grid_shape_raw[0]), int(grid_shape_raw[1]))
    meters_per_cell = meters_per_cell_from_manifest(manifest)
    robot_radius_m = robot_radius_m_from_manifest(manifest)
    return generate_default_candidates(
        grid_shape=grid_shape,
        meters_per_cell=meters_per_cell,
        robot_radius_m=robot_radius_m,
    )


def group_frames_by_sequence(frames: list[SpatialRolloutFrame]) -> dict[tuple[str, str, str], list[SpatialRolloutFrame]]:
    grouped: dict[tuple[str, str, str], list[SpatialRolloutFrame]] = {}
    for frame in frames:
        grouped.setdefault((frame.source_name, frame.sequence_id, frame.camera_id), []).append(frame)
    return {
        key: sorted(values, key=lambda item: (item.timestamp_ns, item.frame_id))
        for key, values in sorted(grouped.items())
    }


def _future_target_frame(
    sequence_frames: list[SpatialRolloutFrame],
    timestamp_ns: int,
    horizon_s: float,
) -> SpatialRolloutFrame | None:
    target_ns = int(timestamp_ns + round(float(horizon_s) * 1_000_000_000))
    future = [frame for frame in sequence_frames if frame.timestamp_ns >= target_ns]
    return future[0] if future else None


def _route_pose_samples(manifest: JsonDict) -> dict[int, RoutePoseSample]:
    source_log = manifest.get("source_log")
    if not isinstance(source_log, str):
        return {}
    root = Path(source_log)
    openloris = root / OPENLORIS_ROUTE_ASSOCIATIONS_FILE
    if openloris.exists():
        return _pose_samples_from_associations(
            openloris,
            source_type="openloris_scene_associations",
            pose_keys=("base_pose", "odom.pose"),
            pose_frame="robot_base_relative_pose",
            pose_source="openloris_robot_base_pose_or_odom_association",
        )
    tum = root / TUM_RGBD_ROUTE_ASSOCIATIONS_FILE
    if tum.exists():
        return _pose_samples_from_associations(
            tum,
            source_type="tum_rgbd_associations",
            pose_keys=("groundtruth",),
            pose_frame="camera_relative_dataset_pose",
            pose_source="tum_rgbd_groundtruth_association",
        )
    return {}


def _pose_samples_from_associations(
    path: Path,
    *,
    source_type: str,
    pose_keys: tuple[str, ...],
    pose_frame: str,
    pose_source: str,
) -> dict[int, RoutePoseSample]:
    data = read_json(path)
    if data.get("source_type") != source_type:
        return {}
    frames = data.get("frames")
    if not isinstance(frames, list):
        return {}
    samples: dict[int, RoutePoseSample] = {}
    for item in frames:
        if not isinstance(item, dict):
            continue
        pose = _pose_from_keys(item, pose_keys)
        if pose is None:
            continue
        frame_id = int(item.get("frame_id", -1))
        timestamp_ns = int(item.get("timestamp_ns", item.get("rgb_timestamp_ns", frame_id)))
        samples[frame_id] = RoutePoseSample(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            pose=pose,
            pose_frame=pose_frame,
            pose_source=pose_source,
        )
    return samples


def _pose_from_keys(item: JsonDict, pose_keys: tuple[str, ...]) -> JsonDict | None:
    for key in pose_keys:
        if "." in key:
            current: Any = item
            for part in key.split("."):
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(part)
            if isinstance(current, dict):
                return current
        else:
            value = item.get(key)
            if isinstance(value, dict):
                return value
    return None


def _relative_delta(origin_pose: JsonDict, target_pose: JsonDict, *, pose_frame: str) -> tuple[float, float, float]:
    origin = _pose_matrix(origin_pose)
    target = _pose_matrix(target_pose)
    relative = np.linalg.inv(origin) @ target
    if pose_frame == "camera_relative_dataset_pose":
        dx = float(relative[0, 3])
        dz_as_dy = float(relative[2, 3])
        dyaw = float(math.atan2(float(relative[0, 2]), float(relative[2, 2])))
        return (dx, dz_as_dy, dyaw)
    return (
        float(relative[0, 3]),
        float(relative[1, 3]),
        float(math.atan2(float(relative[1, 0]), float(relative[0, 0]))),
    )


def _pose_matrix(pose: JsonDict) -> np.ndarray:
    qx = float(pose["qx"])
    qy = float(pose["qy"])
    qz = float(pose["qz"])
    qw = float(pose["qw"])
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        raise ValueError("pose quaternion has zero norm")
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [float(pose["tx"]), float(pose["ty"]), float(pose["tz"])]
    return transform


def _bev_stack(example: dict[str, np.ndarray]) -> np.ndarray:
    free = (np.asarray(example["bev_free"], dtype=np.float32) > 0.0).astype(np.float32)
    occupied = (np.asarray(example["bev_obstacle"], dtype=np.float32) > 0.0).astype(np.float32)
    unknown = (np.asarray(example["bev_unknown"], dtype=np.float32) > 0.0).astype(np.float32)
    traversable = np.asarray(example["bev_traversable"], dtype=np.float32) if "bev_traversable" in example else free.copy()
    risky = np.asarray(example["bev_risky"], dtype=np.float32) if "bev_risky" in example else occupied.copy()
    return np.stack([free, occupied, unknown, traversable, risky], axis=0).astype(np.float32)


def _feature_arrays(record: JsonDict, feature_store: DINOFeatureStore | None) -> dict[str, np.ndarray]:
    if feature_store is None:
        return {
            "dino_patch_features": np.zeros((1, 1, 1), dtype=np.float32),
            "dino_cls_feature": np.zeros((1,), dtype=np.float32),
            "feature_mask": np.asarray([0.0], dtype=np.float32),
        }
    try:
        patch, cls = feature_store.load_for_example(record)
    except KeyError:
        return {
            "dino_patch_features": np.zeros((1, 1, 1), dtype=np.float32),
            "dino_cls_feature": np.zeros((1,), dtype=np.float32),
            "feature_mask": np.asarray([0.0], dtype=np.float32),
        }
    return {
        "dino_patch_features": np.asarray(patch, dtype=np.float32),
        "dino_cls_feature": np.asarray(cls, dtype=np.float32),
        "feature_mask": np.asarray([1.0], dtype=np.float32),
    }


def _feature_tensors_from_example(example: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "dino_patch_features" not in example:
        return (
            np.zeros((1, 1, 1), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
        )
    features = np.asarray(example["dino_patch_features"], dtype=np.float32)
    if features.ndim != 3:
        features = np.zeros((1, 1, 1), dtype=np.float32)
    cls = np.asarray(example.get("dino_cls_feature", np.zeros((features.shape[-1],), dtype=np.float32)), dtype=np.float32)
    if cls.ndim != 1:
        cls = np.zeros((features.shape[-1],), dtype=np.float32)
    mask = np.asarray(example.get("feature_mask", np.asarray([0.0], dtype=np.float32)), dtype=np.float32).reshape(1)
    return features, cls, mask


def _empty_future(height: int, width: int) -> dict[str, Any]:
    zeros = np.zeros((height, width), dtype=np.float32)
    unknown = np.ones((height, width), dtype=np.float32)
    return {
        "free": zeros.copy(),
        "occupied": zeros.copy(),
        "unknown": unknown,
        "valid": zeros.copy(),
        "pose_delta_to_future": (0.0, 0.0, 0.0),
        "pose_delta_future_to_current": (0.0, 0.0, 0.0),
        "target_frame_id": -1,
        "valid_scalar": 0.0,
    }


def _candidate_oracle_cost_from_example(example: dict[str, np.ndarray]) -> np.ndarray:
    if "candidate_oracle_cost" in example:
        return np.asarray(example["candidate_oracle_cost"], dtype=np.float32)
    collision = np.asarray(example["candidate_collision"], dtype=np.float32)
    unsafe_now = np.asarray(example.get("candidate_unsafe_now", np.zeros_like(collision)), dtype=np.float32)
    unknown = np.asarray(example["candidate_unknown_exposure"], dtype=np.float32)
    gain = np.asarray(example["candidate_new_area_gain"], dtype=np.float32)
    progress = np.asarray(example["candidate_progress"], dtype=np.float32)
    return (8.0 * collision + 2.0 * unsafe_now + 1.25 * unknown - 2.4 * gain - 2.0 * progress).astype(np.float32)


def _optional_grid(array: np.ndarray | None, height: int, width: int, *, fill: float) -> np.ndarray:
    if array is None:
        return np.full((height, width), float(fill), dtype=np.float32)
    grid = np.asarray(array, dtype=np.float32)
    if grid.shape != (height, width):
        return np.full((height, width), float(fill), dtype=np.float32)
    return np.clip(grid, 0.0, 1.0).astype(np.float32)


def _candidate_label_cells(candidate: CandidateTrajectory, height: int, width: int) -> tuple[tuple[int, int], ...]:
    cells = set(candidate.footprint_cells)
    linear = abs(float(candidate.cmd_vel_proxy.get("linear_velocity_mps", 0.0)))
    angular = abs(float(candidate.cmd_vel_proxy.get("angular_velocity_radps", 0.0)))
    if angular > 0.0 and linear < 1.0e-6:
        cells = set(_dilate_cells(tuple(cells), height, width, radius_cells=2))
    return tuple(sorted(cells))


def _dilate_cells(
    cells: tuple[tuple[int, int], ...],
    height: int,
    width: int,
    *,
    radius_cells: int,
) -> tuple[tuple[int, int], ...]:
    expanded: set[tuple[int, int]] = set()
    for row, col in cells:
        for row_offset in range(-radius_cells, radius_cells + 1):
            for col_offset in range(-radius_cells, radius_cells + 1):
                if row_offset * row_offset + col_offset * col_offset > radius_cells * radius_cells:
                    continue
                out_row = row + row_offset
                out_col = col + col_offset
                if 0 <= out_row < height and 0 <= out_col < width:
                    expanded.add((out_row, out_col))
    return tuple(sorted(expanded))


def _cells_on_grid(values: np.ndarray, cells: tuple[tuple[int, int], ...], *, default: float) -> np.ndarray:
    result = np.full((max(len(cells), 1),), float(default), dtype=np.float32)
    if not cells:
        return result
    height, width = values.shape[0], values.shape[1]
    for cell_index, (row, col) in enumerate(cells):
        if 0 <= row < height and 0 <= col < width:
            result[cell_index] = float(values[row, col])
    return result


def _cells_over_horizons(values: np.ndarray, cells: tuple[tuple[int, int], ...], *, default: float) -> np.ndarray:
    horizons = int(values.shape[0])
    result = np.full((horizons, max(len(cells), 1)), float(default), dtype=np.float32)
    if not cells:
        return result
    height, width = values.shape[1], values.shape[2]
    for cell_index, (row, col) in enumerate(cells):
        if 0 <= row < height and 0 <= col < width:
            result[:, cell_index] = values[:, row, col].astype(np.float32)
    return result


def _risk_value(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(np.clip(0.6 * float(np.max(array)) + 0.4 * float(np.mean(array)), 0.0, 1.0))


def _mean_array(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(np.clip(float(np.mean(array)), 0.0, 1.0))


def _scalar_float(example: dict[str, np.ndarray], field: str, default: float) -> float:
    if field not in example:
        return default
    try:
        return float(np.asarray(example[field]).item())
    except Exception:  # noqa: BLE001
        return default
