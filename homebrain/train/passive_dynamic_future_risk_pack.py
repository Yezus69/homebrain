from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from homebrain.data.spatial_dataset import DETERMINISTIC_CREATED_AT_UTC, read_json, write_deterministic_npz, write_json
from homebrain.messages.schema import JsonDict
from homebrain.policies.candidate_trajectories import CandidateTrajectory, candidates_hash, generate_default_candidates
from homebrain.teachers.artifacts import file_sha256, relative_to_root
from homebrain.teachers.passive_dynamic_teacher import write_passive_dynamic_fixture_sequences
from homebrain.train.future_bev_rollout_dataset import (
    DEFAULT_FUTURE_HORIZONS_S,
    FUTURE_BEV_CHANNELS,
    FUTURE_DERIVED_CHANNELS,
    FUTURE_RISK_CHANNELS,
    FUTURE_ROLLOUT_EXAMPLE_SCHEMA_VERSION,
    FUTURE_ROLLOUT_PACK_SCHEMA_VERSION,
    ROLLOUT_PROVENANCE_FLAGS,
    candidate_outcome_labels,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm", ".bmp"}
PASSIVE_DYNAMIC_PACK_BUILDER_VERSION = "homebrain.passive_dynamic_future_risk_pack_builder.v0"


@dataclass(frozen=True)
class _PassiveFrame:
    sequence_id: str
    split: str
    scenario: str
    frame_id: int
    timestamp_ns: int
    frame_path: Path | None
    sidecar_path: Path | None
    depth: np.ndarray
    dynamic_mask: np.ndarray
    semantic_mask: np.ndarray
    track_id_mask: np.ndarray
    camera_pose_delta: tuple[float, float, float] | None
    confidence: np.ndarray


def build_passive_dynamic_future_risk_pack(
    *,
    input_dir: str | Path,
    out_dir: str | Path,
    horizons_s: tuple[float, ...] = DEFAULT_FUTURE_HORIZONS_S,
    grid_shape: tuple[int, int] = (16, 16),
    meters_per_cell: float = 0.1,
    robot_radius_m: float = 0.05,
    history_steps: int = 3,
    frame_period_s: float = 0.5,
    max_examples: int | None = None,
) -> Path:
    if not horizons_s or any(float(value) <= 0.0 for value in horizons_s):
        raise ValueError("horizons_s must contain positive values")
    if grid_shape[0] <= 1 or grid_shape[1] <= 1:
        raise ValueError("grid_shape dimensions must be greater than one")
    if meters_per_cell <= 0.0:
        raise ValueError("meters_per_cell must be positive")
    if robot_radius_m < 0.0:
        raise ValueError("robot_radius_m must be non-negative")
    if history_steps <= 0:
        raise ValueError("history_steps must be positive")
    if frame_period_s <= 0.0:
        raise ValueError("frame_period_s must be positive")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive when supplied")

    root = Path(input_dir)
    output = Path(out_dir)
    examples_dir = output / "examples"
    _clear_generated_outputs(output)
    examples_dir.mkdir(parents=True, exist_ok=True)
    sequences = _load_sequences(root, grid_shape=grid_shape)
    if not sequences:
        raise ValueError(f"no passive dynamic sequences found in {root}")

    candidates = generate_default_candidates(
        grid_shape=grid_shape,
        meters_per_cell=meters_per_cell,
        robot_radius_m=robot_radius_m,
    )
    examples: list[JsonDict] = []
    horizon_valid_values: list[float] = []
    candidate_valid_values: list[float] = []
    example_index = 0
    for sequence in sequences:
        max_offset = max(1, max(_horizon_offsets(horizons_s, frame_period_s)))
        usable = sequence[:-max_offset] if len(sequence) > max_offset else sequence
        for frame_position, frame in enumerate(usable):
            if max_examples is not None and example_index >= max_examples:
                break
            arrays, metadata = _example_arrays(
                sequence=sequence,
                frame_position=frame_position,
                frame=frame,
                candidates=candidates,
                horizons_s=horizons_s,
                grid_shape=grid_shape,
                history_steps=history_steps,
                frame_period_s=frame_period_s,
            )
            example_path = examples_dir / f"passive_dynamic_{example_index:06d}.npz"
            write_deterministic_npz(example_path, arrays)
            candidate_valid = np.asarray(arrays["candidate_valid_mask"], dtype=np.float32)
            horizon_valid = [float(value) for value in np.asarray(arrays["horizon_valid"], dtype=np.float32).tolist()]
            horizon_valid_values.extend(horizon_valid)
            candidate_valid_values.extend(float(value) for value in candidate_valid.tolist())
            examples.append(
                {
                    **metadata,
                    "example_path": relative_to_root(example_path, output),
                    "example_sha256": file_sha256(example_path),
                    "candidate_count": len(candidates),
                    "valid_horizon_count": int(sum(1 for value in horizon_valid if value > 0.0)),
                    "feature_mask": 0.0,
                    **ROLLOUT_PROVENANCE_FLAGS,
                }
            )
            example_index += 1
        if max_examples is not None and example_index >= max_examples:
            break
    if not examples:
        raise ValueError("no passive dynamic FutureBEV examples were written")

    candidate_ids = [candidate.id for candidate in candidates]
    sources = _source_records(sequences, examples=examples)
    manifest: JsonDict = {
        "schema_version": FUTURE_ROLLOUT_PACK_SCHEMA_VERSION,
        "passive_dynamic_builder_version": PASSIVE_DYNAMIC_PACK_BUILDER_VERSION,
        "example_schema_version": FUTURE_ROLLOUT_EXAMPLE_SCHEMA_VERSION,
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "package_type": "FutureBEVRolloutPack",
        "version": 1,
        "passive_dynamic_scene_pack": True,
        "input_dir": root.as_posix(),
        "sources": sources,
        "source_dirs": sorted({str(frame.frame_path.parent.parent.as_posix()) for seq in sequences for frame in seq if frame.frame_path}),
        "example_count": len(examples),
        "max_examples": int(max_examples) if max_examples is not None else None,
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "meters_per_cell": round(float(meters_per_cell), 6),
        "robot_radius_m": round(float(robot_radius_m), 6),
        "horizons_s": [round(float(value), 6) for value in horizons_s],
        "frame_period_s": round(float(frame_period_s), 6),
        "history_steps": int(history_steps),
        "future_bev_channels": list(FUTURE_BEV_CHANNELS),
        "future_risk_channels": list(FUTURE_RISK_CHANNELS),
        "future_derived_channels": list(FUTURE_DERIVED_CHANNELS),
        "candidate_count": len(candidates),
        "candidate_ids": candidate_ids,
        "candidate_hash": candidates_hash(candidates),
        "target_generation": {
            "passive_dynamic_video_sidecars": True,
            "sidecar_schema_fields": [
                "depth",
                "dynamic_mask",
                "semantic_mask",
                "track_id_mask",
                "camera_pose_delta",
                "confidence",
            ],
            "future_labels_from_future_sidecars_only_during_training": True,
            "runtime_uses_future_labels": False,
            "uses_model_predictions_as_labels": False,
            "requires_classical_slam": False,
            "weak_label": True,
            "control_safe": False,
        },
        "valid_horizon_fraction": float(sum(1 for value in horizon_valid_values if value > 0.0) / max(len(horizon_valid_values), 1)),
        "candidate_valid_fraction": float(
            sum(1 for value in candidate_valid_values if value > 0.0) / max(len(candidate_valid_values), 1)
        ),
        "route_held_out_split_basis": "sequence_id",
        "examples": examples,
        **ROLLOUT_PROVENANCE_FLAGS,
    }
    write_json(output / "manifest.json", manifest, pretty=True)
    return output / "manifest.json"


def _example_arrays(
    *,
    sequence: list[_PassiveFrame],
    frame_position: int,
    frame: _PassiveFrame,
    candidates: list[CandidateTrajectory],
    horizons_s: tuple[float, ...],
    grid_shape: tuple[int, int],
    history_steps: int,
    frame_period_s: float,
) -> tuple[dict[str, np.ndarray], JsonDict]:
    current = _bev_from_frame(frame, grid_shape=grid_shape)
    memory = _memory_bev(sequence[: frame_position + 1], grid_shape=grid_shape)
    history = _history_bev(sequence, frame_position=frame_position, grid_shape=grid_shape, history_steps=history_steps)
    uncertainty = np.clip(1.0 - _resize_nearest(frame.confidence, grid_shape), 0.0, 1.0).astype(np.float32)
    future_free: list[np.ndarray] = []
    future_occupied: list[np.ndarray] = []
    future_unknown: list[np.ndarray] = []
    future_risk: list[np.ndarray] = []
    future_valid: list[np.ndarray] = []
    pose_delta_to_future: list[tuple[float, float, float]] = []
    pose_delta_future_to_current: list[tuple[float, float, float]] = []
    target_frame_ids: list[int] = []
    horizon_valid: list[float] = []
    offsets = _horizon_offsets(horizons_s, frame_period_s)
    for offset in offsets:
        target_position = frame_position + offset
        if target_position >= len(sequence):
            future = _empty_future(grid_shape)
            pose = (0.0, 0.0, 0.0)
            valid_scalar = 0.0
            target_frame_id = -1
        else:
            target = sequence[target_position]
            future = _bev_from_frame(target, grid_shape=grid_shape)
            pose = _summed_pose_delta(sequence, frame_position + 1, target_position)
            valid_scalar = 1.0
            target_frame_id = target.frame_id
        future_free.append(future[0])
        future_occupied.append(future[1])
        future_unknown.append(future[2])
        future_risk.append(future[4])
        future_valid.append(np.full(grid_shape, valid_scalar, dtype=np.float32))
        pose_delta_to_future.append(pose)
        pose_delta_future_to_current.append(tuple(float(-value) for value in pose))
        target_frame_ids.append(int(target_frame_id))
        horizon_valid.append(float(valid_scalar))

    stacked_free = np.stack(future_free, axis=0).astype(np.float32)
    stacked_occupied = np.stack(future_occupied, axis=0).astype(np.float32)
    stacked_unknown = np.stack(future_unknown, axis=0).astype(np.float32)
    stacked_risk = np.stack(future_risk, axis=0).astype(np.float32)
    stacked_valid = np.stack(future_valid, axis=0).astype(np.float32)
    observed_future = np.clip(stacked_free + stacked_occupied, 0.0, 1.0)
    newly_observed = (
        (current[2][None, ...] >= 0.5)
        & (observed_future >= 0.5)
        & (stacked_valid > 0.5)
    ).astype(np.float32)
    persistent = ((current[1][None, ...] >= 0.5) & (stacked_occupied >= 0.5) & (stacked_valid > 0.5)).astype(np.float32)
    changed = (
        ((np.abs(stacked_occupied - current[1][None, ...]) >= 0.5) | (stacked_risk > current[4][None, ...] + 0.25))
        & (stacked_valid > 0.5)
    ).astype(np.float32)
    labels = candidate_outcome_labels(
        candidates=candidates,
        current_occupied=current[1],
        current_risky=current[4],
        current_unknown=current[2],
        current_free=current[0],
        current_traversable=current[3],
        future_occupied=stacked_occupied,
        future_unknown=stacked_unknown,
        future_risk=stacked_risk,
        newly_observed=newly_observed,
        future_free=stacked_free,
        horizon_valid=np.asarray(horizon_valid, dtype=np.float32),
    )
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(FUTURE_ROLLOUT_EXAMPLE_SCHEMA_VERSION),
        "frame_id": np.asarray(frame.frame_id, dtype=np.int64),
        "timestamp_ns": np.asarray(frame.timestamp_ns, dtype=np.int64),
        "sequence_id": np.asarray(frame.sequence_id),
        "camera_id": np.asarray("front_rgb"),
        "source_name": np.asarray(frame.sequence_id),
        "source_family": np.asarray("passive_dynamic_video_sequence"),
        "supervision_grade": np.asarray("passive_open_model_sidecar_weak_label"),
        "dataset_frame_type": np.asarray("passive_rgb_sidecar_sequence"),
        "split": np.asarray(frame.split),
        "split_unit_id": np.asarray(frame.sequence_id),
        "scenario": np.asarray(frame.scenario),
        "current_bev_free": current[0],
        "current_bev_occupied": current[1],
        "current_bev_unknown": current[2],
        "current_bev_traversable": current[3],
        "current_bev_risky": current[4],
        "current_bev_confidence": np.clip(1.0 - uncertainty, 0.0, 1.0).astype(np.float32),
        "memory_bev_free": memory[0],
        "memory_bev_occupied": memory[1],
        "memory_bev_unknown": memory[2],
        "memory_bev_traversable": memory[3],
        "memory_bev_risky": memory[4],
        "uncertainty_map": uncertainty,
        "bev_history": history,
        "sensor_mask": np.asarray([1.0 if frame.camera_pose_delta is not None else 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "horizons_s": np.asarray(horizons_s, dtype=np.float32),
        "horizon_valid": np.asarray(horizon_valid, dtype=np.float32),
        "future_target_frame_id": np.asarray(target_frame_ids, dtype=np.int64),
        "pose_delta_to_future": np.asarray(pose_delta_to_future, dtype=np.float32),
        "pose_delta_future_to_current": np.asarray(pose_delta_future_to_current, dtype=np.float32),
        "future_free": stacked_free,
        "future_occupied": stacked_occupied,
        "future_unknown": stacked_unknown,
        "future_risk": stacked_risk,
        "future_valid_mask": stacked_valid,
        "newly_observed": newly_observed,
        "persistent_obstacle": persistent,
        "cleared_or_changed": changed,
        "candidate_ids": np.asarray([candidate.id for candidate in candidates]),
        "candidate_footprint_masks": _candidate_masks(candidates, grid_shape),
        **labels,
        "replay_only": np.asarray(True, dtype=np.bool_),
        "not_executed": np.asarray(True, dtype=np.bool_),
        "control_safe": np.asarray(False, dtype=np.bool_),
        "weak_label": np.asarray(True, dtype=np.bool_),
        "product_training_approved": np.asarray(False, dtype=np.bool_),
        "raw_pwm_emitted": np.asarray(False, dtype=np.bool_),
        "hardware_validated": np.asarray(False, dtype=np.bool_),
        "dino_patch_features": np.zeros((1, 1, 1), dtype=np.float32),
        "dino_cls_feature": np.zeros((1,), dtype=np.float32),
        "feature_mask": np.asarray([0.0], dtype=np.float32),
    }
    metadata: JsonDict = {
        "sequence_id": frame.sequence_id,
        "camera_id": "front_rgb",
        "frame_id": int(frame.frame_id),
        "timestamp_ns": int(frame.timestamp_ns),
        "source_name": frame.sequence_id,
        "source_family": "passive_dynamic_video_sequence",
        "supervision_grade": "passive_open_model_sidecar_weak_label",
        "dataset_frame_type": "passive_rgb_sidecar_sequence",
        "split": frame.split,
        "split_unit_id": frame.sequence_id,
        "scenario": frame.scenario,
        "future_target_frame_id": target_frame_ids,
        "horizon_valid": horizon_valid,
        "rgb_path": frame.frame_path.as_posix() if frame.frame_path is not None else None,
        "sidecar_path": frame.sidecar_path.as_posix() if frame.sidecar_path is not None else None,
        "teacher_sidecar_used": frame.sidecar_path is not None,
        "uses_future_labels_at_runtime": False,
    }
    return arrays, metadata


def _load_sequences(root: Path, *, grid_shape: tuple[int, int]) -> list[list[_PassiveFrame]]:
    dirs = _sequence_dirs(root)
    sequences = [_load_sequence(path, grid_shape=grid_shape) for path in dirs]
    return [sequence for sequence in sequences if sequence]


def _sequence_dirs(root: Path) -> list[Path]:
    if _looks_like_sequence(root):
        return [root]
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        records = manifest.get("sequences")
        if isinstance(records, list):
            dirs = []
            for record in records:
                if isinstance(record, dict) and isinstance(record.get("path"), str):
                    path = root / str(record["path"])
                    if _looks_like_sequence(path):
                        dirs.append(path)
            if dirs:
                return sorted(dirs)
    return sorted(path for path in root.iterdir() if path.is_dir() and _looks_like_sequence(path))


def _looks_like_sequence(path: Path) -> bool:
    if (path / "sequence_manifest.json").exists():
        return True
    frame_root = path / "frames" if (path / "frames").exists() else path
    return any(child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES for child in frame_root.glob("*"))


def _load_sequence(path: Path, *, grid_shape: tuple[int, int]) -> list[_PassiveFrame]:
    manifest = read_json(path / "sequence_manifest.json") if (path / "sequence_manifest.json").exists() else {}
    sequence_id = str(manifest.get("sequence_id", path.name))
    split = str(manifest.get("split", _split_from_name(path.name)))
    scenario = str(manifest.get("scenario", "passive_dynamic_sequence"))
    frame_records = manifest.get("frames")
    frames: list[_PassiveFrame] = []
    if isinstance(frame_records, list):
        for index, record in enumerate(frame_records):
            if not isinstance(record, dict):
                continue
            frame_path = path / str(record["rgb_path"]) if isinstance(record.get("rgb_path"), str) else None
            sidecar_path = path / str(record["sidecar_path"]) if isinstance(record.get("sidecar_path"), str) else None
            frames.append(
                _frame_from_paths(
                    sequence_id=sequence_id,
                    split=split,
                    scenario=scenario,
                    frame_id=int(record.get("frame_id", index)),
                    timestamp_ns=int(record.get("timestamp_ns", index)),
                    frame_path=frame_path,
                    sidecar_path=sidecar_path,
                    grid_shape=grid_shape,
                )
            )
    else:
        frame_root = path / "frames" if (path / "frames").exists() else path
        frame_paths = sorted(child for child in frame_root.glob("*") if child.suffix.lower() in IMAGE_SUFFIXES)
        for index, frame_path in enumerate(frame_paths):
            sidecar_path = _matching_sidecar(path, frame_path)
            frames.append(
                _frame_from_paths(
                    sequence_id=sequence_id,
                    split=split,
                    scenario=scenario,
                    frame_id=index,
                    timestamp_ns=index,
                    frame_path=frame_path,
                    sidecar_path=sidecar_path,
                    grid_shape=grid_shape,
                )
            )
    return sorted(frames, key=lambda item: (item.timestamp_ns, item.frame_id))


def _frame_from_paths(
    *,
    sequence_id: str,
    split: str,
    scenario: str,
    frame_id: int,
    timestamp_ns: int,
    frame_path: Path | None,
    sidecar_path: Path | None,
    grid_shape: tuple[int, int],
) -> _PassiveFrame:
    sidecar = _sidecar_arrays(sidecar_path, frame_id=frame_id, timestamp_ns=timestamp_ns, grid_shape=grid_shape)
    return _PassiveFrame(
        sequence_id=str(sidecar.get("sequence_id", sequence_id)),
        split=split,
        scenario=str(sidecar.get("scenario", scenario)),
        frame_id=int(sidecar.get("frame_id", frame_id)),
        timestamp_ns=int(sidecar.get("timestamp_ns", timestamp_ns)),
        frame_path=frame_path,
        sidecar_path=sidecar_path if sidecar_path is not None and sidecar_path.exists() else None,
        depth=np.asarray(sidecar["depth"], dtype=np.float32),
        dynamic_mask=np.asarray(sidecar["dynamic_mask"], dtype=np.float32),
        semantic_mask=np.asarray(sidecar["semantic_mask"], dtype=np.int64),
        track_id_mask=np.asarray(sidecar["track_id_mask"], dtype=np.int64),
        camera_pose_delta=sidecar["camera_pose_delta"],
        confidence=np.asarray(sidecar["confidence"], dtype=np.float32),
    )


def _sidecar_arrays(
    sidecar_path: Path | None,
    *,
    frame_id: int,
    timestamp_ns: int,
    grid_shape: tuple[int, int],
) -> dict[str, object]:
    if sidecar_path is None or not sidecar_path.exists():
        return _blank_sidecar(frame_id=frame_id, timestamp_ns=timestamp_ns, grid_shape=grid_shape)
    with np.load(sidecar_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    depth = np.asarray(arrays.get("depth", np.zeros(grid_shape, dtype=np.float32)), dtype=np.float32)
    dynamic = np.asarray(arrays.get("dynamic_mask", np.zeros_like(depth)), dtype=np.float32)
    semantic = np.asarray(arrays.get("semantic_mask", np.zeros_like(depth, dtype=np.int64)), dtype=np.int64)
    track_id = np.asarray(arrays.get("track_id_mask", np.zeros_like(semantic)), dtype=np.int64)
    confidence = np.asarray(arrays.get("confidence", np.ones_like(depth, dtype=np.float32)), dtype=np.float32)
    pose_raw = arrays.get("camera_pose_delta")
    pose_mask = bool(np.asarray(arrays.get("camera_pose_delta_mask", pose_raw is not None)).item())
    pose = None
    if pose_raw is not None and pose_mask:
        pose_values = np.asarray(pose_raw, dtype=np.float32).reshape(-1)
        if pose_values.size >= 3:
            pose = (float(pose_values[0]), float(pose_values[1]), float(pose_values[2]))
    return {
        "frame_id": int(np.asarray(arrays.get("frame_id", frame_id)).item()),
        "timestamp_ns": int(np.asarray(arrays.get("timestamp_ns", timestamp_ns)).item()),
        "scenario": str(np.asarray(arrays.get("scenario", "passive_dynamic_sequence")).item()),
        "sequence_id": str(np.asarray(arrays.get("sequence_id", "sequence")).item()),
        "depth": depth,
        "dynamic_mask": dynamic,
        "semantic_mask": semantic,
        "track_id_mask": track_id,
        "camera_pose_delta": pose,
        "confidence": confidence,
    }


def _blank_sidecar(*, frame_id: int, timestamp_ns: int, grid_shape: tuple[int, int]) -> dict[str, object]:
    return {
        "frame_id": int(frame_id),
        "timestamp_ns": int(timestamp_ns),
        "scenario": "sidecar_missing_static_sequence",
        "sequence_id": "sequence",
        "depth": np.zeros(grid_shape, dtype=np.float32),
        "dynamic_mask": np.zeros(grid_shape, dtype=np.float32),
        "semantic_mask": np.zeros(grid_shape, dtype=np.int64),
        "track_id_mask": np.zeros(grid_shape, dtype=np.int64),
        "camera_pose_delta": None,
        "confidence": np.ones(grid_shape, dtype=np.float32),
    }


def _matching_sidecar(sequence_root: Path, frame_path: Path) -> Path | None:
    candidates = [
        sequence_root / "sidecars" / f"{frame_path.stem}.npz",
        frame_path.with_suffix(".npz"),
        frame_path.parent.parent / "sidecars" / f"{frame_path.stem}.npz",
    ]
    return next((path for path in candidates if path.exists()), None)


def _bev_from_frame(frame: _PassiveFrame, *, grid_shape: tuple[int, int]) -> np.ndarray:
    dynamic_raw = _resize_nearest(frame.dynamic_mask, grid_shape)
    dynamic = _dilate(dynamic_raw, radius=1)
    dynamic_risk = _dilate(dynamic_raw, radius=2)
    semantic = _resize_nearest(frame.semantic_mask, grid_shape).astype(np.int64)
    confidence = np.clip(_resize_nearest(frame.confidence, grid_shape), 0.0, 1.0)
    unknown = ((confidence < 0.30) | (semantic == 20)).astype(np.float32)
    occupied = np.clip(dynamic, 0.0, 1.0).astype(np.float32)
    risky = np.maximum(dynamic_risk, (semantic == 11).astype(np.float32) * 0.9)
    risky = np.maximum(risky, (semantic == 12).astype(np.float32) * 0.8)
    risky = np.maximum(risky, (semantic == 13).astype(np.float32) * 0.85)
    free = np.clip(1.0 - np.maximum(occupied, unknown), 0.0, 1.0).astype(np.float32)
    traversable = free.copy()
    return np.stack([free, occupied, unknown, traversable, risky.astype(np.float32)], axis=0).astype(np.float32)


def _memory_bev(frames: list[_PassiveFrame], *, grid_shape: tuple[int, int]) -> np.ndarray:
    bevs = [_bev_from_frame(frame, grid_shape=grid_shape) for frame in frames]
    if not bevs:
        return _empty_current(grid_shape)
    stack = np.stack(bevs, axis=0)
    free = np.max(stack[:, 0], axis=0)
    occupied = np.max(stack[:, 1], axis=0)
    unknown = np.min(stack[:, 2], axis=0)
    traversable = np.max(stack[:, 3], axis=0)
    risky = np.max(stack[:, 4], axis=0)
    free = np.where(occupied >= 0.5, 0.0, free)
    return np.stack([free, occupied, unknown, traversable, risky], axis=0).astype(np.float32)


def _history_bev(
    sequence: list[_PassiveFrame],
    *,
    frame_position: int,
    grid_shape: tuple[int, int],
    history_steps: int,
) -> np.ndarray:
    start = max(0, frame_position - history_steps + 1)
    history = [_bev_from_frame(frame, grid_shape=grid_shape) for frame in sequence[start : frame_position + 1]]
    if not history:
        history = [_empty_current(grid_shape)]
    while len(history) < history_steps:
        history.insert(0, history[0].copy())
    return np.stack(history[-history_steps:], axis=0).astype(np.float32)


def _empty_current(grid_shape: tuple[int, int]) -> np.ndarray:
    free = np.ones(grid_shape, dtype=np.float32)
    zeros = np.zeros(grid_shape, dtype=np.float32)
    return np.stack([free, zeros, zeros, free, zeros], axis=0).astype(np.float32)


def _empty_future(grid_shape: tuple[int, int]) -> np.ndarray:
    zeros = np.zeros(grid_shape, dtype=np.float32)
    unknown = np.ones(grid_shape, dtype=np.float32)
    return np.stack([zeros, zeros, unknown, zeros, zeros], axis=0).astype(np.float32)


def _horizon_offsets(horizons_s: tuple[float, ...], frame_period_s: float) -> list[int]:
    return [max(1, int(round(float(horizon) / float(frame_period_s)))) for horizon in horizons_s]


def _summed_pose_delta(sequence: list[_PassiveFrame], start: int, end: int) -> tuple[float, float, float]:
    dx = dy = dyaw = 0.0
    for frame in sequence[start : end + 1]:
        if frame.camera_pose_delta is None:
            continue
        dx += float(frame.camera_pose_delta[0])
        dy += float(frame.camera_pose_delta[1])
        dyaw += float(frame.camera_pose_delta[2])
    return (float(dx), float(dy), float(dyaw))


def _candidate_masks(candidates: list[CandidateTrajectory], grid_shape: tuple[int, int]) -> np.ndarray:
    masks = np.zeros((len(candidates), *grid_shape), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        for row, col in candidate.footprint_cells:
            if 0 <= row < grid_shape[0] and 0 <= col < grid_shape[1]:
                masks[index, row, col] = 1.0
    return masks


def _resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(array)
    if source.shape == shape:
        return source.copy()
    if source.ndim != 2:
        return np.zeros(shape, dtype=np.float32)
    rows = np.linspace(0, source.shape[0] - 1, shape[0]).round().astype(np.int64)
    cols = np.linspace(0, source.shape[1] - 1, shape[1]).round().astype(np.int64)
    return source[np.ix_(rows, cols)].copy()


def _dilate(mask: np.ndarray, *, radius: int) -> np.ndarray:
    values = np.asarray(mask, dtype=np.float32)
    if radius <= 0:
        return np.clip(values, 0.0, 1.0).astype(np.float32)
    out = values.copy()
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if values[row, col] <= 0.0:
                continue
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr = row + dr
                    cc = col + dc
                    if 0 <= rr < values.shape[0] and 0 <= cc < values.shape[1] and dr * dr + dc * dc <= radius * radius:
                        out[rr, cc] = max(out[rr, cc], values[row, col])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _split_from_name(name: str) -> str:
    lowered = name.lower()
    if "val" in lowered or "heldout" in lowered or "test" in lowered:
        return "val"
    if "review" in lowered:
        return "review"
    return "train"


def _source_records(sequences: list[list[_PassiveFrame]], *, examples: list[JsonDict]) -> list[JsonDict]:
    included: dict[str, int] = {}
    for example in examples:
        sequence_id = str(example.get("sequence_id", ""))
        included[sequence_id] = included.get(sequence_id, 0) + 1
    records: dict[str, JsonDict] = {}
    for sequence in sequences:
        if not sequence:
            continue
        first = sequence[0]
        record = records.setdefault(
            first.sequence_id,
            {
                "source_name": first.sequence_id,
                "split": first.split,
                "scenario": first.scenario,
                "source_family": "passive_dynamic_video_sequence",
                "frame_count": 0,
                "included_example_count": 0,
                "control_safe": False,
            },
        )
        record["frame_count"] = int(record["frame_count"]) + len(sequence)
        record["included_example_count"] = int(included.get(first.sequence_id, 0))
    return list(records.values())


def _clear_generated_outputs(output: Path) -> None:
    if (output / "examples").exists():
        shutil.rmtree(output / "examples")
    for filename in ("manifest.json",):
        path = output / filename
        if path.exists():
            path.unlink()


def _parse_shape(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.lower().replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must be H,W")
    return (int(parts[0]), int(parts[1]))


def _parse_horizons(value: str) -> tuple[float, ...]:
    horizons = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not horizons:
        raise argparse.ArgumentTypeError("at least one horizon is required")
    if any(item <= 0.0 for item in horizons):
        raise argparse.ArgumentTypeError("horizons must be positive")
    return horizons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a passive dynamic FutureBEV risk pack.")
    parser.add_argument("--input", required=True, help="Input directory of frame sequences.")
    parser.add_argument("--out", required=True, help="Output FutureBEVRolloutPack directory.")
    parser.add_argument("--horizons-s", type=_parse_horizons, default=DEFAULT_FUTURE_HORIZONS_S)
    parser.add_argument("--grid-shape", type=_parse_shape, default=(16, 16))
    parser.add_argument("--meters-per-cell", type=float, default=0.1)
    parser.add_argument("--robot-radius-m", type=float, default=0.05)
    parser.add_argument("--history-steps", type=int, default=3)
    parser.add_argument("--frame-period-s", type=float, default=0.5)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--write-fixtures-first",
        action="store_true",
        help="Overwrite --input with deterministic passive dynamic fixture sequences before building.",
    )
    args = parser.parse_args(argv)
    if args.write_fixtures_first:
        write_passive_dynamic_fixture_sequences(args.input, shape=args.grid_shape, fps=1.0 / args.frame_period_s)
    manifest = build_passive_dynamic_future_risk_pack(
        input_dir=args.input,
        out_dir=args.out,
        horizons_s=args.horizons_s,
        grid_shape=args.grid_shape,
        meters_per_cell=args.meters_per_cell,
        robot_radius_m=args.robot_radius_m,
        history_steps=args.history_steps,
        frame_period_s=args.frame_period_s,
        max_examples=args.max_examples,
    )
    print(json.dumps({"manifest": manifest.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
