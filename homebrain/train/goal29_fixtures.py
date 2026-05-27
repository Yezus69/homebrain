from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from homebrain.data.spatial_dataset import DETERMINISTIC_CREATED_AT_UTC, write_deterministic_npz, write_json
from homebrain.messages.schema import JsonDict
from homebrain.policies.candidate_trajectories import CandidateTrajectory, candidates_hash, generate_default_candidates
from homebrain.teachers.artifacts import file_sha256, relative_to_root
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

GOAL29_FIXTURE_NAMES: tuple[str, ...] = (
    "moving_obstacle_crossing_path",
    "static_obstacle_in_known_free_space",
    "unknown_corridor",
    "high_uncertainty_near_footprint",
    "future_unsafe_candidate",
)


def write_goal29_fixture_rollout_pack(
    out_dir: str | Path,
    *,
    grid_shape: tuple[int, int] = (16, 16),
    meters_per_cell: float = 0.1,
    robot_radius_m: float = 0.05,
    horizons_s: tuple[float, ...] = DEFAULT_FUTURE_HORIZONS_S,
) -> Path:
    if not horizons_s:
        raise ValueError("Goal29 fixtures require at least one future horizon")
    output = Path(out_dir)
    examples_dir = output / "examples"
    _clear(output)
    examples_dir.mkdir(parents=True, exist_ok=True)
    candidates = generate_default_candidates(
        grid_shape=grid_shape,
        meters_per_cell=meters_per_cell,
        robot_radius_m=robot_radius_m,
    )
    examples: list[JsonDict] = []
    for index, fixture_name in enumerate(GOAL29_FIXTURE_NAMES):
        arrays = _fixture_arrays(
            fixture_name,
            candidates=candidates,
            grid_shape=grid_shape,
            horizons_s=horizons_s,
        )
        example_path = examples_dir / f"goal29_fixture_{index:06d}.npz"
        write_deterministic_npz(example_path, arrays)
        examples.append(
            {
                "sequence_id": "goal29_fixture_sequence",
                "camera_id": "front_rgb",
                "frame_id": index,
                "timestamp_ns": index * 1_000_000_000,
                "split": "train",
                "split_unit_id": fixture_name,
                "source_name": fixture_name,
                "source_family": "deterministic_goal29_fixture",
                "supervision_grade": "synthetic_failure_mode_fixture",
                "dataset_frame_type": "replay_fixture_bev",
                "fixture_name": fixture_name,
                "example_path": relative_to_root(example_path, output),
                "example_sha256": file_sha256(example_path),
                "candidate_count": len(candidates),
                "valid_horizon_count": len(horizons_s),
                "feature_mask": 0.0,
                **ROLLOUT_PROVENANCE_FLAGS,
            }
        )
    manifest: JsonDict = {
        "schema_version": FUTURE_ROLLOUT_PACK_SCHEMA_VERSION,
        "example_schema_version": FUTURE_ROLLOUT_EXAMPLE_SCHEMA_VERSION,
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "package_type": "FutureBEVRolloutPack",
        "version": 1,
        "goal": "Goal29",
        "fixture_pack": True,
        "fixture_names": list(GOAL29_FIXTURE_NAMES),
        "source_dirs": [],
        "sources": [
            {
                "source_name": "goal29_deterministic_fixtures",
                "included_example_count": len(examples),
                "fixture_names": list(GOAL29_FIXTURE_NAMES),
            }
        ],
        "example_count": len(examples),
        "max_examples": len(examples),
        "sampling_strategy": "source_order",
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "meters_per_cell": round(float(meters_per_cell), 6),
        "robot_radius_m": round(float(robot_radius_m), 6),
        "horizons_s": [round(float(value), 6) for value in horizons_s],
        "history_steps": 3,
        "future_bev_channels": list(FUTURE_BEV_CHANNELS),
        "future_risk_channels": list(FUTURE_RISK_CHANNELS),
        "future_derived_channels": list(FUTURE_DERIVED_CHANNELS),
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.id for candidate in candidates],
        "candidate_hash": candidates_hash(candidates),
        "target_generation": {
            "fixture_target_generation": True,
            "moving_obstacle_fixture": "moving_obstacle_crossing_path",
            "uses_model_predictions_as_labels": False,
            "uses_future_labels_at_runtime": False,
            "weak_label": True,
            "control_safe": False,
        },
        "valid_horizon_fraction": 1.0,
        "candidate_valid_fraction": 1.0,
        "invalid_reason_distribution": {"valid": len(examples) * len(horizons_s)},
        "examples": examples,
        **ROLLOUT_PROVENANCE_FLAGS,
    }
    write_json(output / "manifest.json", manifest, pretty=True)
    return output / "manifest.json"


def _fixture_arrays(
    fixture_name: str,
    *,
    candidates: list[CandidateTrajectory],
    grid_shape: tuple[int, int],
    horizons_s: tuple[float, ...],
) -> dict[str, np.ndarray]:
    height, width = grid_shape
    current = _base_bev(grid_shape)
    memory = current.copy()
    uncertainty = np.where(current[2] > 0.5, 0.75, 0.1).astype(np.float32)
    history = np.repeat(current[None, ...], 3, axis=0).astype(np.float32)
    future = np.repeat(current[None, [0, 1, 2], :, :], len(horizons_s), axis=0).astype(np.float32)
    future_risk = np.repeat(current[None, 4, :, :], len(horizons_s), axis=0).astype(np.float32)

    if fixture_name == "moving_obstacle_crossing_path":
        _paint_history_motion(history, candidates, "straight_medium")
        _paint_future_candidate(future, future_risk, candidates, "straight_medium", horizon_index=1, value=1.0)
    elif fixture_name == "static_obstacle_in_known_free_space":
        cells = [*_motion_cells(candidates, "straight_short"), *_motion_cells(candidates, "arc_right_small")]
        _paint_cells(current[1], cells, 1.0)
        _paint_cells(current[4], cells, 1.0)
        _paint_cells(current[0], cells, 0.0)
        memory = current.copy()
        future = np.repeat(current[None, [0, 1, 2], :, :], len(horizons_s), axis=0).astype(np.float32)
        future_risk = np.repeat(current[None, 4, :, :], len(horizons_s), axis=0).astype(np.float32)
    elif fixture_name == "unknown_corridor":
        cells = [*_motion_cells(candidates, "arc_left_small"), *_motion_cells(candidates, "arc_right_small")]
        _paint_cells(current[0], cells, 0.0)
        _paint_cells(current[2], cells, 1.0)
        memory = current.copy()
        future = np.repeat(current[None, [0, 1, 2], :, :], len(horizons_s), axis=0).astype(np.float32)
    elif fixture_name == "high_uncertainty_near_footprint":
        _paint_cells(uncertainty, _motion_cells(candidates, "straight_short"), 0.95)
        motion_cells: list[tuple[int, int]] = []
        for candidate in candidates:
            if candidate.id != "stop":
                motion_cells.extend(_motion_cells(candidates, candidate.id))
        _paint_cells(current[1], motion_cells, 1.0)
        _paint_cells(current[4], motion_cells, 1.0)
        _paint_cells(current[0], motion_cells, 0.0)
        memory = current.copy()
        future = np.repeat(current[None, [0, 1, 2], :, :], len(horizons_s), axis=0).astype(np.float32)
        future_risk = np.repeat(current[None, 4, :, :], len(horizons_s), axis=0).astype(np.float32)
    elif fixture_name == "future_unsafe_candidate":
        _paint_future_candidate(future, future_risk, candidates, "straight_short", horizon_index=len(horizons_s) - 1, value=1.0)
        _paint_future_candidate(future, future_risk, candidates, "arc_right_small", horizon_index=len(horizons_s) - 1, value=1.0)
    else:
        raise ValueError(f"unknown Goal29 fixture: {fixture_name}")

    future_free = future[:, 0]
    future_occupied = future[:, 1]
    future_unknown = future[:, 2]
    future_valid = np.ones((len(horizons_s), height, width), dtype=np.float32)
    newly_observed = ((current[2] >= 0.5) & ((future_free + future_occupied) >= 0.5)).astype(np.float32)
    persistent = ((current[1] >= 0.5) & (future_occupied >= 0.5)).astype(np.float32)
    changed = ((np.abs(future_occupied - current[1]) >= 0.5) | (future_risk > current[4] + 0.25)).astype(np.float32)
    labels = candidate_outcome_labels(
        candidates=candidates,
        current_occupied=current[1],
        current_risky=current[4],
        current_unknown=current[2],
        current_free=current[0],
        current_traversable=current[3],
        future_occupied=future_occupied,
        future_unknown=future_unknown,
        future_risk=future_risk,
        newly_observed=newly_observed,
        future_free=future_free,
        horizon_valid=np.ones((len(horizons_s),), dtype=np.float32),
    )
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(FUTURE_ROLLOUT_EXAMPLE_SCHEMA_VERSION),
        "frame_id": np.asarray(GOAL29_FIXTURE_NAMES.index(fixture_name), dtype=np.int64),
        "timestamp_ns": np.asarray(GOAL29_FIXTURE_NAMES.index(fixture_name) * 1_000_000_000, dtype=np.int64),
        "sequence_id": np.asarray("goal29_fixture_sequence"),
        "camera_id": np.asarray("front_rgb"),
        "source_name": np.asarray(fixture_name),
        "source_family": np.asarray("deterministic_goal29_fixture"),
        "supervision_grade": np.asarray("synthetic_failure_mode_fixture"),
        "dataset_frame_type": np.asarray("replay_fixture_bev"),
        "fixture_name": np.asarray(fixture_name),
        "split": np.asarray("train"),
        "split_unit_id": np.asarray(fixture_name),
        "current_bev_free": current[0],
        "current_bev_occupied": current[1],
        "current_bev_unknown": current[2],
        "current_bev_traversable": current[3],
        "current_bev_risky": current[4],
        "current_bev_confidence": (1.0 - uncertainty).astype(np.float32),
        "memory_bev_free": memory[0],
        "memory_bev_occupied": memory[1],
        "memory_bev_unknown": memory[2],
        "memory_bev_traversable": memory[3],
        "memory_bev_risky": memory[4],
        "uncertainty_map": uncertainty,
        "bev_history": history,
        "sensor_mask": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "horizons_s": np.asarray(horizons_s, dtype=np.float32),
        "horizon_valid": np.ones((len(horizons_s),), dtype=np.float32),
        "future_target_frame_id": np.arange(len(horizons_s), dtype=np.int64),
        "pose_delta_to_future": np.zeros((len(horizons_s), 3), dtype=np.float32),
        "pose_delta_future_to_current": np.zeros((len(horizons_s), 3), dtype=np.float32),
        "future_free": future_free,
        "future_occupied": future_occupied,
        "future_unknown": future_unknown,
        "future_risk": future_risk,
        "future_valid_mask": future_valid,
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
    return arrays


def _base_bev(grid_shape: tuple[int, int]) -> np.ndarray:
    height, width = grid_shape
    bev = np.zeros((5, height, width), dtype=np.float32)
    bev[2] = 1.0
    center = width // 2
    bev[0, height - 7 : height, center - 2 : center + 2] = 1.0
    bev[2, height - 7 : height, center - 2 : center + 2] = 0.0
    bev[3] = bev[0]
    return bev


def _paint_history_motion(history: np.ndarray, candidates: list[CandidateTrajectory], candidate_id: str) -> None:
    cells = _motion_cells(candidates, candidate_id)
    shifted_left = [(row, col - 2) for row, col in cells]
    shifted_center = [(row, col - 1) for row, col in cells]
    _paint_cells(history[0, 1], shifted_left, 1.0)
    _paint_cells(history[0, 4], shifted_left, 1.0)
    _paint_cells(history[1, 1], shifted_center, 1.0)
    _paint_cells(history[1, 4], shifted_center, 1.0)


def _paint_future_candidate(
    future: np.ndarray,
    future_risk: np.ndarray,
    candidates: list[CandidateTrajectory],
    candidate_id: str,
    *,
    horizon_index: int,
    value: float,
) -> None:
    cells = _motion_cells(candidates, candidate_id)
    horizon = int(np.clip(horizon_index, 0, future.shape[0] - 1))
    _paint_cells(future[horizon, 1], cells, value)
    _paint_cells(future[horizon, 0], cells, 0.0)
    _paint_cells(future[horizon, 2], cells, 0.0)
    _paint_cells(future_risk[horizon], cells, value)


def _motion_cells(candidates: list[CandidateTrajectory], candidate_id: str) -> list[tuple[int, int]]:
    candidate = next(item for item in candidates if item.id == candidate_id)
    stop = next(item for item in candidates if item.id == "stop")
    stop_cells = set(stop.footprint_cells)
    return [cell for cell in candidate.footprint_cells if cell not in stop_cells]


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], value: float) -> None:
    for row, col in cells:
        if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
            grid[row, col] = np.float32(value)


def _candidate_masks(candidates: list[CandidateTrajectory], grid_shape: tuple[int, int]) -> np.ndarray:
    masks = np.zeros((len(candidates), *grid_shape), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        _paint_cells(masks[index], list(candidate.footprint_cells), 1.0)
    return masks


def _clear(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.iterdir(), reverse=True):
        if child.is_dir():
            _clear(child)
            child.rmdir()
        else:
            child.unlink()
