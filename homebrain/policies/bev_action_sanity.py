from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.data.spatial_dataset import load_example_npz, np_scalar_to_bool, np_scalar_to_str, read_json, scalar_bool
from homebrain.geometry.bev_projector import BEV_MANIFEST_FILE
from homebrain.geometry.validate_bev import load_bev_manifest
from homebrain.messages.schema import JsonDict
from homebrain.policies.candidate_trajectories import CandidateTrajectory, generate_default_candidates
from homebrain.policies.run_trajectory_scorer import BevDecisionInput, _load_decision_inputs
from homebrain.policies.trajectory_scorer import LocalBev

BEV_ACTION_SANITY_SCHEMA_VERSION = "homebrain.bev_action_sanity.v0"
OCCUPIED_BLOCK_THRESHOLD = 0.35
UNKNOWN_BLOCK_THRESHOLD = 0.50
MIN_FREE_THRESHOLD = 0.50


@dataclass(frozen=True)
class ActionSanityConfig:
    robot_radius_cells: int = 3
    forward_corridor_length_cells: int = 10
    forward_corridor_half_width_cells: int = 3
    near_robot_radius_extra_cells: int = 3
    occupied_block_threshold: float = OCCUPIED_BLOCK_THRESHOLD
    unknown_block_threshold: float = UNKNOWN_BLOCK_THRESHOLD
    min_free_threshold: float = MIN_FREE_THRESHOLD
    min_forward_corridor_free_fraction: float = 0.60

    def to_dict(self) -> JsonDict:
        return {
            "robot_radius_cells": int(self.robot_radius_cells),
            "forward_corridor_length_cells": int(self.forward_corridor_length_cells),
            "forward_corridor_half_width_cells": int(self.forward_corridor_half_width_cells),
            "near_robot_radius_extra_cells": int(self.near_robot_radius_extra_cells),
            "occupied_block_threshold": float(self.occupied_block_threshold),
            "unknown_block_threshold": float(self.unknown_block_threshold),
            "min_free_threshold": float(self.min_free_threshold),
            "min_forward_corridor_free_fraction": float(self.min_forward_corridor_free_fraction),
        }


@dataclass(frozen=True)
class LoadedBevFrame:
    record: BevDecisionInput
    source_root: Path
    source_name: str
    source_family: str
    scenario_name: str
    supervision_grade: str
    source_kind: str
    source_manifest: JsonDict
    frame_record: JsonDict
    meters_per_cell: float
    robot_radius_m: float

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.record.sequence_id, self.record.camera_id, int(self.record.frame_id))


def load_bev_frames_for_action(source: str | Path, *, scratch_dir: str | Path | None = None) -> tuple[list[LoadedBevFrame], JsonDict]:
    root = Path(source)
    if not root.exists():
        raise FileNotFoundError(f"source does not exist: {root}")

    if (root / "trajectory_decisions.jsonl").exists():
        return _load_policy_source(root, scratch_dir=scratch_dir)
    if (root / "manifest.json").exists():
        manifest = read_json(root / "manifest.json")
        package_type = str(manifest.get("package_type", ""))
        if package_type == "SpatialTrainPack":
            return _load_spatial_pack_source(root, manifest)
        if (root / "events.jsonl").exists():
            records, metadata = _load_decision_inputs(
                root,
                Path(scratch_dir) if scratch_dir is not None else root,
                checkpoint=None,
                features=None,
                bev_source="model",
                device_name=None,
            )
            return _wrap_records(records, root=root, source_kind="modeld_output", manifest=manifest, metadata=metadata)
        raise ValueError(f"unsupported manifest package_type for BEV action sanity: {package_type!r}")
    if (root / BEV_MANIFEST_FILE).exists():
        manifest = load_bev_manifest(root)
        records, metadata = _load_decision_inputs(
            root,
            Path(scratch_dir) if scratch_dir is not None else root,
            checkpoint=None,
            features=None,
            bev_source="labels",
            device_name=None,
        )
        return _wrap_records(records, root=root, source_kind="bev_manifest", manifest=manifest, metadata=metadata)

    try:
        records, metadata = _load_decision_inputs(
            root,
            Path(scratch_dir) if scratch_dir is not None else root,
            checkpoint=None,
            features=None,
            bev_source="model",
            device_name=None,
        )
    except Exception as exc:  # noqa: BLE001 - promote to clearer source error.
        raise ValueError(f"could not load BEV frames from {root}: {exc}") from exc
    return _wrap_records(records, root=root, source_kind="modeld_output", manifest={}, metadata=metadata)


def infer_policy_bev_source(source: str | Path) -> str:
    root = Path(source)
    if (root / "manifest.json").exists():
        manifest = read_json(root / "manifest.json")
        if manifest.get("package_type") == "SpatialTrainPack":
            return "labels"
    if (root / BEV_MANIFEST_FILE).exists():
        return "labels"
    return "model"


def evaluate_loaded_frame(
    frame: LoadedBevFrame,
    *,
    config: ActionSanityConfig,
    candidates: list[CandidateTrajectory] | None = None,
) -> JsonDict:
    bev = frame.record.bev
    bev.validate()
    if candidates is None:
        candidates = generate_default_candidates(
            grid_shape=bev.shape,
            meters_per_cell=frame.meters_per_cell,
            robot_radius_m=max(frame.robot_radius_m, config.robot_radius_cells * frame.meters_per_cell),
        )

    free = _free(bev)
    occupied = _occupied(bev)
    risky = _risky(bev)
    unknown = _prob(bev.unknown)
    origin_row, origin_col = robot_origin_cell(bev.shape)
    center_values = _cell_status(free, occupied, risky, unknown, ((origin_row, origin_col),), config=config)
    footprint_cells = disk_cells((origin_row, origin_col), bev.shape, config.robot_radius_cells)
    footprint_status = _cell_status(free, occupied, risky, unknown, footprint_cells, config=config)
    near_cells = disk_cells(
        (origin_row, origin_col),
        bev.shape,
        config.robot_radius_cells + config.near_robot_radius_extra_cells,
    )
    corridor_cells = forward_corridor_cells(
        bev.shape,
        length_cells=config.forward_corridor_length_cells,
        half_width_cells=config.forward_corridor_half_width_cells,
    )
    corridor_status = _cell_status(free, occupied, risky, unknown, corridor_cells, config=config)
    near_status = _cell_ratios(occupied, risky, unknown, near_cells, config=config)
    block_counts, candidate_clear_count = candidate_block_reason_counts(
        candidates,
        bev=bev,
        config=config,
    )
    contract = contract_flags_for_frame(frame)

    center_blocked = bool(center_values["blocked"])
    footprint_blocked = bool(footprint_status["blocked"])
    forward_corridor_free = float(corridor_status["free_fraction"]) >= config.min_forward_corridor_free_fraction
    structural_ok = (not center_blocked) and (not footprint_blocked) and candidate_clear_count > 0
    action_supervision_ok = bool(contract["robot_frame_truth"]) and structural_ok
    if bool(frame.source_manifest.get("derived")) or bool(frame.source_manifest.get("review_required")):
        action_supervision_ok = False

    reasons: list[str] = []
    if not bool(contract["robot_frame_truth"]):
        reasons.append("not_robot_frame_action_truth")
    if center_blocked:
        reasons.append("robot_center_blocked")
    if footprint_blocked:
        reasons.append("robot_footprint_blocked")
    if not forward_corridor_free:
        reasons.append("forward_corridor_not_free")
    if candidate_clear_count <= 0:
        reasons.append("no_clear_motion_or_rotation_candidates")
    if bool(frame.source_manifest.get("derived")) or bool(frame.source_manifest.get("review_required")):
        reasons.append("derived_or_review_required")

    return {
        "sequence_id": frame.record.sequence_id,
        "camera_id": frame.record.camera_id,
        "frame_id": int(frame.record.frame_id),
        "timestamp_ns": int(frame.record.timestamp_ns),
        "source_ref": frame.record.source_ref,
        "source_name": frame.source_name,
        "source_family": frame.source_family,
        "scenario_name": frame.scenario_name,
        "supervision_grade": frame.supervision_grade,
        "origin_frame_status": str(contract["origin_frame_status"]),
        "frame_type": str(contract["frame_type"]),
        "geometry_pretrain_ok": bool(contract["geometry_pretrain_ok"]),
        "pose_pretrain_ok": bool(contract["pose_pretrain_ok"]),
        "robot_frame_truth": bool(contract["robot_frame_truth"]),
        "control_safe": False,
        "robot_center_blocked": center_blocked,
        "footprint_blocked": footprint_blocked,
        "forward_corridor_free": bool(forward_corridor_free),
        "forward_corridor_free_fraction": float(corridor_status["free_fraction"]),
        "occupied_ratio_near_robot": float(near_status["occupied_ratio"]),
        "unknown_ratio_near_robot": float(near_status["unknown_ratio"]),
        "risky_ratio_near_robot": float(near_status["risky_ratio"]),
        "candidate_block_reason_counts": block_counts,
        "candidate_clear_count": int(candidate_clear_count),
        "structural_action_sanity_ok": bool(structural_ok),
        "action_supervision_ok": bool(action_supervision_ok),
        "action_supervision_reject_reasons": reasons,
        "replay_only": True,
        "not_executed": True,
        "raw_pwm_emitted": False,
    }


def contract_flags_for_frame(frame: LoadedBevFrame) -> JsonDict:
    manifest = frame.source_manifest
    source_family = frame.source_family
    supervision_grade = frame.supervision_grade
    source_kind = frame.source_kind

    if source_family == "controlled_bev" or bool(manifest.get("controlled_bev_map_pack")):
        frame_type = "controlled_robot_frame_proxy"
        status = "controlled_algorithmic_robot_frame_proxy"
        robot_frame_truth = True
        geometry_pretrain_ok = True
        pose_pretrain_ok = False
    elif source_kind == "modeld_output" or source_kind == "policy":
        frame_type = "model_prediction"
        status = "model_prediction_not_truth"
        robot_frame_truth = False
        geometry_pretrain_ok = False
        pose_pretrain_ok = False
    elif "public_rgbd" in supervision_grade or "tum" in str(manifest.get("source_depth_teacher_name", "")).lower():
        frame_type = "public_rgbd_camera_pose_geometry"
        status = "not_robot_frame_truth_public_rgbd_camera_pose"
        robot_frame_truth = False
        geometry_pretrain_ok = True
        pose_pretrain_ok = bool(manifest.get("pose_label_count", 0))
    elif "weak_visual" in supervision_grade or str(manifest.get("source_depth_teacher_name", "")).lower() in {"da3", "depth_pro"}:
        frame_type = "phone_or_teacher_estimated_geometry"
        status = "not_robot_frame_truth_phone_teacher_geometry"
        robot_frame_truth = False
        geometry_pretrain_ok = True
        pose_pretrain_ok = False
    else:
        explicit_robot_truth = bool(manifest.get("robot_frame_truth", False))
        not_robot_truth = bool(manifest.get("not_robot_frame_truth", False)) or bool(frame.frame_record.get("not_robot_frame_truth", False))
        frame_type = "explicit_robot_frame_truth" if explicit_robot_truth and not not_robot_truth else "unknown_geometry"
        status = "explicit_robot_frame_truth" if explicit_robot_truth and not not_robot_truth else "unknown_origin_not_action_truth"
        robot_frame_truth = explicit_robot_truth and not not_robot_truth
        geometry_pretrain_ok = True
        pose_pretrain_ok = bool(manifest.get("pose_label_count", 0))

    return {
        "frame_type": frame_type,
        "origin_frame_status": status,
        "geometry_pretrain_ok": bool(geometry_pretrain_ok),
        "pose_pretrain_ok": bool(pose_pretrain_ok),
        "robot_frame_truth": bool(robot_frame_truth),
        "control_safe": False,
    }


def candidate_block_reason_counts(
    candidates: list[CandidateTrajectory],
    *,
    bev: LocalBev,
    config: ActionSanityConfig,
) -> tuple[dict[str, int], int]:
    free = _free(bev)
    occupied = _occupied(bev)
    risky = _risky(bev)
    unknown = _prob(bev.unknown)
    counts = {
        "clear": 0,
        "occupied": 0,
        "risky": 0,
        "unknown": 0,
        "low_free": 0,
        "empty_or_out_of_grid": 0,
    }
    clear_count = 0
    for candidate in candidates:
        if candidate.id == "stop":
            continue
        cells = tuple(candidate.footprint_cells)
        status = _cell_status(free, occupied, risky, unknown, cells, config=config)
        reasons: list[str] = []
        if not cells:
            reasons.append("empty_or_out_of_grid")
        if float(status["occupied_max"]) >= config.occupied_block_threshold:
            reasons.append("occupied")
        if float(status["risky_max"]) >= config.occupied_block_threshold:
            reasons.append("risky")
        if float(status["unknown_max"]) >= config.unknown_block_threshold:
            reasons.append("unknown")
        if float(status["free_fraction"]) < config.min_free_threshold:
            reasons.append("low_free")
        if not reasons:
            reasons.append("clear")
            clear_count += 1
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return counts, clear_count


def aggregate_action_sanity(frame_metrics: list[JsonDict]) -> JsonDict:
    per_source: dict[str, list[JsonDict]] = {}
    for frame in frame_metrics:
        per_source.setdefault(str(frame.get("source_name", "unknown")), []).append(frame)

    aggregate = _aggregate_frames(frame_metrics)
    aggregate["per_source"] = {
        source_name: _aggregate_frames(frames)
        for source_name, frames in sorted(per_source.items())
    }
    return aggregate


def robot_origin_cell(shape: tuple[int, int]) -> tuple[int, int]:
    return (int(shape[0]) - 1, int(shape[1]) // 2)


def disk_cells(center: tuple[int, int], shape: tuple[int, int], radius_cells: int) -> tuple[tuple[int, int], ...]:
    radius = max(0, int(radius_cells))
    row0, col0 = center
    cells: list[tuple[int, int]] = []
    for row_offset in range(-radius, radius + 1):
        for col_offset in range(-radius, radius + 1):
            if row_offset * row_offset + col_offset * col_offset > radius * radius:
                continue
            row = row0 + row_offset
            col = col0 + col_offset
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                cells.append((row, col))
    return tuple(sorted(set(cells)))


def forward_corridor_cells(
    shape: tuple[int, int],
    *,
    length_cells: int,
    half_width_cells: int,
) -> tuple[tuple[int, int], ...]:
    origin_row, origin_col = robot_origin_cell(shape)
    row_start = max(0, origin_row - max(1, int(length_cells)))
    row_end = origin_row + 1
    col_start = max(0, origin_col - max(0, int(half_width_cells)))
    col_end = min(shape[1], origin_col + max(0, int(half_width_cells)) + 1)
    return tuple((row, col) for row in range(row_start, row_end) for col in range(col_start, col_end))


def meters_per_cell_from_manifest(manifest: JsonDict, shape: tuple[int, int]) -> float:
    camera_config = manifest.get("camera_config")
    if isinstance(camera_config, dict):
        value = camera_config.get("meters_per_cell")
        if isinstance(value, (int, float)) and float(value) > 0.0:
            return float(value)
    if min(shape) <= 8:
        return 0.5
    return 0.05


def robot_radius_m_from_manifest(manifest: JsonDict, meters_per_cell: float) -> float:
    camera_config = manifest.get("camera_config")
    if isinstance(camera_config, dict):
        value = camera_config.get("robot_radius_m")
        if isinstance(value, (int, float)) and float(value) >= 0.0:
            return float(value)
    return float(max(meters_per_cell, 3.0 * meters_per_cell))


def source_weight_for_sanity(frame_metric: JsonDict) -> float:
    if bool(frame_metric.get("action_supervision_ok")):
        return 1.0
    if bool(frame_metric.get("robot_frame_truth")) and bool(frame_metric.get("structural_action_sanity_ok")):
        return 0.25
    return 0.0


def frame_metric_severity(frame: JsonDict) -> tuple[float, float, float, float, float]:
    return (
        1.0 if not bool(frame.get("action_supervision_ok")) else 0.0,
        1.0 if bool(frame.get("footprint_blocked")) else 0.0,
        1.0 if bool(frame.get("robot_center_blocked")) else 0.0,
        1.0 - float(frame.get("forward_corridor_free_fraction", 0.0)),
        float(frame.get("occupied_ratio_near_robot", 0.0)) + float(frame.get("risky_ratio_near_robot", 0.0)),
    )


def _load_policy_source(root: Path, *, scratch_dir: str | Path | None) -> tuple[list[LoadedBevFrame], JsonDict]:
    policy_manifest_path = root / "trajectory_policy_manifest.json"
    policy_manifest = read_json(policy_manifest_path) if policy_manifest_path.exists() else {}
    source_value = policy_manifest.get("source")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError(f"policy source is missing source in {policy_manifest_path}")
    source_root = Path(source_value)
    bev_source = str(policy_manifest.get("bev_source", infer_policy_bev_source(source_root)))
    records, metadata = _load_decision_inputs(
        source_root,
        root if scratch_dir is None else Path(scratch_dir),
        checkpoint=None,
        features=None,
        bev_source=bev_source,
        device_name=None,
    )
    frames, source_metadata = _wrap_records(
        records,
        root=source_root,
        source_kind="policy",
        manifest=_read_optional_manifest(source_root),
        metadata=metadata,
    )
    source_metadata["policy_dir"] = root.as_posix()
    source_metadata["policy_manifest"] = policy_manifest
    return frames, source_metadata


def _load_spatial_pack_source(root: Path, manifest: JsonDict) -> tuple[list[LoadedBevFrame], JsonDict]:
    examples = manifest.get("examples") or manifest.get("frames")
    if not isinstance(examples, list):
        raise ValueError(f"SpatialTrainPack manifest examples must be a list: {root}")
    records: list[LoadedBevFrame] = []
    default_source_name = str(manifest.get("source_name") or manifest.get("source_segment_id") or root.name)
    default_source_family = str(manifest.get("source_family") or "unknown")
    default_grade = str(manifest.get("supervision_grade") or manifest.get("robot_supervision_grade") or "unknown")
    meters_per_cell = meters_per_cell_from_manifest(manifest, tuple(manifest.get("grid_shape", [32, 32])))  # type: ignore[arg-type]
    robot_radius_m = robot_radius_m_from_manifest(manifest, meters_per_cell)
    for frame_record in examples:
        if not isinstance(frame_record, dict) or not isinstance(frame_record.get("example_path"), str):
            continue
        example_path = root / str(frame_record["example_path"])
        example = load_example_npz(example_path)
        free = np.asarray(example["bev_free"], dtype=np.float32)
        obstacle = np.asarray(example["bev_obstacle"], dtype=np.float32)
        unknown = np.asarray(example["bev_unknown"], dtype=np.float32)
        confidence = np.asarray(example["bev_confidence"], dtype=np.float32)
        traversable = np.asarray(example["bev_traversable"], dtype=np.float32) if "bev_traversable" in example else free.copy()
        risky = np.asarray(example["bev_risky"], dtype=np.float32) if "bev_risky" in example else obstacle.copy()
        source_name = _scalar_or_record_str(example, frame_record, "source_name", default_source_name)
        source_family = _scalar_or_record_str(example, frame_record, "source_family", default_source_family)
        scenario_name = _scalar_or_record_str(example, frame_record, "scenario_name", "unknown")
        supervision_grade = _scalar_or_record_str(example, frame_record, "supervision_grade", default_grade)
        record = BevDecisionInput(
            sequence_id=str(frame_record.get("sequence_id", manifest.get("source_segment_id", root.name))),
            camera_id=str(frame_record.get("camera_id", "front_rgb")),
            frame_id=int(frame_record.get("frame_id", np.asarray(example["frame_id"]).item())),
            timestamp_ns=int(frame_record.get("timestamp_ns", np.asarray(example["timestamp_ns"]).item())),
            bev=LocalBev(
                free=free,
                occupied=obstacle,
                unknown=unknown,
                traversable=traversable,
                risky=risky,
                confidence=confidence,
                source="labels",
            ),
            pose_delta=None,
            source_ref=str(frame_record["example_path"]),
        )
        records.append(
            LoadedBevFrame(
                record=record,
                source_root=root,
                source_name=source_name,
                source_family=source_family,
                scenario_name=scenario_name,
                supervision_grade=supervision_grade,
                source_kind="spatial_train_pack",
                source_manifest=manifest,
                frame_record=frame_record,
                meters_per_cell=meters_per_cell_from_manifest(manifest, record.bev.shape),
                robot_radius_m=robot_radius_m,
            )
        )
    metadata: JsonDict = {
        "source_kind": "spatial_train_pack",
        "source": root.as_posix(),
        "manifest_schema_version": manifest.get("schema_version"),
        "package_type": manifest.get("package_type"),
        "example_count": len(records),
        "weak_label": manifest.get("weak_label"),
        "control_safe": False,
        "grid_shape": manifest.get("grid_shape", list(records[0].record.bev.shape) if records else []),
    }
    return records, metadata


def _wrap_records(
    records: list[BevDecisionInput],
    *,
    root: Path,
    source_kind: str,
    manifest: JsonDict,
    metadata: JsonDict,
) -> tuple[list[LoadedBevFrame], JsonDict]:
    if not records:
        return [], {
            "source_kind": source_kind,
            "source": root.as_posix(),
            "example_count": 0,
            "control_safe": False,
        }
    meters_per_cell = meters_per_cell_from_manifest(manifest, records[0].bev.shape)
    robot_radius_m = robot_radius_m_from_manifest(manifest, meters_per_cell)
    default_source_name = str(manifest.get("source_name") or manifest.get("source_segment_id") or root.name)
    default_family = str(manifest.get("source_family") or source_kind)
    default_grade = str(manifest.get("supervision_grade") or manifest.get("robot_supervision_grade") or source_kind)
    frames = [
        LoadedBevFrame(
            record=record,
            source_root=root,
            source_name=default_source_name,
            source_family=default_family,
            scenario_name="unknown",
            supervision_grade=default_grade,
            source_kind=source_kind,
            source_manifest=manifest,
            frame_record={},
            meters_per_cell=meters_per_cell,
            robot_radius_m=robot_radius_m,
        )
        for record in records
    ]
    wrapped_metadata = dict(metadata)
    wrapped_metadata.update(
        {
            "source_kind": source_kind,
            "source": root.as_posix(),
            "example_count": len(frames),
            "control_safe": False,
        }
    )
    return frames, wrapped_metadata


def _read_optional_manifest(root: Path) -> JsonDict:
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        return read_json(manifest_path)
    bev_manifest_path = root / BEV_MANIFEST_FILE
    if bev_manifest_path.exists():
        return load_bev_manifest(root)
    return {}


def _scalar_or_record_str(example: dict[str, np.ndarray], record: JsonDict, field: str, default: str) -> str:
    if field in example:
        try:
            return np_scalar_to_str(example[field])
        except Exception:  # noqa: BLE001 - fall through to record/default.
            pass
    value = record.get(field)
    if isinstance(value, str) and value:
        return value
    return default


def _aggregate_frames(frames: list[JsonDict]) -> JsonDict:
    if not frames:
        return {
            "frame_count": 0,
            "robot_center_blocked_rate": 0.0,
            "footprint_blocked_rate": 0.0,
            "forward_corridor_free_rate": 0.0,
            "occupied_ratio_near_robot": 0.0,
            "unknown_ratio_near_robot": 0.0,
            "risky_ratio_near_robot": 0.0,
            "candidate_block_reason_counts": {},
            "origin_frame_status_distribution": {},
            "action_supervision_ok_fraction": 0.0,
        }
    counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for frame in frames:
        for key, value in dict(frame.get("candidate_block_reason_counts", {})).items():
            counts[str(key)] = counts.get(str(key), 0) + int(value)
        status = str(frame.get("origin_frame_status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "frame_count": len(frames),
        "robot_center_blocked_rate": _rate(frames, "robot_center_blocked"),
        "footprint_blocked_rate": _rate(frames, "footprint_blocked"),
        "forward_corridor_free_rate": _rate(frames, "forward_corridor_free"),
        "occupied_ratio_near_robot": _mean(frames, "occupied_ratio_near_robot"),
        "unknown_ratio_near_robot": _mean(frames, "unknown_ratio_near_robot"),
        "risky_ratio_near_robot": _mean(frames, "risky_ratio_near_robot"),
        "candidate_block_reason_counts": dict(sorted(counts.items())),
        "origin_frame_status_distribution": dict(sorted(status_counts.items())),
        "action_supervision_ok_fraction": _rate(frames, "action_supervision_ok"),
    }


def _cell_status(
    free: np.ndarray,
    occupied: np.ndarray,
    risky: np.ndarray,
    unknown: np.ndarray,
    cells: tuple[tuple[int, int], ...],
    *,
    config: ActionSanityConfig,
) -> JsonDict:
    free_values = _cell_values(free, cells, default=0.0)
    occupied_values = _cell_values(occupied, cells, default=1.0)
    risky_values = _cell_values(risky, cells, default=1.0)
    unknown_values = _cell_values(unknown, cells, default=1.0)
    free_fraction = float(np.mean(free_values >= np.float32(config.min_free_threshold)))
    occupied_max = float(np.max(occupied_values))
    risky_max = float(np.max(risky_values))
    unknown_max = float(np.max(unknown_values))
    blocked = (
        occupied_max >= config.occupied_block_threshold
        or risky_max >= config.occupied_block_threshold
        or unknown_max >= config.unknown_block_threshold
        or free_fraction < config.min_free_threshold
    )
    return {
        "free_fraction": free_fraction,
        "occupied_max": occupied_max,
        "risky_max": risky_max,
        "unknown_max": unknown_max,
        "blocked": bool(blocked),
    }


def _cell_ratios(
    occupied: np.ndarray,
    risky: np.ndarray,
    unknown: np.ndarray,
    cells: tuple[tuple[int, int], ...],
    *,
    config: ActionSanityConfig,
) -> JsonDict:
    occupied_values = _cell_values(occupied, cells, default=1.0)
    risky_values = _cell_values(risky, cells, default=1.0)
    unknown_values = _cell_values(unknown, cells, default=1.0)
    return {
        "occupied_ratio": float(np.mean(occupied_values >= np.float32(config.occupied_block_threshold))),
        "risky_ratio": float(np.mean(risky_values >= np.float32(config.occupied_block_threshold))),
        "unknown_ratio": float(np.mean(unknown_values >= np.float32(config.unknown_block_threshold))),
    }


def _cell_values(array: np.ndarray, cells: tuple[tuple[int, int], ...], *, default: float) -> np.ndarray:
    if not cells:
        return np.asarray([default], dtype=np.float32)
    values: list[float] = []
    for row, col in cells:
        if 0 <= row < array.shape[0] and 0 <= col < array.shape[1]:
            values.append(float(array[row, col]))
        else:
            values.append(default)
    if not values:
        values.append(default)
    return np.asarray(values, dtype=np.float32)


def _free(bev: LocalBev) -> np.ndarray:
    traversable = _prob(bev.traversable) if bev.traversable is not None else 0.0
    return np.maximum(_prob(bev.free), traversable)


def _occupied(bev: LocalBev) -> np.ndarray:
    risky = _prob(bev.risky) if bev.risky is not None else 0.0
    return np.maximum(_prob(bev.occupied), risky)


def _risky(bev: LocalBev) -> np.ndarray:
    if bev.risky is not None:
        return _prob(bev.risky)
    return _prob(bev.occupied)


def _prob(array: np.ndarray | float | None) -> np.ndarray:
    if array is None:
        return np.asarray(0.0, dtype=np.float32)
    return np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)


def _rate(frames: list[JsonDict], key: str) -> float:
    return float(sum(1 for frame in frames if bool(frame.get(key))) / max(len(frames), 1))


def _mean(frames: list[JsonDict], key: str) -> float:
    values = [float(frame.get(key, 0.0)) for frame in frames if isinstance(frame.get(key, 0.0), (int, float))]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def sanity_summary_json(
    *,
    source: str | Path,
    source_metadata: JsonDict,
    config: ActionSanityConfig,
    frame_metrics: list[JsonDict],
    contact_sheet_name: str,
) -> JsonDict:
    aggregate = aggregate_action_sanity(frame_metrics)
    return {
        "schema_version": BEV_ACTION_SANITY_SCHEMA_VERSION,
        "source": Path(source).as_posix(),
        "source_metadata": source_metadata,
        "config": config.to_dict(),
        **aggregate,
        "frames": frame_metrics,
        "worst_frame_contact_sheet": contact_sheet_name,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
    }


def parse_frame_metric_json(value: np.ndarray) -> JsonDict:
    loaded = json.loads(np_scalar_to_str(value))
    if not isinstance(loaded, dict):
        raise ValueError("expected action sanity JSON object")
    return loaded


def scalar_action_supervision_ok(value: bool) -> np.ndarray:
    return scalar_bool(bool(value))
