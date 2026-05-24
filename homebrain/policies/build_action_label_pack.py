from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from homebrain.data.spatial_dataset import (
    DETERMINISTIC_CREATED_AT_UTC,
    load_example_npz,
    np_scalar_to_bool,
    np_scalar_to_str,
    read_json,
    scalar_bool,
    scalar_float,
    scalar_int,
    scalar_str,
    write_deterministic_npz,
    write_json,
)
from homebrain.datasets.openloris_scene import OPENLORIS_ROUTE_ASSOCIATIONS_FILE
from homebrain.messages.schema import JsonDict
from homebrain.policies.bev_action_sanity import (
    ActionSanityConfig,
    LoadedBevFrame,
    contract_flags_for_frame,
    evaluate_loaded_frame,
    meters_per_cell_from_manifest,
    robot_radius_m_from_manifest,
    source_weight_for_sanity,
)
from homebrain.policies.candidate_trajectories import CandidateTrajectory, candidates_hash, generate_default_candidates
from homebrain.policies.run_trajectory_scorer import BevDecisionInput
from homebrain.policies.trajectory_scorer import LocalBev, RISKY_CANDIDATE_THRESHOLD
from homebrain.teachers.artifacts import file_sha256, relative_to_root

ACTION_LABEL_PACK_SCHEMA_VERSION = "homebrain.action_label_pack.v0"
ACTION_LABEL_PACK_SCHEMA_VERSION_V1 = "homebrain.action_label_pack.v1"
ACTION_LABEL_PACK_SCHEMA_VERSION_V2 = "homebrain.action_label_pack.v2"
ACTION_LABEL_PACK_SCHEMA_VERSION_V3 = "homebrain.action_label_pack.v3"
ACTION_LABEL_PACK_SCHEMA_VERSION_V4 = "homebrain.action_label_pack.v4"
ACTION_LABEL_EXAMPLE_SCHEMA_VERSION = "homebrain.action_label_example.v0"
COLLISION_THRESHOLD = RISKY_CANDIDATE_THRESHOLD
UNKNOWN_BLOCK_THRESHOLD = 0.95
FUTURE_MOTION_HORIZONS_S: tuple[float, ...] = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class SourceFrame:
    sequence_id: str
    camera_id: str
    frame_id: int
    timestamp_ns: int
    source_name: str
    source_family: str
    source_ref: str
    supervision_grade: str
    scenario_name: str
    bev: LocalBev
    action_supervision_ok: bool
    source_weight: float
    action_sanity: JsonDict
    contract_flags: JsonDict
    base_pose: JsonDict | None


@dataclass(frozen=True)
class ExpertCandidateLabel:
    candidate_id: str
    collision_proxy: float
    near_collision_proxy: float
    unknown_penalty: float
    uncertainty_penalty: float
    coverage_gain: float
    smoothness_cost: float
    total_expert_score: float
    selected_by_expert: bool


@dataclass(frozen=True)
class FutureMotionLabel:
    horizon_s: float
    best_candidate_id: str
    match_error: float
    target_frame_id: int
    dx: float
    dy: float
    dyaw: float
    mask: float


@dataclass(frozen=True)
class PreparedActionExample:
    frame: SourceFrame
    training_labels: tuple[ExpertCandidateLabel, ...]
    coverage_labels: tuple[ExpertCandidateLabel, ...]
    future_labels: tuple[FutureMotionLabel, ...]
    label_source_type: str


def build_action_label_pack(
    *,
    sources: list[str | Path],
    out_dir: str | Path,
    max_examples: int | None = None,
    pack_version: int = 0,
) -> Path:
    if not sources:
        raise ValueError("at least one source is required")
    output = Path(out_dir)
    examples_dir = output / "examples"
    _clear_generated_outputs(output)
    examples_dir.mkdir(parents=True, exist_ok=True)

    source_roots = [Path(source) for source in sources]
    frames: list[SourceFrame] = []
    source_manifests: list[JsonDict] = []
    for source_root in source_roots:
        loaded_frames, manifest = _load_spatial_pack_frames(source_root)
        frames.extend(loaded_frames)
        source_manifests.append(manifest)
    excluded_frames: list[JsonDict] = []
    if pack_version >= 1:
        kept: list[SourceFrame] = []
        for frame in frames:
            if frame.action_supervision_ok:
                kept.append(frame)
            else:
                excluded_frames.append(
                    {
                        "sequence_id": frame.sequence_id,
                        "camera_id": frame.camera_id,
                        "frame_id": frame.frame_id,
                        "timestamp_ns": frame.timestamp_ns,
                        "source_name": frame.source_name,
                        "source_family": frame.source_family,
                        "source_ref": frame.source_ref,
                        "scenario_name": frame.scenario_name,
                        "action_supervision_ok": False,
                        "source_weight": frame.source_weight,
                        "origin_frame_status": frame.action_sanity.get("origin_frame_status"),
                        "reject_reasons": frame.action_sanity.get("action_supervision_reject_reasons", []),
                    }
                )
        frames = kept
    if max_examples is not None:
        frames = frames[:max_examples]
    if not frames:
        raise ValueError("no BEV frames found for action labeling")

    first_shape = frames[0].bev.shape
    meters_per_cell = _meters_per_cell(source_manifests[0], first_shape)
    robot_radius_m = _robot_radius_m(source_manifests[0])
    candidates = generate_default_candidates(
        grid_shape=first_shape,
        meters_per_cell=meters_per_cell,
        robot_radius_m=robot_radius_m,
    )
    candidate_ids = [candidate.id for candidate in candidates]
    prepared: list[PreparedActionExample] = []
    frames_by_motion_key = _frames_by_future_motion_key(frames)

    for frame in frames:
        if frame.bev.shape != first_shape:
            raise ValueError(f"all ActionLabelPack examples must share one grid shape; got {frame.bev.shape}")
        coverage_labels = expert_labels_for_frame(bev=frame.bev, candidates=candidates)
        future_labels = _future_motion_labels_for_frame(
            frame,
            frames_by_key=frames_by_motion_key,
            candidates=candidates,
        )
        training_labels, label_source_type = _training_labels_for_v4(
            coverage_labels=coverage_labels,
            future_labels=future_labels,
            enabled=pack_version >= 4,
        )
        prepared.append(
            PreparedActionExample(
                frame=frame,
                training_labels=training_labels,
                coverage_labels=coverage_labels,
                future_labels=future_labels,
                label_source_type=label_source_type,
            )
        )

    raw_source_distribution = _distribution([item.frame.source_name for item in prepared])
    raw_selected_distribution = _distribution([_selected_label(item.training_labels).candidate_id for item in prepared])
    raw_label_source_distribution = _distribution([item.label_source_type for item in prepared])
    balanced = _balance_prepared_examples(prepared) if pack_version >= 4 else prepared

    examples: list[JsonDict] = []
    source_distribution: dict[str, int] = {}
    selected_distribution: dict[str, int] = {}
    label_source_distribution: dict[str, int] = {}

    for index, item in enumerate(balanced):
        frame = item.frame
        labels = item.training_labels
        selected = _selected_label(labels)
        source_distribution[frame.source_name] = source_distribution.get(frame.source_name, 0) + 1
        selected_distribution[selected.candidate_id] = selected_distribution.get(selected.candidate_id, 0) + 1
        label_source_distribution[item.label_source_type] = label_source_distribution.get(item.label_source_type, 0) + 1
        example_path = examples_dir / f"action_{index:06d}.npz"
        write_deterministic_npz(
            example_path,
            _example_arrays(
                frame,
                labels,
                coverage_labels=item.coverage_labels,
                future_labels=item.future_labels,
                label_source_type=item.label_source_type,
                pack_version=pack_version,
            ),
        )
        examples.append(
            {
                "example_path": relative_to_root(example_path, output),
                "example_sha256": file_sha256(example_path),
                "sequence_id": frame.sequence_id,
                "camera_id": frame.camera_id,
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "source_name": frame.source_name,
                "source_family": frame.source_family,
                "source_ref": frame.source_ref,
                "supervision_grade": frame.supervision_grade,
                "scenario_name": frame.scenario_name,
                "selected_candidate_id": selected.candidate_id,
                "selected_by_expert_count": 1,
                "candidate_count": len(labels),
                "label_source_type": item.label_source_type,
                "coverage_expert_selected_candidate_id": _selected_label(item.coverage_labels).candidate_id,
                "future_motion_primary_candidate_id": _primary_future_label(item.future_labels).best_candidate_id
                if _primary_future_label(item.future_labels).mask > 0.0
                else None,
                "future_motion_primary_match_error": _primary_future_label(item.future_labels).match_error
                if _primary_future_label(item.future_labels).mask > 0.0
                else None,
                "source_weight": round(float(frame.source_weight), 6),
                "action_supervision_ok": bool(frame.action_supervision_ok),
                "origin_frame_status": frame.action_sanity.get("origin_frame_status"),
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
            }
        )

    manifest: JsonDict = {
        "schema_version": _pack_schema_version(pack_version),
        "example_schema_version": ACTION_LABEL_EXAMPLE_SCHEMA_VERSION,
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "package_type": "ActionLabelPack",
        "version": int(pack_version),
        "source_dirs": [source.as_posix() for source in source_roots],
        "source_manifest_sha256": [
            file_sha256(source / "manifest.json") if (source / "manifest.json").exists() else "missing"
            for source in source_roots
        ],
        "example_count": len(examples),
        "candidate_count": len(candidates),
        "candidate_ids": candidate_ids,
        "candidate_hash": candidates_hash(candidates),
        "grid_shape": [first_shape[0], first_shape[1]],
        "meters_per_cell": round(float(meters_per_cell), 6),
        "robot_radius_m": round(float(robot_radius_m), 6),
        "source_distribution": dict(sorted(source_distribution.items())),
        "excluded_source_distribution": _excluded_source_distribution(excluded_frames),
        "source_frame_counts": _source_frame_counts(frames, excluded_frames),
        "raw_source_distribution": raw_source_distribution,
        "raw_selected_distribution": raw_selected_distribution,
        "raw_label_source_distribution": raw_label_source_distribution,
        "selected_distribution": dict(sorted(selected_distribution.items())),
        "label_source_distribution": dict(sorted(label_source_distribution.items())),
        "balanced_source_distribution": dict(sorted(source_distribution.items())),
        "balanced_selected_distribution": dict(sorted(selected_distribution.items())),
        "balancing": _balancing_record(
            enabled=pack_version >= 4,
            raw_count=len(prepared),
            balanced_count=len(balanced),
            raw_source_distribution=raw_source_distribution,
            raw_selected_distribution=raw_selected_distribution,
            balanced_source_distribution=source_distribution,
            balanced_selected_distribution=selected_distribution,
        ),
        "scoring": _scoring_description(),
        "action_sanity_filter": {
            "enabled": bool(pack_version >= 1),
            "schema_version": "homebrain.bev_action_sanity.v0",
            "excluded_frame_count": len(excluded_frames),
            "excluded_by_source": _excluded_source_distribution(excluded_frames),
            "included_by_source": dict(sorted(source_distribution.items())),
            "policy": "v1 includes only action_supervision_ok frames; excluded frames are retained as manifest audit records",
        },
        "excluded_frames": excluded_frames,
        "supervision_type": "deterministic_candidate_trajectory_scores",
        "future_motion_supervision": {
            "enabled": bool(pack_version >= 4),
            "horizons_s": list(FUTURE_MOTION_HORIZONS_S),
            "source": "robot_base_pose_or_odom_matching_to_candidate_trajectories",
            "primary_horizon_s": 1.0,
            "kept_separate_from_coverage_risk_expert": True,
            "control_safe": False,
        },
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "control_safe_claim": False,
        "examples": examples,
    }
    write_json(output / "manifest.json", manifest, pretty=True)
    return output / "manifest.json"


def _pack_schema_version(pack_version: int) -> str:
    if pack_version >= 4:
        return ACTION_LABEL_PACK_SCHEMA_VERSION_V4
    if pack_version >= 3:
        return ACTION_LABEL_PACK_SCHEMA_VERSION_V3
    if pack_version >= 2:
        return ACTION_LABEL_PACK_SCHEMA_VERSION_V2
    if pack_version >= 1:
        return ACTION_LABEL_PACK_SCHEMA_VERSION_V1
    return ACTION_LABEL_PACK_SCHEMA_VERSION


def _excluded_source_distribution(excluded_frames: list[JsonDict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in excluded_frames:
        source = str(frame.get("source_name", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _source_frame_counts(frames: list[SourceFrame], excluded_frames: list[JsonDict]) -> dict[str, JsonDict]:
    counts: dict[str, JsonDict] = {}
    for frame in frames:
        record = counts.setdefault(frame.source_name, {"included": 0, "excluded": 0, "total": 0})
        record["included"] = int(record["included"]) + 1
        record["total"] = int(record["total"]) + 1
    for frame in excluded_frames:
        source = str(frame.get("source_name", "unknown"))
        record = counts.setdefault(source, {"included": 0, "excluded": 0, "total": 0})
        record["excluded"] = int(record["excluded"]) + 1
        record["total"] = int(record["total"]) + 1
    return dict(sorted(counts.items()))


def expert_labels_for_frame(
    *,
    bev: LocalBev,
    candidates: list[CandidateTrajectory],
) -> tuple[ExpertCandidateLabel, ...]:
    bev.validate()
    if not candidates:
        raise ValueError("at least one candidate is required")
    raw = [
        _raw_expert_label(candidate, bev=bev, selected_by_expert=False)
        for candidate in candidates
    ]
    motion = [
        label
        for candidate, label in zip(candidates, raw)
        if label.candidate_id != "stop" and abs(float(candidate.cmd_vel_proxy.get("linear_velocity_mps", 0.0))) > 0.0
    ]
    all_motion_blocked = bool(motion) and all(
        label.collision_proxy >= COLLISION_THRESHOLD or label.unknown_penalty >= UNKNOWN_BLOCK_THRESHOLD
        or label.coverage_gain <= 0.0
        for label in motion
    )
    straight_motion = [
        label
        for label in raw
        if label.candidate_id in {"straight_short", "straight_medium"}
    ]
    straight_blocked = bool(straight_motion) and all(
        label.collision_proxy >= COLLISION_THRESHOLD or label.unknown_penalty >= UNKNOWN_BLOCK_THRESHOLD
        for label in straight_motion
    )
    all_motion_blocked = all_motion_blocked or straight_blocked
    adjusted = [_adjust_stop_label(label, all_motion_blocked=all_motion_blocked) for label in raw]
    selected_index = min(
        range(len(adjusted)),
        key=lambda index: (adjusted[index].total_expert_score, index),
    )
    return tuple(
        ExpertCandidateLabel(
            candidate_id=label.candidate_id,
            collision_proxy=label.collision_proxy,
            near_collision_proxy=label.near_collision_proxy,
            unknown_penalty=label.unknown_penalty,
            uncertainty_penalty=label.uncertainty_penalty,
            coverage_gain=label.coverage_gain,
            smoothness_cost=label.smoothness_cost,
            total_expert_score=label.total_expert_score,
            selected_by_expert=index == selected_index,
        )
        for index, label in enumerate(adjusted)
    )


def _raw_expert_label(
    candidate: CandidateTrajectory,
    *,
    bev: LocalBev,
    selected_by_expert: bool,
) -> ExpertCandidateLabel:
    cells = tuple(candidate.footprint_cells)
    occupied = np.maximum(_as_probability(bev.occupied), _as_probability(bev.risky) if bev.risky is not None else 0.0)
    free = np.maximum(_as_probability(bev.free), _as_probability(bev.traversable) if bev.traversable is not None else 0.0)
    unknown = _as_probability(bev.unknown)
    if bev.uncertainty is not None:
        uncertainty = _as_probability(bev.uncertainty)
    elif bev.confidence is not None:
        uncertainty = 1.0 - _as_probability(bev.confidence)
    else:
        uncertainty = unknown

    collision = float(np.max(_cell_values(occupied, cells, default=1.0)))
    near_cells = _inflate_cells(cells, shape=bev.shape, radius_cells=1)
    near_collision = float(np.max(_cell_values(occupied, near_cells, default=1.0)))
    unknown_penalty = float(np.mean(_cell_values(unknown, cells, default=1.0)))
    uncertainty_penalty = float(np.mean(_cell_values(uncertainty, cells, default=1.0)))
    coverage_gain = _coverage_gain(candidate, free=free)
    smoothness = _smoothness_cost(candidate)
    total = (
        25.0 * collision
        + 4.0 * near_collision
        + 1.6 * unknown_penalty
        + 0.8 * uncertainty_penalty
        + 0.25 * smoothness
        - 0.18 * coverage_gain
    )
    if not cells:
        total += 10.0
    return ExpertCandidateLabel(
        candidate_id=candidate.id,
        collision_proxy=collision,
        near_collision_proxy=near_collision,
        unknown_penalty=unknown_penalty,
        uncertainty_penalty=uncertainty_penalty,
        coverage_gain=coverage_gain,
        smoothness_cost=smoothness,
        total_expert_score=total,
        selected_by_expert=selected_by_expert,
    )


def _adjust_stop_label(label: ExpertCandidateLabel, *, all_motion_blocked: bool) -> ExpertCandidateLabel:
    if label.candidate_id != "stop":
        return label
    stop_delta = -8.0 if all_motion_blocked else 8.0
    return ExpertCandidateLabel(
        candidate_id=label.candidate_id,
        collision_proxy=label.collision_proxy,
        near_collision_proxy=label.near_collision_proxy,
        unknown_penalty=label.unknown_penalty,
        uncertainty_penalty=label.uncertainty_penalty,
        coverage_gain=label.coverage_gain,
        smoothness_cost=label.smoothness_cost,
        total_expert_score=label.total_expert_score + stop_delta,
        selected_by_expert=label.selected_by_expert,
    )


def _example_arrays(
    frame: SourceFrame,
    labels: tuple[ExpertCandidateLabel, ...],
    *,
    coverage_labels: tuple[ExpertCandidateLabel, ...],
    future_labels: tuple[FutureMotionLabel, ...],
    label_source_type: str,
    pack_version: int,
) -> dict[str, np.ndarray]:
    selected = _selected_label(labels)
    coverage_selected = _selected_label(coverage_labels)
    arrays = {
        "schema_version": scalar_str(ACTION_LABEL_EXAMPLE_SCHEMA_VERSION),
        "frame_id": scalar_int(frame.frame_id),
        "timestamp_ns": scalar_int(frame.timestamp_ns),
        "sequence_id": scalar_str(frame.sequence_id),
        "camera_id": scalar_str(frame.camera_id),
        "source_name": scalar_str(frame.source_name),
        "source_family": scalar_str(frame.source_family),
        "source_ref": scalar_str(frame.source_ref),
        "supervision_grade": scalar_str(frame.supervision_grade),
        "scenario_name": scalar_str(frame.scenario_name),
        "source_weight": scalar_float(frame.source_weight),
        "action_supervision_ok": scalar_bool(frame.action_supervision_ok),
        "origin_frame_status": scalar_str(str(frame.action_sanity.get("origin_frame_status", "unknown"))),
        "robot_frame_truth": scalar_bool(bool(frame.action_sanity.get("robot_frame_truth", False))),
        "action_sanity_json": scalar_str(json.dumps(frame.action_sanity, sort_keys=True, separators=(",", ":"))),
        "candidate_ids": np.asarray([label.candidate_id for label in labels]),
        "candidate_count": scalar_int(len(labels)),
        "collision_proxy": np.asarray([label.collision_proxy for label in labels], dtype=np.float32),
        "near_collision_proxy": np.asarray([label.near_collision_proxy for label in labels], dtype=np.float32),
        "unknown_penalty": np.asarray([label.unknown_penalty for label in labels], dtype=np.float32),
        "uncertainty_penalty": np.asarray([label.uncertainty_penalty for label in labels], dtype=np.float32),
        "coverage_gain": np.asarray([label.coverage_gain for label in labels], dtype=np.float32),
        "smoothness_cost": np.asarray([label.smoothness_cost for label in labels], dtype=np.float32),
        "total_expert_score": np.asarray([label.total_expert_score for label in labels], dtype=np.float32),
        "selected_by_expert": np.asarray([label.selected_by_expert for label in labels], dtype=np.bool_),
        "selected_candidate_id": scalar_str(selected.candidate_id),
        "label_source_type": scalar_str(label_source_type),
        "replay_only": scalar_bool(True),
        "not_executed": scalar_bool(True),
        "control_safe": scalar_bool(False),
        "raw_pwm_emitted": scalar_bool(False),
    }
    if pack_version >= 4:
        arrays.update(
            {
                "coverage_expert_score": np.asarray(
                    [label.total_expert_score for label in coverage_labels],
                    dtype=np.float32,
                ),
                "coverage_selected_by_expert": np.asarray(
                    [label.selected_by_expert for label in coverage_labels],
                    dtype=np.bool_,
                ),
                "coverage_expert_selected_candidate_id": scalar_str(coverage_selected.candidate_id),
                "future_horizons_s": np.asarray([label.horizon_s for label in future_labels], dtype=np.float32),
                "future_best_candidate_by_odom": np.asarray(
                    [label.best_candidate_id for label in future_labels],
                ),
                "future_match_error": np.asarray([label.match_error for label in future_labels], dtype=np.float32),
                "future_label_mask": np.asarray([label.mask for label in future_labels], dtype=np.float32),
                "future_target_frame_id": np.asarray([label.target_frame_id for label in future_labels], dtype=np.int64),
                "future_dx_dy_dyaw": np.asarray(
                    [[label.dx, label.dy, label.dyaw] for label in future_labels],
                    dtype=np.float32,
                ),
                "future_motion_primary_candidate_id": scalar_str(_primary_future_label(future_labels).best_candidate_id),
                "future_motion_primary_match_error": scalar_float(_primary_future_label(future_labels).match_error),
                "future_motion_primary_mask": scalar_float(_primary_future_label(future_labels).mask),
            }
        )
    return arrays


def _load_spatial_pack_frames(root: Path) -> tuple[list[SourceFrame], JsonDict]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest.json in source pack: {root}")
    manifest = read_json(manifest_path)
    examples = manifest.get("examples") or manifest.get("frames")
    if not isinstance(examples, list):
        raise ValueError(f"source pack manifest examples must be a list: {root}")
    records: list[SourceFrame] = []
    default_source_name = str(manifest.get("source_name") or manifest.get("source_segment_id") or root.name)
    default_source_family = str(manifest.get("source_family") or "unknown")
    default_grade = str(manifest.get("supervision_grade") or manifest.get("robot_supervision_grade") or "unknown")
    meters_per_cell = meters_per_cell_from_manifest(manifest, tuple(manifest.get("grid_shape", [32, 32])))  # type: ignore[arg-type]
    robot_radius_m = robot_radius_m_from_manifest(manifest, meters_per_cell)
    sanity_config = ActionSanityConfig()
    for record in examples:
        if not isinstance(record, dict) or not isinstance(record.get("example_path"), str):
            continue
        example_path = root / str(record["example_path"])
        example = load_example_npz(example_path)
        _validate_source_flags(example, example_path)
        source_name = _scalar_or_record_str(example, record, "source_name", default_source_name)
        source_family = _scalar_or_record_str(example, record, "source_family", default_source_family)
        supervision_grade = _scalar_or_record_str(example, record, "supervision_grade", default_grade)
        scenario_name = _scalar_or_record_str(example, record, "scenario_name", "unknown")
        free = np.asarray(example["bev_free"], dtype=np.float32)
        obstacle = np.asarray(example["bev_obstacle"], dtype=np.float32)
        unknown = np.asarray(example["bev_unknown"], dtype=np.float32)
        confidence = np.asarray(example["bev_confidence"], dtype=np.float32)
        traversable = np.asarray(example["bev_traversable"], dtype=np.float32) if "bev_traversable" in example else free.copy()
        risky = np.asarray(example["bev_risky"], dtype=np.float32) if "bev_risky" in example else obstacle.copy()
        bev = LocalBev(
            free=free,
            occupied=obstacle,
            unknown=unknown,
            traversable=traversable,
            risky=risky,
            confidence=confidence,
            source="labels",
        )
        decision_input = BevDecisionInput(
            sequence_id=str(record.get("sequence_id", manifest.get("source_segment_id", root.name))),
            camera_id=str(record.get("camera_id", "front_rgb")),
            frame_id=int(record.get("frame_id", np.asarray(example["frame_id"]).item())),
            timestamp_ns=int(record.get("timestamp_ns", np.asarray(example["timestamp_ns"]).item())),
            bev=bev,
            pose_delta=None,
            source_ref=str(record["example_path"]),
        )
        loaded_frame = LoadedBevFrame(
            record=decision_input,
            source_root=root,
            source_name=source_name,
            source_family=source_family,
            scenario_name=scenario_name,
            supervision_grade=supervision_grade,
            source_kind="spatial_train_pack",
            source_manifest=manifest,
            frame_record=record,
            meters_per_cell=meters_per_cell,
            robot_radius_m=robot_radius_m,
        )
        action_sanity = evaluate_loaded_frame(loaded_frame, config=sanity_config)
        contract_flags = contract_flags_for_frame(loaded_frame)
        records.append(
            SourceFrame(
                sequence_id=decision_input.sequence_id,
                camera_id=decision_input.camera_id,
                frame_id=decision_input.frame_id,
                timestamp_ns=decision_input.timestamp_ns,
                source_name=source_name,
                source_family=source_family,
                source_ref=str(record["example_path"]),
                supervision_grade=supervision_grade,
                scenario_name=scenario_name,
                bev=bev,
                action_supervision_ok=bool(action_sanity.get("action_supervision_ok", False)),
                source_weight=source_weight_for_sanity(action_sanity),
                action_sanity=action_sanity,
                contract_flags=contract_flags,
                base_pose=_base_pose_for_source_frame(manifest, decision_input.frame_id),
            )
        )
    return records, manifest


def _base_pose_for_source_frame(manifest: JsonDict, frame_id: int) -> JsonDict | None:
    source_log = manifest.get("source_log")
    if not isinstance(source_log, str):
        return None
    associations_path = Path(source_log) / OPENLORIS_ROUTE_ASSOCIATIONS_FILE
    if not associations_path.exists():
        return None
    cache = getattr(_base_pose_for_source_frame, "_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(_base_pose_for_source_frame, "_cache", cache)
    cached = cache.get(associations_path.as_posix())
    if isinstance(cached, dict):
        return cached.get(int(frame_id))
    try:
        associations = read_json(associations_path)
    except Exception:  # noqa: BLE001
        return None
    frames = associations.get("frames")
    if not isinstance(frames, list):
        return None
    pose_by_frame: dict[int, JsonDict] = {}
    for record in frames:
        if not isinstance(record, dict):
            continue
        record_frame_id = int(record.get("frame_id", -1))
        base_pose = record.get("base_pose")
        if isinstance(base_pose, dict):
            pose_by_frame[record_frame_id] = base_pose
            continue
        odom = record.get("odom")
        if isinstance(odom, dict) and isinstance(odom.get("pose"), dict):
            pose_by_frame[record_frame_id] = odom["pose"]  # type: ignore[assignment]
    cache[associations_path.as_posix()] = pose_by_frame
    return pose_by_frame.get(int(frame_id))


def _frames_by_future_motion_key(frames: list[SourceFrame]) -> dict[tuple[str, str], list[SourceFrame]]:
    grouped: dict[tuple[str, str], list[SourceFrame]] = {}
    for frame in frames:
        grouped.setdefault((frame.source_name, frame.sequence_id), []).append(frame)
    return {key: sorted(values, key=lambda item: item.timestamp_ns) for key, values in grouped.items()}


def _future_motion_labels_for_frame(
    frame: SourceFrame,
    *,
    frames_by_key: dict[tuple[str, str], list[SourceFrame]],
    candidates: list[CandidateTrajectory],
) -> tuple[FutureMotionLabel, ...]:
    if frame.base_pose is None:
        return tuple(_missing_future_label(horizon) for horizon in FUTURE_MOTION_HORIZONS_S)
    sequence = frames_by_key.get((frame.source_name, frame.sequence_id), [])
    labels: list[FutureMotionLabel] = []
    for horizon_s in FUTURE_MOTION_HORIZONS_S:
        target = _future_target_frame(sequence, frame.timestamp_ns, horizon_s)
        if target is None or target.base_pose is None:
            labels.append(_missing_future_label(horizon_s))
            continue
        dx, dy, dyaw = _relative_robot_delta(frame.base_pose, target.base_pose)
        best_id, error = _best_matching_candidate(
            dx=dx,
            dy=dy,
            dyaw=dyaw,
            horizon_s=horizon_s,
            candidates=candidates,
        )
        labels.append(
            FutureMotionLabel(
                horizon_s=float(horizon_s),
                best_candidate_id=best_id,
                match_error=float(error),
                target_frame_id=target.frame_id,
                dx=float(dx),
                dy=float(dy),
                dyaw=float(dyaw),
                mask=1.0,
            )
        )
    return tuple(labels)


def _missing_future_label(horizon_s: float) -> FutureMotionLabel:
    return FutureMotionLabel(
        horizon_s=float(horizon_s),
        best_candidate_id="missing",
        match_error=float("inf"),
        target_frame_id=-1,
        dx=0.0,
        dy=0.0,
        dyaw=0.0,
        mask=0.0,
    )


def _future_target_frame(sequence: list[SourceFrame], timestamp_ns: int, horizon_s: float) -> SourceFrame | None:
    target_ns = int(timestamp_ns + round(horizon_s * 1_000_000_000))
    candidates = [frame for frame in sequence if frame.timestamp_ns >= target_ns and frame.base_pose is not None]
    return candidates[0] if candidates else None


def _best_matching_candidate(
    *,
    dx: float,
    dy: float,
    dyaw: float,
    horizon_s: float,
    candidates: list[CandidateTrajectory],
) -> tuple[str, float]:
    best_id = candidates[0].id if candidates else "missing"
    best_error = float("inf")
    for candidate in candidates:
        pose = _candidate_pose_at_horizon(candidate, horizon_s)
        yaw_error = _angle_abs_diff(dyaw, pose.yaw_rad)
        trans_error = math.hypot(dx - pose.x_m, dy - pose.y_m)
        error = trans_error + 0.25 * yaw_error
        if error < best_error:
            best_error = float(error)
            best_id = candidate.id
    return best_id, best_error


def _candidate_pose_at_horizon(candidate: CandidateTrajectory, horizon_s: float):
    if not candidate.poses:
        raise ValueError(f"candidate {candidate.id} has no poses")
    if candidate.duration_s <= 0.0 or horizon_s >= candidate.duration_s:
        return candidate.poses[-1]
    index = int(round((horizon_s / candidate.duration_s) * (len(candidate.poses) - 1)))
    index = max(0, min(index, len(candidate.poses) - 1))
    return candidate.poses[index]


def _angle_abs_diff(a: float, b: float) -> float:
    diff = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(float(diff))


def _relative_robot_delta(current_pose: JsonDict, target_pose: JsonDict) -> tuple[float, float, float]:
    current = _pose_matrix(current_pose)
    target = _pose_matrix(target_pose)
    relative = np.linalg.inv(current) @ target
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
        raise ValueError("robot pose quaternion has zero norm")
    x = qx / norm
    y = qy / norm
    z = qz / norm
    w = qw / norm
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [float(pose["tx"]), float(pose["ty"]), float(pose["tz"])]
    return transform


def _training_labels_for_v4(
    *,
    coverage_labels: tuple[ExpertCandidateLabel, ...],
    future_labels: tuple[FutureMotionLabel, ...],
    enabled: bool,
) -> tuple[tuple[ExpertCandidateLabel, ...], str]:
    if not enabled:
        return coverage_labels, "coverage_risk_expert"
    primary = _primary_future_label(future_labels)
    if primary.mask <= 0.0 or primary.best_candidate_id == "missing":
        return coverage_labels, "coverage_risk_expert"
    return _labels_with_selected_candidate(coverage_labels, primary.best_candidate_id), "future_motion_odom"


def _primary_future_label(labels: tuple[FutureMotionLabel, ...]) -> FutureMotionLabel:
    valid = [label for label in labels if label.mask > 0.0]
    if not valid:
        return _missing_future_label(1.0)
    return min(valid, key=lambda label: (abs(label.horizon_s - 1.0), label.match_error))


def _labels_with_selected_candidate(
    labels: tuple[ExpertCandidateLabel, ...],
    selected_candidate_id: str,
) -> tuple[ExpertCandidateLabel, ...]:
    if selected_candidate_id not in {label.candidate_id for label in labels}:
        return labels
    min_score = min(label.total_expert_score for label in labels)
    adjusted: list[ExpertCandidateLabel] = []
    for label in labels:
        selected = label.candidate_id == selected_candidate_id
        score = min_score - 0.01 if selected else max(label.total_expert_score, min_score + 0.01)
        adjusted.append(
            ExpertCandidateLabel(
                candidate_id=label.candidate_id,
                collision_proxy=label.collision_proxy,
                near_collision_proxy=label.near_collision_proxy,
                unknown_penalty=label.unknown_penalty,
                uncertainty_penalty=label.uncertainty_penalty,
                coverage_gain=label.coverage_gain,
                smoothness_cost=label.smoothness_cost,
                total_expert_score=float(score),
                selected_by_expert=selected,
            )
        )
    return tuple(adjusted)


def _balance_prepared_examples(examples: list[PreparedActionExample]) -> list[PreparedActionExample]:
    if not examples:
        return []
    source_counts = _distribution([item.frame.source_name for item in examples])
    source_total = sum(source_counts.values())
    dominant_source, dominant_source_count = max(source_counts.items(), key=lambda item: item[1])
    source_fraction = dominant_source_count / max(source_total, 1)
    if source_fraction > 0.65:
        other_source_count = source_total - dominant_source_count
        source_cap = max(1, int((0.65 / 0.35) * max(other_source_count, 1)))
    else:
        source_cap = len(examples)
    source_seen: dict[str, int] = {}
    source_capped: list[PreparedActionExample] = []
    for item in examples:
        source = item.frame.source_name
        if source == dominant_source and source_seen.get(source, 0) >= source_cap:
            continue
        source_seen[source] = source_seen.get(source, 0) + 1
        source_capped.append(item)

    action_counts = _distribution([_selected_label(item.training_labels).candidate_id for item in source_capped])
    if len(action_counts) <= 1:
        return source_capped
    total = sum(action_counts.values())
    dominant_action, dominant_count = max(action_counts.items(), key=lambda item: item[1])
    dominant_fraction = dominant_count / max(total, 1)
    if dominant_fraction <= 0.65:
        return source_capped
    other_count = total - dominant_count
    action_cap = max(1, int((0.65 / 0.35) * max(other_count, 1)))
    action_seen: dict[str, int] = {}
    balanced: list[PreparedActionExample] = []
    for item in source_capped:
        action = _selected_label(item.training_labels).candidate_id
        if action == dominant_action and action_seen.get(action, 0) >= action_cap:
            continue
        action_seen[action] = action_seen.get(action, 0) + 1
        balanced.append(item)
    return balanced


def _balancing_record(
    *,
    enabled: bool,
    raw_count: int,
    balanced_count: int,
    raw_source_distribution: dict[str, int],
    raw_selected_distribution: dict[str, int],
    balanced_source_distribution: dict[str, int],
    balanced_selected_distribution: dict[str, int],
) -> JsonDict:
    return {
        "enabled": bool(enabled),
        "method": "deterministic_source_cap_then_action_cap" if enabled else "none",
        "raw_example_count": int(raw_count),
        "balanced_example_count": int(balanced_count),
        "raw_source_distribution": dict(sorted(raw_source_distribution.items())),
        "raw_selected_distribution": dict(sorted(raw_selected_distribution.items())),
        "balanced_source_distribution": dict(sorted(balanced_source_distribution.items())),
        "balanced_selected_distribution": dict(sorted(balanced_selected_distribution.items())),
    }


def _distribution(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _validate_source_flags(example: dict[str, np.ndarray], example_path: Path) -> None:
    if "control_safe" in example and np_scalar_to_bool(example["control_safe"]) is not False:
        raise ValueError(f"source example must be control_safe=false: {example_path}")
    if "weak_label" in example and np_scalar_to_bool(example["weak_label"]) is not True:
        raise ValueError(f"source example must be weak_label=true: {example_path}")


def _selected_label(labels: tuple[ExpertCandidateLabel, ...]) -> ExpertCandidateLabel:
    selected = [label for label in labels if label.selected_by_expert]
    if len(selected) != 1:
        raise ValueError("expected exactly one selected expert label")
    return selected[0]


def _meters_per_cell(manifest: JsonDict, shape: tuple[int, int]) -> float:
    camera_config = manifest.get("camera_config")
    if isinstance(camera_config, dict):
        value = camera_config.get("meters_per_cell")
        if isinstance(value, (int, float)) and float(value) > 0.0:
            return float(value)
    if min(shape) <= 8:
        return 0.5
    return 0.05


def _robot_radius_m(manifest: JsonDict) -> float:
    camera_config = manifest.get("camera_config")
    if isinstance(camera_config, dict):
        value = camera_config.get("robot_radius_m")
        if isinstance(value, (int, float)) and float(value) >= 0.0:
            return float(value)
    return 0.18


def _scalar_or_record_str(
    example: dict[str, np.ndarray],
    record: JsonDict,
    field: str,
    default: str,
) -> str:
    if field in example:
        try:
            return np_scalar_to_str(example[field])
        except Exception:  # noqa: BLE001
            pass
    value = record.get(field)
    if isinstance(value, str) and value:
        return value
    return default


def _coverage_gain(candidate: CandidateTrajectory, *, free: np.ndarray) -> float:
    gain = 0
    for row, col in candidate.footprint_cells:
        if 0 <= row < free.shape[0] and 0 <= col < free.shape[1] and free[row, col] >= 0.5:
            gain += 1
    return float(gain)


def _smoothness_cost(candidate: CandidateTrajectory) -> float:
    angular = abs(float(candidate.cmd_vel_proxy.get("angular_velocity_radps", 0.0)))
    linear = abs(float(candidate.cmd_vel_proxy.get("linear_velocity_mps", 0.0)))
    return float(angular * candidate.duration_s + (0.05 if linear == 0.0 and angular > 0.0 else 0.0))


def _inflate_cells(
    cells: tuple[tuple[int, int], ...],
    *,
    shape: tuple[int, int],
    radius_cells: int,
) -> tuple[tuple[int, int], ...]:
    inflated: set[tuple[int, int]] = set()
    for row, col in cells:
        for row_offset in range(-radius_cells, radius_cells + 1):
            for col_offset in range(-radius_cells, radius_cells + 1):
                rr = row + row_offset
                cc = col + col_offset
                if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                    inflated.add((rr, cc))
    return tuple(sorted(inflated))


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


def _as_probability(array: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)


def _scoring_description() -> JsonDict:
    return {
        "collision_weight": 25.0,
        "near_collision_weight": 4.0,
        "unknown_weight": 1.6,
        "uncertainty_weight": 0.8,
        "smoothness_weight": 0.25,
        "coverage_gain_weight": -0.18,
        "stop_bonus_when_all_motion_blocked": -8.0,
        "stop_penalty_when_motion_available": 8.0,
        "collision_threshold": COLLISION_THRESHOLD,
        "unknown_block_threshold": UNKNOWN_BLOCK_THRESHOLD,
        "not_learned": True,
        "not_control_safe": True,
    }


def _clear_generated_outputs(output: Path) -> None:
    if (output / "examples").exists():
        shutil.rmtree(output / "examples")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic replay-only ActionLabelPack v0.")
    parser.add_argument("--source", action="append", required=True, help="Input controlled/reviewed BEV pack.")
    parser.add_argument("--out", required=True, help="Output ActionLabelPack directory.")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--pack-version", type=int, choices=(0, 1, 2, 3, 4), default=0)
    args = parser.parse_args(argv)
    manifest = build_action_label_pack(
        sources=[Path(source) for source in args.source],
        out_dir=args.out,
        max_examples=args.max_examples,
        pack_version=args.pack_version,
    )
    print(json.dumps({"manifest": manifest.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
