from __future__ import annotations

import argparse
import json
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
from homebrain.messages.schema import JsonDict
from homebrain.policies.candidate_trajectories import CandidateTrajectory, candidates_hash, generate_default_candidates
from homebrain.policies.trajectory_scorer import LocalBev, RISKY_CANDIDATE_THRESHOLD
from homebrain.teachers.artifacts import file_sha256, relative_to_root

ACTION_LABEL_PACK_SCHEMA_VERSION = "homebrain.action_label_pack.v0"
ACTION_LABEL_EXAMPLE_SCHEMA_VERSION = "homebrain.action_label_example.v0"
COLLISION_THRESHOLD = RISKY_CANDIDATE_THRESHOLD
UNKNOWN_BLOCK_THRESHOLD = 0.95


@dataclass(frozen=True)
class SourceFrame:
    sequence_id: str
    camera_id: str
    frame_id: int
    timestamp_ns: int
    source_name: str
    source_ref: str
    supervision_grade: str
    scenario_name: str
    bev: LocalBev


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


def build_action_label_pack(
    *,
    sources: list[str | Path],
    out_dir: str | Path,
    max_examples: int | None = None,
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
    examples: list[JsonDict] = []
    source_distribution: dict[str, int] = {}
    selected_distribution: dict[str, int] = {}

    for index, frame in enumerate(frames):
        if frame.bev.shape != first_shape:
            raise ValueError(f"all ActionLabelPack examples must share one grid shape; got {frame.bev.shape}")
        labels = expert_labels_for_frame(bev=frame.bev, candidates=candidates)
        selected = _selected_label(labels)
        source_distribution[frame.source_name] = source_distribution.get(frame.source_name, 0) + 1
        selected_distribution[selected.candidate_id] = selected_distribution.get(selected.candidate_id, 0) + 1
        example_path = examples_dir / f"action_{index:06d}.npz"
        write_deterministic_npz(example_path, _example_arrays(frame, labels))
        examples.append(
            {
                "example_path": relative_to_root(example_path, output),
                "example_sha256": file_sha256(example_path),
                "sequence_id": frame.sequence_id,
                "camera_id": frame.camera_id,
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "source_name": frame.source_name,
                "source_ref": frame.source_ref,
                "supervision_grade": frame.supervision_grade,
                "scenario_name": frame.scenario_name,
                "selected_candidate_id": selected.candidate_id,
                "selected_by_expert_count": 1,
                "candidate_count": len(labels),
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
            }
        )

    manifest: JsonDict = {
        "schema_version": ACTION_LABEL_PACK_SCHEMA_VERSION,
        "example_schema_version": ACTION_LABEL_EXAMPLE_SCHEMA_VERSION,
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "package_type": "ActionLabelPack",
        "version": 0,
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
        "selected_distribution": dict(sorted(selected_distribution.items())),
        "scoring": _scoring_description(),
        "supervision_type": "deterministic_candidate_trajectory_scores",
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "control_safe_claim": False,
        "examples": examples,
    }
    write_json(output / "manifest.json", manifest, pretty=True)
    return output / "manifest.json"


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
    motion = [label for label in raw if label.candidate_id != "stop"]
    all_motion_blocked = bool(motion) and all(
        label.collision_proxy >= COLLISION_THRESHOLD or label.unknown_penalty >= UNKNOWN_BLOCK_THRESHOLD
        for label in motion
    )
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


def _example_arrays(frame: SourceFrame, labels: tuple[ExpertCandidateLabel, ...]) -> dict[str, np.ndarray]:
    selected = _selected_label(labels)
    return {
        "schema_version": scalar_str(ACTION_LABEL_EXAMPLE_SCHEMA_VERSION),
        "frame_id": scalar_int(frame.frame_id),
        "timestamp_ns": scalar_int(frame.timestamp_ns),
        "sequence_id": scalar_str(frame.sequence_id),
        "camera_id": scalar_str(frame.camera_id),
        "source_name": scalar_str(frame.source_name),
        "source_ref": scalar_str(frame.source_ref),
        "supervision_grade": scalar_str(frame.supervision_grade),
        "scenario_name": scalar_str(frame.scenario_name),
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
        "replay_only": scalar_bool(True),
        "not_executed": scalar_bool(True),
        "control_safe": scalar_bool(False),
        "raw_pwm_emitted": scalar_bool(False),
    }


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
    default_grade = str(manifest.get("supervision_grade") or manifest.get("robot_supervision_grade") or "unknown")
    for record in examples:
        if not isinstance(record, dict) or not isinstance(record.get("example_path"), str):
            continue
        example_path = root / str(record["example_path"])
        example = load_example_npz(example_path)
        _validate_source_flags(example, example_path)
        source_name = _scalar_or_record_str(example, record, "source_name", default_source_name)
        supervision_grade = _scalar_or_record_str(example, record, "supervision_grade", default_grade)
        scenario_name = _scalar_or_record_str(example, record, "scenario_name", "unknown")
        free = np.asarray(example["bev_free"], dtype=np.float32)
        obstacle = np.asarray(example["bev_obstacle"], dtype=np.float32)
        unknown = np.asarray(example["bev_unknown"], dtype=np.float32)
        confidence = np.asarray(example["bev_confidence"], dtype=np.float32)
        records.append(
            SourceFrame(
                sequence_id=str(record.get("sequence_id", manifest.get("source_segment_id", root.name))),
                camera_id=str(record.get("camera_id", "front_rgb")),
                frame_id=int(record.get("frame_id", np.asarray(example["frame_id"]).item())),
                timestamp_ns=int(record.get("timestamp_ns", np.asarray(example["timestamp_ns"]).item())),
                source_name=source_name,
                source_ref=str(record["example_path"]),
                supervision_grade=supervision_grade,
                scenario_name=scenario_name,
                bev=LocalBev(
                    free=free,
                    occupied=obstacle,
                    unknown=unknown,
                    traversable=free.copy(),
                    risky=obstacle.copy(),
                    confidence=confidence,
                    source="labels",
                ),
            )
        )
    return records, manifest


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
    args = parser.parse_args(argv)
    manifest = build_action_label_pack(sources=[Path(source) for source in args.source], out_dir=args.out, max_examples=args.max_examples)
    print(json.dumps({"manifest": manifest.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
