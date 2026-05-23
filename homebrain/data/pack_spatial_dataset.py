from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import atan2
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.data.spatial_dataset import (
    DETERMINISTIC_CREATED_AT_UTC,
    SPATIAL_DATASET_SCHEMA_VERSION,
    SPATIAL_EXAMPLE_SCHEMA_VERSION,
    SPATIAL_MANIFEST_FILE,
    json_sha256,
    read_json,
    scalar_bool,
    scalar_float,
    scalar_int,
    scalar_str,
    write_deterministic_npz,
    write_json,
)
from homebrain.geometry.bev_projector import BEV_ARTIFACT_KINDS
from homebrain.geometry.validate_bev import load_bev_manifest
from homebrain.messages.schema import FrameEvent, JsonDict, deterministic_json
from homebrain.replay.replayd import order_events_for_replay
from homebrain.replay.segment_log import DEFAULT_EVENT_FILE, load_manifest, read_events
from homebrain.teachers.artifacts import file_sha256, frame_key, load_array, relative_to_root


@dataclass(frozen=True)
class SplitAssignment:
    split: str
    split_unit_id: str
    leakage_guard_frames: int


@dataclass(frozen=True)
class PoseDeltaLabel:
    pose_delta: tuple[float, float, float]
    mask: float
    target_frame_id: int
    source: str
    frame: str


def pack_spatial_dataset(
    *,
    log_dir: str | Path,
    bev_dir: str | Path,
    out_dir: str | Path,
    teacher_artifacts: str | Path | None = None,
) -> Path:
    log_root = Path(log_dir)
    bev_root = Path(bev_dir)
    output = Path(out_dir)
    examples_dir = output / "examples"
    _clear_generated_outputs(output)
    examples_dir.mkdir(parents=True, exist_ok=True)

    route_manifest = load_manifest(log_root)
    event_file = log_root / (route_manifest.event_file or DEFAULT_EVENT_FILE)
    bev_manifest = load_bev_manifest(bev_root)
    frames_by_key = _route_frames_by_key(log_root)

    if bev_manifest.get("weak_label") is not True:
        raise ValueError("BEV manifest must be marked weak_label=true")
    if bev_manifest.get("control_safe") is not False:
        raise ValueError("BEV manifest must be marked control_safe=false")

    camera_config = _camera_config(bev_manifest)
    camera_config_hash = json_sha256(camera_config)
    teacher_manifest_hash = _teacher_manifest_hash(bev_manifest, teacher_artifacts)
    source_bev_manifest_hash = file_sha256(bev_root / "bev_manifest.json")
    source_log_manifest_hash = file_sha256(log_root / "manifest.json")
    source_event_file_hash = file_sha256(event_file)

    frame_records = _manifest_frames(bev_manifest)
    split_assignments = _sequence_aware_split_assignments(frame_records)
    pose_labels = _pose_delta_labels(log_root)
    pose_label_count = 0
    not_robot_frame_truth_count = 0
    example_records: list[JsonDict] = []
    split_counts: dict[str, int] = {"train": 0, "val": 0, "review": 0}

    for index, frame_record in enumerate(frame_records):
        frame = frames_by_key.get(frame_key(frame_record))
        if frame is None:
            raise ValueError(f"BEV frame is missing from log: {frame_key(frame_record)}")
        _validate_frame_flags(frame_record)

        arrays = _load_bev_arrays(bev_root, frame_record)
        frame_id = int(frame_record["frame_id"])
        camera_id = str(frame_record["camera_id"])
        split_assignment = split_assignments[index]
        split = split_assignment.split
        split_counts[split] += 1
        not_robot_frame_truth = _not_robot_frame_truth(bev_manifest, frame_record)
        if not_robot_frame_truth:
            not_robot_frame_truth_count += 1
        pose_label = pose_labels.get(frame_id, _missing_pose_label())
        if pose_label.mask > 0.0:
            pose_label_count += 1
        rgb_ref = frame.data_ref
        rgb_path = (log_root / rgb_ref).as_posix()
        provenance = _example_provenance(
            log_root=log_root,
            bev_root=bev_root,
            frame=frame,
            frame_record=frame_record,
            source_bev_manifest_hash=source_bev_manifest_hash,
            source_log_manifest_hash=source_log_manifest_hash,
            camera_config_hash=camera_config_hash,
            teacher_manifest_hash=teacher_manifest_hash,
            not_robot_frame_truth=not_robot_frame_truth,
        )

        example_arrays: dict[str, np.ndarray] = {
            "frame_id": scalar_int(frame_id),
            "timestamp_ns": scalar_int(int(frame_record["timestamp_ns"])),
            "timestamp": scalar_int(int(frame_record["timestamp_ns"])),
            "rgb_ref": scalar_str(rgb_ref),
            "rgb_path": scalar_str(rgb_path),
            "bev_free": arrays["bev_free"],
            "bev_obstacle": arrays["bev_obstacle"],
            "bev_unknown": arrays["bev_unknown"],
            "bev_confidence": arrays["bev_confidence"],
            "provenance": scalar_str(deterministic_json(provenance)),
            "weak_label": scalar_bool(True),
            "control_safe": scalar_bool(False),
            "not_robot_frame_truth": scalar_bool(not_robot_frame_truth),
            "camera_config_hash": scalar_str(camera_config_hash),
            "teacher_manifest_hash": scalar_str(teacher_manifest_hash),
            "split": scalar_str(split),
            "split_unit_id": scalar_str(split_assignment.split_unit_id),
            "pose_delta": np.asarray(pose_label.pose_delta, dtype=np.float32),
            "pose_delta_mask": scalar_float(pose_label.mask),
            "pose_label_frame": scalar_str(pose_label.frame),
            "pose_label_source": scalar_str(pose_label.source),
            "pose_label_target_frame_id": scalar_int(pose_label.target_frame_id),
            "action_label_mask": scalar_float(0.0),
            "imu_label_mask": scalar_float(0.0),
            "wheel_label_mask": scalar_float(0.0),
        }
        if "bev_height" in arrays:
            example_arrays["bev_height"] = arrays["bev_height"]
        if "bev_floor_candidate" in arrays:
            example_arrays["bev_floor_candidate"] = arrays["bev_floor_candidate"]

        example_path = examples_dir / f"{camera_id}_{frame_id:06d}.npz"
        write_deterministic_npz(example_path, example_arrays)
        example_hash = file_sha256(example_path)
        stats = frame_record.get("stats") if isinstance(frame_record.get("stats"), dict) else {}
        example_records.append(
            {
                "sequence_id": frame.sequence_id,
                "camera_id": camera_id,
                "frame_id": frame_id,
                "timestamp_ns": int(frame_record["timestamp_ns"]),
                "rgb_ref": rgb_ref,
                "rgb_path": rgb_path,
                "split": split,
                "split_unit_id": split_assignment.split_unit_id,
                "split_leakage_guard_frames": split_assignment.leakage_guard_frames,
                "weak_label": True,
                "control_safe": False,
                "not_robot_frame_truth": not_robot_frame_truth,
                "example_path": relative_to_root(example_path, output),
                "example_sha256": example_hash,
                "camera_config_hash": camera_config_hash,
                "teacher_manifest_hash": teacher_manifest_hash,
                "pose_delta_mask": pose_label.mask,
                "pose_label_frame": pose_label.frame,
                "pose_label_source": pose_label.source,
                "pose_label_target_frame_id": pose_label.target_frame_id,
                "source_bev_metadata_path": frame_record.get("metadata_path"),
                "source_bev_stats": dict(stats),
            }
        )

    manifest: JsonDict = {
        "schema_version": SPATIAL_DATASET_SCHEMA_VERSION,
        "example_schema_version": SPATIAL_EXAMPLE_SCHEMA_VERSION,
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "package_type": "SpatialTrainPack",
        "quality_review_status": "unreviewed_requires_qa",
        "source_log": log_root.as_posix(),
        "source_segment_id": route_manifest.segment_id,
        "source_log_manifest_sha256": source_log_manifest_hash,
        "source_event_file_sha256": source_event_file_hash,
        "source_bev": bev_root.as_posix(),
        "source_bev_manifest_sha256": source_bev_manifest_hash,
        "source_bev_schema_version": bev_manifest.get("schema_version"),
        "source_depth_teacher_name": bev_manifest.get("source_depth_teacher_name"),
        "source_depth_backend": bev_manifest.get("source_depth_backend"),
        "source_depth_mock": bool(bev_manifest.get("source_depth_mock", False)),
        "source_depth_real_perception": bool(bev_manifest.get("source_depth_real_perception", False)),
        "robot_supervision_grade": _robot_supervision_grade(bev_manifest),
        "teacher_artifacts": Path(teacher_artifacts).as_posix() if teacher_artifacts is not None else None,
        "teacher_manifest_hash": teacher_manifest_hash,
        "camera_config_hash": camera_config_hash,
        "camera_config": camera_config,
        "grid_shape": list(bev_manifest.get("grid_shape", [])),
        "artifact_kinds": list(BEV_ARTIFACT_KINDS),
        "example_fields": [
            "frame_id",
            "timestamp_ns",
            "timestamp",
            "rgb_ref",
            "rgb_path",
            "bev_free",
            "bev_obstacle",
            "bev_unknown",
            "bev_confidence",
            "bev_height",
            "bev_floor_candidate",
            "provenance",
            "weak_label",
            "control_safe",
            "not_robot_frame_truth",
            "camera_config_hash",
            "teacher_manifest_hash",
            "split",
            "split_unit_id",
            "pose_delta",
            "pose_delta_mask",
            "pose_label_frame",
            "pose_label_source",
            "pose_label_target_frame_id",
            "action_label_mask",
            "imu_label_mask",
            "wheel_label_mask",
        ],
        "weak_label": True,
        "control_safe": False,
        "not_robot_frame_truth": bool(not_robot_frame_truth_count == len(example_records) and example_records),
        "not_robot_frame_truth_count": not_robot_frame_truth_count,
        "control_safety": "not_control_safe_reviewed_training_data_candidate_only",
        "example_count": len(example_records),
        "split_counts": split_counts,
        "split_strategy": {
            "name": "sequence_aware_contiguous_blocks_with_review_embargo",
            "version": 1,
            "description": (
                "Examples are grouped by sequence/camera/stable window/contiguous frame span. "
                "Validation frames are contiguous blocks and neighboring frames are assigned to "
                "review, preventing direct train/val adjacency for contiguous windows."
            ),
            "leakage_guard_frames": 1,
        },
        "pose_label_count": pose_label_count,
        "pose_label_frame": "camera_relative_dataset_pose" if pose_label_count else "none",
        "pose_label_convention": _pose_label_convention(pose_label_count),
        "frames": example_records,
        "examples": example_records,
    }
    manifest_path = output / SPATIAL_MANIFEST_FILE
    write_json(manifest_path, manifest)
    return manifest_path


def _clear_generated_outputs(output: Path) -> None:
    if (output / "examples").exists():
        shutil.rmtree(output / "examples")
    manifest_path = output / SPATIAL_MANIFEST_FILE
    if manifest_path.exists():
        manifest_path.unlink()


def _route_frames_by_key(log_dir: Path) -> dict[str, FrameEvent]:
    events = order_events_for_replay(read_events(log_dir))
    return {frame_key(event): event for event in events if isinstance(event, FrameEvent)}


def _manifest_frames(manifest: JsonDict) -> list[JsonDict]:
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("BEV manifest frames must be a list")
    return [frame for frame in frames if isinstance(frame, dict)]


def _validate_frame_flags(frame_record: JsonDict) -> None:
    if frame_record.get("weak_label") is not True:
        raise ValueError(f"frame {frame_record.get('frame_id')} must be weak_label=true")
    if frame_record.get("control_safe") is not False:
        raise ValueError(f"frame {frame_record.get('frame_id')} must be control_safe=false")


def _load_bev_arrays(bev_root: Path, frame_record: JsonDict) -> dict[str, np.ndarray]:
    artifacts = frame_record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"frame {frame_record.get('frame_id')} is missing artifacts")
    arrays: dict[str, np.ndarray] = {}
    for kind in BEV_ARTIFACT_KINDS:
        artifact = artifacts.get(kind)
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValueError(f"frame {frame_record.get('frame_id')} missing {kind} artifact")
        arrays[kind] = load_array(bev_root / str(artifact["path"]))
    return arrays


def _camera_config(bev_manifest: JsonDict) -> JsonDict:
    camera_config = bev_manifest.get("camera_config")
    if isinstance(camera_config, dict):
        return dict(camera_config)
    return {}


def _teacher_manifest_hash(bev_manifest: JsonDict, teacher_artifacts: str | Path | None) -> str:
    if teacher_artifacts is not None:
        teacher_manifest_path = Path(teacher_artifacts) / "teacher_manifest.json"
        if not teacher_manifest_path.exists():
            raise FileNotFoundError(f"missing teacher manifest: {teacher_manifest_path}")
        return file_sha256(teacher_manifest_path)
    value = bev_manifest.get("source_depth_manifest_sha256")
    if isinstance(value, str) and value:
        return value
    return "not_supplied"


def _sequence_aware_split_assignments(frame_records: list[JsonDict]) -> list[SplitAssignment]:
    assignments: list[SplitAssignment] = [SplitAssignment("train", "unassigned", 1) for _ in frame_records]
    groups = _contiguous_split_groups(frame_records)
    for indices in groups:
        if not indices:
            continue
        group_frames = [frame_records[index] for index in indices]
        split_unit_id = _split_unit_id(group_frames)
        local_splits = _split_group(len(indices))
        for local_index, record_index in enumerate(indices):
            assignments[record_index] = SplitAssignment(
                split=local_splits[local_index],
                split_unit_id=split_unit_id,
                leakage_guard_frames=1,
            )
    return assignments


def _contiguous_split_groups(frame_records: list[JsonDict]) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    previous_key: tuple[str, str, str] | None = None
    previous_frame_id: int | None = None
    for index, record in enumerate(frame_records):
        key = (
            str(record.get("sequence_id", "")),
            str(record.get("camera_id", "")),
            str(record.get("stable_window_id", "")),
        )
        frame_id = int(record.get("frame_id", index))
        starts_new = (
            previous_key is not None
            and (
                key != previous_key
                or previous_frame_id is None
                or frame_id != previous_frame_id + 1
            )
        )
        if starts_new and current:
            groups.append(current)
            current = []
        current.append(index)
        previous_key = key
        previous_frame_id = frame_id
    if current:
        groups.append(current)
    return groups


def _split_unit_id(group_frames: list[JsonDict]) -> str:
    first = group_frames[0]
    last = group_frames[-1]
    stable_window_id = first.get("stable_window_id")
    if isinstance(stable_window_id, str) and stable_window_id:
        suffix = stable_window_id
    else:
        suffix = f"frames_{int(first.get('frame_id', 0)):06d}_{int(last.get('frame_id', 0)):06d}"
    return f"{first.get('sequence_id')}:{first.get('camera_id')}:{suffix}"


def _split_group(count: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return ["train"]
    if count == 2:
        return ["train", "review"]
    if count == 3:
        return ["train", "val", "review"]

    splits = ["train"] * count
    guard = 1
    val_count = max(1, int(round(count * 0.15)))
    review_tail_count = max(1, int(round(count * 0.10)))
    usable_end = max(guard + val_count, count - review_tail_count)
    val_start = int(round(count * 0.70))
    val_start = max(guard, min(val_start, usable_end - val_count - guard))
    val_start = max(guard, min(val_start, count - val_count - guard))
    val_end = min(count, val_start + val_count)

    for index in range(val_start, val_end):
        splits[index] = "val"
    for index in range(max(0, val_start - guard), val_start):
        splits[index] = "review"
    for index in range(val_end, min(count, val_end + guard)):
        splits[index] = "review"
    for index in range(max(0, count - review_tail_count), count):
        if splits[index] != "val":
            splits[index] = "review"
    return splits


def _not_robot_frame_truth(bev_manifest: JsonDict, frame_record: JsonDict) -> bool:
    explicit = frame_record.get("not_robot_frame_truth")
    if isinstance(explicit, bool):
        return explicit
    manifest_explicit = bev_manifest.get("not_robot_frame_truth")
    if isinstance(manifest_explicit, bool):
        return manifest_explicit
    return _robot_supervision_grade(bev_manifest) != "robot_frame_metric"


def _missing_pose_label() -> PoseDeltaLabel:
    return PoseDeltaLabel(
        pose_delta=(0.0, 0.0, 0.0),
        mask=0.0,
        target_frame_id=-1,
        source="missing",
        frame="none",
    )


def _pose_delta_labels(log_root: Path) -> dict[int, PoseDeltaLabel]:
    associations_path = log_root / "tum_rgbd_associations.json"
    if not associations_path.exists():
        return {}
    associations = read_json(associations_path)
    if associations.get("source_type") != "tum_rgbd_associations":
        return {}
    frames = associations.get("frames")
    if not isinstance(frames, list):
        return {}

    ordered = sorted(
        [frame for frame in frames if isinstance(frame, dict)],
        key=lambda frame: int(frame.get("frame_id", 0)),
    )
    labels: dict[int, PoseDeltaLabel] = {}
    for current, nxt in zip(ordered, ordered[1:]):
        current_gt = current.get("groundtruth")
        next_gt = nxt.get("groundtruth")
        if not isinstance(current_gt, dict) or not isinstance(next_gt, dict):
            continue
        delta = _relative_camera_pose_delta(current_gt, next_gt)
        labels[int(current["frame_id"])] = PoseDeltaLabel(
            pose_delta=delta,
            mask=1.0,
            target_frame_id=int(nxt["frame_id"]),
            source="tum_rgbd_groundtruth_association",
            frame="camera_relative_dataset_pose",
        )
    if ordered:
        last = ordered[-1]
        labels[int(last["frame_id"])] = PoseDeltaLabel(
            pose_delta=(0.0, 0.0, 0.0),
            mask=0.0,
            target_frame_id=-1,
            source="tum_rgbd_groundtruth_association_no_next_frame",
            frame="camera_relative_dataset_pose",
        )
    return labels


def _relative_camera_pose_delta(current_gt: JsonDict, next_gt: JsonDict) -> tuple[float, float, float]:
    current_pose = _tum_pose_matrix(current_gt)
    next_pose = _tum_pose_matrix(next_gt)
    relative = np.linalg.inv(current_pose) @ next_pose
    dx = float(relative[0, 3])
    dz_as_dy = float(relative[2, 3])
    dyaw = float(atan2(float(relative[0, 2]), float(relative[2, 2])))
    return (dx, dz_as_dy, dyaw)


def _tum_pose_matrix(groundtruth: JsonDict) -> np.ndarray:
    rotation = _quat_to_rotation(
        float(groundtruth["qx"]),
        float(groundtruth["qy"]),
        float(groundtruth["qz"]),
        float(groundtruth["qw"]),
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [
        float(groundtruth["tx"]),
        float(groundtruth["ty"]),
        float(groundtruth["tz"]),
    ]
    return transform


def _quat_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = float((qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5)
    if norm <= 0.0:
        raise ValueError("TUM groundtruth quaternion has zero norm")
    x = qx / norm
    y = qy / norm
    z = qz / norm
    w = qw / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_label_convention(pose_label_count: int) -> JsonDict:
    if pose_label_count <= 0:
        return {
            "available": False,
            "pose_label_frame": "none",
            "note": "No pose labels were packed; missing pose targets are masked and are not fake labels.",
        }
    return {
        "available": True,
        "pose_label_frame": "camera_relative_dataset_pose",
        "source": "TUM RGB-D groundtruth associations",
        "transform": "inv(T_dataset_camera_current) @ T_dataset_camera_next",
        "pose_delta_components": [
            "camera_relative_x_m",
            "camera_relative_z_m_stored_as_dy",
            "relative_yaw_about_camera_y_rad",
        ],
        "not_robot_odometry": True,
        "control_safe": False,
        "note": (
            "This is a dataset camera-pose delta for representation pretraining/eval only. "
            "It is not robot odometry and not robot-frame truth."
        ),
    }


def _example_provenance(
    *,
    log_root: Path,
    bev_root: Path,
    frame: FrameEvent,
    frame_record: JsonDict,
    source_bev_manifest_hash: str,
    source_log_manifest_hash: str,
    camera_config_hash: str,
    teacher_manifest_hash: str,
    not_robot_frame_truth: bool,
) -> JsonDict:
    return {
        "schema_version": SPATIAL_EXAMPLE_SCHEMA_VERSION,
        "source_log": log_root.as_posix(),
        "source_bev": bev_root.as_posix(),
        "source_log_manifest_sha256": source_log_manifest_hash,
        "source_bev_manifest_sha256": source_bev_manifest_hash,
        "source_bev_metadata_path": frame_record.get("metadata_path"),
        "sequence_id": frame.sequence_id,
        "camera_id": frame.camera_id,
        "frame_id": frame.frame_id,
        "timestamp_ns": frame.timestamp_ns,
        "rgb_ref": frame.data_ref,
        "weak_label": True,
        "control_safe": False,
        "not_robot_frame_truth": not_robot_frame_truth,
        "camera_config_hash": camera_config_hash,
        "teacher_manifest_hash": teacher_manifest_hash,
        "source_bev_stats": frame_record.get("stats") if isinstance(frame_record.get("stats"), dict) else {},
        "source_bev_assumptions": frame_record.get("assumptions") if isinstance(frame_record.get("assumptions"), list) else [],
        "source_bev_warnings": frame_record.get("warnings") if isinstance(frame_record.get("warnings"), list) else [],
    }


def _robot_supervision_grade(bev_manifest: JsonDict) -> str:
    source_name = str(bev_manifest.get("source_depth_teacher_name", "")).lower()
    source_backend = str(bev_manifest.get("source_depth_backend", "")).lower()
    source_path = str(bev_manifest.get("source_log", "")).lower()
    if "robot_frame_metric" in {source_name, source_backend}:
        return "robot_frame_metric"
    if (
        "tum" in source_name
        or "rgbd_truth" in source_name
        or "public_rgbd" in source_backend
        or "tum" in source_path
    ):
        return "public_rgbd_anchor"
    if source_name in {"da3", "depth_pro"} or source_backend in {"real", "fake"}:
        return "weak_visual_geometry"
    return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pack BEV weak labels into a deterministic SpatialTrainPack.")
    parser.add_argument("--log", required=True, help="Input HomeBrain route log directory.")
    parser.add_argument("--bev", required=True, help="Input depth-to-BEV artifact directory.")
    parser.add_argument("--teacher-artifacts", default=None, help="Optional teacher artifact directory.")
    parser.add_argument("--out", required=True, help="Output SpatialTrainPack directory.")
    args = parser.parse_args(argv)
    manifest_path = pack_spatial_dataset(
        log_dir=args.log,
        bev_dir=args.bev,
        teacher_artifacts=args.teacher_artifacts,
        out_dir=args.out,
    )
    print(f"wrote SpatialTrainPack manifest to {manifest_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
