from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from homebrain.geometry.bev_projector import BEV_ARTIFACT_KINDS, BEV_MANIFEST_FILE, BEV_SCHEMA_VERSION, bev_array_stats
from homebrain.messages.schema import JsonDict
from homebrain.replay.segment_log import load_manifest
from homebrain.teachers.artifacts import file_sha256, load_array, relative_to_root, save_array, write_json
from homebrain.teachers.scene_teacher import SCENE_TEACHER_MANIFEST_FILE, load_scene_teacher_manifest

DEFAULT_GRID_SHAPE = (32, 32)


def convert_scene_teacher_to_bev(
    *,
    log_dir: str | Path,
    scene_teacher_dir: str | Path,
    out_dir: str | Path,
) -> Path:
    log_root = Path(log_dir)
    scene_root = Path(scene_teacher_dir)
    output = Path(out_dir)
    route_manifest = load_manifest(log_root)
    scene_manifest = load_scene_teacher_manifest(scene_root)
    route_has_measured_robot_frame = _route_has_measured_robot_frame_transforms(log_root)
    robot_frame_truth = bool(
        route_has_measured_robot_frame
        and scene_manifest.get("robot_frame_truth_source") == "measured_robot_frame"
        and scene_manifest.get("robot_frame_truth") is True
    )

    output.mkdir(parents=True, exist_ok=True)
    frame_records: list[JsonDict] = []
    warnings: list[str] = []
    for source_record in scene_manifest["frames"]:
        if not isinstance(source_record, dict):
            warnings.append("skipped non-object scene teacher frame record")
            continue
        try:
            frame_records.append(
                _write_bev_frame(
                    output=output,
                    scene_root=scene_root,
                    source_record=source_record,
                    robot_frame_truth=robot_frame_truth,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep review conversion robust.
            warnings.append(f"frame_{source_record.get('frame_id')}_failed:{exc}")

    if not frame_records:
        raise ValueError("no valid scene-teacher geometry frames were available for review BEV conversion")

    manifest: JsonDict = {
        "schema_version": BEV_SCHEMA_VERSION,
        "source_log": log_root.as_posix(),
        "source_segment_id": route_manifest.segment_id,
        "source_schema_version": route_manifest.schema_version,
        "source_scene_teacher": scene_root.as_posix(),
        "source_scene_teacher_manifest": (scene_root / SCENE_TEACHER_MANIFEST_FILE).as_posix(),
        "source_scene_teacher_manifest_sha256": file_sha256(scene_root / SCENE_TEACHER_MANIFEST_FILE),
        "source_scene_teacher_name": scene_manifest.get("teacher_name"),
        "source_scene_teacher_backend": scene_manifest.get("backend"),
        "source_scene_teacher_mock": bool(scene_manifest.get("mock", False)),
        "source_scene_teacher_synthetic": bool(scene_manifest.get("synthetic", False)),
        "source_scene_teacher_real_perception": bool(scene_manifest.get("real_perception", False)),
        "source_scene_teacher_scale_status": scene_manifest.get("scale_status"),
        "grid_shape": list(DEFAULT_GRID_SHAPE),
        "grid_orientation": {
            "note": "review_only_image_plane_scene_teacher_proxy_not_robot_action_frame",
        },
        "coordinate_frame": "scene_teacher_visual_frame_not_robot_frame_truth",
        "projection_model": "scene_teacher_depth_or_point_map_review_proxy",
        "artifact_kinds": list(BEV_ARTIFACT_KINDS),
        "weak_label": True,
        "review_only": True,
        "geometry_pretrain_ok": True,
        "pose_pretrain_ok": bool(scene_manifest.get("frame_count", 0)),
        "robot_frame_truth_candidate": route_has_measured_robot_frame,
        "route_has_measured_robot_frame_transforms": route_has_measured_robot_frame,
        "robot_frame_truth": robot_frame_truth,
        "not_robot_frame_truth": not robot_frame_truth,
        "action_supervision_ok": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "control_safe_claim": False,
        "product_training_approved": False,
        "raw_pwm_emitted": False,
        "trainable_for": "geometry_pretrain_review_only",
        "control_safety": "not_control_safe_scene_teacher_review_bev_only",
        "warnings": sorted(set(warnings)),
        "assumptions": [
            "scene teacher geometry is weak supervision unless measured robot-frame transforms are present",
            "action supervision remains disabled",
            "floor/risk/dynamic masks may be teacher placeholders",
        ],
        "frame_count": len(frame_records),
        "frames": frame_records,
    }
    manifest_path = output / BEV_MANIFEST_FILE
    write_json(manifest_path, manifest)
    return manifest_path


def _write_bev_frame(
    *,
    output: Path,
    scene_root: Path,
    source_record: JsonDict,
    robot_frame_truth: bool,
) -> JsonDict:
    artifacts = source_record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("missing artifacts object")
    arrays = _review_bev_arrays(scene_root, artifacts)
    frame_id = int(source_record["frame_id"])
    camera_id = str(source_record["camera_id"])
    frame_dir = output / "frames" / f"{camera_id}_{frame_id:06d}"

    artifact_records: dict[str, JsonDict] = {}
    for kind in BEV_ARTIFACT_KINDS:
        target = frame_dir / f"{kind}.npy"
        record = save_array(target, arrays[kind])
        artifact_records[kind] = {
            "kind": kind,
            "path": relative_to_root(target, output),
            **record,
        }

    metadata: JsonDict = {
        "schema_version": BEV_SCHEMA_VERSION,
        "sequence_id": source_record["sequence_id"],
        "camera_id": camera_id,
        "frame_id": frame_id,
        "timestamp_ns": source_record["timestamp_ns"],
        "source_scene_teacher_artifacts": {
            "depth": _artifact_path(artifacts, "depth"),
            "point_map": _artifact_path(artifacts, "point_map"),
            "intrinsics": _artifact_path(artifacts, "intrinsics"),
            "extrinsics": _artifact_path(artifacts, "extrinsics"),
            "confidence": _artifact_path(artifacts, "confidence"),
            "validity_mask": _artifact_path(artifacts, "validity_mask"),
            "floor_traversable_mask": _artifact_path(artifacts, "floor_traversable_mask"),
            "obstacle_risk_mask": _artifact_path(artifacts, "obstacle_risk_mask"),
            "dynamic_motion_mask": _artifact_path(artifacts, "dynamic_motion_mask"),
        },
        "source_data_ref": source_record.get("source_data_ref"),
        "width": int(source_record["width"]),
        "height": int(source_record["height"]),
        "weak_label": True,
        "review_only": True,
        "geometry_pretrain_ok": True,
        "pose_pretrain_ok": True,
        "robot_frame_truth": robot_frame_truth,
        "not_robot_frame_truth": not robot_frame_truth,
        "action_supervision_ok": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "control_safe_claim": False,
        "product_training_approved": False,
        "raw_pwm_emitted": False,
        "trainable_for": "geometry_pretrain_review_only",
        "control_safety": "not_control_safe_scene_teacher_review_bev_only",
        "projection_model": "scene_teacher_depth_or_point_map_review_proxy",
        "grid_shape": list(DEFAULT_GRID_SHAPE),
        "stats": bev_array_stats(arrays),
        "warnings": ["review_only_weak_scene_teacher_geometry", "action_supervision_disabled"],
        "artifact_shapes": {kind: artifact_records[kind]["shape"] for kind in BEV_ARTIFACT_KINDS},
        "artifact_dtypes": {kind: artifact_records[kind]["dtype"] for kind in BEV_ARTIFACT_KINDS},
    }
    metadata_path = frame_dir / "metadata.json"
    write_json(metadata_path, metadata)

    return {
        "sequence_id": source_record["sequence_id"],
        "camera_id": camera_id,
        "frame_id": frame_id,
        "timestamp_ns": source_record["timestamp_ns"],
        "width": int(source_record["width"]),
        "height": int(source_record["height"]),
        "weak_label": True,
        "review_only": True,
        "geometry_pretrain_ok": True,
        "robot_frame_truth": robot_frame_truth,
        "not_robot_frame_truth": not robot_frame_truth,
        "action_supervision_ok": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "product_training_approved": False,
        "raw_pwm_emitted": False,
        "trainable_for": "geometry_pretrain_review_only",
        "metadata_path": relative_to_root(metadata_path, output),
        "metadata_sha256": file_sha256(metadata_path),
        "artifacts": artifact_records,
        "stats": metadata["stats"],
        "warnings": metadata["warnings"],
    }


def _review_bev_arrays(scene_root: Path, artifacts: JsonDict) -> dict[str, np.ndarray]:
    depth = _scene_depth(scene_root, artifacts)
    confidence = _load_optional_mask_or_float(scene_root, artifacts, "confidence", depth.shape, default=1.0)
    validity = _load_optional_mask_or_float(scene_root, artifacts, "validity_mask", depth.shape, default=1.0) > 0.0
    floor_mask = _load_optional_mask_or_float(scene_root, artifacts, "floor_traversable_mask", depth.shape, default=np.nan)
    obstacle_mask = _load_optional_mask_or_float(scene_root, artifacts, "obstacle_risk_mask", depth.shape, default=np.nan)
    dynamic_mask = _load_optional_mask_or_float(scene_root, artifacts, "dynamic_motion_mask", depth.shape, default=0.0) > 0.0

    valid = np.isfinite(depth) & (depth > np.float32(0.0)) & validity & np.isfinite(confidence) & (confidence > np.float32(0.05))
    if not np.any(valid):
        raise ValueError("scene geometry has no valid depth or point-map samples")
    finite_depth = depth[valid]
    lo = float(np.percentile(finite_depth, 5))
    hi = float(np.percentile(finite_depth, 95))
    scale = max(hi - lo, 1.0e-6)
    norm = np.clip((depth - np.float32(lo)) / np.float32(scale), np.float32(0.0), np.float32(1.0))

    if np.isnan(floor_mask).all():
        floor = valid & (norm >= np.float32(0.35))
    else:
        floor = valid & (floor_mask > 0.0)
    if np.isnan(obstacle_mask).all():
        obstacle = valid & (norm <= np.float32(0.20))
    else:
        obstacle = valid & (obstacle_mask > 0.0)
    obstacle = obstacle | (valid & dynamic_mask)
    free = floor & ~obstacle
    unknown = ~(free | obstacle)
    depth_grid = _resize_nearest(norm.astype(np.float32), DEFAULT_GRID_SHAPE)
    conf_grid = _resize_nearest(np.clip(confidence, 0.0, 1.0).astype(np.float32), DEFAULT_GRID_SHAPE)
    valid_grid = _resize_nearest(valid.astype(np.float32), DEFAULT_GRID_SHAPE) > 0.0
    free_grid = _resize_nearest(free.astype(np.float32), DEFAULT_GRID_SHAPE) > 0.0
    obstacle_grid = _resize_nearest(obstacle.astype(np.float32), DEFAULT_GRID_SHAPE) > 0.0
    unknown_grid = ~(free_grid | obstacle_grid)
    floor_grid = _resize_nearest(floor.astype(np.float32), DEFAULT_GRID_SHAPE) > 0.0
    height = (np.float32(0.5) - depth_grid).astype(np.float32)

    return {
        "bev_free": (free_grid & valid_grid).astype(np.uint8),
        "bev_obstacle": (obstacle_grid & valid_grid).astype(np.uint8),
        "bev_unknown": unknown_grid.astype(np.uint8),
        "bev_floor_candidate": (floor_grid & valid_grid).astype(np.uint8),
        "bev_height": height,
        "bev_confidence": np.where(valid_grid, conf_grid, np.float32(0.0)).astype(np.float32),
    }


def _scene_depth(scene_root: Path, artifacts: JsonDict) -> np.ndarray:
    depth = _load_optional(scene_root, artifacts, "depth")
    if depth is not None:
        values = np.asarray(depth, dtype=np.float32)
        values = np.squeeze(values)
        if values.ndim != 2:
            raise ValueError(f"depth must be 2D, got {values.shape}")
        return values.astype(np.float32)
    point_map = _load_optional(scene_root, artifacts, "point_map")
    if point_map is None:
        raise ValueError("missing depth or point_map scene geometry")
    values = np.asarray(point_map, dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"point_map must be HxWx3, got {values.shape}")
    z = values[:, :, 2]
    norm = np.linalg.norm(values, axis=2)
    return np.where(np.isfinite(z) & (z > np.float32(0.0)), z, norm).astype(np.float32)


def _load_optional_mask_or_float(
    scene_root: Path,
    artifacts: JsonDict,
    kind: str,
    shape: tuple[int, int],
    *,
    default: float,
) -> np.ndarray:
    array = _load_optional(scene_root, artifacts, kind)
    if array is None:
        return np.full(shape, np.float32(default), dtype=np.float32)
    values = np.asarray(array, dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"{kind} must be 2D, got {values.shape}")
    if values.shape != shape:
        return _resize_nearest(values, shape).astype(np.float32)
    return values.astype(np.float32)


def _load_optional(scene_root: Path, artifacts: JsonDict, kind: str) -> np.ndarray | None:
    record = artifacts.get(kind)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return None
    return load_array(scene_root / str(record["path"]))


def _artifact_path(artifacts: JsonDict, kind: str) -> str | None:
    record = artifacts.get(kind)
    if isinstance(record, dict) and isinstance(record.get("path"), str):
        return str(record["path"])
    return None


def _resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(array)
    values = np.squeeze(values)
    if values.ndim != 2:
        values = values.reshape(1, -1)
    out_h, out_w = shape
    row_index = np.linspace(0, values.shape[0] - 1, out_h).round().astype(np.int64)
    col_index = np.linspace(0, values.shape[1] - 1, out_w).round().astype(np.int64)
    return values[row_index[:, None], col_index[None, :]]


def _route_has_measured_robot_frame_transforms(log_root: Path) -> bool:
    metadata_path = log_root / "route_metadata.json"
    if not metadata_path.exists():
        return False
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    return bool(
        data.get("has_camera_to_base_transform") is True
        and (data.get("has_robot_base_pose") is True or data.get("has_wheel_odometry") is True)
        and data.get("review_assumed_extrinsics") is not True
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert SceneTeacherPack geometry into review-only weak BEV targets.")
    parser.add_argument("--log", required=True, help="Input HomeBrain route log directory.")
    parser.add_argument("--scene-teacher", required=True, help="Input SceneTeacherPack directory.")
    parser.add_argument("--out", required=True, help="Output review BEV directory.")
    args = parser.parse_args(argv)
    manifest_path = convert_scene_teacher_to_bev(
        log_dir=args.log,
        scene_teacher_dir=args.scene_teacher,
        out_dir=args.out,
    )
    print(f"wrote scene-teacher review BEV manifest to {manifest_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
