from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from homebrain.datasets.tum_rgbd import (
    TUM_RGBD_LICENSE_REVIEW_STATUS,
    TUM_RGBD_ROUTE_ASSOCIATIONS_FILE,
    read_depth_png_m,
)
from homebrain.geometry.bev_projector import BEV_ARTIFACT_KINDS, BEV_MANIFEST_FILE, BEV_SCHEMA_VERSION, bev_array_stats
from homebrain.messages.schema import FrameEvent, JsonDict
from homebrain.replay.replayd import order_events_for_replay
from homebrain.replay.segment_log import load_manifest, read_events
from homebrain.teachers.artifacts import file_sha256, relative_to_root, save_array, write_json

DEFAULT_GRID_SHAPE = (32, 32)


def rgbd_truth_to_bev(*, log_dir: str | Path, out_dir: str | Path) -> Path:
    log_root = Path(log_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    route_manifest = load_manifest(log_root)
    associations_path = log_root / TUM_RGBD_ROUTE_ASSOCIATIONS_FILE
    associations = _read_json(associations_path)
    frames_by_id = _route_frames_by_id(log_root)

    frame_records: list[JsonDict] = []
    warnings: list[str] = []
    for record in associations.get("frames", []):
        if not isinstance(record, dict):
            continue
        frame_id = int(record.get("frame_id", -1))
        frame = frames_by_id.get(frame_id)
        if frame is None:
            warnings.append(f"frame_{frame_id}_missing_from_route_log")
            continue
        try:
            frame_record = _write_frame(output=output, log_root=log_root, frame=frame, association=record)
        except Exception as exc:  # noqa: BLE001 - preserve failures in manifest.
            warnings.append(f"frame_{frame_id}_failed:{exc}")
            continue
        frame_records.append(frame_record)

    manifest: JsonDict = {
        "schema_version": BEV_SCHEMA_VERSION,
        "source_log": log_root.as_posix(),
        "source_segment_id": route_manifest.segment_id,
        "source_schema_version": route_manifest.schema_version,
        "source_depth_artifacts": associations_path.as_posix(),
        "source_depth_manifest_sha256": file_sha256(associations_path),
        "source_depth_teacher_name": "tum_rgbd_truth",
        "source_depth_backend": "public_rgbd_pose_truth",
        "source_depth_mock": False,
        "source_depth_real_perception": True,
        "source_geometry_truth": True,
        "depth_truth_metric": True,
        "pose_truth_metric": True,
        "license_review_status": TUM_RGBD_LICENSE_REVIEW_STATUS,
        "grid_shape": list(DEFAULT_GRID_SHAPE),
        "grid_orientation": {
            "note": "metric_depth_image_plane_anchor_not_robot_frame_truth",
        },
        "coordinate_frame": "tum_camera_image_plane_not_robot_frame_truth",
        "projection_model": "tum_rgbd_metric_depth_image_plane_geometry_anchor",
        "artifact_kinds": list(BEV_ARTIFACT_KINDS),
        "weak_label": True,
        "control_safe": False,
        "calibration_class": "dataset_metric_rgbd_pose",
        "intrinsics_source": "tum_rgbd_calibration",
        "extrinsics_source": "tum_rgbd_groundtruth",
        "pose_source": "tum_rgbd_groundtruth",
        "scale_source": "dataset_metric_depth",
        "gravity_floor_source": "not_robot_floor_calibrated",
        "not_robot_frame_truth": True,
        "trainable_for": "geometry_pretrain_anchor_only",
        "control_safety": "not_control_safe_public_geometry_anchor_only",
        "warnings": sorted(set(warnings)),
        "assumptions": [
            "TUM RGB-D depth and pose are metric dataset truth, but not robot-frame traversability truth",
            "BEV labels are image-plane geometry anchors for pretraining/review only",
        ],
        "frame_count": len(frame_records),
        "frames": frame_records,
    }
    manifest_path = output / BEV_MANIFEST_FILE
    write_json(manifest_path, manifest)
    return manifest_path


def _write_frame(*, output: Path, log_root: Path, frame: FrameEvent, association: JsonDict) -> JsonDict:
    depth_ref = association.get("depth_ref")
    if not isinstance(depth_ref, str):
        raise ValueError("association is missing depth_ref")
    depth_m = read_depth_png_m(log_root / depth_ref)
    arrays = _truth_bev_arrays(depth_m)
    frame_dir = output / "frames" / f"{frame.camera_id}_{frame.frame_id:06d}"

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
        "sequence_id": frame.sequence_id,
        "camera_id": frame.camera_id,
        "frame_id": frame.frame_id,
        "timestamp_ns": frame.timestamp_ns,
        "source_data_ref": frame.data_ref,
        "source_depth_ref": depth_ref,
        "source_groundtruth_pose": association.get("groundtruth"),
        "width": frame.width,
        "height": frame.height,
        "weak_label": True,
        "control_safe": False,
        "calibration_class": "dataset_metric_rgbd_pose",
        "intrinsics_source": "tum_rgbd_calibration",
        "extrinsics_source": "tum_rgbd_groundtruth",
        "pose_source": "tum_rgbd_groundtruth",
        "scale_source": "dataset_metric_depth",
        "gravity_floor_source": "not_robot_floor_calibrated",
        "not_robot_frame_truth": True,
        "trainable_for": "geometry_pretrain_anchor_only",
        "control_safety": "not_control_safe_public_geometry_anchor_only",
        "projection_model": "tum_rgbd_metric_depth_image_plane_geometry_anchor",
        "grid_shape": list(DEFAULT_GRID_SHAPE),
        "stats": {
            **bev_array_stats(arrays),
            "depth_valid_ratio": _depth_valid_ratio(depth_m),
            "depth_min_m": _finite_min(depth_m),
            "depth_max_m": _finite_max(depth_m),
        },
        "warnings": ["not_robot_frame_truth", "image_plane_depth_anchor"],
        "artifact_shapes": {kind: artifact_records[kind]["shape"] for kind in BEV_ARTIFACT_KINDS},
        "artifact_dtypes": {kind: artifact_records[kind]["dtype"] for kind in BEV_ARTIFACT_KINDS},
    }
    metadata_path = frame_dir / "metadata.json"
    write_json(metadata_path, metadata)
    return {
        "sequence_id": frame.sequence_id,
        "camera_id": frame.camera_id,
        "frame_id": frame.frame_id,
        "timestamp_ns": frame.timestamp_ns,
        "width": frame.width,
        "height": frame.height,
        "weak_label": True,
        "control_safe": False,
        "calibration_class": "dataset_metric_rgbd_pose",
        "not_robot_frame_truth": True,
        "trainable_for": "geometry_pretrain_anchor_only",
        "metadata_path": relative_to_root(metadata_path, output),
        "metadata_sha256": file_sha256(metadata_path),
        "artifacts": artifact_records,
        "stats": metadata["stats"],
        "warnings": metadata["warnings"],
    }


def _truth_bev_arrays(depth_m: np.ndarray) -> dict[str, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > np.float32(0.0))
    clipped = np.where(valid, np.clip(depth, np.float32(0.0), np.float32(5.0)), np.float32(0.0))
    norm = clipped / np.float32(5.0)
    depth_grid = _resize_nearest(norm, DEFAULT_GRID_SHAPE).astype(np.float32)
    valid_grid = _resize_nearest(valid.astype(np.uint8), DEFAULT_GRID_SHAPE) > 0
    observed = valid_grid
    obstacle = observed & (depth_grid < np.float32(0.20))
    floor_candidate = observed & (depth_grid >= np.float32(0.24))
    free = floor_candidate & ~obstacle
    unknown = ~(free | obstacle)
    height = (np.float32(1.0) - depth_grid).astype(np.float32)
    confidence = np.where(observed, np.float32(1.0), np.float32(0.0)).astype(np.float32)
    return {
        "bev_free": free.astype(np.uint8),
        "bev_obstacle": obstacle.astype(np.uint8),
        "bev_unknown": unknown.astype(np.uint8),
        "bev_floor_candidate": floor_candidate.astype(np.uint8),
        "bev_height": height,
        "bev_confidence": confidence,
    }


def _route_frames_by_id(log_dir: Path) -> dict[int, FrameEvent]:
    events = order_events_for_replay(read_events(log_dir))
    return {event.frame_id: event for event in events if isinstance(event, FrameEvent)}


def _resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(array)
    values = np.squeeze(values)
    if values.ndim != 2:
        values = values.reshape(1, -1)
    out_h, out_w = shape
    row_index = np.linspace(0, values.shape[0] - 1, out_h).round().astype(np.int64)
    col_index = np.linspace(0, values.shape[1] - 1, out_w).round().astype(np.int64)
    return values[row_index[:, None], col_index[None, :]]


def _depth_valid_ratio(depth_m: np.ndarray) -> float:
    depth = np.asarray(depth_m, dtype=np.float32)
    return float(np.count_nonzero(np.isfinite(depth) & (depth > np.float32(0.0))) / max(depth.size, 1))


def _finite_min(depth_m: np.ndarray) -> float:
    values = np.asarray(depth_m, dtype=np.float32)
    finite = values[np.isfinite(values) & (values > np.float32(0.0))]
    return float(np.min(finite)) if finite.size else 0.0


def _finite_max(depth_m: np.ndarray) -> float:
    values = np.asarray(depth_m, dtype=np.float32)
    finite = values[np.isfinite(values) & (values > np.float32(0.0))]
    return float(np.max(finite)) if finite.size else 0.0


def _read_json(path: str | Path) -> JsonDict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert TUM RGB-D metric depth/pose truth into BEV anchor labels.")
    parser.add_argument("--log", required=True, help="Input TUM-imported HomeBrain route.")
    parser.add_argument("--out", required=True, help="Output BEV directory.")
    args = parser.parse_args(argv)
    manifest_path = rgbd_truth_to_bev(log_dir=args.log, out_dir=args.out)
    print(f"wrote RGB-D truth BEV manifest to {manifest_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

