from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from homebrain.datasets.openloris_scene import (
    OPENLORIS_DEPTH_SCALE,
    OPENLORIS_LICENSE_REVIEW_STATUS,
    OPENLORIS_ROUTE_ASSOCIATIONS_FILE,
)
from homebrain.datasets.tum_rgbd import read_depth_png_m
from homebrain.geometry.bev_projector import BEV_ARTIFACT_KINDS, BEV_MANIFEST_FILE, BEV_SCHEMA_VERSION, bev_array_stats
from homebrain.messages.schema import FrameEvent, JsonDict
from homebrain.replay.replayd import order_events_for_replay
from homebrain.replay.segment_log import load_manifest, read_events
from homebrain.teachers.artifacts import file_sha256, relative_to_root, save_array, write_json

DEFAULT_GRID_CELLS = 32
DEFAULT_METERS_PER_CELL = 0.05
DEFAULT_ROBOT_RADIUS_M = 0.18
DEFAULT_FLOOR_HEIGHT_TOL_M = 0.08
DEFAULT_OBSTACLE_HEIGHT_MIN_M = 0.08
DEFAULT_OBSTACLE_DILATION_CELLS = 1


def robot_rgbd_to_bev(
    *,
    log_dir: str | Path,
    out_dir: str | Path,
    review_assumed_extrinsics: bool = False,
    max_frames: int | None = None,
    grid_cells: int = DEFAULT_GRID_CELLS,
    meters_per_cell: float = DEFAULT_METERS_PER_CELL,
    robot_radius_m: float = DEFAULT_ROBOT_RADIUS_M,
) -> Path:
    log_root = Path(log_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    route_manifest = load_manifest(log_root)
    associations_path = log_root / OPENLORIS_ROUTE_ASSOCIATIONS_FILE
    associations = _read_json(associations_path)
    if associations.get("source_type") != "openloris_scene_associations":
        raise ValueError(f"expected OpenLORIS associations in {associations_path}")
    frames_by_id = _route_frames_by_id(log_root)
    frame_records: list[JsonDict] = []
    warnings: list[str] = []
    source_frames = associations.get("frames")
    if not isinstance(source_frames, list):
        raise ValueError("OpenLORIS associations frames must be a list")
    if max_frames is not None:
        source_frames = source_frames[:max_frames]

    for record in source_frames:
        if not isinstance(record, dict):
            continue
        frame_id = int(record.get("frame_id", -1))
        frame = frames_by_id.get(frame_id)
        if frame is None:
            warnings.append(f"frame_{frame_id}_missing_from_route_log")
            continue
        try:
            frame_record = _write_frame(
                output=output,
                log_root=log_root,
                frame=frame,
                association=record,
                review_assumed_extrinsics=review_assumed_extrinsics,
                grid_cells=grid_cells,
                meters_per_cell=meters_per_cell,
                robot_radius_m=robot_radius_m,
            )
        except Exception as exc:  # noqa: BLE001 - preserve failures in manifest.
            warnings.append(f"frame_{frame_id}_failed:{exc}")
            continue
        frame_records.append(frame_record)

    if not frame_records:
        if warnings:
            raise ValueError(f"no robot-frame BEV frames written; first warning: {warnings[0]}")
        raise ValueError("no robot-frame BEV frames written")

    robot_truth_count = sum(1 for frame in frame_records if bool(frame.get("robot_frame_truth")))
    pose_count = sum(1 for frame in frame_records if bool(frame.get("pose_pretrain_ok")))
    manifest: JsonDict = {
        "schema_version": BEV_SCHEMA_VERSION,
        "source_log": log_root.as_posix(),
        "source_segment_id": route_manifest.segment_id,
        "source_schema_version": route_manifest.schema_version,
        "source_depth_artifacts": associations_path.as_posix(),
        "source_depth_manifest_sha256": file_sha256(associations_path),
        "source_depth_teacher_name": "openloris_scene_rgbd",
        "source_depth_backend": "public_robot_mounted_rgbd_pose",
        "source_depth_mock": False,
        "source_depth_real_perception": True,
        "source_name": str(associations.get("source_sequence", route_manifest.segment_id)),
        "source_family": "public_robot_mounted",
        "supervision_grade": "public_robot_frame_geometry",
        "dataset_frame_type": "public_robot_mounted",
        "frame_type": "public_robot_frame_geometry",
        "source_geometry_truth": True,
        "depth_truth_metric": True,
        "pose_truth_metric": bool(pose_count > 0),
        "license_review_status": OPENLORIS_LICENSE_REVIEW_STATUS,
        "grid_shape": [grid_cells, grid_cells],
        "camera_config": {
            "meters_per_cell": float(meters_per_cell),
            "robot_radius_m": float(robot_radius_m),
            "grid_cell_count": int(grid_cells),
            "grid_size_m": float(grid_cells * meters_per_cell),
            "floor_height_tol_m": DEFAULT_FLOOR_HEIGHT_TOL_M,
            "obstacle_height_min_m": DEFAULT_OBSTACLE_HEIGHT_MIN_M,
            "obstacle_dilation_cells": DEFAULT_OBSTACLE_DILATION_CELLS,
        },
        "grid_orientation": {
            "origin": "bottom_center_robot_base_link",
            "forward": "decreasing_row_index",
            "left": "increasing_col_index",
        },
        "coordinate_frame": "base_link_local_robot_frame",
        "projection_model": "pinhole_rgbd_camera_to_base_link_local_bev",
        "artifact_kinds": list(BEV_ARTIFACT_KINDS),
        "weak_label": True,
        "control_safe": False,
        "control_safe_claim": False,
        "geometry_pretrain_ok": True,
        "pose_pretrain_ok": bool(pose_count > 0),
        "action_supervision_ok": "unknown_until_action_sanity_audit",
        "robot_frame_truth_candidate": bool(robot_truth_count > 0),
        "robot_frame_truth": bool(robot_truth_count == len(frame_records)),
        "not_robot_frame_truth": bool(robot_truth_count == 0),
        "trainable_for": "geometry_pose_pretrain_and_action_sanity_review_only",
        "control_safety": "not_control_safe_public_robot_frame_replay_only",
        "warnings": sorted(set(warnings)),
        "assumptions": [
            "Current robot footprint cells are marked free only when robot_frame_truth is present, because the robot occupies that footprint.",
            "Action supervision remains unknown until audit_bev_action_sanity evaluates center, footprint, corridor, and candidates.",
        ],
        "frame_count": len(frame_records),
        "frames": frame_records,
    }
    manifest_path = output / BEV_MANIFEST_FILE
    write_json(manifest_path, manifest)
    return manifest_path


def _write_frame(
    *,
    output: Path,
    log_root: Path,
    frame: FrameEvent,
    association: JsonDict,
    review_assumed_extrinsics: bool,
    grid_cells: int,
    meters_per_cell: float,
    robot_radius_m: float,
) -> JsonDict:
    depth_ref = association.get("depth_ref")
    if not isinstance(depth_ref, str):
        raise ValueError("association is missing depth_ref; robot RGB-D BEV requires depth")
    intrinsics = association.get("intrinsics") if isinstance(association.get("intrinsics"), dict) else frame.intrinsics
    if not isinstance(intrinsics, dict) or not intrinsics.get("available"):
        raise ValueError("association/frame is missing camera intrinsics; refusing robot-frame BEV")
    camera_to_base = association.get("camera_to_base")
    assumed_extrinsics = False
    if not _valid_matrix(camera_to_base):
        if not review_assumed_extrinsics:
            raise ValueError(
                "camera_to_base transform is missing; rerun with --review-assumed-extrinsics "
                "only for review artifacts marked robot_frame_truth=false"
            )
        camera_to_base = _assumed_camera_to_base()
        assumed_extrinsics = True

    depth_scale = float(intrinsics.get("depth_scale", OPENLORIS_DEPTH_SCALE))
    depth_m = read_depth_png_m(log_root / depth_ref, scale=depth_scale)
    arrays, stats = _project_depth_to_bev(
        depth_m=depth_m,
        intrinsics=intrinsics,
        camera_to_base=np.asarray(camera_to_base, dtype=np.float64),
        grid_cells=grid_cells,
        meters_per_cell=meters_per_cell,
    )
    base_pose = _base_pose(association)
    pose_pretrain_ok = base_pose is not None
    robot_frame_truth = bool((not assumed_extrinsics) and pose_pretrain_ok)
    if robot_frame_truth:
        _carve_robot_footprint(arrays, robot_radius_m=robot_radius_m, meters_per_cell=meters_per_cell)
        stats = {**bev_array_stats(arrays), **{key: value for key, value in stats.items() if key not in bev_array_stats(arrays)}}

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
        "source_base_pose": base_pose,
        "width": frame.width,
        "height": frame.height,
        "weak_label": True,
        "control_safe": False,
        "control_safe_claim": False,
        "dataset_frame_type": "public_robot_mounted",
        "frame_type": "public_robot_frame_geometry",
        "source_family": "public_robot_mounted",
        "supervision_grade": "public_robot_frame_geometry",
        "geometry_pretrain_ok": True,
        "pose_pretrain_ok": bool(pose_pretrain_ok),
        "action_supervision_ok": "unknown_until_action_sanity_audit",
        "robot_frame_truth_candidate": bool(pose_pretrain_ok and not assumed_extrinsics),
        "robot_frame_truth": bool(robot_frame_truth),
        "not_robot_frame_truth": not bool(robot_frame_truth),
        "calibration_class": "dataset_robot_mounted_rgbd",
        "intrinsics_source": intrinsics.get("source", "openloris"),
        "extrinsics_source": "openloris_camera_to_base" if not assumed_extrinsics else "assumed_review_only",
        "pose_source": "openloris_groundtruth_or_odom" if pose_pretrain_ok else "missing",
        "scale_source": "openloris_metric_depth",
        "gravity_floor_source": "camera_to_base_robot_mount",
        "trainable_for": "geometry_pose_pretrain_and_action_sanity_review_only",
        "control_safety": "not_control_safe_public_robot_frame_replay_only",
        "projection_model": "pinhole_rgbd_camera_to_base_link_local_bev",
        "camera_to_base": camera_to_base,
        "assumed_extrinsics": bool(assumed_extrinsics),
        "robot_footprint_prior_applied": bool(robot_frame_truth),
        "grid_shape": [grid_cells, grid_cells],
        "stats": stats,
        "warnings": ["assumed_extrinsics_not_robot_frame_truth"] if assumed_extrinsics else [],
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
        "dataset_frame_type": "public_robot_mounted",
        "frame_type": "public_robot_frame_geometry",
        "source_family": "public_robot_mounted",
        "supervision_grade": "public_robot_frame_geometry",
        "geometry_pretrain_ok": True,
        "pose_pretrain_ok": bool(pose_pretrain_ok),
        "action_supervision_ok": "unknown_until_action_sanity_audit",
        "robot_frame_truth_candidate": bool(pose_pretrain_ok and not assumed_extrinsics),
        "robot_frame_truth": bool(robot_frame_truth),
        "not_robot_frame_truth": not bool(robot_frame_truth),
        "metadata_path": relative_to_root(metadata_path, output),
        "metadata_sha256": file_sha256(metadata_path),
        "artifacts": artifact_records,
        "stats": metadata["stats"],
        "warnings": metadata["warnings"],
    }


def _project_depth_to_bev(
    *,
    depth_m: np.ndarray,
    intrinsics: JsonDict,
    camera_to_base: np.ndarray,
    grid_cells: int,
    meters_per_cell: float,
) -> tuple[dict[str, np.ndarray], JsonDict]:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth image must be 2D")
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    rows, cols = depth.shape
    stride = max(1, int(min(rows, cols) // 96))
    vv, uu = np.mgrid[0:rows:stride, 0:cols:stride]
    z = depth[vv, uu]
    valid = np.isfinite(z) & (z > np.float32(0.0))
    if not np.any(valid):
        raise ValueError("depth image has no valid positive metric values")
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    z_valid = z[valid].astype(np.float64)
    x_cam = (u - cx) * z_valid / fx
    y_cam = (v - cy) * z_valid / fy
    points_camera = np.stack([x_cam, y_cam, z_valid, np.ones_like(z_valid)], axis=0)
    points_base = camera_to_base @ points_camera
    forward = points_base[0].astype(np.float32)
    left = points_base[1].astype(np.float32)
    height = points_base[2].astype(np.float32)

    grid_shape = (grid_cells, grid_cells)
    floor_counts = np.zeros(grid_shape, dtype=np.uint16)
    obstacle_counts = np.zeros(grid_shape, dtype=np.uint16)
    point_counts = np.zeros(grid_shape, dtype=np.uint16)
    confidence_sum = np.zeros(grid_shape, dtype=np.float32)
    height_max = np.full(grid_shape, -np.inf, dtype=np.float32)
    half_width = np.float32(grid_cells * meters_per_cell / 2.0)
    in_grid = (
        (forward >= np.float32(0.0))
        & (forward < np.float32(grid_cells * meters_per_cell))
        & (left >= -half_width)
        & (left < half_width)
    )
    if np.any(in_grid):
        forward_grid = forward[in_grid]
        left_grid = left[in_grid]
        height_grid = height[in_grid]
        row = grid_cells - 1 - np.floor(forward_grid / np.float32(meters_per_cell)).astype(np.int64)
        col = np.floor((left_grid + half_width) / np.float32(meters_per_cell)).astype(np.int64)
        row = np.clip(row, 0, grid_cells - 1)
        col = np.clip(col, 0, grid_cells - 1)
        floor_point = np.abs(height_grid) <= np.float32(DEFAULT_FLOOR_HEIGHT_TOL_M)
        obstacle_point = height_grid >= np.float32(DEFAULT_OBSTACLE_HEIGHT_MIN_M)
        np.add.at(point_counts, (row, col), 1)
        np.add.at(confidence_sum, (row, col), 1.0)
        np.maximum.at(height_max, (row, col), height_grid)
        if np.any(floor_point):
            np.add.at(floor_counts, (row[floor_point], col[floor_point]), 1)
        if np.any(obstacle_point):
            np.add.at(obstacle_counts, (row[obstacle_point], col[obstacle_point]), 1)

    floor_candidate = floor_counts > 0
    raw_obstacle = obstacle_counts > 0
    obstacle = _dilate_mask(raw_obstacle, DEFAULT_OBSTACLE_DILATION_CELLS)
    free = floor_candidate & ~obstacle
    unknown = ~(free | obstacle)
    observed = point_counts > 0
    bev_height = np.where(observed, height_max, np.float32(0.0)).astype(np.float32)
    bev_confidence = np.divide(
        confidence_sum,
        point_counts,
        out=np.zeros(grid_shape, dtype=np.float32),
        where=point_counts > 0,
    ).astype(np.float32)
    arrays = {
        "bev_free": free.astype(np.uint8),
        "bev_obstacle": obstacle.astype(np.uint8),
        "bev_unknown": unknown.astype(np.uint8),
        "bev_floor_candidate": floor_candidate.astype(np.uint8),
        "bev_height": bev_height,
        "bev_confidence": bev_confidence,
    }
    stats = {
        **bev_array_stats(arrays),
        "sampled_pixel_count": int(vv.size),
        "valid_depth_point_count": int(np.count_nonzero(valid)),
        "points_in_grid_count": int(np.count_nonzero(in_grid)),
        "observed_cell_count": int(np.count_nonzero(observed)),
        "raw_obstacle_cell_count": int(np.count_nonzero(raw_obstacle)),
        "obstacle_dilation_cells": DEFAULT_OBSTACLE_DILATION_CELLS,
    }
    return arrays, stats


def _carve_robot_footprint(arrays: dict[str, np.ndarray], *, robot_radius_m: float, meters_per_cell: float) -> None:
    radius_cells = max(1, int(round(robot_radius_m / meters_per_cell)))
    shape = arrays["bev_free"].shape
    center = (shape[0] - 1, shape[1] // 2)
    for row_offset in range(-radius_cells, radius_cells + 1):
        for col_offset in range(-radius_cells, radius_cells + 1):
            if row_offset * row_offset + col_offset * col_offset > radius_cells * radius_cells:
                continue
            row = center[0] + row_offset
            col = center[1] + col_offset
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                arrays["bev_free"][row, col] = np.uint8(1)
                arrays["bev_floor_candidate"][row, col] = np.uint8(1)
                arrays["bev_obstacle"][row, col] = np.uint8(0)
                arrays["bev_unknown"][row, col] = np.uint8(0)
                arrays["bev_confidence"][row, col] = np.float32(1.0)
                arrays["bev_height"][row, col] = np.float32(0.0)


def _base_pose(association: JsonDict) -> JsonDict | None:
    pose = association.get("base_pose")
    if isinstance(pose, dict):
        return dict(pose)
    odom = association.get("odom")
    if isinstance(odom, dict) and isinstance(odom.get("pose"), dict):
        return dict(odom["pose"])
    return None


def _route_frames_by_id(log_dir: Path) -> dict[int, FrameEvent]:
    events = order_events_for_replay(read_events(log_dir))
    return {event.frame_id: event for event in events if isinstance(event, FrameEvent)}


def _valid_matrix(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    return all(isinstance(row, list) and len(row) == 4 for row in value)


def _assumed_camera_to_base() -> list[list[float]]:
    return [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _dilate_mask(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0 or not np.any(mask):
        return mask.copy()
    height, width = mask.shape
    dilated = mask.copy()
    source_rows, source_cols = np.nonzero(mask)
    radius_sq = radius_cells * radius_cells
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_sq:
                continue
            rows = source_rows + dy
            cols = source_cols + dx
            valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            dilated[rows[valid], cols[valid]] = True
    return dilated


def _read_json(path: str | Path) -> JsonDict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert robot-mounted RGB-D route frames into robot-frame BEV labels.")
    parser.add_argument("--log", required=True, help="Input OpenLORIS-imported HomeBrain route.")
    parser.add_argument("--out", required=True, help="Output BEV directory.")
    parser.add_argument("--review-assumed-extrinsics", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--grid-cells", type=int, default=DEFAULT_GRID_CELLS)
    parser.add_argument("--meters-per-cell", type=float, default=DEFAULT_METERS_PER_CELL)
    parser.add_argument("--robot-radius-m", type=float, default=DEFAULT_ROBOT_RADIUS_M)
    args = parser.parse_args(argv)
    manifest_path = robot_rgbd_to_bev(
        log_dir=args.log,
        out_dir=args.out,
        review_assumed_extrinsics=args.review_assumed_extrinsics,
        max_frames=args.max_frames,
        grid_cells=args.grid_cells,
        meters_per_cell=args.meters_per_cell,
        robot_radius_m=args.robot_radius_m,
    )
    print(f"wrote robot RGB-D BEV manifest to {manifest_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
