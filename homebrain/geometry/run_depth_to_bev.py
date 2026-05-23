from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.geometry.bev_projector import (
    BEV_ARTIFACT_KINDS,
    BEV_MANIFEST_FILE,
    BEV_SCHEMA_VERSION,
    BevProjection,
    build_bev_from_depth,
)
from homebrain.geometry.camera_config import CameraConfig, camera_config_with_overrides, load_camera_config
from homebrain.geometry.validate_bev import validate_bev_artifacts
from homebrain.messages.schema import FrameEvent, JsonDict, deterministic_json
from homebrain.replay.replayd import order_events_for_replay
from homebrain.replay.segment_log import load_manifest, read_events
from homebrain.teachers.artifacts import (
    file_sha256,
    frame_key,
    load_array,
    load_teacher_manifest,
    relative_to_root,
    save_array,
    write_json,
)


def convert_depth_artifacts_to_bev(
    *,
    log_dir: str | Path,
    depth_artifacts_dir: str | Path,
    camera_config_path: str | Path,
    out_dir: str | Path,
    pixel_stride: int = 4,
) -> Path:
    log_path = Path(log_dir)
    depth_root = Path(depth_artifacts_dir)
    output = Path(out_dir)
    camera_config = load_camera_config(camera_config_path)
    source_manifest = load_manifest(log_path)
    depth_manifest = load_teacher_manifest(depth_root)
    frames_by_key = _route_frames_by_key(log_path)
    output.mkdir(parents=True, exist_ok=True)

    frame_records: list[JsonDict] = []
    warnings: list[str] = []
    assumptions: list[str] = [
        "camera_extrinsics_from_config_not_calibrated",
        "weak_depth_to_bev_labels_for_training_only",
    ]

    for source_record in depth_manifest["frames"]:
        if not isinstance(source_record, dict):
            warnings.append("skipped non-object source depth frame record")
            continue
        frame = frames_by_key.get(frame_key(source_record))
        record = _write_bev_frame(
            depth_root=depth_root,
            output=output,
            source_record=source_record,
            route_frame=frame,
            camera_config=camera_config,
            pixel_stride=pixel_stride,
        )
        frame_records.append(record)
        warnings.extend(str(item) for item in record.get("warnings", []))
        assumptions.extend(str(item) for item in record.get("assumptions", []))

    manifest: JsonDict = {
        "schema_version": BEV_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "source_log": log_path.as_posix(),
        "source_segment_id": source_manifest.segment_id,
        "source_schema_version": source_manifest.schema_version,
        "source_depth_artifacts": depth_root.as_posix(),
        "source_depth_manifest_sha256": file_sha256(depth_root / "teacher_manifest.json"),
        "source_depth_teacher_name": depth_manifest.get("teacher_name"),
        "source_depth_backend": depth_manifest.get("backend"),
        "source_depth_mock": bool(depth_manifest.get("mock", False)),
        "source_depth_real_perception": bool(depth_manifest.get("real_perception", False)),
        "camera_config_path": Path(camera_config_path).as_posix(),
        "camera_config_schema_version": "homebrain.camera_config.v0",
        "camera_config": camera_config.to_dict(),
        "grid_shape": list(camera_config.grid_shape),
        "grid_orientation": {
            "row_0": "far_forward",
            "last_row": "robot_near_field",
            "col_0": "robot_right",
            "last_col": "robot_left",
            "robot_reference": "bottom_center_forward_only_grid",
        },
        "coordinate_frame": "robot_x_forward_y_left_z_up",
        "projection_model": "pinhole_depth_m_with_single_focallength_px",
        "rotation_convention": "camera x-right y-down z-forward mapped to robot x-forward y-left z-up, then roll/pitch/yaw in robot frame; positive pitch points camera downward",
        "pixel_stride": pixel_stride,
        "artifact_kinds": list(BEV_ARTIFACT_KINDS),
        "weak_label": True,
        "control_safe": False,
        "control_safety": "not_control_safe_weak_offline_geometry_label",
        "calibration_status": "assumed_from_camera_config_and_depth_pro_focal_length",
        "warnings": sorted(set(warnings)),
        "assumptions": sorted(set(assumptions)),
        "frame_count": len(frame_records),
        "frames": frame_records,
    }
    manifest_path = output / BEV_MANIFEST_FILE
    write_json(manifest_path, manifest)
    return manifest_path


def run_camera_config_sweep(
    *,
    depth_artifacts_dir: str | Path,
    camera_config_path: str | Path,
    report_path: str | Path,
    pixel_stride: int = 4,
    max_frames: int | None = None,
) -> Path:
    depth_root = Path(depth_artifacts_dir)
    base_config = load_camera_config(camera_config_path)
    depth_manifest = load_teacher_manifest(depth_root)
    frame_records = [record for record in depth_manifest["frames"] if isinstance(record, dict)]
    if max_frames is not None and max_frames > 0:
        frame_records = frame_records[:max_frames]

    variants = _sweep_variants(base_config)
    results = []
    for name, config in variants:
        results.append(_score_sweep_variant(name, config, depth_root, frame_records, pixel_stride))

    report: JsonDict = {
        "schema_version": "homebrain.geometry.bev_sweep.v0",
        "created_at_utc": _utc_now(),
        "source_depth_artifacts": depth_root.as_posix(),
        "source_depth_manifest_sha256": file_sha256(depth_root / "teacher_manifest.json"),
        "camera_config_path": Path(camera_config_path).as_posix(),
        "pixel_stride": pixel_stride,
        "max_frames": max_frames,
        "not_ground_truth": True,
        "weak_label": True,
        "control_safe": False,
        "score_note": "sanity_score rewards nonempty, stable weak labels only; it is not accuracy, traversability, or safety",
        "variant_count": len(results),
        "variants": results,
    }
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(deterministic_json(report))
        handle.write("\n")
    return target


def _write_bev_frame(
    *,
    depth_root: Path,
    output: Path,
    source_record: JsonDict,
    route_frame: FrameEvent | None,
    camera_config: CameraConfig,
    pixel_stride: int,
) -> JsonDict:
    projection = _project_source_record(
        depth_root=depth_root,
        source_record=source_record,
        route_frame=route_frame,
        camera_config=camera_config,
        pixel_stride=pixel_stride,
    )
    frame_id = int(source_record["frame_id"])
    camera_id = str(source_record["camera_id"])
    frame_dir = output / "frames" / f"{camera_id}_{frame_id:06d}"

    artifacts: dict[str, JsonDict] = {}
    for kind, array in projection.arrays().items():
        target = frame_dir / f"{kind}.npy"
        record = save_array(target, array)
        artifacts[kind] = {
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
        "source_depth_artifacts": {
            "depth_m": source_record["artifacts"]["depth_m"]["path"],
            "depth_confidence": source_record["artifacts"].get("depth_confidence", {}).get("path"),
            "focallength_px": source_record["artifacts"]["focallength_px"]["path"],
        },
        "source_data_ref": source_record.get("source_data_ref"),
        "width": int(source_record["width"]),
        "height": int(source_record["height"]),
        "weak_label": True,
        "control_safe": False,
        "control_safety": "not_control_safe_weak_offline_geometry_label",
        "camera_config": camera_config.to_dict(),
        "grid_shape": list(camera_config.grid_shape),
        "meters_per_cell": camera_config.meters_per_cell,
        "classification_rules": {
            "floor_candidate": f"abs(height_m) <= {camera_config.floor_height_tol_m}",
            "obstacle": f"height_m >= {camera_config.obstacle_height_min_m}, then dilated by robot_radius_m for weak clearance context",
            "free": "floor_candidate and not obstacle",
            "unknown": "not free and not obstacle",
        },
        "principal_point_px": list(projection.projected.principal_point_px),
        "focal_length_px": projection.projected.focal_length_px,
        "pixel_stride": pixel_stride,
        "assumptions": list(projection.assumptions),
        "warnings": list(projection.warnings),
        "stats": projection.stats,
        "artifact_shapes": {kind: artifacts[kind]["shape"] for kind in BEV_ARTIFACT_KINDS},
        "artifact_dtypes": {kind: artifacts[kind]["dtype"] for kind in BEV_ARTIFACT_KINDS},
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
        "control_safe": False,
        "metadata_path": relative_to_root(metadata_path, output),
        "metadata_sha256": file_sha256(metadata_path),
        "artifacts": artifacts,
        "stats": projection.stats,
        "assumptions": list(projection.assumptions),
        "warnings": list(projection.warnings),
    }


def _project_source_record(
    *,
    depth_root: Path,
    source_record: JsonDict,
    route_frame: FrameEvent | None,
    camera_config: CameraConfig,
    pixel_stride: int,
) -> BevProjection:
    artifacts = source_record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"depth source frame {source_record.get('frame_id')} missing artifacts object")
    depth = _load_required_array(depth_root, artifacts, "depth_m")
    focal_array = _load_required_array(depth_root, artifacts, "focallength_px")
    focal = _focal_scalar(focal_array)
    confidence = _load_optional_confidence(depth_root, artifacts, depth.shape)
    principal_point, principal_source = _principal_point_from_frame(route_frame)

    projection = build_bev_from_depth(
        depth,
        focal,
        camera_config,
        depth_confidence=confidence,
        principal_point_px=principal_point,
        pixel_stride=pixel_stride,
    )
    assumptions = list(projection.assumptions)
    warnings = list(projection.warnings)
    if route_frame is None:
        warnings.append("source frame missing from log; calibration intrinsics unavailable")
    if principal_source == "frame_intrinsics":
        assumptions.append("principal_point_from_frame_intrinsics")
    else:
        assumptions.append("principal_point_from_image_center")
    return replace(
        projection,
        assumptions=tuple(dict.fromkeys(assumptions)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _route_frames_by_key(log_dir: Path) -> dict[str, FrameEvent]:
    events = order_events_for_replay(read_events(log_dir))
    return {frame_key(event): event for event in events if isinstance(event, FrameEvent)}


def _load_required_array(root: Path, artifacts: JsonDict, kind: str) -> np.ndarray:
    record = artifacts.get(kind)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError(f"missing required Depth Pro artifact {kind}")
    return load_array(root / str(record["path"]))


def _load_optional_confidence(root: Path, artifacts: JsonDict, depth_shape: tuple[int, ...]) -> np.ndarray | None:
    record = artifacts.get("depth_confidence")
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return None
    confidence = load_array(root / str(record["path"]))
    if confidence.shape != depth_shape:
        raise ValueError(f"depth_confidence shape {confidence.shape} does not match depth shape {depth_shape}")
    return confidence


def _focal_scalar(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=np.float32).reshape(-1)
    if values.size != 1:
        raise ValueError(f"focallength_px must contain one scalar, got shape {array.shape}")
    value = float(values[0])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"focallength_px must be positive and finite, got {value!r}")
    return value


def _principal_point_from_frame(frame: FrameEvent | None) -> tuple[tuple[float, float] | None, str]:
    if frame is None or not isinstance(frame.intrinsics, dict):
        return None, "assumed_center"
    intrinsics = frame.intrinsics
    cx = _optional_number(intrinsics.get("cx"))
    cy = _optional_number(intrinsics.get("cy"))
    if cx is not None and cy is not None:
        return (cx, cy), "frame_intrinsics"
    principal = intrinsics.get("principal_point_px") or intrinsics.get("principal_point")
    if isinstance(principal, list) and len(principal) == 2:
        px = _optional_number(principal[0])
        py = _optional_number(principal[1])
        if px is not None and py is not None:
            return (px, py), "frame_intrinsics"
    camera_matrix = intrinsics.get("camera_matrix")
    if (
        isinstance(camera_matrix, list)
        and len(camera_matrix) >= 2
        and isinstance(camera_matrix[0], list)
        and isinstance(camera_matrix[1], list)
        and len(camera_matrix[0]) >= 3
        and len(camera_matrix[1]) >= 3
    ):
        px = _optional_number(camera_matrix[0][2])
        py = _optional_number(camera_matrix[1][2])
        if px is not None and py is not None:
            return (px, py), "frame_intrinsics"
    return None, "assumed_center"


def _optional_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _sweep_variants(base: CameraConfig) -> list[tuple[str, CameraConfig]]:
    return [
        ("base", base),
        (
            "low_phone_pitch20",
            camera_config_with_overrides(base, camera_height_m=0.32, pitch_deg=20.0, roll_deg=0.0, yaw_deg=0.0),
        ),
        (
            "mid_phone_pitch30",
            camera_config_with_overrides(base, camera_height_m=0.45, pitch_deg=30.0, roll_deg=0.0, yaw_deg=0.0),
        ),
        (
            "high_phone_pitch35",
            camera_config_with_overrides(base, camera_height_m=0.62, pitch_deg=35.0, roll_deg=0.0, yaw_deg=0.0),
        ),
    ]


def _score_sweep_variant(
    name: str,
    config: CameraConfig,
    depth_root: Path,
    frame_records: list[JsonDict],
    pixel_stride: int,
) -> JsonDict:
    free_ratios: list[float] = []
    obstacle_ratios: list[float] = []
    unknown_ratios: list[float] = []
    confidence_means: list[float] = []
    occupancy: list[np.ndarray] = []
    error_count = 0

    for record in frame_records:
        try:
            projection = _project_source_record(
                depth_root=depth_root,
                source_record=record,
                route_frame=None,
                camera_config=config,
                pixel_stride=pixel_stride,
            )
        except Exception:  # noqa: BLE001 - sweep report counts frame failures.
            error_count += 1
            continue
        free_ratios.append(float(projection.stats["free_ratio"]))
        obstacle_ratios.append(float(projection.stats["obstacle_ratio"]))
        unknown_ratios.append(float(projection.stats["unknown_ratio"]))
        confidence_means.append(float(projection.stats["confidence_mean"]))
        occupancy.append(
            (
                (projection.bev_free > 0).astype(np.uint8)
                + (projection.bev_obstacle > 0).astype(np.uint8) * 2
            ).astype(np.uint8)
        )

    jitter_values: list[float] = []
    for previous, current in zip(occupancy, occupancy[1:]):
        if previous.shape == current.shape and previous.size:
            jitter_values.append(float(np.count_nonzero(previous != current) / previous.size))

    unknown_mean = _mean(unknown_ratios)
    jitter_mean = _mean(jitter_values)
    confidence_mean = _mean(confidence_means)
    sanity_score = confidence_mean * max(0.0, 1.0 - unknown_mean) * max(0.0, 1.0 - jitter_mean)
    return {
        "name": name,
        "camera_config": config.to_dict(),
        "processed_frame_count": len(free_ratios),
        "frame_error_count": error_count,
        "free_ratio_mean": _mean(free_ratios),
        "obstacle_ratio_mean": _mean(obstacle_ratios),
        "unknown_ratio_mean": unknown_mean,
        "confidence_mean": confidence_mean,
        "temporal_jitter_mean": jitter_mean,
        "sanity_score": float(sanity_score),
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert Depth Pro depth artifacts into weak egocentric BEV labels.")
    parser.add_argument("--log", required=True, help="Input HomeBrain route log directory.")
    parser.add_argument("--depth-artifacts", required=True, help="Input Depth Pro teacher artifact directory.")
    parser.add_argument("--camera-config", required=True, help="Camera config JSON path.")
    parser.add_argument("--out", required=True, help="Output BEV artifact directory.")
    parser.add_argument("--pixel-stride", type=int, default=4, help="Depth pixel stride used for projection.")
    parser.add_argument("--sweep-report", default=None, help="Optional output JSON for plausible camera-config sweep.")
    parser.add_argument("--sweep-max-frames", type=int, default=0, help="Optional frame cap for sweep scoring.")
    args = parser.parse_args(argv)

    manifest_path = convert_depth_artifacts_to_bev(
        log_dir=args.log,
        depth_artifacts_dir=args.depth_artifacts,
        camera_config_path=args.camera_config,
        out_dir=args.out,
        pixel_stride=args.pixel_stride,
    )
    validation = validate_bev_artifacts(args.out)
    print(f"wrote BEV manifest to {manifest_path.as_posix()}")
    print(json.dumps(validation.to_metrics(), sort_keys=True))
    if args.sweep_report:
        report_path = run_camera_config_sweep(
            depth_artifacts_dir=args.depth_artifacts,
            camera_config_path=args.camera_config,
            report_path=args.sweep_report,
            pixel_stride=args.pixel_stride,
            max_frames=args.sweep_max_frames if args.sweep_max_frames > 0 else None,
        )
        print(f"wrote BEV camera-config sweep report to {report_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

