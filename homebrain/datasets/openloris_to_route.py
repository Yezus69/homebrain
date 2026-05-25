from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from homebrain.datasets.openloris_scene import (
    OPENLORIS_DEPTH_SCALE,
    OPENLORIS_LICENSE_NAME,
    OPENLORIS_LICENSE_REVIEW_STATUS,
    OPENLORIS_ROUTE_ASSOCIATIONS_FILE,
    OPENLORIS_SCHEMA_VERSION,
    OPENLORIS_SOURCE_URL,
    OpenLorisAssociation,
    discover_sequence_root,
    load_associations,
    load_calibration,
    merge_imu_samples,
    parse_imu_samples,
    sequence_scene,
    timestamp_ns,
    validate_sequence_dir,
)
from homebrain.datasets.usage_policy import openloris_usage_policy
from homebrain.ingest.image_sequence import read_image_size
from homebrain.ingest.metadata import ROUTE_SOURCE_METADATA_FILE, ROUTE_SOURCE_SCHEMA_VERSION, write_route_metadata
from homebrain.messages.schema import Event, FrameEvent, ImuEvent, JsonDict, OdomEvent, PoseEvent, deterministic_json
from homebrain.replay.segment_log import write_segment
from homebrain.teachers.artifacts import file_sha256

OPENLORIS_IMPORT_SOURCE = "openloris_scene_import"


def openloris_to_route(
    *,
    source_dir: str | Path,
    out_dir: str | Path,
    max_frames: int | None = None,
    camera_id: str = "d400_color",
    require_depth: bool = True,
    require_robot_frame_calibration: bool = True,
    include_imu: bool = True,
) -> Path:
    source = discover_sequence_root(source_dir)
    output = Path(out_dir)
    if max_frames is not None and max_frames < 1:
        raise ValueError("max-frames must be at least 1 when supplied")
    validate_sequence_dir(source, require_depth=require_depth)
    scene = sequence_scene(source)
    calibration = load_calibration(source, camera_id=camera_id)
    intrinsics = _frame_intrinsics(calibration)
    camera_to_base = calibration.get("camera_to_base")
    if require_robot_frame_calibration:
        _require_openloris_robot_frame_calibration(
            intrinsics=intrinsics,
            camera_to_base=camera_to_base,
            source=source,
        )

    associations = load_associations(source, max_frames=max_frames)
    if not associations:
        raise ValueError(f"no OpenLORIS RGB associations found in {source}")

    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir(parents=True, exist_ok=True)
    (output / "depth").mkdir(parents=True, exist_ok=True)

    events: list[Event] = []
    artifact_files: list[str] = [ROUTE_SOURCE_METADATA_FILE, OPENLORIS_ROUTE_ASSOCIATIONS_FILE]
    artifact_checksums: dict[str, str] = {}
    frame_records: list[JsonDict] = []
    copied_depth_count = 0
    for frame_id, association in enumerate(associations):
        rgb_source = source / association.rgb.path
        if not rgb_source.exists():
            raise FileNotFoundError(f"missing OpenLORIS RGB frame: {rgb_source}")
        width, height = read_image_size(rgb_source)
        rgb_ref = f"frames/frame_{frame_id:06d}{rgb_source.suffix.lower()}"
        shutil.copy2(rgb_source, output / rgb_ref)
        artifact_files.append(rgb_ref)
        artifact_checksums[rgb_ref] = file_sha256(output / rgb_ref)

        depth_ref: str | None = None
        depth_sha256: str | None = None
        if association.depth is not None:
            depth_source = source / association.depth.path
            if not depth_source.exists():
                raise FileNotFoundError(f"missing OpenLORIS depth frame: {depth_source}")
            depth_ref = f"depth/depth_{frame_id:06d}{depth_source.suffix.lower()}"
            shutil.copy2(depth_source, output / depth_ref)
            artifact_files.append(depth_ref)
            depth_sha256 = file_sha256(output / depth_ref)
            artifact_checksums[depth_ref] = depth_sha256
            copied_depth_count += 1

        ts_ns = timestamp_ns(association.rgb.timestamp)
        events.append(
            FrameEvent(
                timestamp_ns=ts_ns,
                sequence_id=output.name,
                source=OPENLORIS_IMPORT_SOURCE,
                camera_id=camera_id,
                frame_id=frame_id,
                width=width,
                height=height,
                format=_frame_format(rgb_source.suffix),
                data_ref=rgb_ref,
                intrinsics=intrinsics,
            )
        )
        if association.odom is not None:
            events.append(_odom_event(output.name, association.odom))
        if association.pose is not None:
            events.append(_pose_event(output.name, association.pose))
        frame_records.append(
            _association_record(
                frame_id,
                rgb_ref,
                depth_ref,
                association,
                intrinsics,
                camera_to_base,
                rgb_sha256=artifact_checksums[rgb_ref],
                depth_sha256=depth_sha256,
            )
        )

    if require_depth and copied_depth_count != len(frame_records):
        raise ValueError(
            f"OpenLORIS import requires synchronized depth, but copied {copied_depth_count}/"
            f"{len(frame_records)} depth frames"
        )

    imu_pairs = (
        merge_imu_samples(
            parse_imu_samples(source / "d400_accelerometer.txt"),
            parse_imu_samples(source / "d400_gyroscope.txt"),
        )
        if include_imu
        else []
    )
    if imu_pairs:
        min_ts = min(record["timestamp_ns"] for record in frame_records)
        max_ts = max(record["timestamp_ns"] for record in frame_records)
        for pair in imu_pairs:
            pair_ts = timestamp_ns(pair.timestamp)
            if min_ts <= pair_ts <= max_ts:
                events.append(
                    ImuEvent(
                        timestamp_ns=pair_ts,
                        sequence_id=output.name,
                        source=OPENLORIS_IMPORT_SOURCE,
                        accel_mps2=pair.accel,
                        gyro_radps=pair.gyro,
                    )
                )

    robot_frame_truth_candidate = bool(camera_to_base and any(record.get("base_pose") for record in frame_records))
    associations_manifest: JsonDict = {
        "schema_version": OPENLORIS_SCHEMA_VERSION,
        "source_type": "openloris_scene_associations",
        "source_sequence": source.name,
        "source_scene": scene,
        "source_dir": source.resolve().as_posix(),
        "official_page": OPENLORIS_SOURCE_URL,
        "license_name": OPENLORIS_LICENSE_NAME,
        "license_review_status": OPENLORIS_LICENSE_REVIEW_STATUS,
        "usage_policy": openloris_usage_policy(),
        "owned_or_license_approved": False,
        "poc_training_eval_allowed": True,
        "product_training_approved": False,
        "runtime_dependency": False,
        "derived_dataset_redistribution_allowed": False,
        "attribution_required": True,
        "depth_scale": OPENLORIS_DEPTH_SCALE,
        "intrinsics": intrinsics,
        "camera_to_base": camera_to_base,
        "camera_to_base_source": calibration.get("camera_to_base_source"),
        "calibration_sources": _calibration_sources(source, calibration),
        "dataset_frame_type": "public_robot_mounted",
        "robot_frame_truth_candidate": robot_frame_truth_candidate,
        "robot_frame_truth": bool(robot_frame_truth_candidate),
        "control_safe": False,
        "frame_count": len(frame_records),
        "artifact_sha256": artifact_checksums,
        "frames": frame_records,
    }
    _write_json(output / OPENLORIS_ROUTE_ASSOCIATIONS_FILE, associations_manifest)

    missing_sensor_notices = []
    if not include_imu:
        missing_sensor_notices.append(
            {
                "sensor": "imu",
                "status": "not_imported",
                "reason": "IMU parsing was explicitly skipped; odom/pose events remain available when associated",
            }
        )
    elif not imu_pairs:
        missing_sensor_notices.append(
            {"sensor": "imu", "status": "unavailable", "reason": "no paired d400 accelerometer/gyroscope samples"}
        )
    if not any(record.get("odom") for record in frame_records):
        missing_sensor_notices.append(
            {"sensor": "odometry", "status": "unavailable", "reason": "odom.txt missing or no matching odom rows"}
        )
    if not any(record.get("base_pose") for record in frame_records):
        missing_sensor_notices.append(
            {
                "sensor": "groundtruth_pose",
                "status": "unavailable",
                "reason": "groundtruth.txt missing or no matching pose rows",
            }
        )
    if not camera_to_base:
        missing_sensor_notices.append(
            {
                "sensor": "camera_to_base",
                "status": "unavailable",
                "reason": "camera-to-base transform missing; robot-frame BEV must refuse unless reviewed as assumed",
            }
        )

    metadata: JsonDict = {
        "schema_version": ROUTE_SOURCE_SCHEMA_VERSION,
        "source_type": "openloris_scene",
        "source_path": source.resolve().as_posix(),
        "official_page": OPENLORIS_SOURCE_URL,
        "sequence": source.name,
        "scene": scene,
        "frame_count": len(frame_records),
        "imported_frame_count": len(frame_records),
        "has_rgb": True,
        "has_depth": copied_depth_count > 0,
        "depth_frame_count": copied_depth_count,
        "has_intrinsics": bool(intrinsics.get("available")),
        "has_imu": bool(imu_pairs),
        "has_odometry": any(record.get("odom") for record in frame_records),
        "has_wheel_odometry": False,
        "has_groundtruth_pose": any(record.get("base_pose") for record in frame_records),
        "has_camera_to_base_transform": bool(camera_to_base),
        "has_commands": False,
        "dataset_frame_type": "public_robot_mounted",
        "calibration_class": "dataset_robot_mounted_rgbd",
        "intrinsics_source": intrinsics.get("source", "missing"),
        "extrinsics_source": _extrinsics_source_name(calibration),
        "pose_source": "openloris_groundtruth_or_odom",
        "scale_source": "openloris_metric_depth",
        "gravity_floor_source": "camera_to_base_robot_mount",
        "depth_scale": OPENLORIS_DEPTH_SCALE,
        "associations_file": OPENLORIS_ROUTE_ASSOCIATIONS_FILE,
        "robot_frame_truth_candidate": robot_frame_truth_candidate,
        "robot_frame_truth": bool(robot_frame_truth_candidate),
        "license_name": OPENLORIS_LICENSE_NAME,
        "license_review_status": OPENLORIS_LICENSE_REVIEW_STATUS,
        "usage_policy": openloris_usage_policy(),
        "owned_or_license_approved": False,
        "poc_training_eval_allowed": True,
        "product_training_approved": False,
        "runtime_dependency": False,
        "derived_dataset_redistribution_allowed": False,
        "attribution_required": True,
        "control_safe": False,
        "control_safe_claim": False,
        "user_owned_or_license_unknown": False,
        "missing_sensor_notices": missing_sensor_notices,
        "artifact_sha256": artifact_checksums,
        "calibration_sources": _calibration_sources(source, calibration),
    }
    write_route_metadata(output, metadata)
    write_segment(output, events, segment_id=output.name, artifact_files=artifact_files)
    return output


def _association_record(
    frame_id: int,
    rgb_ref: str,
    depth_ref: str | None,
    association: OpenLorisAssociation,
    intrinsics: JsonDict,
    camera_to_base: object,
    *,
    rgb_sha256: str,
    depth_sha256: str | None,
) -> JsonDict:
    return {
        "frame_id": frame_id,
        "timestamp_ns": timestamp_ns(association.rgb.timestamp),
        "rgb_timestamp": association.rgb.timestamp,
        "rgb_source_path": association.rgb.path,
        "rgb_ref": rgb_ref,
        "rgb_sha256": rgb_sha256,
        "depth_timestamp": association.depth.timestamp if association.depth is not None else None,
        "depth_source_path": association.depth.path if association.depth is not None else None,
        "depth_ref": depth_ref,
        "depth_sha256": depth_sha256,
        "intrinsics": intrinsics,
        "camera_to_base": camera_to_base,
        "base_pose": association.pose.to_dict() if association.pose is not None else None,
        "odom": association.odom.to_dict() if association.odom is not None else None,
        "dataset_frame_type": "public_robot_mounted",
        "robot_frame_truth_candidate": bool(camera_to_base and association.pose is not None),
        "robot_frame_truth": bool(camera_to_base and association.pose is not None),
        "control_safe": False,
    }


def _odom_event(sequence_id: str, odom: object) -> OdomEvent:
    typed = odom  # keeps call sites readable without importing the dataclass only for typing.
    return OdomEvent(
        timestamp_ns=timestamp_ns(typed.timestamp),  # type: ignore[attr-defined]
        sequence_id=sequence_id,
        source=OPENLORIS_IMPORT_SOURCE,
        position_m=(typed.pose.tx, typed.pose.ty, typed.pose.tz),  # type: ignore[attr-defined]
        orientation_xyzw=(typed.pose.qx, typed.pose.qy, typed.pose.qz, typed.pose.qw),  # type: ignore[attr-defined]
        linear_velocity_mps=typed.linear_velocity,  # type: ignore[attr-defined]
        angular_velocity_radps=typed.angular_velocity,  # type: ignore[attr-defined]
    )


def _pose_event(sequence_id: str, pose: object) -> PoseEvent:
    typed = pose
    return PoseEvent(
        timestamp_ns=timestamp_ns(typed.timestamp),  # type: ignore[attr-defined]
        sequence_id=sequence_id,
        source=OPENLORIS_IMPORT_SOURCE,
        position_m=(typed.tx, typed.ty, typed.tz),  # type: ignore[attr-defined]
        orientation_xyzw=(typed.qx, typed.qy, typed.qz, typed.qw),  # type: ignore[attr-defined]
        frame_id="map",
        child_frame_id="base_link",
        pose_kind="groundtruth_robot_base_pose",
    )


def _frame_intrinsics(calibration: JsonDict) -> JsonDict:
    intrinsics = calibration.get("intrinsics")
    if isinstance(intrinsics, dict):
        return {
            "available": bool(intrinsics.get("available", False)),
            "status": "provided_by_openloris" if intrinsics.get("available") else str(intrinsics.get("status", "missing")),
            **intrinsics,
        }
    return {"available": False, "status": "missing"}


def _frame_format(suffix: str) -> str:
    value = suffix.lower()
    if value in {".jpg", ".jpeg"}:
        return "encoded_jpeg"
    if value == ".png":
        return "encoded_png"
    if value == ".pgm":
        return "encoded_pgm"
    if value == ".ppm":
        return "encoded_ppm"
    raise ValueError(f"unsupported RGB frame extension: {suffix}")


def _write_json(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(deterministic_json(data))
        handle.write("\n")


def _require_openloris_robot_frame_calibration(
    *,
    intrinsics: JsonDict,
    camera_to_base: object,
    source: Path,
) -> None:
    missing: list[str] = []
    if not intrinsics.get("available"):
        missing.append("camera intrinsics")
    if not _valid_matrix(camera_to_base):
        missing.append("camera_to_base transform")
    if missing:
        raise ValueError(
            "OpenLORIS robot-frame import requires measured calibration; missing "
            + ", ".join(missing)
            + f" in {source}. Expected sensors.yaml plus trans_matrix.yaml or reviewed calibration.json."
        )


def _valid_matrix(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in value)
    )


def _extrinsics_source_name(calibration: JsonDict) -> str:
    source = calibration.get("camera_to_base_source")
    if not isinstance(source, dict):
        return "missing"
    if source.get("source") == "trans_matrix.yaml":
        return "openloris_trans_matrix_yaml"
    if source.get("enabled_by_env") == "HOMEBRAIN_OPENLORIS_ALLOW_OFFICIAL_STATIC_TF":
        return "openloris_static_tf_env_reviewed"
    if source.get("source"):
        return str(source["source"])
    if source.get("source_url"):
        return "openloris_static_tf"
    return "provided"


def _calibration_sources(source: Path, calibration: JsonDict) -> JsonDict:
    files: JsonDict = {}
    for filename in ("sensors.yaml", "trans_matrix.yaml", "openloris_calibration.json", "calibration.json"):
        path = source / filename
        if path.exists():
            files[filename] = {
                "path": path.as_posix(),
                "sha256": file_sha256(path),
            }
    return {
        "intrinsics": calibration.get("intrinsics_source"),
        "camera_to_base": calibration.get("camera_to_base_source"),
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import an extracted OpenLORIS-Scene package into a HomeBrain route log.")
    parser.add_argument("--source", required=True, help="Extracted OpenLORIS sequence directory.")
    parser.add_argument("--out", required=True, help="Output HomeBrain route directory.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera-id", default="d400_color")
    parser.add_argument(
        "--allow-incomplete-calibration",
        action="store_true",
        help="Import for review only when robot-frame RGB-D calibration is incomplete.",
    )
    parser.add_argument("--allow-missing-depth", action="store_true", help="Import RGB-only review routes.")
    parser.add_argument(
        "--skip-imu",
        action="store_true",
        help="Do not parse full IMU files; frame-paired odom/pose are still imported when available.",
    )
    args = parser.parse_args(argv)
    out = openloris_to_route(
        source_dir=args.source,
        out_dir=args.out,
        max_frames=args.max_frames,
        camera_id=args.camera_id,
        require_depth=not args.allow_missing_depth,
        require_robot_frame_calibration=not args.allow_incomplete_calibration,
        include_imu=not args.skip_imu,
    )
    print(f"imported OpenLORIS route to {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
