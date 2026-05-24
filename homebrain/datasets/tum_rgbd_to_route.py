from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from homebrain.datasets.tum_rgbd import (
    TUM_RGBD_DEPTH_SCALE,
    TUM_RGBD_LICENSE_NAME,
    TUM_RGBD_LICENSE_REVIEW_STATUS,
    TUM_RGBD_ROUTE_ASSOCIATIONS_FILE,
    TUM_RGBD_SCHEMA_VERSION,
    TUM_RGBD_SOURCE_URL,
    TumAssociation,
    TumAccelerometer,
    load_associations,
    parse_accelerometer,
    read_png_shape,
    sequence_intrinsics,
    validate_sequence_dir,
)
from homebrain.ingest.image_sequence import read_image_size
from homebrain.ingest.metadata import ROUTE_SOURCE_METADATA_FILE, ROUTE_SOURCE_SCHEMA_VERSION, write_route_metadata
from homebrain.messages.schema import Event, FrameEvent, JsonDict, PoseEvent, deterministic_json
from homebrain.replay.segment_log import write_segment

TUM_RGBD_IMPORT_SOURCE = "tum_rgbd_import"


def tum_rgbd_to_route(
    *,
    source_dir: str | Path,
    out_dir: str | Path,
    max_frames: int | None = None,
) -> Path:
    source = Path(source_dir)
    output = Path(out_dir)
    if max_frames is not None and max_frames < 1:
        raise ValueError("max-frames must be at least 1 when supplied")
    validate_sequence_dir(source)
    sequence = _sequence_name(source)
    intrinsics = sequence_intrinsics(sequence)
    associations = load_associations(source)
    accelerometer_samples = parse_accelerometer(source / "accelerometer.txt")
    if max_frames is not None:
        associations = associations[:max_frames]
    if not associations:
        raise ValueError(f"no RGB-D/groundtruth associations found in {source}")

    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir(parents=True, exist_ok=True)
    (output / "depth").mkdir(parents=True, exist_ok=True)

    events: list[Event] = []
    association_records: list[JsonDict] = []
    artifact_files: list[str] = [ROUTE_SOURCE_METADATA_FILE, TUM_RGBD_ROUTE_ASSOCIATIONS_FILE]
    for frame_id, association in enumerate(associations):
        rgb_source = source / association.rgb.path
        depth_source = source / association.depth.path
        if not rgb_source.exists():
            raise FileNotFoundError(f"missing TUM RGB frame: {rgb_source}")
        if not depth_source.exists():
            raise FileNotFoundError(f"missing TUM depth frame: {depth_source}")
        width, height = read_image_size(rgb_source)
        depth_width, depth_height = read_png_shape(depth_source)
        if (depth_width, depth_height) != (width, height):
            raise ValueError(
                f"RGB/depth shape mismatch for frame {frame_id}: "
                f"rgb={width}x{height}, depth={depth_width}x{depth_height}"
            )

        rgb_ref = f"frames/frame_{frame_id:06d}{rgb_source.suffix.lower()}"
        depth_ref = f"depth/depth_{frame_id:06d}{depth_source.suffix.lower()}"
        shutil.copy2(rgb_source, output / rgb_ref)
        shutil.copy2(depth_source, output / depth_ref)
        timestamp_ns = int(round(association.rgb.timestamp * 1_000_000_000))
        events.append(
            FrameEvent(
                timestamp_ns=timestamp_ns,
                sequence_id=output.name,
                source=TUM_RGBD_IMPORT_SOURCE,
                camera_id="tum_rgb",
                frame_id=frame_id,
                width=width,
                height=height,
                format=_frame_format(rgb_source.suffix),
                data_ref=rgb_ref,
                intrinsics={
                    "available": True,
                    "status": "provided_by_dataset_calibration",
                    **intrinsics,
                },
            )
        )
        events.append(_pose_event(output.name, association))
        artifact_files.extend([rgb_ref, depth_ref])
        accelerometer = _nearest_accelerometer(
            accelerometer_samples,
            association.rgb.timestamp,
            max_difference=0.03,
        )
        association_records.append(_association_record(frame_id, rgb_ref, depth_ref, association, accelerometer, sequence))

    has_accelerometer = bool(accelerometer_samples)
    dataset_frame_type = "public_robot_mounted" if "pioneer" in sequence else "public_rgbd_camera_pose_geometry"
    associations_manifest: JsonDict = {
        "schema_version": TUM_RGBD_SCHEMA_VERSION,
        "source_type": "tum_rgbd_associations",
        "source_sequence": sequence,
        "source_dir": source.resolve().as_posix(),
        "official_page": TUM_RGBD_SOURCE_URL,
        "license_name": TUM_RGBD_LICENSE_NAME,
        "license_review_status": TUM_RGBD_LICENSE_REVIEW_STATUS,
        "depth_scale": TUM_RGBD_DEPTH_SCALE,
        "intrinsics": intrinsics,
        "dataset_frame_type": dataset_frame_type,
        "sensor_masks": {
            "rgb": True,
            "depth": True,
            "camera_groundtruth_pose": True,
            "accelerometer": has_accelerometer,
            "imu_6dof": False,
            "wheel_odometry": False,
            "commands": False,
            "camera_to_base": False,
            "robot_base_pose": False,
        },
        "available_sensor_fields": [
            "rgb",
            "depth",
            "camera_groundtruth_pose",
            *(["accelerometer"] if has_accelerometer else []),
        ],
        "accelerometer_file": (source / "accelerometer.txt").as_posix() if has_accelerometer else None,
        "frame_count": len(association_records),
        "frames": association_records,
        "control_safe": False,
    }
    _write_json(output / TUM_RGBD_ROUTE_ASSOCIATIONS_FILE, associations_manifest)

    metadata: JsonDict = {
        "schema_version": ROUTE_SOURCE_SCHEMA_VERSION,
        "source_type": "tum_rgbd",
        "source_path": source.resolve().as_posix(),
        "official_page": TUM_RGBD_SOURCE_URL,
        "sequence": sequence,
        "frame_count": len(association_records),
        "imported_frame_count": len(association_records),
        "has_rgb": True,
        "has_depth": True,
        "has_groundtruth_pose": True,
        "has_intrinsics": True,
        "has_accelerometer": has_accelerometer,
        "has_imu": False,
        "has_wheel_odometry": False,
        "has_commands": False,
        "has_camera_to_base_transform": False,
        "has_robot_base_pose": False,
        "dataset_frame_type": dataset_frame_type,
        "sensor_masks": {
            "rgb": True,
            "depth": True,
            "camera_groundtruth_pose": True,
            "accelerometer": has_accelerometer,
            "imu_6dof": False,
            "wheel_odometry": False,
            "commands": False,
            "camera_to_base": False,
            "robot_base_pose": False,
        },
        "available_sensor_fields": [
            "rgb",
            "depth",
            "camera_groundtruth_pose",
            *(["accelerometer"] if has_accelerometer else []),
        ],
        "calibration_class": "dataset_metric_rgbd_pose",
        "intrinsics_source": "tum_rgbd_calibration",
        "extrinsics_source": "missing_camera_to_base",
        "pose_source": "tum_rgbd_groundtruth_camera_pose",
        "scale_source": "dataset_metric_depth",
        "gravity_floor_source": "not_robot_floor_calibrated",
        "depth_scale": TUM_RGBD_DEPTH_SCALE,
        "associations_file": TUM_RGBD_ROUTE_ASSOCIATIONS_FILE,
        "license_name": TUM_RGBD_LICENSE_NAME,
        "license_review_status": TUM_RGBD_LICENSE_REVIEW_STATUS,
        "control_safe": False,
        "user_owned_or_license_unknown": False,
        "missing_sensor_notices": [
            {
                "sensor": "imu",
                "status": "unavailable",
                "reason": "tum_rgbd_route_import_has_accelerometer_only_no_gyro_stream_no_imu_event_emitted",
            },
            {
                "sensor": "wheel_odometry",
                "status": "unavailable",
                "reason": "tum_rgbd_route_import_does_not_include_wheel_odometry_stream",
            },
            {
                "sensor": "commands",
                "status": "unavailable",
                "reason": "tum_rgbd_route_import_does_not_include_command_stream",
            },
            {
                "sensor": "camera_to_base",
                "status": "unavailable",
                "reason": "tum_rgbd_route_import_does_not_include_camera_to_base_transform",
            },
            {
                "sensor": "robot_base_pose",
                "status": "unavailable",
                "reason": "tum_rgbd_groundtruth_is_preserved_as_dataset_camera_pose_not_robot_base_pose",
            },
        ],
    }
    write_route_metadata(output, metadata)
    write_segment(output, events, segment_id=output.name, artifact_files=artifact_files)
    return output


def _association_record(
    frame_id: int,
    rgb_ref: str,
    depth_ref: str,
    association: TumAssociation,
    accelerometer: TumAccelerometer | None,
    sequence: str,
) -> JsonDict:
    return {
        "frame_id": frame_id,
        "timestamp_ns": int(round(association.rgb.timestamp * 1_000_000_000)),
        "rgb_timestamp": association.rgb.timestamp,
        "rgb_source_path": association.rgb.path,
        "rgb_ref": rgb_ref,
        "depth_timestamp": association.depth.timestamp,
        "depth_source_path": association.depth.path,
        "depth_ref": depth_ref,
        "groundtruth": association.groundtruth.to_dict(),
        "camera_pose": association.groundtruth.to_dict(),
        "accelerometer": accelerometer.to_dict() if accelerometer is not None else None,
        "sensor_masks": {
            "rgb": True,
            "depth": True,
            "camera_groundtruth_pose": True,
            "accelerometer": accelerometer is not None,
            "imu_6dof": False,
            "wheel_odometry": False,
            "commands": False,
            "camera_to_base": False,
            "robot_base_pose": False,
        },
        "dataset_frame_type": "public_robot_mounted" if "pioneer" in sequence else "public_rgbd_camera_pose_geometry",
        "robot_frame_truth_candidate": False,
        "robot_frame_truth": False,
        "control_safe": False,
    }


def _pose_event(sequence_id: str, association: TumAssociation) -> PoseEvent:
    pose = association.groundtruth
    return PoseEvent(
        timestamp_ns=int(round(pose.timestamp * 1_000_000_000)),
        sequence_id=sequence_id,
        source=TUM_RGBD_IMPORT_SOURCE,
        position_m=(pose.tx, pose.ty, pose.tz),
        orientation_xyzw=(pose.qx, pose.qy, pose.qz, pose.qw),
        frame_id="dataset_world",
        child_frame_id="tum_rgb_camera",
        pose_kind="tum_rgbd_groundtruth_camera_pose",
    )


def _nearest_accelerometer(
    samples: list[TumAccelerometer],
    timestamp: float,
    *,
    max_difference: float,
) -> TumAccelerometer | None:
    best: TumAccelerometer | None = None
    best_diff = max_difference
    for sample in samples:
        diff = abs(sample.timestamp - timestamp)
        if diff <= best_diff:
            best = sample
            best_diff = diff
    return best


def _sequence_name(source: Path) -> str:
    name = source.name
    if name.startswith("rgbd_dataset_"):
        name = name.removeprefix("rgbd_dataset_")
    return name


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import a TUM RGB-D sequence into a HomeBrain route log.")
    parser.add_argument("--source", required=True, help="TUM sequence directory.")
    parser.add_argument("--out", required=True, help="Output HomeBrain route directory.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap.")
    args = parser.parse_args(argv)
    out = tum_rgbd_to_route(source_dir=args.source, out_dir=args.out, max_frames=args.max_frames)
    print(f"imported TUM RGB-D route to {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
