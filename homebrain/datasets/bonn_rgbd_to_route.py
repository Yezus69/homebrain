from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from homebrain.data.spatial_dataset import write_json
from homebrain.datasets.bonn_rgbd import (
    BONN_RGBD_DATASET_NAME,
    BONN_RGBD_DEPTH_SCALE,
    BONN_RGBD_LICENSE_NAME,
    BONN_RGBD_LICENSE_REVIEW_STATUS,
    BONN_RGBD_ROUTE_ASSOCIATIONS_FILE,
    BONN_RGBD_SCHEMA_VERSION,
    BONN_RGBD_SOURCE_URL,
    load_bonn_rgbd_sequence,
    missing_bonn_report,
)
from homebrain.ingest.metadata import ROUTE_SOURCE_METADATA_FILE, ROUTE_SOURCE_SCHEMA_VERSION, write_route_metadata
from homebrain.messages.schema import Event, FrameEvent, JsonDict, PoseEvent
from homebrain.replay.segment_log import write_segment

BONN_RGBD_IMPORT_SOURCE = "bonn_rgbd_dynamic_import"


def bonn_rgbd_to_route(
    *,
    source_dir: str | Path,
    out_dir: str | Path,
    max_frames: int | None = None,
    synthetic_or_fixture: bool = False,
) -> Path:
    source = Path(source_dir)
    output = Path(out_dir)
    if not source.exists():
        missing_bonn_report(
            output,
            source_root=source,
            reason="missing_bonn_rgbd_root",
            synthetic_or_fixture=synthetic_or_fixture,
        )
        return output
    if max_frames is not None and max_frames < 1:
        raise ValueError("max-frames must be at least 1 when supplied")

    sequence = load_bonn_rgbd_sequence(source, route_id=output.name)
    frames = list(sequence.frames[:max_frames])
    if not frames:
        missing_bonn_report(
            output,
            source_root=source,
            reason="missing_bonn_rgbd_frames",
            synthetic_or_fixture=synthetic_or_fixture,
        )
        return output

    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir(parents=True, exist_ok=True)
    (output / "depth").mkdir(parents=True, exist_ok=True)
    events: list[Event] = []
    records: list[JsonDict] = []
    artifact_files = [ROUTE_SOURCE_METADATA_FILE, BONN_RGBD_ROUTE_ASSOCIATIONS_FILE]
    for frame in frames:
        rgb_ref = f"frames/frame_{frame.frame_index:06d}{frame.rgb_path.suffix.lower()}"
        shutil.copy2(frame.rgb_path, output / rgb_ref)
        artifact_files.append(rgb_ref)
        depth_ref = None
        if frame.depth_path is not None:
            depth_ref = f"depth/depth_{frame.frame_index:06d}{frame.depth_path.suffix.lower()}"
            shutil.copy2(frame.depth_path, output / depth_ref)
            artifact_files.append(depth_ref)
        events.append(
            FrameEvent(
                timestamp_ns=frame.timestamp_ns,
                sequence_id=output.name,
                source=BONN_RGBD_IMPORT_SOURCE,
                camera_id="bonn_rgb",
                frame_id=frame.frame_index,
                width=frame.width,
                height=frame.height,
                format=_frame_format(frame.rgb_path.suffix),
                data_ref=rgb_ref,
                intrinsics={
                    "available": True,
                    "status": "provided_by_route_or_bonn_default",
                    **frame.intrinsics,
                },
            )
        )
        if frame.pose is not None:
            events.append(_pose_event(output.name, frame.pose, frame.timestamp_ns))
        records.append(_association_record(frame, rgb_ref, depth_ref, synthetic_or_fixture=synthetic_or_fixture))

    has_depth = any(record.get("depth_ref") for record in records)
    has_pose = any(record.get("groundtruth") for record in records)
    missing_sensor_notices = _missing_sensor_notices(has_depth=has_depth, has_pose=has_pose)
    associations: JsonDict = {
        "schema_version": BONN_RGBD_SCHEMA_VERSION,
        "source_type": "bonn_rgbd_dynamic_associations",
        "source_sequence": source.name,
        "source_dir": source.resolve().as_posix(),
        "official_page": BONN_RGBD_SOURCE_URL,
        "license_name": BONN_RGBD_LICENSE_NAME,
        "license_review_status": BONN_RGBD_LICENSE_REVIEW_STATUS,
        "depth_scale": BONN_RGBD_DEPTH_SCALE,
        "dataset_frame_type": "public_rgbd_camera_pose_dynamic_people",
        "sensor_masks": {
            "rgb": True,
            "depth": has_depth,
            "camera_groundtruth_pose": has_pose,
            "dynamic_people": True,
            "imu_6dof": False,
            "wheel_odometry": False,
            "commands": False,
            "camera_to_base": False,
            "robot_base_pose": False,
        },
        "missing_sensor_notices": missing_sensor_notices,
        "synthetic_or_fixture": bool(synthetic_or_fixture),
        "accepted": False if synthetic_or_fixture else bool(has_depth and has_pose),
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
        "frames": records,
    }
    write_json(output / BONN_RGBD_ROUTE_ASSOCIATIONS_FILE, associations, pretty=True)
    metadata: JsonDict = {
        "schema_version": ROUTE_SOURCE_SCHEMA_VERSION,
        "source_type": BONN_RGBD_DATASET_NAME,
        "source_path": source.resolve().as_posix(),
        "official_page": BONN_RGBD_SOURCE_URL,
        "sequence": source.name,
        "frame_count": len(records),
        "imported_frame_count": len(records),
        "has_rgb": True,
        "has_depth": has_depth,
        "has_groundtruth_pose": has_pose,
        "has_imu": False,
        "has_wheel_odometry": False,
        "has_commands": False,
        "has_camera_to_base_transform": False,
        "has_robot_base_pose": False,
        "dataset_frame_type": "public_rgbd_camera_pose_dynamic_people",
        "calibration_class": "dataset_metric_rgbd_pose",
        "intrinsics_source": frames[0].intrinsics.get("source", "bonn_rgbd"),
        "extrinsics_source": "missing_camera_to_base",
        "pose_source": "bonn_rgbd_groundtruth_camera_pose",
        "scale_source": "dataset_metric_depth",
        "depth_scale": BONN_RGBD_DEPTH_SCALE,
        "associations_file": BONN_RGBD_ROUTE_ASSOCIATIONS_FILE,
        "license_name": BONN_RGBD_LICENSE_NAME,
        "license_review_status": BONN_RGBD_LICENSE_REVIEW_STATUS,
        "synthetic_or_fixture": bool(synthetic_or_fixture),
        "accepted": False if synthetic_or_fixture else bool(has_depth and has_pose),
        "product_training_approved": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
        "missing_sensor_notices": missing_sensor_notices,
    }
    write_route_metadata(output, metadata)
    write_segment(output, events, segment_id=output.name, artifact_files=artifact_files)
    return output


def _association_record(frame: object, rgb_ref: str, depth_ref: str | None, *, synthetic_or_fixture: bool) -> JsonDict:
    return {
        "frame_id": int(frame.frame_index),
        "timestamp_ns": int(frame.timestamp_ns),
        "rgb_source_path": frame.rgb_path.as_posix(),
        "rgb_ref": rgb_ref,
        "depth_source_path": frame.depth_path.as_posix() if frame.depth_path is not None else None,
        "depth_ref": depth_ref,
        "groundtruth": frame.pose,
        "dataset_frame_type": "public_rgbd_camera_pose_dynamic_people",
        "synthetic_or_fixture": bool(synthetic_or_fixture),
        "robot_frame_truth_candidate": False,
        "robot_frame_truth": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
    }


def _pose_event(sequence_id: str, pose: JsonDict, timestamp_ns: int) -> PoseEvent:
    return PoseEvent(
        timestamp_ns=timestamp_ns,
        sequence_id=sequence_id,
        source=BONN_RGBD_IMPORT_SOURCE,
        position_m=(float(pose["tx"]), float(pose["ty"]), float(pose["tz"])),
        orientation_xyzw=(float(pose["qx"]), float(pose["qy"]), float(pose["qz"]), float(pose["qw"])),
        frame_id="dataset_world",
        child_frame_id="bonn_rgb_camera",
        pose_kind="bonn_rgbd_groundtruth_camera_pose",
    )


def _missing_sensor_notices(*, has_depth: bool, has_pose: bool) -> list[JsonDict]:
    notices: list[JsonDict] = [
        {"sensor": "imu", "status": "unavailable", "reason": "bonn_rgbd_dynamic_has_no_synchronized_imu_stream"},
        {
            "sensor": "wheel_odometry",
            "status": "unavailable",
            "reason": "bonn_rgbd_dynamic_has_camera_trajectory_not_robot_wheel_odometry",
        },
        {"sensor": "commands", "status": "unavailable", "reason": "bonn_rgbd_dynamic_has_no_command_stream"},
        {
            "sensor": "camera_to_base",
            "status": "unavailable",
            "reason": "bonn_rgbd_dynamic_is_not_a_robot_base_frame_dataset",
        },
    ]
    if not has_depth:
        notices.append({"sensor": "depth", "status": "unavailable", "reason": "no_depth_frames_associated"})
    if not has_pose:
        notices.append({"sensor": "groundtruth_pose", "status": "unavailable", "reason": "no_groundtruth_rows_associated"})
    return notices


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import a Bonn RGB-D Dynamic sequence into a HomeBrain route log.")
    parser.add_argument("--source", required=True, help="Bonn RGB-D Dynamic sequence directory.")
    parser.add_argument("--out", required=True, help="Output HomeBrain route directory.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--synthetic-or-fixture", action="store_true")
    args = parser.parse_args(argv)
    out = bonn_rgbd_to_route(
        source_dir=args.source,
        out_dir=args.out,
        max_frames=args.max_frames,
        synthetic_or_fixture=args.synthetic_or_fixture,
    )
    print(f"processed Bonn RGB-D route at {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
