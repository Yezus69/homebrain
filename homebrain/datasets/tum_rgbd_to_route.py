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
    load_associations,
    read_png_shape,
    sequence_intrinsics,
    validate_sequence_dir,
)
from homebrain.ingest.image_sequence import read_image_size
from homebrain.ingest.metadata import ROUTE_SOURCE_METADATA_FILE, ROUTE_SOURCE_SCHEMA_VERSION, write_route_metadata
from homebrain.messages.schema import Event, FrameEvent, JsonDict, deterministic_json
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
        artifact_files.extend([rgb_ref, depth_ref])
        association_records.append(_association_record(frame_id, rgb_ref, depth_ref, association))

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
        "has_imu": False,
        "has_wheel_odometry": False,
        "has_commands": False,
        "calibration_class": "dataset_metric_rgbd_pose",
        "intrinsics_source": "tum_rgbd_calibration",
        "extrinsics_source": "tum_rgbd_groundtruth",
        "pose_source": "tum_rgbd_groundtruth",
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
                "reason": "tum_rgbd_route_import_does_not_include_imu_stream",
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
        ],
    }
    write_route_metadata(output, metadata)
    write_segment(output, events, segment_id=output.name, artifact_files=artifact_files)
    return output


def _association_record(frame_id: int, rgb_ref: str, depth_ref: str, association: TumAssociation) -> JsonDict:
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
    }


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

