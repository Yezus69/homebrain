from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from homebrain.data.spatial_dataset import (
    SPATIAL_DATASET_SCHEMA_VERSION,
    SPATIAL_EXAMPLE_SCHEMA_VERSION,
    DETERMINISTIC_CREATED_AT_UTC,
    scalar_bool,
    scalar_float,
    scalar_int,
    scalar_str,
    write_deterministic_npz,
    write_json,
)


def write_synthetic_policy_fixture(out_dir: str | Path, *, count: int = 6) -> Path:
    root = Path(out_dir)
    examples_dir = root / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for frame_id in range(count):
        arrays = _fixture_arrays(frame_id)
        example_path = examples_dir / f"front_rgb_{frame_id:06d}.npz"
        write_deterministic_npz(
            example_path,
            {
                "frame_id": scalar_int(frame_id),
                "timestamp_ns": scalar_int(1_700_000_000_000_000_000 + frame_id * 100_000_000),
                "timestamp": scalar_int(1_700_000_000_000_000_000 + frame_id * 100_000_000),
                "rgb_ref": scalar_str(f"frames/frame_{frame_id:06d}.ppm"),
                "rgb_path": scalar_str(f"synthetic/frames/frame_{frame_id:06d}.ppm"),
                "bev_free": arrays["free"],
                "bev_obstacle": arrays["obstacle"],
                "bev_unknown": arrays["unknown"],
                "bev_confidence": arrays["confidence"],
                "provenance": scalar_str("synthetic_policy_fixture"),
                "weak_label": scalar_bool(True),
                "control_safe": scalar_bool(False),
                "not_robot_frame_truth": scalar_bool(True),
                "camera_config_hash": scalar_str("synthetic"),
                "teacher_manifest_hash": scalar_str("synthetic"),
                "split": scalar_str("review"),
                "pose_delta": np.asarray([0.10, 0.0, 0.0], dtype=np.float32),
                "pose_delta_mask": scalar_float(1.0),
                "pose_label_frame": scalar_str("synthetic_fixture_pose"),
                "pose_label_source": scalar_str("synthetic_fixture"),
                "pose_label_target_frame_id": scalar_int(frame_id + 1 if frame_id + 1 < count else -1),
                "action_label_mask": scalar_float(0.0),
                "imu_label_mask": scalar_float(0.0),
                "wheel_label_mask": scalar_float(0.0),
            },
        )
        records.append(
            {
                "sequence_id": "synthetic_policy_fixture",
                "camera_id": "front_rgb",
                "frame_id": frame_id,
                "timestamp_ns": 1_700_000_000_000_000_000 + frame_id * 100_000_000,
                "rgb_ref": f"frames/frame_{frame_id:06d}.ppm",
                "rgb_path": f"synthetic/frames/frame_{frame_id:06d}.ppm",
                "split": "review",
                "weak_label": True,
                "control_safe": False,
                "not_robot_frame_truth": True,
                "example_path": example_path.relative_to(root).as_posix(),
                "pose_delta_mask": 1.0,
                "pose_label_frame": "synthetic_fixture_pose",
                "pose_label_source": "synthetic_fixture",
                "pose_label_target_frame_id": frame_id + 1 if frame_id + 1 < count else -1,
            }
        )
    manifest = {
        "schema_version": SPATIAL_DATASET_SCHEMA_VERSION,
        "example_schema_version": SPATIAL_EXAMPLE_SCHEMA_VERSION,
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "package_type": "SpatialTrainPack",
        "quality_review_status": "synthetic_policy_fixture",
        "source_log": "synthetic_policy_fixture",
        "source_segment_id": "synthetic_policy_fixture",
        "robot_supervision_grade": "unknown",
        "camera_config": {
            "grid_size_m": 2.4,
            "meters_per_cell": 0.1,
            "robot_radius_m": 0.15,
        },
        "grid_shape": [24, 24],
        "weak_label": True,
        "control_safe": False,
        "not_robot_frame_truth": True,
        "control_safety": "not_control_safe_synthetic_policy_fixture",
        "example_count": len(records),
        "split_counts": {"train": 0, "val": 0, "review": len(records)},
        "pose_label_count": len(records),
        "pose_label_frame": "synthetic_fixture_pose",
        "frames": records,
        "examples": records,
    }
    write_json(root / "manifest.json", manifest, pretty=True)
    return root / "manifest.json"


def _fixture_arrays(frame_id: int) -> dict[str, np.ndarray]:
    shape = (24, 24)
    free = np.zeros(shape, dtype=np.uint8)
    obstacle = np.zeros(shape, dtype=np.uint8)
    unknown = np.ones(shape, dtype=np.uint8)
    center_col = shape[1] // 2
    free[shape[0] - 8 : shape[0], center_col - 2 : center_col + 3] = 1
    unknown[free > 0] = 0
    if frame_id % 3 == 2:
        obstacle[shape[0] - 5 : shape[0] - 2, center_col - 2 : center_col + 3] = 1
        free[obstacle > 0] = 0
        unknown[obstacle > 0] = 0
    confidence = np.where(unknown > 0, 0.0, 0.9).astype(np.float32)
    return {
        "free": free,
        "obstacle": obstacle,
        "unknown": unknown,
        "confidence": confidence,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a deterministic synthetic policy SpatialTrainPack.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args(argv)
    manifest = write_synthetic_policy_fixture(args.out, count=args.count)
    print(f"wrote synthetic policy fixture manifest to {manifest.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

