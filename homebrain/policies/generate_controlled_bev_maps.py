from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from homebrain.data.spatial_dataset import (
    DETERMINISTIC_CREATED_AT_UTC,
    SPATIAL_DATASET_SCHEMA_VERSION,
    SPATIAL_EXAMPLE_SCHEMA_VERSION,
    json_sha256,
    scalar_bool,
    scalar_float,
    scalar_int,
    scalar_str,
    write_deterministic_npz,
    write_json,
)
from homebrain.messages.schema import JsonDict, deterministic_json
from homebrain.teachers.artifacts import file_sha256, relative_to_root

CONTROLLED_BEV_SCHEMA_VERSION = "homebrain.controlled_bev_maps.v0"
DEFAULT_SCENARIOS: tuple[str, ...] = (
    "open_room",
    "corridor",
    "narrow_passage",
    "blocked_path",
    "clutter",
    "cul_de_sac",
    "unknown_frontier",
)


def generate_controlled_bev_maps(
    *,
    out_dir: str | Path,
    examples_per_scenario: int = 160,
    grid_size: int = 32,
    meters_per_cell: float = 0.1,
) -> Path:
    if examples_per_scenario <= 0:
        raise ValueError("examples_per_scenario must be positive")
    if grid_size < 16:
        raise ValueError("grid_size must be at least 16")
    if meters_per_cell <= 0.0:
        raise ValueError("meters_per_cell must be positive")

    root = Path(out_dir)
    examples_dir = root / "examples"
    _clear_generated_outputs(root)
    examples_dir.mkdir(parents=True, exist_ok=True)

    camera_config: JsonDict = {
        "grid_size_m": round(float(grid_size * meters_per_cell), 6),
        "meters_per_cell": round(float(meters_per_cell), 6),
        "robot_radius_m": 0.15,
    }
    camera_config_hash = json_sha256(camera_config)
    records: list[JsonDict] = []
    split_counts: dict[str, int] = {"train": 0, "val": 0, "review": 0}
    timestamp_base = 1_800_000_000_000_000_000
    frame_index = 0

    for scenario in DEFAULT_SCENARIOS:
        for variant in range(examples_per_scenario):
            arrays = controlled_bev_arrays(scenario, variant, shape=(grid_size, grid_size))
            split = _split_for_variant(variant)
            split_counts[split] += 1
            timestamp_ns = timestamp_base + frame_index * 100_000_000
            example_path = examples_dir / f"{scenario}_{variant:06d}.npz"
            source_name = f"controlled_bev:{scenario}"
            provenance = {
                "schema_version": CONTROLLED_BEV_SCHEMA_VERSION,
                "source_family": "algorithmic_coverage_planner",
                "source_name": source_name,
                "scenario_name": scenario,
                "variant_index": variant,
                "weak_label": True,
                "control_safe": False,
                "replay_only": True,
                "not_executed": True,
            }
            write_deterministic_npz(
                example_path,
                {
                    "frame_id": scalar_int(frame_index),
                    "timestamp_ns": scalar_int(timestamp_ns),
                    "timestamp": scalar_int(timestamp_ns),
                    "rgb_ref": scalar_str(f"controlled/{scenario}/{variant:06d}.ppm"),
                    "rgb_path": scalar_str(f"controlled/{scenario}/{variant:06d}.ppm"),
                    "bev_free": arrays["free"],
                    "bev_obstacle": arrays["obstacle"],
                    "bev_unknown": arrays["unknown"],
                    "bev_confidence": arrays["confidence"],
                    "provenance": scalar_str(deterministic_json(provenance)),
                    "weak_label": scalar_bool(True),
                    "control_safe": scalar_bool(False),
                    "not_robot_frame_truth": scalar_bool(True),
                    "camera_config_hash": scalar_str(camera_config_hash),
                    "teacher_manifest_hash": scalar_str("controlled_algorithmic_grid_v0"),
                    "split": scalar_str(split),
                    "split_unit_id": scalar_str(source_name),
                    "pose_delta": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                    "pose_delta_mask": scalar_float(0.0),
                    "pose_label_frame": scalar_str("none"),
                    "pose_label_source": scalar_str("none"),
                    "pose_label_target_frame_id": scalar_int(-1),
                    "action_label_mask": scalar_float(0.0),
                    "imu_label_mask": scalar_float(0.0),
                    "wheel_label_mask": scalar_float(0.0),
                    "source_name": scalar_str(source_name),
                    "source_family": scalar_str("controlled_bev"),
                    "scenario_name": scalar_str(scenario),
                    "supervision_grade": scalar_str("controlled_algorithmic_grid_v0"),
                    "review_required": scalar_bool(True),
                    "replay_only": scalar_bool(True),
                    "not_executed": scalar_bool(True),
                },
            )
            records.append(
                {
                    "sequence_id": "controlled_bev_maps_v0",
                    "camera_id": "front_rgb",
                    "frame_id": frame_index,
                    "timestamp_ns": timestamp_ns,
                    "rgb_ref": f"controlled/{scenario}/{variant:06d}.ppm",
                    "rgb_path": f"controlled/{scenario}/{variant:06d}.ppm",
                    "split": split,
                    "split_unit_id": source_name,
                    "weak_label": True,
                    "control_safe": False,
                    "not_robot_frame_truth": True,
                    "source_name": source_name,
                    "source_family": "controlled_bev",
                    "scenario_name": scenario,
                    "supervision_grade": "controlled_algorithmic_grid_v0",
                    "example_path": relative_to_root(example_path, root),
                    "example_sha256": file_sha256(example_path),
                    "camera_config_hash": camera_config_hash,
                    "teacher_manifest_hash": "controlled_algorithmic_grid_v0",
                }
            )
            frame_index += 1

    manifest: JsonDict = {
        "schema_version": SPATIAL_DATASET_SCHEMA_VERSION,
        "controlled_bev_schema_version": CONTROLLED_BEV_SCHEMA_VERSION,
        "example_schema_version": SPATIAL_EXAMPLE_SCHEMA_VERSION,
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "package_type": "SpatialTrainPack",
        "controlled_bev_map_pack": True,
        "quality_review_status": "controlled_algorithmic_grid_requires_review",
        "source_log": "controlled_bev_maps_v0",
        "source_segment_id": "controlled_bev_maps_v0",
        "source_family": "controlled_bev",
        "source_name": "controlled_bev_maps_v0",
        "supervision_grade": "controlled_algorithmic_grid_v0",
        "robot_supervision_grade": "unknown",
        "camera_config": camera_config,
        "camera_config_hash": camera_config_hash,
        "grid_shape": [grid_size, grid_size],
        "weak_label": True,
        "control_safe": False,
        "not_robot_frame_truth": True,
        "replay_only": True,
        "not_executed": True,
        "raw_pwm_emitted": False,
        "control_safety": "not_control_safe_controlled_algorithmic_labels_only",
        "example_count": len(records),
        "examples_per_scenario": examples_per_scenario,
        "scenarios": list(DEFAULT_SCENARIOS),
        "split_counts": split_counts,
        "pose_label_count": 0,
        "pose_label_frame": "none",
        "frames": records,
        "examples": records,
    }
    write_json(root / "manifest.json", manifest, pretty=True)
    return root / "manifest.json"


def controlled_bev_arrays(
    scenario: str,
    variant: int,
    *,
    shape: tuple[int, int] = (32, 32),
) -> dict[str, np.ndarray]:
    if scenario not in DEFAULT_SCENARIOS:
        raise ValueError(f"unknown controlled BEV scenario: {scenario}")
    free = np.zeros(shape, dtype=np.uint8)
    obstacle = np.zeros(shape, dtype=np.uint8)
    unknown = np.ones(shape, dtype=np.uint8)

    if scenario == "open_room":
        _open_room(free, obstacle, variant)
    elif scenario == "corridor":
        _corridor(free, obstacle, variant)
    elif scenario == "narrow_passage":
        _narrow_passage(free, obstacle, variant)
    elif scenario == "blocked_path":
        _blocked_path(free, obstacle, variant)
    elif scenario == "clutter":
        _clutter(free, obstacle, variant)
    elif scenario == "cul_de_sac":
        _cul_de_sac(free, obstacle, variant)
    elif scenario == "unknown_frontier":
        _unknown_frontier(free, obstacle, variant)

    free[obstacle > 0] = 0
    unknown[(free > 0) | (obstacle > 0)] = 0
    confidence = np.where(unknown > 0, 0.05, 0.95).astype(np.float32)
    return {
        "free": free,
        "obstacle": obstacle,
        "unknown": unknown,
        "confidence": confidence,
    }


def _open_room(free: np.ndarray, obstacle: np.ndarray, variant: int) -> None:
    rows, cols = free.shape
    _set_free(free, rows - 24, rows, 4, cols - 4)
    _room_walls(obstacle, rows - 25, rows - 1, 3, cols - 3)
    if variant % 5 == 0:
        col = 6 + (variant * 3) % max(cols - 16, 1)
        _set_obstacle(obstacle, rows - 18, rows - 15, col, min(col + 3, cols - 5))


def _corridor(free: np.ndarray, obstacle: np.ndarray, variant: int) -> None:
    rows, cols = free.shape
    center = cols // 2 + ((variant % 3) - 1)
    half_width = 3
    _set_free(free, 4, rows, center - half_width, center + half_width + 1)
    _set_obstacle(obstacle, 4, rows - 1, center - half_width - 1, center - half_width)
    _set_obstacle(obstacle, 4, rows - 1, center + half_width + 1, center + half_width + 2)


def _narrow_passage(free: np.ndarray, obstacle: np.ndarray, variant: int) -> None:
    rows, cols = free.shape
    center = cols // 2 + ((variant % 3) - 1)
    _set_free(free, rows - 11, rows, center - 6, center + 7)
    _set_free(free, rows - 22, rows - 11, center - 2, center + 3)
    _set_free(free, 6, rows - 22, center - 7, center + 8)
    _set_obstacle(obstacle, rows - 21, rows - 12, center - 6, center - 3)
    _set_obstacle(obstacle, rows - 21, rows - 12, center + 3, center + 7)


def _blocked_path(free: np.ndarray, obstacle: np.ndarray, variant: int) -> None:
    rows, cols = free.shape
    center = cols // 2
    _set_free(free, rows - 12, rows, center - 5, center + 6)
    _set_free(free, rows - 22, rows - 12, center - 4, center + 5)
    barrier_row = rows - 8 - (variant % 3)
    _set_obstacle(obstacle, barrier_row - 1, barrier_row + 2, center - 5, center + 6)


def _clutter(free: np.ndarray, obstacle: np.ndarray, variant: int) -> None:
    rows, cols = free.shape
    _set_free(free, rows - 25, rows, 4, cols - 4)
    _room_walls(obstacle, rows - 26, rows - 1, 3, cols - 3)
    for index in range(5):
        height = 2 + ((variant + index) % 3)
        width = 2 + ((variant + 2 * index) % 4)
        row = rows - 23 + ((variant * 5 + index * 7) % 15)
        col = 5 + ((variant * 7 + index * 5) % max(cols - 12, 1))
        if row > rows - 9 and abs((col + width // 2) - cols // 2) < 5:
            continue
        _set_obstacle(obstacle, row, min(row + height, rows - 4), col, min(col + width, cols - 4))


def _cul_de_sac(free: np.ndarray, obstacle: np.ndarray, variant: int) -> None:
    rows, cols = free.shape
    center = cols // 2 + ((variant % 3) - 1)
    _set_free(free, rows - 18, rows, center - 3, center + 4)
    _set_free(free, rows - 22, rows - 16, center - 7, center + 8)
    _set_obstacle(obstacle, rows - 23, rows - 21, center - 8, center + 9)
    _set_obstacle(obstacle, rows - 22, rows - 15, center - 8, center - 7)
    _set_obstacle(obstacle, rows - 22, rows - 15, center + 8, center + 9)


def _unknown_frontier(free: np.ndarray, obstacle: np.ndarray, variant: int) -> None:
    rows, cols = free.shape
    center = cols // 2 + ((variant % 5) - 2)
    _set_free(free, rows - 9, rows, center - 6, center + 7)
    _set_free(free, rows - 15, rows - 8, center - 3, center + 4)
    if variant % 4 == 0:
        _set_obstacle(obstacle, rows - 12, rows - 10, center + 5, center + 8)


def _room_walls(obstacle: np.ndarray, top: int, bottom: int, left: int, right: int) -> None:
    _set_obstacle(obstacle, top, top + 1, left, right + 1)
    _set_obstacle(obstacle, top, bottom + 1, left, left + 1)
    _set_obstacle(obstacle, top, bottom + 1, right, right + 1)


def _set_free(array: np.ndarray, row0: int, row1: int, col0: int, col1: int) -> None:
    rows, cols = array.shape
    array[max(0, row0) : min(rows, row1), max(0, col0) : min(cols, col1)] = 1


def _set_obstacle(array: np.ndarray, row0: int, row1: int, col0: int, col1: int) -> None:
    rows, cols = array.shape
    array[max(0, row0) : min(rows, row1), max(0, col0) : min(cols, col1)] = 1


def _split_for_variant(variant: int) -> str:
    if variant % 10 == 9:
        return "review"
    if variant % 10 == 8:
        return "val"
    return "train"


def _clear_generated_outputs(root: Path) -> None:
    if (root / "examples").exists():
        shutil.rmtree(root / "examples")
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic controlled indoor-like BEV maps.")
    parser.add_argument("--out", required=True, help="Output controlled BEV SpatialTrainPack directory.")
    parser.add_argument("--examples-per-scenario", type=int, default=160)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--meters-per-cell", type=float, default=0.1)
    args = parser.parse_args(argv)
    manifest = generate_controlled_bev_maps(
        out_dir=args.out,
        examples_per_scenario=args.examples_per_scenario,
        grid_size=args.grid_size,
        meters_per_cell=args.meters_per_cell,
    )
    print(f"wrote controlled BEV map manifest to {manifest.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
