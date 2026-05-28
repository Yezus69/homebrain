from __future__ import annotations

from pathlib import Path

import numpy as np

from homebrain.data.spatial_dataset import DETERMINISTIC_CREATED_AT_UTC, write_deterministic_npz, write_json
from homebrain.policies.candidate_trajectories import generate_default_candidates


def write_real_rgbd_source_pack_fixture(
    root: Path,
    *,
    grid_shape: tuple[int, int] = (16, 16),
    meters_per_cell: float = 0.1,
    frame_count: int = 6,
    synthetic_or_fixture: bool = True,
    train_val_overlap: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidates = generate_default_candidates(grid_shape=grid_shape, meters_per_cell=meters_per_cell, robot_radius_m=0.05)
    route_ids = ["route_train_a", "route_train_b", "route_val"]
    examples = []
    for route_id in route_ids:
        split = "val" if route_id == "route_val" else "train"
        for frame_index in range(frame_count):
            arrays = _source_arrays(
                route_id=route_id,
                frame_index=frame_index,
                grid_shape=grid_shape,
                candidates=candidates,
            )
            rel = Path("examples") / split / route_id / f"{frame_index:08d}.npz"
            write_deterministic_npz(root / rel, arrays)
            examples.append(
                {
                    "example_path": rel.as_posix(),
                    "route_id": route_id,
                    "split": split,
                    "timestamp_ns": int(frame_index * 500_000_000),
                    "dynamic_positive": bool(np.count_nonzero(arrays["target_dynamic_residual_risk"] > 0.5) > 0),
                }
            )
    train_ids = ["route_train_a", "route_train_b"]
    val_ids = ["route_train_a"] if train_val_overlap else ["route_val"]
    manifest = {
        "schema_version": "homebrain.real_rgbd_route_bev_pack.v0",
        "package_type": "RealRGBDRouteBEVPack",
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "synthetic_or_fixture": synthetic_or_fixture,
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "meters_per_cell": float(meters_per_cell),
        "real_source_route_count": len(route_ids),
        "real_source_frame_count": len(examples),
        "train_route_ids": train_ids,
        "heldout_route_ids": val_ids,
        "route_held_out_split_basis": "route_id",
        "control_safe": False,
        "replay_only": True,
        "not_executed": True,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
        "examples": examples,
    }
    write_json(root / "manifest.json", manifest, pretty=True)
    return root / "manifest.json"


def _source_arrays(
    *,
    route_id: str,
    frame_index: int,
    grid_shape: tuple[int, int],
    candidates: list[object],
) -> dict[str, np.ndarray]:
    height, width = grid_shape
    free = np.zeros(grid_shape, dtype=np.float32)
    occupied = np.zeros(grid_shape, dtype=np.float32)
    unknown = np.ones(grid_shape, dtype=np.float32)
    center = width // 2
    free[height - 8 : height, center - 2 : center + 2] = 1.0
    unknown[height - 8 : height, center - 2 : center + 2] = 0.0
    risky = occupied.copy()
    dynamic = np.zeros(grid_shape, dtype=np.float32)
    if frame_index % 3 == 1:
        straight = next(candidate for candidate in candidates if getattr(candidate, "id") == "straight_short")
        stop = next(candidate for candidate in candidates if getattr(candidate, "id") == "stop")
        stop_cells = set(getattr(stop, "footprint_cells"))
        for row, col in getattr(straight, "footprint_cells"):
            if (row, col) in stop_cells:
                continue
            if 0 <= row < height and 0 <= col < width:
                occupied[row, col] = 1.0
                risky[row, col] = 1.0
                dynamic[row, col] = 1.0
                free[row, col] = 0.0
                unknown[row, col] = 0.0
    return {
        "schema_version": np.asarray("homebrain.real_rgbd_route_bev_example.v0"),
        "route_id": np.asarray(route_id),
        "split": np.asarray("val" if route_id == "route_val" else "train"),
        "frame_index": np.asarray(frame_index, dtype=np.int64),
        "timestamp_ns": np.asarray(frame_index * 500_000_000, dtype=np.int64),
        "pose_delta_prev": np.asarray([0.02, 0.0, 0.0], dtype=np.float32),
        "sensor_mask": np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        "previous_action": np.asarray([0.15, 0.0], dtype=np.float32),
        "target_current_bev_free": free,
        "target_current_bev_occupied": occupied,
        "target_current_bev_unknown": unknown,
        "target_current_bev_traversable": free.copy(),
        "target_current_bev_risky": risky,
        "target_uncertainty_map": np.where(unknown > 0.5, 0.8, 0.1).astype(np.float32),
        "target_dynamic_residual_risk": dynamic,
        "replay_only": np.asarray([True], dtype=np.bool_),
        "not_executed": np.asarray([True], dtype=np.bool_),
        "control_safe": np.asarray([False], dtype=np.bool_),
        "raw_pwm_emitted": np.asarray([False], dtype=np.bool_),
        "hardware_validated": np.asarray([False], dtype=np.bool_),
    }
