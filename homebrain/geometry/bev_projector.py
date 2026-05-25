from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from homebrain.geometry.camera_config import CameraConfig
from homebrain.geometry.depth_to_points import ProjectedPoints, project_depth_to_robot_points
from homebrain.messages.schema import JsonDict


BEV_SCHEMA_VERSION = "homebrain.geometry.bev.v0"
BEV_MANIFEST_FILE = "bev_manifest.json"
BEV_ARTIFACT_KINDS: tuple[str, ...] = (
    "bev_free",
    "bev_obstacle",
    "bev_unknown",
    "bev_floor_candidate",
    "bev_height",
    "bev_confidence",
)
BEV_MASK_KINDS = {
    "bev_free",
    "bev_obstacle",
    "bev_unknown",
    "bev_floor_candidate",
}


@dataclass(frozen=True)
class BevProjection:
    bev_free: np.ndarray
    bev_obstacle: np.ndarray
    bev_unknown: np.ndarray
    bev_floor_candidate: np.ndarray
    bev_height: np.ndarray
    bev_confidence: np.ndarray
    projected: ProjectedPoints
    raw_obstacle_count: int
    stats: JsonDict
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "bev_free": self.bev_free,
            "bev_obstacle": self.bev_obstacle,
            "bev_unknown": self.bev_unknown,
            "bev_floor_candidate": self.bev_floor_candidate,
            "bev_height": self.bev_height,
            "bev_confidence": self.bev_confidence,
        }


def build_bev_from_depth(
    depth_m: np.ndarray,
    focal_length_px: float,
    config: CameraConfig,
    *,
    depth_confidence: np.ndarray | None = None,
    principal_point_px: tuple[float, float] | None = None,
    pixel_stride: int = 1,
) -> BevProjection:
    points = project_depth_to_robot_points(
        depth_m,
        focal_length_px,
        config,
        confidence=depth_confidence,
        principal_point_px=principal_point_px,
        pixel_stride=pixel_stride,
    )
    return points_to_bev(points, config)


def points_to_bev(points: ProjectedPoints, config: CameraConfig) -> BevProjection:
    arrays, stats, raw_obstacle_count = robot_points_to_bev_arrays(
        forward_m=points.forward_m,
        left_m=points.left_m,
        height_m=points.height_m,
        confidence=points.confidence,
        grid_cells=config.grid_cell_count,
        meters_per_cell=config.meters_per_cell,
        grid_size_m=config.grid_size_m,
        floor_height_tol_m=config.floor_height_tol_m,
        obstacle_height_min_m=config.obstacle_height_min_m,
        obstacle_dilation_cells=config.obstacle_dilation_cells,
        sampled_pixel_count=points.sampled_pixel_count,
        valid_point_count=points.valid_point_count,
    )

    return BevProjection(
        bev_free=arrays["bev_free"],
        bev_obstacle=arrays["bev_obstacle"],
        bev_unknown=arrays["bev_unknown"],
        bev_floor_candidate=arrays["bev_floor_candidate"],
        bev_height=arrays["bev_height"],
        bev_confidence=arrays["bev_confidence"],
        projected=points,
        raw_obstacle_count=raw_obstacle_count,
        stats=stats,
        assumptions=points.assumptions,
        warnings=points.warnings,
    )


def robot_points_to_bev_arrays(
    *,
    forward_m: np.ndarray,
    left_m: np.ndarray,
    height_m: np.ndarray,
    confidence: np.ndarray,
    grid_cells: int,
    meters_per_cell: float,
    grid_size_m: float | None = None,
    floor_height_tol_m: float,
    obstacle_height_min_m: float,
    obstacle_dilation_cells: int,
    sampled_pixel_count: int | None = None,
    valid_point_count: int | None = None,
) -> tuple[dict[str, np.ndarray], JsonDict, int]:
    """Bin robot-frame points into the standard HomeBrain local BEV arrays."""
    cells = int(grid_cells)
    if cells <= 0:
        raise ValueError(f"grid_cells must be positive, got {grid_cells}")
    meters = np.float32(meters_per_cell)
    if not np.isfinite(meters) or float(meters) <= 0.0:
        raise ValueError(f"meters_per_cell must be positive and finite, got {meters_per_cell!r}")
    grid_size = np.float32(grid_size_m if grid_size_m is not None else cells * float(meters))
    if not np.isfinite(grid_size) or float(grid_size) <= 0.0:
        raise ValueError(f"grid_size_m must be positive and finite, got {grid_size_m!r}")

    forward = np.asarray(forward_m, dtype=np.float32).reshape(-1)
    left = np.asarray(left_m, dtype=np.float32).reshape(-1)
    height = np.asarray(height_m, dtype=np.float32).reshape(-1)
    point_confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if not (forward.shape == left.shape == height.shape == point_confidence.shape):
        raise ValueError(
            "forward_m, left_m, height_m, and confidence must have matching flat shapes; "
            f"got {forward.shape}, {left.shape}, {height.shape}, {point_confidence.shape}"
        )

    grid_shape = (cells, cells)
    floor_counts = np.zeros(grid_shape, dtype=np.uint16)
    obstacle_counts = np.zeros(grid_shape, dtype=np.uint16)
    point_counts = np.zeros(grid_shape, dtype=np.uint16)
    confidence_sum = np.zeros(grid_shape, dtype=np.float32)
    height_max = np.full(grid_shape, -np.inf, dtype=np.float32)

    finite = np.isfinite(forward) & np.isfinite(left) & np.isfinite(height) & np.isfinite(point_confidence)
    half_width = np.float32(grid_size / np.float32(2.0))
    in_grid = (
        finite
        & (forward >= np.float32(0.0))
        & (forward < grid_size)
        & (left >= -half_width)
        & (left < half_width)
    )
    if np.any(in_grid):
        forward_grid = forward[in_grid]
        left_grid = left[in_grid]
        height_grid = height[in_grid]
        confidence_grid = np.clip(point_confidence[in_grid], np.float32(0.0), np.float32(1.0))

        row = cells - 1 - np.floor(forward_grid / meters).astype(np.int64)
        col = np.floor((left_grid + half_width) / meters).astype(np.int64)
        row = np.clip(row, 0, cells - 1)
        col = np.clip(col, 0, cells - 1)

        floor_point = np.abs(height_grid) <= np.float32(floor_height_tol_m)
        obstacle_point = height_grid >= np.float32(obstacle_height_min_m)

        np.add.at(point_counts, (row, col), 1)
        np.add.at(confidence_sum, (row, col), confidence_grid)
        np.maximum.at(height_max, (row, col), height_grid)
        if np.any(floor_point):
            np.add.at(floor_counts, (row[floor_point], col[floor_point]), 1)
        if np.any(obstacle_point):
            np.add.at(obstacle_counts, (row[obstacle_point], col[obstacle_point]), 1)

    floor_candidate = floor_counts > 0
    raw_obstacle = obstacle_counts > 0
    obstacle = _dilate_mask(raw_obstacle, int(obstacle_dilation_cells))
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
    stats = bev_array_stats(arrays)
    stats.update(
        {
            "sampled_pixel_count": int(sampled_pixel_count if sampled_pixel_count is not None else forward.size),
            "valid_depth_point_count": int(valid_point_count if valid_point_count is not None else np.count_nonzero(finite)),
            "points_in_grid_count": int(np.count_nonzero(in_grid)),
            "observed_cell_count": int(np.count_nonzero(observed)),
            "raw_obstacle_cell_count": int(np.count_nonzero(raw_obstacle)),
            "obstacle_dilation_cells": int(obstacle_dilation_cells),
        }
    )

    return arrays, stats, int(np.count_nonzero(raw_obstacle))


def bev_array_stats(arrays: dict[str, np.ndarray]) -> JsonDict:
    free = arrays["bev_free"] > 0
    obstacle = arrays["bev_obstacle"] > 0
    unknown = arrays["bev_unknown"] > 0
    floor = arrays["bev_floor_candidate"] > 0
    confidence = arrays["bev_confidence"].astype(np.float32)
    total = max(int(free.size), 1)
    return {
        "free_ratio": float(np.count_nonzero(free) / total),
        "obstacle_ratio": float(np.count_nonzero(obstacle) / total),
        "unknown_ratio": float(np.count_nonzero(unknown) / total),
        "floor_candidate_ratio": float(np.count_nonzero(floor) / total),
        "confidence_mean": float(np.mean(confidence)) if confidence.size else 0.0,
        "height_min_m": float(np.min(arrays["bev_height"])) if arrays["bev_height"].size else 0.0,
        "height_max_m": float(np.max(arrays["bev_height"])) if arrays["bev_height"].size else 0.0,
    }


def _dilate_mask(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0 or not np.any(mask):
        return mask.copy()
    height, width = mask.shape
    dilated = mask.copy()
    offsets: list[tuple[int, int]] = []
    radius_sq = radius_cells * radius_cells
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy <= radius_sq:
                offsets.append((dy, dx))
    source_rows, source_cols = np.nonzero(mask)
    for dy, dx in offsets:
        rows = source_rows + dy
        cols = source_cols + dx
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        dilated[rows[valid], cols[valid]] = True
    return dilated
