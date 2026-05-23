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
    cells = config.grid_cell_count
    grid_shape = (cells, cells)
    floor_counts = np.zeros(grid_shape, dtype=np.uint16)
    obstacle_counts = np.zeros(grid_shape, dtype=np.uint16)
    point_counts = np.zeros(grid_shape, dtype=np.uint16)
    confidence_sum = np.zeros(grid_shape, dtype=np.float32)
    height_max = np.full(grid_shape, -np.inf, dtype=np.float32)

    half_width = np.float32(config.grid_size_m / 2.0)
    in_grid = (
        (points.forward_m >= np.float32(0.0))
        & (points.forward_m < np.float32(config.grid_size_m))
        & (points.left_m >= -half_width)
        & (points.left_m < half_width)
    )
    if np.any(in_grid):
        forward = points.forward_m[in_grid]
        left = points.left_m[in_grid]
        height = points.height_m[in_grid]
        confidence = points.confidence[in_grid]

        row = cells - 1 - np.floor(forward / np.float32(config.meters_per_cell)).astype(np.int64)
        col = np.floor((left + half_width) / np.float32(config.meters_per_cell)).astype(np.int64)
        row = np.clip(row, 0, cells - 1)
        col = np.clip(col, 0, cells - 1)

        floor_point = np.abs(height) <= np.float32(config.floor_height_tol_m)
        obstacle_point = height >= np.float32(config.obstacle_height_min_m)

        np.add.at(point_counts, (row, col), 1)
        np.add.at(confidence_sum, (row, col), confidence)
        np.maximum.at(height_max, (row, col), height)
        if np.any(floor_point):
            np.add.at(floor_counts, (row[floor_point], col[floor_point]), 1)
        if np.any(obstacle_point):
            np.add.at(obstacle_counts, (row[obstacle_point], col[obstacle_point]), 1)

    floor_candidate = floor_counts > 0
    raw_obstacle = obstacle_counts > 0
    obstacle = _dilate_mask(raw_obstacle, config.obstacle_dilation_cells)
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
            "sampled_pixel_count": points.sampled_pixel_count,
            "valid_depth_point_count": points.valid_point_count,
            "points_in_grid_count": int(np.count_nonzero(in_grid)),
            "observed_cell_count": int(np.count_nonzero(observed)),
            "raw_obstacle_cell_count": int(np.count_nonzero(raw_obstacle)),
            "obstacle_dilation_cells": config.obstacle_dilation_cells,
        }
    )

    return BevProjection(
        bev_free=arrays["bev_free"],
        bev_obstacle=arrays["bev_obstacle"],
        bev_unknown=arrays["bev_unknown"],
        bev_floor_candidate=arrays["bev_floor_candidate"],
        bev_height=arrays["bev_height"],
        bev_confidence=arrays["bev_confidence"],
        projected=points,
        raw_obstacle_count=int(np.count_nonzero(raw_obstacle)),
        stats=stats,
        assumptions=points.assumptions,
        warnings=points.warnings,
    )


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

