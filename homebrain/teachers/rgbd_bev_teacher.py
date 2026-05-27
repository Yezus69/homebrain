from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np

from homebrain.data.tum_rgbd_route import RouteFrame
from homebrain.datasets.tum_rgbd import read_depth_png_m
from homebrain.messages.schema import JsonDict

RGBD_BEV_TEACHER_SCHEMA_VERSION = "homebrain.rgbd_bev_teacher.v0"
BEV_LABEL_NAMES: tuple[str, ...] = (
    "free",
    "occupied",
    "unknown",
    "traversable",
    "risky",
)


@dataclass(frozen=True)
class RGBDBEVTeacherConfig:
    grid_shape: tuple[int, int] = (64, 64)
    meters_per_cell: float = 0.05
    camera_height_m: float = 0.35
    floor_height_tolerance_m: float = 0.08
    obstacle_height_min_m: float = 0.10
    camera_forward_offset_m: float = 0.0
    camera_left_offset_m: float = 0.0
    camera_yaw_rad: float = 0.0
    max_depth_m: float = 6.0
    pixel_stride: int = 4
    dynamic_residual_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.grid_shape[0] <= 0 or self.grid_shape[1] <= 0:
            raise ValueError("grid_shape must be positive")
        if self.meters_per_cell <= 0.0:
            raise ValueError("meters_per_cell must be positive")
        if self.pixel_stride <= 0:
            raise ValueError("pixel_stride must be positive")

    def to_dict(self) -> JsonDict:
        return {
            "grid_shape": [int(self.grid_shape[0]), int(self.grid_shape[1])],
            "meters_per_cell": float(self.meters_per_cell),
            "camera_height_m": float(self.camera_height_m),
            "floor_height_tolerance_m": float(self.floor_height_tolerance_m),
            "obstacle_height_min_m": float(self.obstacle_height_min_m),
            "camera_forward_offset_m": float(self.camera_forward_offset_m),
            "camera_left_offset_m": float(self.camera_left_offset_m),
            "camera_yaw_rad": float(self.camera_yaw_rad),
            "max_depth_m": float(self.max_depth_m),
            "pixel_stride": int(self.pixel_stride),
            "dynamic_residual_threshold": float(self.dynamic_residual_threshold),
        }


@dataclass(frozen=True)
class RGBDBEVLabels:
    current_bev_free: np.ndarray
    current_bev_occupied: np.ndarray
    current_bev_unknown: np.ndarray
    current_bev_traversable: np.ndarray
    current_bev_risky: np.ndarray
    uncertainty_map: np.ndarray
    dynamic_residual_risk: np.ndarray
    metadata: JsonDict

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "target_current_bev_free": self.current_bev_free.astype(np.float32),
            "target_current_bev_occupied": self.current_bev_occupied.astype(np.float32),
            "target_current_bev_unknown": self.current_bev_unknown.astype(np.float32),
            "target_current_bev_traversable": self.current_bev_traversable.astype(np.float32),
            "target_current_bev_risky": self.current_bev_risky.astype(np.float32),
            "target_uncertainty_map": self.uncertainty_map.astype(np.float32),
            "target_dynamic_residual_risk": self.dynamic_residual_risk.astype(np.float32),
        }


def load_depth_for_route_frame(frame: RouteFrame) -> np.ndarray | None:
    if frame.depth_path is None or not frame.depth_path.exists():
        return None
    scale = float(frame.intrinsics.get("depth_scale", 5000.0))
    return read_depth_png_m(frame.depth_path, scale=scale)


def build_rgbd_bev_labels(
    *,
    depth_m: np.ndarray,
    intrinsics: JsonDict,
    config: RGBDBEVTeacherConfig,
    previous_labels: RGBDBEVLabels | None = None,
    pose_delta_prev: tuple[float, float, float] | None = None,
) -> RGBDBEVLabels:
    free, occupied, observed, stats = _project_depth(depth_m, intrinsics, config)
    occupied = _dilate(occupied, radius=1)
    free = free & ~occupied
    unknown = ~(free | occupied | observed)
    unknown = unknown | (~observed & ~free & ~occupied)
    dynamic = np.zeros(config.grid_shape, dtype=np.float32)
    if previous_labels is not None and pose_delta_prev is not None:
        prev_occ = warp_bev_by_pose_delta(
            previous_labels.current_bev_occupied,
            pose_delta_prev,
            meters_per_cell=config.meters_per_cell,
        )
        prev_free = warp_bev_by_pose_delta(
            previous_labels.current_bev_free,
            pose_delta_prev,
            meters_per_cell=config.meters_per_cell,
        )
        appeared = (occupied.astype(np.float32) - prev_occ) >= np.float32(config.dynamic_residual_threshold)
        disappeared = (prev_occ - occupied.astype(np.float32) >= np.float32(config.dynamic_residual_threshold)) & free
        free_to_occ = (prev_free >= np.float32(0.5)) & occupied
        dynamic = (appeared | disappeared | free_to_occ).astype(np.float32)
        dynamic = _dilate(dynamic > 0.0, radius=1).astype(np.float32)

    traversable = free.astype(np.float32)
    risky = np.maximum(occupied.astype(np.float32), dynamic).astype(np.float32)
    unknown_f = unknown.astype(np.float32)
    observed_f = np.clip(free.astype(np.float32) + occupied.astype(np.float32), 0.0, 1.0)
    uncertainty = np.clip(0.85 * unknown_f + 0.25 * (1.0 - observed_f), 0.0, 1.0).astype(np.float32)
    metadata: JsonDict = {
        "schema_version": RGBD_BEV_TEACHER_SCHEMA_VERSION,
        "weak_label": True,
        "product_training_approved": False,
        "teacher_runtime_dependency": False,
        "uses_depth": True,
        "uses_dataset_pose_offline_for_dynamic_residual": bool(previous_labels is not None and pose_delta_prev is not None),
        "config": config.to_dict(),
        "stats": stats,
    }
    return RGBDBEVLabels(
        current_bev_free=free.astype(np.float32),
        current_bev_occupied=occupied.astype(np.float32),
        current_bev_unknown=unknown_f,
        current_bev_traversable=traversable,
        current_bev_risky=risky,
        uncertainty_map=uncertainty,
        dynamic_residual_risk=dynamic.astype(np.float32),
        metadata=metadata,
    )


def warp_bev_by_pose_delta(
    bev: np.ndarray,
    pose_delta: tuple[float, float, float],
    *,
    meters_per_cell: float,
) -> np.ndarray:
    source = np.asarray(bev, dtype=np.float32)
    height, width = source.shape
    rows, cols = np.indices((height, width), dtype=np.float32)
    forward = (np.float32(height - 1) - rows + np.float32(0.5)) * np.float32(meters_per_cell)
    left = (cols - np.float32(width) / np.float32(2.0) + np.float32(0.5)) * np.float32(meters_per_cell)
    dx, dy, dyaw = (float(value) for value in pose_delta)
    cos_y = math.cos(-dyaw)
    sin_y = math.sin(-dyaw)
    prev_forward = cos_y * (forward + dx) - sin_y * (left + dy)
    prev_left = sin_y * (forward + dx) + cos_y * (left + dy)
    src_rows = np.rint(np.float32(height - 1) - prev_forward / np.float32(meters_per_cell)).astype(np.int64)
    src_cols = np.rint(prev_left / np.float32(meters_per_cell) + np.float32(width) / np.float32(2.0)).astype(np.int64)
    valid = (src_rows >= 0) & (src_cols >= 0) & (src_rows < height) & (src_cols < width)
    warped = np.zeros_like(source, dtype=np.float32)
    warped[valid] = source[src_rows[valid], src_cols[valid]]
    return warped


def _project_depth(
    depth_m: np.ndarray,
    intrinsics: JsonDict,
    config: RGBDBEVTeacherConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, JsonDict]:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2D array")
    height_px, width_px = depth.shape
    fx = _positive_float(intrinsics.get("fx"), max(float(width_px), 1.0))
    fy = _positive_float(intrinsics.get("fy"), fx)
    cx = _positive_float(intrinsics.get("cx"), (float(width_px) - 1.0) / 2.0)
    cy = _positive_float(intrinsics.get("cy"), (float(height_px) - 1.0) / 2.0)
    grid_h, grid_w = config.grid_shape
    free = np.zeros(config.grid_shape, dtype=np.bool_)
    occupied = np.zeros(config.grid_shape, dtype=np.bool_)
    observed = np.zeros(config.grid_shape, dtype=np.bool_)
    valid_count = 0
    endpoint_count = 0
    for v in range(0, height_px, config.pixel_stride):
        for u in range(0, width_px, config.pixel_stride):
            z = float(depth[v, u])
            if not np.isfinite(z) or z <= 0.0 or z > float(config.max_depth_m):
                continue
            x_cam = (float(u) - cx) * z / fx
            y_cam = (float(v) - cy) * z / fy
            camera_forward = z
            camera_left = -x_cam
            forward = (
                float(config.camera_forward_offset_m)
                + math.cos(float(config.camera_yaw_rad)) * camera_forward
                - math.sin(float(config.camera_yaw_rad)) * camera_left
            )
            left = (
                float(config.camera_left_offset_m)
                + math.sin(float(config.camera_yaw_rad)) * camera_forward
                + math.cos(float(config.camera_yaw_rad)) * camera_left
            )
            height = float(config.camera_height_m) - y_cam
            cell = _cell(forward, left, grid_shape=config.grid_shape, meters_per_cell=config.meters_per_cell)
            if cell is None:
                continue
            valid_count += 1
            endpoint_count += 1
            row, col = cell
            for ray_row, ray_col in _ray_cells(grid_h - 1, grid_w // 2, row, col)[:-1]:
                free[ray_row, ray_col] = True
                observed[ray_row, ray_col] = True
            observed[row, col] = True
            if height >= float(config.obstacle_height_min_m):
                occupied[row, col] = True
            elif abs(height) <= float(config.floor_height_tolerance_m) or height < float(config.obstacle_height_min_m):
                free[row, col] = True
    stats: JsonDict = {
        "depth_shape": [int(height_px), int(width_px)],
        "sampled_depth_count": int(math.ceil(height_px / config.pixel_stride) * math.ceil(width_px / config.pixel_stride)),
        "valid_depth_in_grid_count": int(valid_count),
        "endpoint_count": int(endpoint_count),
        "free_cell_count": int(np.count_nonzero(free)),
        "occupied_cell_count": int(np.count_nonzero(occupied)),
    }
    return free, occupied, observed, stats


def _cell(
    forward_m: float,
    left_m: float,
    *,
    grid_shape: tuple[int, int],
    meters_per_cell: float,
) -> tuple[int, int] | None:
    rows, cols = grid_shape
    width_m = cols * meters_per_cell
    if forward_m < 0.0 or forward_m >= rows * meters_per_cell:
        return None
    if left_m < -width_m / 2.0 or left_m >= width_m / 2.0:
        return None
    row = rows - 1 - int(math.floor(forward_m / meters_per_cell))
    col = int(math.floor((left_m + width_m / 2.0) / meters_per_cell))
    if row < 0 or col < 0 or row >= rows or col >= cols:
        return None
    return row, col


def _ray_cells(row0: int, col0: int, row1: int, col1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    dr = abs(row1 - row0)
    dc = abs(col1 - col0)
    step_r = 1 if row0 < row1 else -1
    step_c = 1 if col0 < col1 else -1
    err = dr - dc
    row, col = row0, col0
    while True:
        cells.append((row, col))
        if row == row1 and col == col1:
            return cells
        twice = 2 * err
        if twice > -dc:
            err -= dc
            row += step_r
        if twice < dr:
            err += dr
            col += step_c


def _dilate(mask: np.ndarray, *, radius: int) -> np.ndarray:
    source = np.asarray(mask, dtype=np.bool_)
    if radius <= 0 or not np.any(source):
        return source.copy()
    height, width = source.shape
    out = source.copy()
    rows, cols = np.nonzero(source)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc > radius * radius:
                continue
            rr = rows + dr
            cc = cols + dc
            valid = (rr >= 0) & (cc >= 0) & (rr < height) & (cc < width)
            out[rr[valid], cc[valid]] = True
    return out


def _positive_float(value: object, fallback: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if np.isfinite(number) and number > 0.0:
            return number
    return float(fallback)
