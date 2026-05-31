from __future__ import annotations

import math
from typing import Any

import numpy as np

from homebrain.geometry.camera_config import CameraConfig
from homebrain.messages.schema import JsonDict


def project_hazard_masks_to_bev(
    *,
    hazard_masks: np.ndarray,
    frame_width: int,
    frame_height: int,
    intrinsics: JsonDict,
    config: CameraConfig,
) -> tuple[np.ndarray, np.ndarray, JsonDict]:
    masks = np.asarray(hazard_masks, dtype=np.float32)
    if masks.ndim != 3:
        raise ValueError(f"hazard_masks must be CxHxW, got {masks.shape}")
    if masks.shape[1:] != (frame_height, frame_width):
        raise ValueError(f"hazard_masks shape {masks.shape[1:]} does not match frame {frame_height}x{frame_width}")
    focal_x, focal_y, cx, cy, intrinsics_source = _intrinsics(intrinsics, frame_width=frame_width, frame_height=frame_height)
    rows, cols = np.indices((frame_height, frame_width), dtype=np.float32)
    forward, left, valid = ground_plane_rays_to_robot_frame(
        u_px=cols,
        v_px=rows,
        focal_x_px=focal_x,
        focal_y_px=focal_y,
        principal_point_px=(cx, cy),
        config=config,
    )
    bev_hazard, valid_mask = _project_values_to_grid(
        values=np.max(masks, axis=0),
        forward_m=forward,
        left_m=left,
        valid=valid,
        config=config,
    )
    positive_cells = int(np.count_nonzero(bev_hazard >= np.float32(0.5)))
    valid_cells = int(np.count_nonzero(valid_mask > 0))
    stats: JsonDict = {
        "hazard_positive_cell_count": positive_cells,
        "hazard_valid_cell_count": valid_cells,
        "hazard_positive_cell_fraction": float(positive_cells / max(bev_hazard.size, 1)),
        "hazard_valid_cell_fraction": float(valid_cells / max(valid_mask.size, 1)),
        "intrinsics_source": intrinsics_source,
    }
    return bev_hazard, valid_mask, stats


def ground_plane_rays_to_robot_frame(
    *,
    u_px: np.ndarray,
    v_px: np.ndarray,
    focal_x_px: float,
    focal_y_px: float,
    principal_point_px: tuple[float, float],
    config: CameraConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy = principal_point_px
    x_cam = (np.asarray(u_px, dtype=np.float32) - np.float32(cx)) / np.float32(focal_x_px)
    y_cam = (np.asarray(v_px, dtype=np.float32) - np.float32(cy)) / np.float32(focal_y_px)
    forward_dir, left_dir, up_dir = _apply_robot_frame_rotation(
        np.ones_like(x_cam, dtype=np.float32),
        -x_cam,
        -y_cam,
        pitch_deg=config.pitch_deg,
        roll_deg=config.roll_deg,
        yaw_deg=config.yaw_deg,
    )
    downward = up_dir < np.float32(-1.0e-6)
    t = np.divide(
        -np.float32(config.camera_height_m),
        up_dir,
        out=np.full(up_dir.shape, np.nan, dtype=np.float32),
        where=downward,
    )
    forward = (t * forward_dir).astype(np.float32)
    left = (t * left_dir).astype(np.float32)
    valid = downward & np.isfinite(forward) & np.isfinite(left) & (t > np.float32(0.0))
    return forward, left, valid


def _project_values_to_grid(
    *,
    values: np.ndarray,
    forward_m: np.ndarray,
    left_m: np.ndarray,
    valid: np.ndarray,
    config: CameraConfig,
) -> tuple[np.ndarray, np.ndarray]:
    cells = config.grid_cell_count
    meters = np.float32(config.meters_per_cell)
    grid_size = np.float32(config.grid_size_m)
    half_width = grid_size / np.float32(2.0)
    value_array = np.asarray(values, dtype=np.float32)
    in_grid = (
        valid
        & np.isfinite(value_array)
        & (forward_m >= np.float32(0.0))
        & (forward_m < grid_size)
        & (left_m >= -half_width)
        & (left_m < half_width)
    )
    bev = np.zeros((cells, cells), dtype=np.float32)
    valid_mask = np.zeros((cells, cells), dtype=np.uint8)
    if np.any(in_grid):
        row = cells - 1 - np.floor(forward_m[in_grid] / meters).astype(np.int64)
        col = np.floor((left_m[in_grid] + half_width) / meters).astype(np.int64)
        row = np.clip(row, 0, cells - 1)
        col = np.clip(col, 0, cells - 1)
        valid_mask[row, col] = 1
        np.maximum.at(bev, (row, col), np.clip(value_array[in_grid], 0.0, 1.0))
    return bev.astype(np.float32), valid_mask.astype(np.uint8)


def _intrinsics(intrinsics: JsonDict, *, frame_width: int, frame_height: int) -> tuple[float, float, float, float, str]:
    fx = _optional_number(intrinsics.get("fx"))
    fy = _optional_number(intrinsics.get("fy"))
    cx = _optional_number(intrinsics.get("cx"))
    cy = _optional_number(intrinsics.get("cy"))
    matrix = intrinsics.get("camera_matrix")
    if isinstance(matrix, list) and len(matrix) >= 3:
        row0 = matrix[0] if isinstance(matrix[0], list) else []
        row1 = matrix[1] if isinstance(matrix[1], list) else []
        fx = fx if fx is not None else _optional_number(row0[0] if len(row0) >= 1 else None)
        cx = cx if cx is not None else _optional_number(row0[2] if len(row0) >= 3 else None)
        fy = fy if fy is not None else _optional_number(row1[1] if len(row1) >= 2 else None)
        cy = cy if cy is not None else _optional_number(row1[2] if len(row1) >= 3 else None)
    focal = _optional_number(intrinsics.get("focallength_px")) or _optional_number(intrinsics.get("focal_length_px"))
    fx = fx if fx is not None else focal
    fy = fy if fy is not None else focal
    source = "frame_intrinsics"
    if fx is None or fy is None:
        fx = fy = float(max(frame_width, frame_height))
        source = "heuristic_max_image_dimension"
    if cx is None:
        cx = (frame_width - 1) / 2.0
        source = f"{source}_principal_point_center"
    if cy is None:
        cy = (frame_height - 1) / 2.0
        source = f"{source}_principal_point_center"
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera intrinsics focal lengths must be positive")
    return float(fx), float(fy), float(cx), float(cy), source


def _optional_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _apply_robot_frame_rotation(
    forward: np.ndarray,
    left: np.ndarray,
    up: np.ndarray,
    *,
    pitch_deg: float,
    roll_deg: float,
    yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    x1 = forward
    y1 = np.float32(cr) * left - np.float32(sr) * up
    z1 = np.float32(sr) * left + np.float32(cr) * up
    x2 = np.float32(cp) * x1 + np.float32(sp) * z1
    y2 = y1
    z2 = -np.float32(sp) * x1 + np.float32(cp) * z1
    x3 = np.float32(cy) * x2 - np.float32(sy) * y2
    y3 = np.float32(sy) * x2 + np.float32(cy) * y2
    return x3, y3, z2
