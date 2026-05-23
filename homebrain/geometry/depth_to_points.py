from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from homebrain.geometry.camera_config import CameraConfig


@dataclass(frozen=True)
class ProjectedPoints:
    forward_m: np.ndarray
    left_m: np.ndarray
    height_m: np.ndarray
    confidence: np.ndarray
    depth_m: np.ndarray
    image_shape: tuple[int, int]
    focal_length_px: float
    principal_point_px: tuple[float, float]
    pixel_stride: int
    sampled_pixel_count: int
    valid_point_count: int
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]


def project_depth_to_robot_points(
    depth_m: np.ndarray,
    focal_length_px: float,
    config: CameraConfig,
    *,
    confidence: np.ndarray | None = None,
    principal_point_px: tuple[float, float] | None = None,
    pixel_stride: int = 1,
) -> ProjectedPoints:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be 2D, got shape {depth.shape}")
    height, width = depth.shape
    focal = float(focal_length_px)
    if not math.isfinite(focal) or focal <= 0.0:
        raise ValueError(f"focal_length_px must be positive and finite, got {focal_length_px!r}")
    if pixel_stride <= 0:
        raise ValueError(f"pixel_stride must be positive, got {pixel_stride}")

    assumptions: list[str] = []
    warnings: list[str] = []
    if principal_point_px is None:
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        assumptions.append("principal_point_assumed_image_center")
        warnings.append("camera principal point missing; assumed image center")
    else:
        cx, cy = principal_point_px
        if not all(math.isfinite(value) for value in (cx, cy)):
            raise ValueError(f"principal_point_px must be finite, got {principal_point_px!r}")

    if pixel_stride > 1:
        assumptions.append(f"depth_sampled_with_pixel_stride_{pixel_stride}")

    sampled_depth = depth[::pixel_stride, ::pixel_stride]
    row_index, col_index = np.indices(sampled_depth.shape, dtype=np.float32)
    v = row_index * np.float32(pixel_stride)
    u = col_index * np.float32(pixel_stride)

    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth > np.float32(0.0))
        & (sampled_depth <= np.float32(config.max_depth_m))
    )
    sampled_count = int(sampled_depth.size)
    if not np.any(valid):
        return ProjectedPoints(
            forward_m=np.zeros((0,), dtype=np.float32),
            left_m=np.zeros((0,), dtype=np.float32),
            height_m=np.zeros((0,), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
            depth_m=np.zeros((0,), dtype=np.float32),
            image_shape=(height, width),
            focal_length_px=focal,
            principal_point_px=(float(cx), float(cy)),
            pixel_stride=pixel_stride,
            sampled_pixel_count=sampled_count,
            valid_point_count=0,
            assumptions=tuple(assumptions),
            warnings=tuple(warnings),
        )

    z_cam = sampled_depth[valid].astype(np.float32)
    x_cam = ((u[valid] - np.float32(cx)) / np.float32(focal)) * z_cam
    y_cam = ((v[valid] - np.float32(cy)) / np.float32(focal)) * z_cam

    # Base camera convention is x right, y down, z forward.
    # Robot/local convention is x forward, y left, z up.
    base_forward = z_cam
    base_left = -x_cam
    base_up = -y_cam
    forward, left, up = _apply_robot_frame_rotation(
        base_forward,
        base_left,
        base_up,
        pitch_deg=config.pitch_deg,
        roll_deg=config.roll_deg,
        yaw_deg=config.yaw_deg,
    )
    point_height = (np.float32(config.camera_height_m) + up).astype(np.float32)

    if confidence is None:
        point_confidence = np.ones(z_cam.shape, dtype=np.float32)
        assumptions.append("depth_confidence_missing_assumed_one")
    else:
        conf = np.asarray(confidence, dtype=np.float32)
        if conf.shape != depth.shape:
            raise ValueError(f"depth_confidence shape {conf.shape} does not match depth shape {depth.shape}")
        point_confidence = np.clip(conf[::pixel_stride, ::pixel_stride][valid], 0.0, 1.0).astype(np.float32)

    return ProjectedPoints(
        forward_m=forward.astype(np.float32),
        left_m=left.astype(np.float32),
        height_m=point_height,
        confidence=point_confidence,
        depth_m=z_cam,
        image_shape=(height, width),
        focal_length_px=focal,
        principal_point_px=(float(cx), float(cy)),
        pixel_stride=pixel_stride,
        sampled_pixel_count=sampled_count,
        valid_point_count=int(z_cam.size),
        assumptions=tuple(assumptions),
        warnings=tuple(warnings),
    )


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

    # Roll around robot-forward x.
    x1 = forward
    y1 = np.float32(cr) * left - np.float32(sr) * up
    z1 = np.float32(sr) * left + np.float32(cr) * up

    # Positive pitch points the camera downward from horizontal.
    x2 = np.float32(cp) * x1 + np.float32(sp) * z1
    y2 = y1
    z2 = -np.float32(sp) * x1 + np.float32(cp) * z1

    # Yaw around robot-up z.
    x3 = np.float32(cy) * x2 - np.float32(sy) * y2
    y3 = np.float32(sy) * x2 + np.float32(cy) * y2
    z3 = z2
    return x3, y3, z3

