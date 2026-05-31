from __future__ import annotations

from math import cos, sin
from typing import Any

import numpy as np

from homebrain.viz.checkpoint_3d_scene_types_v0 import CameraFrustumV0, TransformV0


def build_T_map_base_from_pose_sequence(
    pose: dict[str, Any] | tuple[float, float, float] | list[float],
    *,
    timestamp_ns: int | None = None,
    source: str = "odom",
    is_assumed: bool = False,
    is_oracle: bool = False,
) -> TransformV0:
    if isinstance(pose, dict):
        x_m = float(pose.get("x_m", pose.get("x", 0.0)))
        y_m = float(pose.get("y_m", pose.get("y", 0.0)))
        yaw_rad = float(pose.get("yaw_rad", pose.get("theta", 0.0)))
    else:
        if len(pose) != 3:
            raise ValueError("pose must contain x_m, y_m, yaw_rad")
        x_m, y_m, yaw_rad = (float(pose[0]), float(pose[1]), float(pose[2]))
    c = cos(yaw_rad)
    s = sin(yaw_rad)
    matrix = [
        [c, -s, 0.0, x_m],
        [s, c, 0.0, y_m],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return TransformV0(
        frame_from="map",
        frame_to="base_link",
        matrix_4x4=matrix,
        source=source,
        timestamp_ns=timestamp_ns,
        is_assumed=is_assumed,
        is_oracle=is_oracle,
    )


def compose_transforms(a_from_b: TransformV0, b_from_c: TransformV0, *, source: str | None = None) -> TransformV0:
    if a_from_b.frame_to != b_from_c.frame_from:
        raise ValueError(f"cannot compose {a_from_b.frame_from}->{a_from_b.frame_to} with {b_from_c.frame_from}->{b_from_c.frame_to}")
    matrix = np.asarray(a_from_b.matrix_4x4, dtype=np.float64) @ np.asarray(b_from_c.matrix_4x4, dtype=np.float64)
    return TransformV0(
        frame_from=a_from_b.frame_from,
        frame_to=b_from_c.frame_to,
        matrix_4x4=_matrix_list(matrix),
        source=source or f"{a_from_b.source}+{b_from_c.source}",
        timestamp_ns=b_from_c.timestamp_ns if b_from_c.timestamp_ns is not None else a_from_b.timestamp_ns,
        is_assumed=bool(a_from_b.is_assumed or b_from_c.is_assumed),
        is_oracle=bool(a_from_b.is_oracle or b_from_c.is_oracle),
    )


def invert_transform(transform: TransformV0, *, source: str | None = None) -> TransformV0:
    matrix = np.linalg.inv(np.asarray(transform.matrix_4x4, dtype=np.float64))
    return TransformV0(
        frame_from=transform.frame_to,
        frame_to=transform.frame_from,
        matrix_4x4=_matrix_list(matrix),
        source=source or f"inverse({transform.source})",
        timestamp_ns=transform.timestamp_ns,
        covariance=transform.covariance,
        is_assumed=transform.is_assumed,
        is_oracle=transform.is_oracle,
    )


def transform_points(transform: TransformV0, points: np.ndarray | list[list[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] != 3:
        raise ValueError("points must be Nx3")
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    out = (np.asarray(transform.matrix_4x4, dtype=np.float64) @ np.concatenate([pts, ones], axis=1).T).T
    return out[:, :3]


def camera_frustum_from_intrinsics(
    *,
    T_map_camera: TransformV0,
    intrinsics: dict[str, Any],
    timestamp_ns: int | None = None,
    camera_id: str = "camera",
    near_m: float = 0.05,
    far_m: float = 1.5,
) -> CameraFrustumV0:
    fx = _positive_float(intrinsics.get("fx"), "fx")
    fy = _positive_float(intrinsics.get("fy"), "fy")
    cx = float(intrinsics.get("cx", 0.0))
    cy = float(intrinsics.get("cy", 0.0))
    width = int(intrinsics.get("width", max(2, int(round(cx * 2.0)))))
    height = int(intrinsics.get("height", max(2, int(round(cy * 2.0)))))
    corners_camera = []
    for u, v in ((0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))):
        x = (u - cx) * far_m / fx
        y = (v - cy) * far_m / fy
        corners_camera.append([x, y, far_m])
    origin = transform_points(T_map_camera, [[0.0, 0.0, 0.0]])[0].tolist()
    corners_map = transform_points(T_map_camera, corners_camera).tolist()
    lines = [[origin, corner] for corner in corners_map]
    lines.extend(
        [
            [corners_map[0], corners_map[1]],
            [corners_map[1], corners_map[2]],
            [corners_map[2], corners_map[3]],
            [corners_map[3], corners_map[0]],
        ]
    )
    return CameraFrustumV0(
        timestamp_ns=timestamp_ns,
        camera_id=camera_id,
        T_map_camera=T_map_camera,
        intrinsics={"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height},
        near_m=float(near_m),
        far_m=float(far_m),
        frustum_lines_map=_lines_list(lines),
        valid=True,
        warnings=[],
    )


def robot_footprint_polygon(T_map_base: TransformV0, *, length_m: float = 0.36, width_m: float = 0.32) -> list[list[float]]:
    half_l = float(length_m) / 2.0
    half_w = float(width_m) / 2.0
    local = np.asarray(
        [[half_l, half_w, 0.0], [half_l, -half_w, 0.0], [-half_l, -half_w, 0.0], [-half_l, half_w, 0.0], [half_l, half_w, 0.0]],
        dtype=np.float64,
    )
    return _points_list(transform_points(T_map_base, local))


def base_axes_lines(T_map_base: TransformV0, *, axis_len_m: float = 0.25) -> list[list[list[float]]]:
    points = transform_points(T_map_base, [[0.0, 0.0, 0.04], [axis_len_m, 0.0, 0.04], [0.0, axis_len_m, 0.04]])
    return [[points[0].tolist(), points[1].tolist()], [points[0].tolist(), points[2].tolist()]]


def bev_cell_centers_to_robot_frame(
    grid_shape: tuple[int, int],
    resolution_m: float,
    cells: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    if rows <= 0 or cols <= 0 or resolution_m <= 0.0:
        raise ValueError("grid_shape and resolution_m must be positive")
    selected = cells if cells is not None else [(row, col) for row in range(rows) for col in range(cols)]
    points = []
    for row, col in selected:
        if row < 0 or row >= rows or col < 0 or col >= cols:
            raise ValueError(f"cell out of range: {(row, col)}")
        forward_m = (rows - float(row) - 0.5) * float(resolution_m)
        left_m = (float(col) + 0.5 - (float(cols) / 2.0)) * float(resolution_m)
        points.append([forward_m, left_m, 0.0])
    return np.asarray(points, dtype=np.float64)


def robot_frame_to_map_frame(T_map_base: TransformV0, points_robot: np.ndarray | list[list[float]]) -> np.ndarray:
    return transform_points(T_map_base, points_robot)


def transform_from_matrix(
    *,
    frame_from: str,
    frame_to: str,
    matrix_4x4: list[list[float]],
    source: str,
    timestamp_ns: int | None = None,
    is_assumed: bool = False,
    is_oracle: bool = False,
) -> TransformV0:
    return TransformV0(
        frame_from=frame_from,
        frame_to=frame_to,
        matrix_4x4=_matrix_list(np.asarray(matrix_4x4, dtype=np.float64)),
        source=source,
        timestamp_ns=timestamp_ns,
        is_assumed=is_assumed,
        is_oracle=is_oracle,
    )


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _matrix_list(matrix: np.ndarray) -> list[list[float]]:
    if matrix.shape != (4, 4):
        raise ValueError("transform matrix must be 4x4")
    return [[round(float(value), 9) for value in row] for row in matrix.tolist()]


def _points_list(points: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 6) for value in row] for row in np.asarray(points, dtype=np.float64)]


def _lines_list(lines: list[list[list[float]]]) -> list[list[list[float]]]:
    return [[_points_list(np.asarray(segment, dtype=np.float64))[0], _points_list(np.asarray(segment, dtype=np.float64))[1]] for segment in lines]

