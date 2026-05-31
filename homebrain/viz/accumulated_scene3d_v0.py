from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from homebrain.viz.checkpoint_3d_scene_types_v0 import AccumulatedMapStateV0, CameraFrustumV0, LocalBEVFrameV0, TransformV0
from homebrain.viz.pose_camera_geometry_v0 import bev_cell_centers_to_robot_frame, robot_frame_to_map_frame, transform_points


@dataclass
class _CellState:
    ix: int
    iy: int
    x_m: float
    y_m: float
    free: bool = False
    obstacle: bool = False
    unknown: bool = False
    risky: bool = False
    hazard: bool = False
    stale: bool = False
    first_frame: int | str = 0
    last_frame: int | str = 0
    last_timestamp_ns: int | None = None
    observations: int = 0

    def record(self, state: str) -> dict[str, Any]:
        return {
            "ix": int(self.ix),
            "iy": int(self.iy),
            "x_m": round(float(self.x_m), 6),
            "y_m": round(float(self.y_m), 6),
            "z_m": _z_for_state(state),
            "state": state,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "observations": int(self.observations),
        }


class AccumulatedScene3DV0:
    def __init__(
        self,
        resolution_m: float,
        decay_sec: float,
        max_points: int,
        geometry_source: str,
        *,
        pose_source: str = "fixture_pose",
        dense_3d_claimed: bool = False,
    ) -> None:
        if resolution_m <= 0.0:
            raise ValueError("resolution_m must be positive")
        if max_points <= 0:
            raise ValueError("max_points must be positive")
        self.resolution_m = float(resolution_m)
        self.decay_sec = float(decay_sec)
        self.max_points = int(max_points)
        self.geometry_source = str(geometry_source)
        self.pose_source = str(pose_source)
        self.dense_3d_claimed = bool(dense_3d_claimed)
        self._cells: dict[tuple[int, int], _CellState] = {}
        self._trajectory: list[list[float]] = []
        self._camera_frustums: list[CameraFrustumV0] = []
        self._point_cloud: list[list[float]] = []
        self._conflicts = 0
        self._stale_updates = 0
        self._observations = 0

    def record_frame_pose(self, T_map_base: TransformV0, *, camera_frustum: CameraFrustumV0 | None = None) -> None:
        base = np.asarray(T_map_base.matrix_4x4, dtype=np.float64)[:3, 3]
        point = [round(float(base[0]), 6), round(float(base[1]), 6), round(float(base[2]), 6)]
        if not self._trajectory or self._trajectory[-1] != point:
            self._trajectory.append(point)
        if camera_frustum is not None:
            self._camera_frustums.append(camera_frustum)

    def update_from_local_bev(self, frame: LocalBEVFrameV0, *, camera_frustum: CameraFrustumV0 | None = None) -> None:
        channels = frame.channels
        free = _channel(channels, "free")
        obstacle = _channel(channels, "obstacle", fallback="occupied")
        unknown = _channel(channels, "unknown")
        risky = _channel(channels, "risky", fallback="risk")
        hazard = _channel(channels, "hazard")
        memory_age = _channel(channels, "memory_age")
        shape = _first_shape([free, obstacle, unknown, risky, hazard, memory_age])
        if shape is None:
            return

        cells = [(row, col) for row in range(shape[0]) for col in range(shape[1])]
        local = bev_cell_centers_to_robot_frame(shape, frame.resolution_m, cells)
        mapped = robot_frame_to_map_frame(frame.T_map_base, local)
        for index, (row, col) in enumerate(cells):
            values = {
                "free": _value(free, row, col),
                "obstacle": _value(obstacle, row, col),
                "unknown": _value(unknown, row, col),
                "risky": _value(risky, row, col),
                "hazard": _value(hazard, row, col),
                "memory_age": _value(memory_age, row, col),
            }
            if max(values["free"], values["obstacle"], values["unknown"], values["risky"], values["hazard"]) <= 0.5:
                continue
            point = mapped[index]
            cell = self._cell_for_point(point[0], point[1], frame.frame_id, frame.timestamp_ns)
            was_free = cell.free
            was_obstacle = cell.obstacle
            if values["free"] > 0.5:
                cell.free = True
            if values["obstacle"] > 0.5:
                cell.obstacle = True
            if values["unknown"] > 0.5 and not (cell.free or cell.obstacle):
                cell.unknown = True
            if values["risky"] > 0.5:
                cell.risky = True
            if values["hazard"] > 0.5:
                cell.hazard = True
            if values["memory_age"] > self.decay_sec > 0.0:
                cell.stale = True
                self._stale_updates += 1
            if (cell.free and was_obstacle and not was_free) or (cell.obstacle and was_free and not was_obstacle):
                self._conflicts += 1
            cell.last_frame = frame.frame_id
            cell.last_timestamp_ns = frame.timestamp_ns
            cell.observations += 1
        self.record_frame_pose(frame.T_map_base, camera_frustum=camera_frustum)
        self._observations += 1

    def update_from_depth_pointcloud(
        self,
        points_camera: np.ndarray | list[list[float]],
        *,
        T_map_camera: Any,
        frame_id: int | str,
        timestamp_ns: int | None,
    ) -> None:
        raw = np.asarray(points_camera, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        colors = raw[:, 3:6] if raw.shape[1] >= 6 else None
        points = transform_points(T_map_camera, raw[:, :3])
        if points.shape[0] > self.max_points:
            step = max(1, int(np.ceil(points.shape[0] / self.max_points)))
            points = points[::step]
            if colors is not None:
                colors = colors[::step]
        for index, point in enumerate(points[: self.max_points]):
            record = [round(float(point[0]), 6), round(float(point[1]), 6), round(float(point[2]), 6)]
            if colors is not None and index < colors.shape[0]:
                record.extend([int(np.clip(round(float(value)), 0, 255)) for value in colors[index, :3]])
            self._point_cloud.append(record)
            cell = self._cell_for_point(point[0], point[1], frame_id, timestamp_ns)
            cell.obstacle = True
            cell.last_frame = frame_id
            cell.last_timestamp_ns = timestamp_ns
            cell.observations += 1
        self._point_cloud = self._point_cloud[-self.max_points :]
        self._observations += 1

    def snapshot(self, frame_id: int | str, timestamp_ns: int | None) -> AccumulatedMapStateV0:
        stale_keys = self._stale_keys(timestamp_ns)
        for key in stale_keys:
            self._cells[key].stale = True
        free = self.export_cells("free")
        obstacle = self.export_cells("obstacle")
        unknown = self.export_cells("unknown")
        risky = self.export_cells("risky")
        hazard = self.export_cells("hazard")
        stale = self.export_cells("stale")
        return AccumulatedMapStateV0(
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
            map_resolution_m=self.resolution_m,
            global_bounds_m=_bounds([*free, *obstacle, *unknown, *risky, *hazard, *stale]),
            free_cells=free,
            obstacle_cells=obstacle,
            unknown_cells=unknown,
            risky_cells=risky,
            hazard_cells=hazard,
            stale_cells=stale,
            trajectory_points=list(self._trajectory),
            camera_frustums=[frustum.to_dict() for frustum in self._camera_frustums],
            point_cloud=self.export_point_cloud(max_points=self.max_points),
            conflict_count=int(self._conflicts),
            stale_update_count=int(self._stale_updates + len(stale_keys)),
            observation_count=int(self._observations),
            geometry_source=self.geometry_source,
            pose_source=self.pose_source,
            dense_3d_claimed=self.dense_3d_claimed,
        )

    def export_cells(self, state: str, *, max_cells: int | None = None) -> list[dict[str, Any]]:
        records = [cell.record(state) for cell in self._sorted_cells() if _has_state(cell, state)]
        if max_cells is not None:
            return records[: max(0, int(max_cells))]
        return records

    def export_point_cloud(self, *, max_points: int | None = None) -> list[list[float]]:
        limit = self.max_points if max_points is None else max(0, int(max_points))
        if self._point_cloud:
            return self._point_cloud[:limit]
        points: list[list[float]] = []
        for state in ("free", "obstacle", "unknown", "risky", "hazard", "stale"):
            for cell in self.export_cells(state):
                points.append([cell["x_m"], cell["y_m"], cell["z_m"]])
                if len(points) >= limit:
                    return points
        return points

    def _cell_for_point(self, x_m: float, y_m: float, frame_id: int | str, timestamp_ns: int | None) -> _CellState:
        ix = int(round(float(x_m) / self.resolution_m))
        iy = int(round(float(y_m) / self.resolution_m))
        key = (ix, iy)
        if key not in self._cells:
            self._cells[key] = _CellState(
                ix=ix,
                iy=iy,
                x_m=ix * self.resolution_m,
                y_m=iy * self.resolution_m,
                first_frame=frame_id,
                last_frame=frame_id,
                last_timestamp_ns=timestamp_ns,
            )
        return self._cells[key]

    def _sorted_cells(self) -> list[_CellState]:
        return [self._cells[key] for key in sorted(self._cells)]

    def _stale_keys(self, timestamp_ns: int | None) -> list[tuple[int, int]]:
        if timestamp_ns is None or self.decay_sec <= 0.0:
            return []
        max_age_ns = int(self.decay_sec * 1_000_000_000)
        stale = []
        for key, cell in self._cells.items():
            if cell.last_timestamp_ns is not None and int(timestamp_ns) - int(cell.last_timestamp_ns) > max_age_ns:
                stale.append(key)
        return stale


def _channel(channels: dict[str, Any], name: str, *, fallback: str | None = None) -> np.ndarray | None:
    value = channels.get(name)
    if value is None and fallback is not None:
        value = channels.get(fallback)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"channel {name} must be a 2D array")
    return array


def _first_shape(arrays: list[np.ndarray | None]) -> tuple[int, int] | None:
    shapes = [array.shape for array in arrays if array is not None]
    if not shapes:
        return None
    first = shapes[0]
    if any(shape != first for shape in shapes):
        raise ValueError(f"BEV channels must share a shape, got {shapes}")
    return (int(first[0]), int(first[1]))


def _value(array: np.ndarray | None, row: int, col: int) -> float:
    if array is None:
        return 0.0
    value = float(array[row, col])
    return value if np.isfinite(value) else 0.0


def _has_state(cell: _CellState, state: str) -> bool:
    return bool(getattr(cell, state))


def _z_for_state(state: str) -> float:
    return {
        "free": 0.0,
        "obstacle": 0.18,
        "unknown": 0.03,
        "risky": 0.02,
        "hazard": 0.04,
        "stale": 0.01,
    }.get(state, 0.0)


def _bounds(cells: list[dict[str, Any]]) -> dict[str, float]:
    if not cells:
        return {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0}
    xs = [float(cell["x_m"]) for cell in cells]
    ys = [float(cell["y_m"]) for cell in cells]
    return {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}
