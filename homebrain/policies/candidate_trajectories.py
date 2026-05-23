from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, floor, sin

from homebrain.messages.schema import JsonDict, deterministic_json


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float

    def to_dict(self) -> JsonDict:
        return {
            "x_m": round(float(self.x_m), 6),
            "y_m": round(float(self.y_m), 6),
            "yaw_rad": round(float(self.yaw_rad), 6),
        }


@dataclass(frozen=True)
class CandidateTrajectory:
    id: str
    cmd_vel_proxy: JsonDict
    duration_s: float
    poses: tuple[Pose2D, ...]
    footprint_cells: tuple[tuple[int, int], ...]

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "cmd_vel_proxy": dict(self.cmd_vel_proxy),
            "duration_s": round(float(self.duration_s), 6),
            "poses": [pose.to_dict() for pose in self.poses],
            "footprint_cells": [[int(row), int(col)] for row, col in self.footprint_cells],
        }


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    linear_velocity_mps: float
    angular_velocity_radps: float
    duration_s: float


DEFAULT_CANDIDATE_SPECS: tuple[CandidateSpec, ...] = (
    CandidateSpec("stop", 0.0, 0.0, 0.5),
    CandidateSpec("straight_short", 0.25, 0.0, 1.2),
    CandidateSpec("straight_medium", 0.25, 0.0, 2.4),
    CandidateSpec("arc_left_small", 0.20, 0.45, 1.6),
    CandidateSpec("arc_right_small", 0.20, -0.45, 1.6),
    CandidateSpec("arc_left_medium", 0.20, 0.85, 1.6),
    CandidateSpec("arc_right_medium", 0.20, -0.85, 1.6),
    CandidateSpec("rotate_left", 0.0, 1.0, 0.9),
    CandidateSpec("rotate_right", 0.0, -1.0, 0.9),
)


def generate_default_candidates(
    *,
    grid_shape: tuple[int, int],
    meters_per_cell: float,
    robot_radius_m: float = 0.18,
    integration_steps: int = 12,
) -> list[CandidateTrajectory]:
    if grid_shape[0] <= 0 or grid_shape[1] <= 0:
        raise ValueError("grid_shape dimensions must be positive")
    if meters_per_cell <= 0.0:
        raise ValueError("meters_per_cell must be positive")
    if robot_radius_m < 0.0:
        raise ValueError("robot_radius_m must be non-negative")
    if integration_steps <= 0:
        raise ValueError("integration_steps must be positive")

    return [
        _candidate_from_spec(
            spec,
            grid_shape=grid_shape,
            meters_per_cell=meters_per_cell,
            robot_radius_m=robot_radius_m,
            integration_steps=integration_steps,
        )
        for spec in DEFAULT_CANDIDATE_SPECS
    ]


def candidates_hash(candidates: list[CandidateTrajectory]) -> str:
    import hashlib

    text = deterministic_json({"candidates": [candidate.to_dict() for candidate in candidates]})
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _candidate_from_spec(
    spec: CandidateSpec,
    *,
    grid_shape: tuple[int, int],
    meters_per_cell: float,
    robot_radius_m: float,
    integration_steps: int,
) -> CandidateTrajectory:
    poses = _integrate_unicycle(
        linear_velocity_mps=spec.linear_velocity_mps,
        angular_velocity_radps=spec.angular_velocity_radps,
        duration_s=spec.duration_s,
        integration_steps=integration_steps,
    )
    cells = _footprint_cells_for_poses(
        poses,
        grid_shape=grid_shape,
        meters_per_cell=meters_per_cell,
        robot_radius_m=robot_radius_m,
    )
    return CandidateTrajectory(
        id=spec.candidate_id,
        cmd_vel_proxy={
            "linear_velocity_mps": round(float(spec.linear_velocity_mps), 6),
            "angular_velocity_radps": round(float(spec.angular_velocity_radps), 6),
            "not_hardware_control": True,
        },
        duration_s=spec.duration_s,
        poses=tuple(poses),
        footprint_cells=tuple(cells),
    )


def _integrate_unicycle(
    *,
    linear_velocity_mps: float,
    angular_velocity_radps: float,
    duration_s: float,
    integration_steps: int,
) -> list[Pose2D]:
    dt = duration_s / float(integration_steps)
    x_m = 0.0
    y_m = 0.0
    yaw_rad = 0.0
    poses = [Pose2D(x_m=0.0, y_m=0.0, yaw_rad=0.0)]
    for _step in range(integration_steps):
        x_m += linear_velocity_mps * cos(yaw_rad) * dt
        y_m += linear_velocity_mps * sin(yaw_rad) * dt
        yaw_rad += angular_velocity_radps * dt
        poses.append(Pose2D(x_m=x_m, y_m=y_m, yaw_rad=yaw_rad))
    return poses


def _footprint_cells_for_poses(
    poses: list[Pose2D],
    *,
    grid_shape: tuple[int, int],
    meters_per_cell: float,
    robot_radius_m: float,
) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    radius_cells = max(0, int(ceil(robot_radius_m / meters_per_cell)))
    for pose in poses:
        center = _pose_to_cell(
            pose.x_m,
            pose.y_m,
            grid_shape=grid_shape,
            meters_per_cell=meters_per_cell,
        )
        if center is None:
            continue
        center_row, center_col = center
        for row_offset in range(-radius_cells, radius_cells + 1):
            for col_offset in range(-radius_cells, radius_cells + 1):
                if (row_offset * meters_per_cell) ** 2 + (col_offset * meters_per_cell) ** 2 > robot_radius_m**2:
                    continue
                row = center_row + row_offset
                col = center_col + col_offset
                if 0 <= row < grid_shape[0] and 0 <= col < grid_shape[1]:
                    cells.add((row, col))
    return sorted(cells)


def _pose_to_cell(
    forward_m: float,
    left_m: float,
    *,
    grid_shape: tuple[int, int],
    meters_per_cell: float,
) -> tuple[int, int] | None:
    rows, cols = grid_shape
    grid_forward_m = rows * meters_per_cell
    grid_width_m = cols * meters_per_cell
    half_width_m = grid_width_m / 2.0
    if forward_m < 0.0 or forward_m >= grid_forward_m:
        return None
    if left_m < -half_width_m or left_m >= half_width_m:
        return None
    row = rows - 1 - int(floor(forward_m / meters_per_cell))
    col = int(floor((left_m + half_width_m) / meters_per_cell))
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return None
    return row, col

