from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from homebrain.messages.schema import JsonDict, deterministic_json


CAMERA_CONFIG_SCHEMA_VERSION = "homebrain.camera_config.v0"

REQUIRED_CAMERA_CONFIG_FIELDS: tuple[str, ...] = (
    "camera_height_m",
    "pitch_deg",
    "roll_deg",
    "yaw_deg",
    "grid_size_m",
    "meters_per_cell",
    "max_depth_m",
    "floor_height_tol_m",
    "obstacle_height_min_m",
    "robot_radius_m",
)


@dataclass(frozen=True)
class CameraConfig:
    camera_height_m: float
    pitch_deg: float
    roll_deg: float
    yaw_deg: float
    grid_size_m: float
    meters_per_cell: float
    max_depth_m: float
    floor_height_tol_m: float
    obstacle_height_min_m: float
    robot_radius_m: float

    @classmethod
    def from_dict(cls, data: JsonDict) -> "CameraConfig":
        missing = [field for field in REQUIRED_CAMERA_CONFIG_FIELDS if field not in data]
        if missing:
            raise ValueError(f"camera config missing required fields: {missing}")

        unknown = sorted(set(data) - set(REQUIRED_CAMERA_CONFIG_FIELDS))
        if unknown:
            raise ValueError(f"camera config has unknown fields: {unknown}")

        config = cls(
            camera_height_m=_finite_float(data, "camera_height_m"),
            pitch_deg=_finite_float(data, "pitch_deg"),
            roll_deg=_finite_float(data, "roll_deg"),
            yaw_deg=_finite_float(data, "yaw_deg"),
            grid_size_m=_positive_float(data, "grid_size_m"),
            meters_per_cell=_positive_float(data, "meters_per_cell"),
            max_depth_m=_positive_float(data, "max_depth_m"),
            floor_height_tol_m=_positive_float(data, "floor_height_tol_m"),
            obstacle_height_min_m=_positive_float(data, "obstacle_height_min_m"),
            robot_radius_m=_nonnegative_float(data, "robot_radius_m"),
        )
        if config.camera_height_m <= 0.0:
            raise ValueError("camera_height_m must be positive")
        if config.obstacle_height_min_m <= config.floor_height_tol_m:
            raise ValueError("obstacle_height_min_m must be greater than floor_height_tol_m")
        _ = config.grid_cell_count
        return config

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @property
    def grid_cell_count(self) -> int:
        cells = self.grid_size_m / self.meters_per_cell
        rounded = round(cells)
        if not math.isclose(cells, rounded, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(
                "grid_size_m must be an integer multiple of meters_per_cell; "
                f"got {self.grid_size_m} / {self.meters_per_cell} = {cells}"
            )
        if rounded <= 0:
            raise ValueError("grid must have at least one cell")
        return int(rounded)

    @property
    def grid_shape(self) -> tuple[int, int]:
        cells = self.grid_cell_count
        return (cells, cells)

    @property
    def obstacle_dilation_cells(self) -> int:
        return int(math.ceil(self.robot_radius_m / self.meters_per_cell))


def load_camera_config(path: str | Path) -> CameraConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"camera config must be a JSON object: {config_path}")
    return CameraConfig.from_dict(data)


def write_camera_config(path: str | Path, config: CameraConfig) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(deterministic_json(config.to_dict()))
        handle.write("\n")


def _finite_float(data: JsonDict, key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _positive_float(data: JsonDict, key: str) -> float:
    number = _finite_float(data, key)
    if number <= 0.0:
        raise ValueError(f"{key} must be positive")
    return number


def _nonnegative_float(data: JsonDict, key: str) -> float:
    number = _finite_float(data, key)
    if number < 0.0:
        raise ValueError(f"{key} must be nonnegative")
    return number


def camera_config_with_overrides(config: CameraConfig, **overrides: Any) -> CameraConfig:
    data = config.to_dict()
    data.update(overrides)
    return CameraConfig.from_dict(data)

