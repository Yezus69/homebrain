from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import json
import math
from typing import Any

JsonValue = Any

SCENE_TIMELINE_SCHEMA_VERSION = "Checkpoint3DSceneTimelineV0"
SAFETY_FLAGS_V0: dict[str, bool] = {
    "replay_only": True,
    "not_executed": True,
    "control_safe": False,
    "raw_pwm_emitted": False,
    "hardware_validated": False,
}


def deterministic_scene_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def to_jsonable(value: Any) -> JsonValue:
    if is_dataclass(value):
        return {item.name: to_jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): to_jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return to_jsonable(value.tolist())
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(float(value), 6)
    return str(value)


@dataclass(frozen=True)
class SerializableSceneDataclass:
    def to_dict(self) -> dict[str, Any]:
        data = to_jsonable(self)
        if not isinstance(data, dict):
            raise TypeError("scene dataclass did not serialize to an object")
        return data

    def to_json(self) -> str:
        return deterministic_scene_json(self)


@dataclass(frozen=True)
class TransformV0(SerializableSceneDataclass):
    frame_from: str
    frame_to: str
    matrix_4x4: list[list[float]]
    source: str
    timestamp_ns: int | None
    covariance: list[list[float]] | list[float] | None = None
    is_assumed: bool = False
    is_oracle: bool = False


@dataclass(frozen=True)
class CameraFrustumV0(SerializableSceneDataclass):
    timestamp_ns: int | None
    camera_id: str
    T_map_camera: TransformV0
    intrinsics: dict[str, float | int]
    near_m: float
    far_m: float
    frustum_lines_map: list[list[list[float]]]
    valid: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RobotPoseVizV0(SerializableSceneDataclass):
    timestamp_ns: int | None
    T_map_base: TransformV0
    T_map_camera: TransformV0 | None
    base_axes_lines_map: list[list[list[float]]]
    footprint_polygon_map: list[list[float]]
    pose_confidence: float | None
    pose_source: str
    pose_source_is_oracle: bool


@dataclass(frozen=True)
class ModelChannelSummaryV0(SerializableSceneDataclass):
    name: str
    shape: list[int]
    min: float | None
    max: float | None
    mean: float | None
    valid_fraction: float | None
    source: str
    rendered: bool
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LocalBEVFrameV0(SerializableSceneDataclass):
    timestamp_ns: int | None
    frame_id: int | str
    T_map_base: TransformV0
    resolution_m: float
    origin_convention: str
    channels: dict[str, Any]
    channel_summaries: list[ModelChannelSummaryV0]
    rendered_cells_map: list[dict[str, Any]] = field(default_factory=list)
    valid: bool = True
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AccumulatedMapStateV0(SerializableSceneDataclass):
    timestamp_ns: int | None
    frame_id: int | str
    map_resolution_m: float
    global_bounds_m: dict[str, float]
    free_cells: list[dict[str, Any]]
    obstacle_cells: list[dict[str, Any]]
    unknown_cells: list[dict[str, Any]]
    risky_cells: list[dict[str, Any]]
    hazard_cells: list[dict[str, Any]]
    stale_cells: list[dict[str, Any]]
    trajectory_points: list[list[float]]
    camera_frustums: list[dict[str, Any]]
    point_cloud: list[list[float]]
    conflict_count: int
    stale_update_count: int
    observation_count: int
    geometry_source: str
    pose_source: str
    dense_3d_claimed: bool


@dataclass(frozen=True)
class CandidateTrajectoryVizV0(SerializableSceneDataclass):
    timestamp_ns: int | None
    candidate_id: str
    points_map: list[list[float]]
    footprint_polygons_map: list[list[list[float]]] | None = None
    score: float | None = None
    risk: float | None = None
    hazard_exposure: float | None = None
    unknown_exposure: float | None = None
    selected: bool = False
    vetoed: bool = False
    stop_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SafetyDebugVizV0(SerializableSceneDataclass):
    timestamp_ns: int | None
    proposed_cmd_vel: dict[str, float] | None = None
    safe_cmd_vel: dict[str, float] | None = None
    accepted: bool = False
    stop_reasons: list[str] = field(default_factory=list)
    recovery_proposal: dict[str, Any] | None = None
    stale_sensor: bool | None = None
    high_uncertainty: bool | None = None
    high_risk: bool | None = None
    obstacle_too_close: bool | None = None
    replay_only: bool = True
    not_executed: bool = True


@dataclass(frozen=True)
class SceneTimelineFrameV0(SerializableSceneDataclass):
    schema_version: str = field(default=SCENE_TIMELINE_SCHEMA_VERSION, init=False)
    route_id: str = ""
    frame_id: int | str = 0
    timestamp_ns: int | None = None
    robot_pose: RobotPoseVizV0 | None = None
    camera_frustum: CameraFrustumV0 | None = None
    local_bev: LocalBEVFrameV0 | None = None
    rgb_frame: dict[str, Any] | None = None
    depth_point_cloud_camera: list[list[float]] = field(default_factory=list)
    accumulated_map_summary: dict[str, Any] = field(default_factory=dict)
    candidate_trajectories: list[CandidateTrajectoryVizV0] = field(default_factory=list)
    safety_debug: SafetyDebugVizV0 | None = None
    model_channel_summaries: list[ModelChannelSummaryV0] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    replay_only: bool = True
    not_executed: bool = True
    control_safe: bool = False
    raw_pwm_emitted: bool = False
    hardware_validated: bool = False


def safety_flags() -> dict[str, bool]:
    return dict(SAFETY_FLAGS_V0)
