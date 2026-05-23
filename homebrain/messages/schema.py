from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Union

SCHEMA_VERSION = "homebrain.segment.v0"

JsonDict = dict[str, Any]


def deterministic_json(data: JsonDict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _require_event_type(data: JsonDict, expected: str) -> None:
    actual = data.get("event_type")
    if actual != expected:
        raise ValueError(f"expected event_type {expected!r}, got {actual!r}")


def _require_str(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _require_int(data: JsonDict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an int")
    return value


def _require_float(data: JsonDict, key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _optional_float(data: JsonDict, key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number or null")
    return float(value)


def _require_dict(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _optional_dict(data: JsonDict, key: str) -> JsonDict | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object or null")
    return value


def _require_str_list(data: JsonDict, key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return list(value)


def _optional_dict_list(data: JsonDict, key: str) -> list[JsonDict] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{key} must be a list of objects or null")
    return [dict(item) for item in value]


def _vec2(data: JsonDict, key: str) -> tuple[float, float]:
    value = data.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{key} must be a length-2 list")
    return (_number(value[0], key), _number(value[1], key))


def _optional_vec2(data: JsonDict, key: str) -> tuple[float, float] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{key} must be a length-2 list or null")
    return (_number(value[0], key), _number(value[1], key))


def _vec3(data: JsonDict, key: str) -> tuple[float, float, float]:
    value = data.get(key)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{key} must be a length-3 list")
    return (_number(value[0], key), _number(value[1], key), _number(value[2], key))


def _optional_vec3(data: JsonDict, key: str) -> tuple[float, float, float] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{key} must be a length-3 list or null")
    return (_number(value[0], key), _number(value[1], key), _number(value[2], key))


def _number(value: Any, key: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} values must be numbers")
    return float(value)


def _optional_str(data: JsonDict, key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value


@dataclass(frozen=True)
class FrameEvent:
    timestamp_ns: int
    sequence_id: str
    source: str
    camera_id: str
    frame_id: int
    width: int
    height: int
    format: Literal["rgb8", "bgr8", "gray8", "encoded_jpeg"]
    data_ref: str
    intrinsics: JsonDict | None = None
    event_type: Literal["frame"] = field(default="frame", init=False)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "FrameEvent":
        _require_event_type(data, "frame")
        fmt = _require_str(data, "format")
        if fmt not in {"rgb8", "bgr8", "gray8", "encoded_jpeg"}:
            raise ValueError(f"unsupported frame format {fmt!r}")
        return cls(
            timestamp_ns=_require_int(data, "timestamp_ns"),
            sequence_id=_require_str(data, "sequence_id"),
            source=_require_str(data, "source"),
            camera_id=_require_str(data, "camera_id"),
            frame_id=_require_int(data, "frame_id"),
            width=_require_int(data, "width"),
            height=_require_int(data, "height"),
            format=fmt,  # type: ignore[arg-type]
            data_ref=_require_str(data, "data_ref"),
            intrinsics=_optional_dict(data, "intrinsics"),
        )


@dataclass(frozen=True)
class ImuEvent:
    timestamp_ns: int
    sequence_id: str
    source: str
    accel_mps2: tuple[float, float, float]
    gyro_radps: tuple[float, float, float]
    event_type: Literal["imu"] = field(default="imu", init=False)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "ImuEvent":
        _require_event_type(data, "imu")
        return cls(
            timestamp_ns=_require_int(data, "timestamp_ns"),
            sequence_id=_require_str(data, "sequence_id"),
            source=_require_str(data, "source"),
            accel_mps2=_vec3(data, "accel_mps2"),
            gyro_radps=_vec3(data, "gyro_radps"),
        )


@dataclass(frozen=True)
class WheelEvent:
    timestamp_ns: int
    sequence_id: str
    source: str
    left_ticks: int
    right_ticks: int
    left_velocity: float | None = None
    right_velocity: float | None = None
    odom_delta: tuple[float, float, float] | None = None
    event_type: Literal["wheel"] = field(default="wheel", init=False)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "WheelEvent":
        _require_event_type(data, "wheel")
        return cls(
            timestamp_ns=_require_int(data, "timestamp_ns"),
            sequence_id=_require_str(data, "sequence_id"),
            source=_require_str(data, "source"),
            left_ticks=_require_int(data, "left_ticks"),
            right_ticks=_require_int(data, "right_ticks"),
            left_velocity=_optional_float(data, "left_velocity"),
            right_velocity=_optional_float(data, "right_velocity"),
            odom_delta=_optional_vec3(data, "odom_delta"),
        )


@dataclass(frozen=True)
class CommandEvent:
    timestamp_ns: int
    sequence_id: str
    source: str
    linear_velocity_mps: float
    angular_velocity_radps: float
    command_id: str
    event_type: Literal["command"] = field(default="command", init=False)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "CommandEvent":
        _require_event_type(data, "command")
        return cls(
            timestamp_ns=_require_int(data, "timestamp_ns"),
            sequence_id=_require_str(data, "sequence_id"),
            source=_require_str(data, "source"),
            linear_velocity_mps=_require_float(data, "linear_velocity_mps"),
            angular_velocity_radps=_require_float(data, "angular_velocity_radps"),
            command_id=_require_str(data, "command_id"),
        )


@dataclass(frozen=True)
class BrainOutputEvent:
    timestamp_ns: int
    sequence_id: str
    source: str
    input_event_ids: list[str]
    pose_confidence: float
    uncertainty: float
    debug: JsonDict
    pose_delta: tuple[float, float, float] | None = None
    local_bev_ref: str | None = None
    dynamic_risk_ref: str | None = None
    coverage_update_ref: str | None = None
    candidate_trajectories: list[JsonDict] | None = None
    selected_trajectory_id: str | None = None
    cmd_vel: tuple[float, float] | None = None
    stop_reason: str | None = None
    event_type: Literal["brain_output"] = field(default="brain_output", init=False)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "BrainOutputEvent":
        _require_event_type(data, "brain_output")
        return cls(
            timestamp_ns=_require_int(data, "timestamp_ns"),
            sequence_id=_require_str(data, "sequence_id"),
            source=_require_str(data, "source"),
            input_event_ids=_require_str_list(data, "input_event_ids"),
            pose_delta=_optional_vec3(data, "pose_delta"),
            pose_confidence=_require_float(data, "pose_confidence"),
            local_bev_ref=_optional_str(data, "local_bev_ref"),
            dynamic_risk_ref=_optional_str(data, "dynamic_risk_ref"),
            coverage_update_ref=_optional_str(data, "coverage_update_ref"),
            candidate_trajectories=_optional_dict_list(data, "candidate_trajectories"),
            selected_trajectory_id=_optional_str(data, "selected_trajectory_id"),
            cmd_vel=_optional_vec2(data, "cmd_vel"),
            uncertainty=_require_float(data, "uncertainty"),
            stop_reason=_optional_str(data, "stop_reason"),
            debug=_require_dict(data, "debug"),
        )


@dataclass(frozen=True)
class EvalEvent:
    timestamp_ns: int
    sequence_id: str
    source: str
    metric_name: str
    metric_value: float | int | bool | str
    metadata: JsonDict
    event_type: Literal["eval"] = field(default="eval", init=False)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "EvalEvent":
        _require_event_type(data, "eval")
        value = data.get("metric_value")
        if not isinstance(value, (float, int, bool, str)):
            raise ValueError("metric_value must be a scalar")
        return cls(
            timestamp_ns=_require_int(data, "timestamp_ns"),
            sequence_id=_require_str(data, "sequence_id"),
            source=_require_str(data, "source"),
            metric_name=_require_str(data, "metric_name"),
            metric_value=value,
            metadata=_require_dict(data, "metadata"),
        )


Event = Union[
    FrameEvent,
    ImuEvent,
    WheelEvent,
    CommandEvent,
    BrainOutputEvent,
    EvalEvent,
]

_EVENT_BY_TYPE = {
    "frame": FrameEvent,
    "imu": ImuEvent,
    "wheel": WheelEvent,
    "command": CommandEvent,
    "brain_output": BrainOutputEvent,
    "eval": EvalEvent,
}


@dataclass(frozen=True)
class SegmentManifest:
    segment_id: str
    created_at_utc: str
    schema_version: str
    event_file: str
    artifact_files: list[str]
    start_timestamp_ns: int
    end_timestamp_ns: int
    event_count: int

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "SegmentManifest":
        return cls(
            segment_id=_require_str(data, "segment_id"),
            created_at_utc=_require_str(data, "created_at_utc"),
            schema_version=_require_str(data, "schema_version"),
            event_file=_require_str(data, "event_file"),
            artifact_files=_require_str_list(data, "artifact_files"),
            start_timestamp_ns=_require_int(data, "start_timestamp_ns"),
            end_timestamp_ns=_require_int(data, "end_timestamp_ns"),
            event_count=_require_int(data, "event_count"),
        )


def event_to_dict(event: Event) -> JsonDict:
    if not hasattr(event, "to_dict"):
        raise TypeError(f"unsupported event object {type(event)!r}")
    return event.to_dict()


def event_from_dict(data: JsonDict) -> Event:
    event_type = data.get("event_type")
    event_cls = _EVENT_BY_TYPE.get(event_type)
    if event_cls is None:
        raise ValueError(f"unknown event_type {event_type!r}")
    return event_cls.from_dict(data)  # type: ignore[return-value]


def event_to_json_line(event: Event) -> str:
    return deterministic_json(event_to_dict(event))


def event_from_json_line(line: str) -> Event:
    data = json.loads(line)
    if not isinstance(data, dict):
        raise ValueError("event JSON line must decode to an object")
    return event_from_dict(data)


def event_identity(event: Event) -> str:
    base = f"{event.sequence_id}:{event.event_type}:{event.timestamp_ns}:{event.source}"
    if isinstance(event, FrameEvent):
        return f"{base}:camera={event.camera_id}:frame={event.frame_id}"
    if isinstance(event, CommandEvent):
        return f"{base}:command={event.command_id}"
    return base
