# SCHEMA.md — Initial Event Schema

Codex may implement this using dataclasses, pydantic, msgpack, JSONL, Cap'n Proto, FlatBuffers, or another simple typed format.

Do not overcomplicate schema in Goal 0. Correctness and deterministic replay matter most.

## Common fields

Every event:
```text
event_type: string
timestamp_ns: int
sequence_id: string
source: string
```

## FrameEvent

```text
event_type: "frame"
timestamp_ns: int
sequence_id: string
camera_id: string
frame_id: int
width: int
height: int
format: "rgb8" | "bgr8" | "gray8" | "encoded_jpeg"
data_ref: string
intrinsics: optional dict
```

`data_ref` may point to an image file inside the segment.

## ImuEvent

```text
event_type: "imu"
timestamp_ns: int
sequence_id: string
accel_mps2: [float, float, float]
gyro_radps: [float, float, float]
```

## WheelEvent

```text
event_type: "wheel"
timestamp_ns: int
sequence_id: string
left_ticks: int
right_ticks: int
left_velocity: optional float
right_velocity: optional float
odom_delta: optional [dx, dy, dyaw]
```

## CommandEvent

```text
event_type: "command"
timestamp_ns: int
sequence_id: string
linear_velocity_mps: float
angular_velocity_radps: float
command_id: string
```

## BrainOutputEvent

```text
event_type: "brain_output"
timestamp_ns: int
sequence_id: string
input_event_ids: list[string]
pose_delta: optional [dx, dy, dyaw]
pose_confidence: float
local_bev_ref: optional string
dynamic_risk_ref: optional string
coverage_update_ref: optional string
candidate_trajectories: optional list[dict]
selected_trajectory_id: optional string
cmd_vel: optional [linear_mps, angular_radps]
uncertainty: float
stop_reason: optional string
debug: dict
```

## EvalEvent

```text
event_type: "eval"
timestamp_ns: int
sequence_id: string
metric_name: string
metric_value: float | int | bool | string
metadata: dict
```

## SegmentManifest

```text
segment_id: string
created_at_utc: string
schema_version: string
event_file: string
artifact_files: list[string]
start_timestamp_ns: int
end_timestamp_ns: int
event_count: int
```

## Determinism rule

Replaying the same segment with the same model/config must produce byte-identical output artifacts unless the config explicitly allows nondeterminism.
