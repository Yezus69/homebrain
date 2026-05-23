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

Goal 2 extends imported image-frame formats to include:

```text
encoded_png
encoded_pgm
encoded_ppm
```

For image-sequence imports, `intrinsics` must explicitly mark calibration as missing instead of implying a real calibration exists.

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

## Route source metadata

Image-sequence imports write `route_metadata.json` as a manifest artifact:

```text
schema_version: "homebrain.route_source.v0"
source_type: "image_sequence"
source_path: string
camera_name: string
fps: number
effective_fps: number
stride: int
max_frames: optional int
source_frame_count: int
selected_frame_count: int
frame_count: int
imported_frame_count: int
width: optional int
height: optional int
dimensions_consistent: bool
source_frame_interval_ns: int
expected_timestamp_interval_ns: int
has_imu: false
has_wheel_odometry: false
has_commands: false
has_intrinsics: false
user_owned_or_license_unknown: bool
image_load_error_count: int
image_load_errors: list[dict]
missing_sensor_notices: list[dict]
frames: list[dict]
```

This metadata records absent sensors as unavailable. It must not be replaced by fake IMU, wheel odometry, or command events.
