from __future__ import annotations

import argparse
from pathlib import Path

from homebrain.messages.schema import CommandEvent, Event, FrameEvent, ImuEvent, WheelEvent
from homebrain.replay.segment_log import write_segment

BASE_TIMESTAMP_NS = 1_700_000_000_000_000_000
FRAME_PERIOD_NS = 100_000_000
FRAME_COUNT = 6
SEQUENCE_ID = "dummy-route-000"


def _frame_bytes(frame_id: int) -> bytes:
    values: list[int] = []
    for pixel_index in range(4):
        values.extend(
            [
                (frame_id * 17 + pixel_index * 3) % 256,
                (frame_id * 17 + pixel_index * 5 + 1) % 256,
                (frame_id * 17 + pixel_index * 7 + 2) % 256,
            ]
        )
    return bytes(values)


def write_dummy_artifacts(out_dir: str | Path, frame_count: int = FRAME_COUNT) -> list[str]:
    path = Path(out_dir)
    frames_dir = path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    artifact_files: list[str] = []
    for frame_id in range(frame_count):
        relative_path = f"frames/frame_{frame_id:06d}.rgb"
        (path / relative_path).write_bytes(_frame_bytes(frame_id))
        artifact_files.append(relative_path)
    return artifact_files


def build_dummy_events(frame_count: int = FRAME_COUNT) -> list[Event]:
    events: list[Event] = []
    for frame_id in range(frame_count):
        tick = BASE_TIMESTAMP_NS + frame_id * FRAME_PERIOD_NS
        command_linear = 0.05 if frame_id < frame_count - 1 else 0.0
        events.append(
            CommandEvent(
                timestamp_ns=tick,
                sequence_id=SEQUENCE_ID,
                source="dummy_logger",
                linear_velocity_mps=command_linear,
                angular_velocity_radps=0.02 if frame_id % 2 else 0.0,
                command_id=f"cmd-{frame_id:06d}",
            )
        )
        events.append(
            FrameEvent(
                timestamp_ns=tick + 1_000_000,
                sequence_id=SEQUENCE_ID,
                source="dummy_camera",
                camera_id="front_rgb",
                frame_id=frame_id,
                width=2,
                height=2,
                format="rgb8",
                data_ref=f"frames/frame_{frame_id:06d}.rgb",
                intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5},
            )
        )
        events.append(
            ImuEvent(
                timestamp_ns=tick + 2_000_000,
                sequence_id=SEQUENCE_ID,
                source="dummy_imu",
                accel_mps2=(0.0, 0.0, 9.81),
                gyro_radps=(0.0, 0.0, 0.01 * frame_id),
            )
        )
        events.append(
            WheelEvent(
                timestamp_ns=tick + 3_000_000,
                sequence_id=SEQUENCE_ID,
                source="dummy_wheels",
                left_ticks=frame_id * 12,
                right_ticks=frame_id * 12 + (frame_id % 2),
                left_velocity=command_linear,
                right_velocity=command_linear,
                odom_delta=(0.005 * frame_id, 0.0, 0.001 * frame_id),
            )
        )
    return events


def generate_dummy_log(out_dir: str | Path) -> None:
    path = Path(out_dir)
    artifact_files = write_dummy_artifacts(path)
    events = build_dummy_events()
    write_segment(path, events, segment_id=path.name, artifact_files=artifact_files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a deterministic dummy HomeBrain log.")
    parser.add_argument("--out", required=True, help="Output segment log directory.")
    args = parser.parse_args(argv)
    generate_dummy_log(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
