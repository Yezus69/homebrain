from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from homebrain.messages.schema import BrainOutputEvent, Event, FrameEvent, event_identity
from homebrain.replay.segment_log import load_manifest, read_events, write_segment

DUMMY_MODELD_SOURCE = "modeld_dummy_v0"


def dummy_brain_output_for_frame(frame: FrameEvent) -> BrainOutputEvent:
    input_id = event_identity(frame)
    stop_candidate = {
        "trajectory_id": "dummy_stop",
        "linear_velocity_mps": 0.0,
        "angular_velocity_radps": 0.0,
        "duration_sec": 0.5,
        "score": 0.0,
        "mock": True,
    }
    forward_candidate = {
        "trajectory_id": "dummy_forward_slow",
        "linear_velocity_mps": 0.05,
        "angular_velocity_radps": 0.0,
        "duration_sec": 0.5,
        "score": -1.0,
        "mock": True,
    }
    return BrainOutputEvent(
        timestamp_ns=frame.timestamp_ns,
        sequence_id=frame.sequence_id,
        source=DUMMY_MODELD_SOURCE,
        input_event_ids=[input_id],
        pose_delta=(0.0, 0.0, 0.0),
        pose_confidence=0.0,
        candidate_trajectories=[stop_candidate, forward_candidate],
        selected_trajectory_id="dummy_stop",
        cmd_vel=(0.0, 0.0),
        uncertainty=1.0,
        stop_reason="dummy_model_no_real_perception",
        debug={
            "mock": True,
            "model": DUMMY_MODELD_SOURCE,
            "note": "Deterministic placeholder; not model performance.",
            "input_frame_id": frame.frame_id,
        },
    )


def dummy_model_outputs(events: Iterable[Event]) -> list[BrainOutputEvent]:
    return [dummy_brain_output_for_frame(event) for event in events if isinstance(event, FrameEvent)]


def replay_events_with_dummy_model(events: Iterable[Event]) -> list[Event]:
    replayed: list[Event] = []
    for event in events:
        replayed.append(event)
        if isinstance(event, FrameEvent):
            replayed.append(dummy_brain_output_for_frame(event))
    return replayed


def write_dummy_model_outputs(log_dir: str | Path, out_dir: str | Path) -> None:
    manifest = load_manifest(log_dir)
    events = read_events(log_dir)
    outputs = dummy_model_outputs(events)
    write_segment(
        out_dir,
        outputs,
        segment_id=f"{manifest.segment_id}-dummy-model",
        artifact_files=[],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic dummy modeld.")
    parser.add_argument("--log", required=True, help="Input segment log directory.")
    parser.add_argument("--out", required=True, help="Output segment log directory.")
    args = parser.parse_args(argv)
    write_dummy_model_outputs(args.log, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
