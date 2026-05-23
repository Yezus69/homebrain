from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from homebrain.brain.modeld import replay_events_with_dummy_model
from homebrain.messages.schema import BrainOutputEvent, Event, FrameEvent
from homebrain.replay.replayd import order_events_for_replay
from homebrain.replay.segment_log import canonical_event_text, read_events
from homebrain.teachers.artifacts import validate_teacher_artifacts


def count_event_ordering_errors(events: list[Event]) -> int:
    errors = 0
    previous_timestamp: int | None = None
    for event in events:
        if previous_timestamp is not None and event.timestamp_ns < previous_timestamp:
            errors += 1
        previous_timestamp = event.timestamp_ns
    return errors


def count_dropped_frames(events: list[Event]) -> int:
    frame_ids_by_stream: dict[tuple[str, str], list[int]] = {}
    for event in events:
        if isinstance(event, FrameEvent):
            key = (event.sequence_id, event.camera_id)
            frame_ids_by_stream.setdefault(key, []).append(event.frame_id)

    dropped = 0
    for frame_ids in frame_ids_by_stream.values():
        ordered_ids = sorted(frame_ids)
        previous: int | None = None
        for frame_id in ordered_ids:
            if previous is not None and frame_id > previous + 1:
                dropped += frame_id - previous - 1
            previous = frame_id
    return dropped


def evaluate_log(
    log_dir: str | Path,
    *,
    teacher_artifacts: str | Path | None = None,
    eval_runtime_sec: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    events = read_events(log_dir)
    ordered_events = order_events_for_replay(events)
    replay_a = replay_events_with_dummy_model(ordered_events)
    replay_b = replay_events_with_dummy_model(ordered_events)
    deterministic = canonical_event_text(replay_a) == canonical_event_text(replay_b)
    elapsed = time.perf_counter() - started if eval_runtime_sec is None else eval_runtime_sec

    metrics = {
        "event_count": len(events),
        "frame_count": sum(1 for event in events if isinstance(event, FrameEvent)),
        "dropped_frame_count": count_dropped_frames(events),
        "event_ordering_error_count": count_event_ordering_errors(events),
        "replay_determinism_pass": deterministic,
        "brain_output_count": sum(1 for event in replay_a if isinstance(event, BrainOutputEvent)),
        "eval_runtime_sec": round(elapsed, 6),
    }
    if teacher_artifacts is not None:
        frames = [event for event in ordered_events if isinstance(event, FrameEvent)]
        validation = validate_teacher_artifacts(teacher_artifacts, frames=frames)
        metrics.update(validation.to_metrics())
    return metrics


def write_eval_metrics(metrics: dict[str, Any], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metrics, handle, sort_keys=True, indent=2)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run HomeBrain Goal 0 eval metrics.")
    parser.add_argument("--log", required=True, help="Input segment log directory.")
    parser.add_argument(
        "--teacher-artifacts",
        default=None,
        help="Optional teacher artifact directory to validate alongside the log.",
    )
    parser.add_argument("--out", required=True, help="Output metrics JSON path.")
    args = parser.parse_args(argv)
    metrics = evaluate_log(args.log, teacher_artifacts=args.teacher_artifacts)
    write_eval_metrics(metrics, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
