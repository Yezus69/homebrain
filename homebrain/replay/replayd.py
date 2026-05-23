from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from homebrain.brain.modeld import replay_events_with_dummy_model
from homebrain.messages.schema import Event
from homebrain.replay.segment_log import load_manifest, read_events, write_segment


def order_events_for_replay(events: list[Event]) -> list[Event]:
    return [
        event
        for _index, event in sorted(
            enumerate(events),
            key=lambda indexed: (indexed[1].timestamp_ns, indexed[0]),
        )
    ]


def _copy_artifacts(source_dir: Path, out_dir: Path, artifact_files: list[str]) -> None:
    for relative in artifact_files:
        source_path = source_dir / relative
        target_path = out_dir / relative
        if not source_path.exists():
            raise FileNotFoundError(f"manifest artifact is missing: {source_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def replay_log(log_dir: str | Path, out_dir: str | Path) -> None:
    source_path = Path(log_dir)
    output_path = Path(out_dir)
    manifest = load_manifest(source_path)
    events = read_events(source_path)
    ordered_events = order_events_for_replay(events)
    replayed_events = replay_events_with_dummy_model(ordered_events)

    output_path.mkdir(parents=True, exist_ok=True)
    _copy_artifacts(source_path, output_path, manifest.artifact_files)
    write_segment(
        output_path,
        replayed_events,
        segment_id=f"{manifest.segment_id}-replay",
        artifact_files=list(manifest.artifact_files),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministically replay a HomeBrain segment.")
    parser.add_argument("--log", required=True, help="Input segment log directory.")
    parser.add_argument("--out", required=True, help="Output segment log directory.")
    args = parser.parse_args(argv)
    replay_log(args.log, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
