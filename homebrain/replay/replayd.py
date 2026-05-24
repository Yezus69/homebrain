from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from homebrain.brain.modeld import replay_events_with_dummy_model, replay_events_with_spatial_model
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


def replay_log(
    log_dir: str | Path,
    out_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    feature_dir: str | Path | None = None,
    trajectory_scorer_checkpoint: str | Path | None = None,
    device_name: str | None = None,
    v1_pose_warp_source: str = "route_pose",
) -> None:
    source_path = Path(log_dir)
    output_path = Path(out_dir)
    manifest = load_manifest(source_path)
    events = read_events(source_path)
    ordered_events = order_events_for_replay(events)

    output_path.mkdir(parents=True, exist_ok=True)
    _copy_artifacts(source_path, output_path, manifest.artifact_files)
    if checkpoint is not None:
        replayed_events, model_artifacts = replay_events_with_spatial_model(
            ordered_events,
            log_dir=source_path,
            out_dir=output_path,
            checkpoint=checkpoint,
            feature_dir=feature_dir,
            trajectory_scorer_checkpoint=trajectory_scorer_checkpoint,
            device_name=device_name,
            v1_pose_warp_source=v1_pose_warp_source,
        )
    else:
        replayed_events = replay_events_with_dummy_model(ordered_events)
        model_artifacts = []
    write_segment(
        output_path,
        replayed_events,
        segment_id=f"{manifest.segment_id}-replay",
        artifact_files=[*manifest.artifact_files, *model_artifacts],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministically replay a HomeBrain segment.")
    parser.add_argument("--log", required=True, help="Input segment log directory.")
    parser.add_argument("--out", required=True, help="Output segment log directory.")
    parser.add_argument("--checkpoint", default=None, help="Optional SpatialMemoryNet v0 checkpoint.")
    parser.add_argument("--features", default=None, help="Optional DINO feature artifact directory.")
    parser.add_argument("--trajectory-scorer-checkpoint", default=None, help="Optional TrajectoryScorerNet v0 checkpoint.")
    parser.add_argument(
        "--v1-pose-warp-source",
        choices=("route_pose", "predicted_pose", "none"),
        default="route_pose",
        help="Pose source for SpatialMemoryNet v1 memory warp; predicted_pose is an explicit ablation.",
    )
    parser.add_argument("--device", default=None, help="Optional torch device for checkpoint inference.")
    args = parser.parse_args(argv)
    replay_log(
        args.log,
        args.out,
        checkpoint=args.checkpoint,
        feature_dir=args.features,
        trajectory_scorer_checkpoint=args.trajectory_scorer_checkpoint,
        device_name=args.device,
        v1_pose_warp_source=args.v1_pose_warp_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
