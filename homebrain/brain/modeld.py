from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from homebrain.brain.spatial_memory_v0 import SPATIAL_MEMORY_V0_SOURCE, load_checkpoint
from homebrain.data.spatial_dataset import write_deterministic_npz
from homebrain.messages.schema import BrainOutputEvent, Event, FrameEvent, event_identity
from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.policies.trajectory_scorer import CoverageMemory, LocalBev
from homebrain.policies.trajectory_scorer_net_v0 import (
    TRAJECTORY_SCORER_V0_SOURCE,
    candidate_score_records,
    load_trajectory_scorer_checkpoint,
    score_local_bev_with_model,
    scorer_checkpoint_hash,
)
from homebrain.replay.segment_log import load_manifest, read_events, write_segment
from homebrain.train.spatial_dataset import BEV_OUTPUT_CHANNELS, DINOFeatureStore

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


def replay_events_with_spatial_model(
    events: Iterable[Event],
    *,
    log_dir: str | Path,
    out_dir: str | Path,
    checkpoint: str | Path,
    feature_dir: str | Path | None = None,
    trajectory_scorer_checkpoint: str | Path | None = None,
    device_name: str | None = None,
) -> tuple[list[Event], list[str]]:
    ordered_events = list(events)
    outputs, artifacts = spatial_model_outputs(
        ordered_events,
        log_dir=log_dir,
        out_dir=out_dir,
        checkpoint=checkpoint,
        feature_dir=feature_dir,
        trajectory_scorer_checkpoint=trajectory_scorer_checkpoint,
        device_name=device_name,
    )
    by_identity = {event.input_event_ids[0]: event for event in outputs}
    replayed: list[Event] = []
    for event in ordered_events:
        replayed.append(event)
        if isinstance(event, FrameEvent):
            output = by_identity.get(event_identity(event))
            if output is not None:
                replayed.append(output)
    return replayed, artifacts


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


def write_spatial_model_outputs(
    log_dir: str | Path,
    out_dir: str | Path,
    *,
    checkpoint: str | Path,
    feature_dir: str | Path | None = None,
    trajectory_scorer_checkpoint: str | Path | None = None,
    device_name: str | None = None,
) -> None:
    manifest = load_manifest(log_dir)
    events = read_events(log_dir)
    outputs, artifacts = spatial_model_outputs(
        events,
        log_dir=log_dir,
        out_dir=out_dir,
        checkpoint=checkpoint,
        feature_dir=feature_dir,
        trajectory_scorer_checkpoint=trajectory_scorer_checkpoint,
        device_name=device_name,
    )
    write_segment(
        out_dir,
        outputs,
        segment_id=f"{manifest.segment_id}-spatial-v0-model",
        artifact_files=artifacts,
    )


def spatial_model_outputs(
    events: Iterable[Event],
    *,
    log_dir: str | Path,
    out_dir: str | Path,
    checkpoint: str | Path,
    feature_dir: str | Path | None = None,
    trajectory_scorer_checkpoint: str | Path | None = None,
    device_name: str | None = None,
) -> tuple[list[BrainOutputEvent], list[str]]:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model.to(device)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    resolved_feature_dir = feature_dir or metadata.get("feature_artifacts")
    if not isinstance(resolved_feature_dir, (str, Path)):
        raise ValueError("DINO feature artifacts must be supplied with --features or checkpoint metadata")
    feature_store = DINOFeatureStore(resolved_feature_dir)
    trajectory_scorer: torch.nn.Module | None = None
    trajectory_scorer_metadata: dict[str, object] = {}
    trajectory_candidates = None
    trajectory_coverage: CoverageMemory | None = None
    if trajectory_scorer_checkpoint is not None:
        trajectory_scorer, trajectory_payload = load_trajectory_scorer_checkpoint(
            trajectory_scorer_checkpoint,
            map_location=device,
        )
        trajectory_scorer.to(device)
        trajectory_scorer_metadata = (
            trajectory_payload.get("metadata") if isinstance(trajectory_payload.get("metadata"), dict) else {}
        )
        meters_per_cell = _metadata_float(trajectory_scorer_metadata, "meters_per_cell", 0.05)
        robot_radius_m = _metadata_float(trajectory_scorer_metadata, "robot_radius_m", 0.18)
        trajectory_candidates = generate_default_candidates(
            grid_shape=model.config.bev_shape,
            meters_per_cell=meters_per_cell,
            robot_radius_m=robot_radius_m,
        )
        trajectory_coverage = CoverageMemory(model.config.bev_shape, meters_per_cell=meters_per_cell)
    output_root = Path(out_dir)
    frames = [event for event in events if isinstance(event, FrameEvent)]
    start_timestamp_ns = min((frame.timestamp_ns for frame in frames), default=0)
    brain_outputs: list[BrainOutputEvent] = []
    artifact_files: list[str] = []
    for frame in frames:
        output, artifact = _spatial_output_for_frame(
            model=model,
            frame=frame,
            feature_store=feature_store,
            output_root=output_root,
            checkpoint=Path(checkpoint),
            feature_dir=Path(resolved_feature_dir),
            trajectory_scorer=trajectory_scorer,
            trajectory_scorer_checkpoint=Path(trajectory_scorer_checkpoint)
            if trajectory_scorer_checkpoint is not None
            else None,
            trajectory_scorer_metadata=trajectory_scorer_metadata,
            trajectory_candidates=trajectory_candidates,
            trajectory_coverage=trajectory_coverage,
            start_timestamp_ns=start_timestamp_ns,
            device=device,
        )
        brain_outputs.append(output)
        artifact_files.append(artifact)
    return brain_outputs, artifact_files


def _spatial_output_for_frame(
    *,
    model: torch.nn.Module,
    frame: FrameEvent,
    feature_store: DINOFeatureStore,
    output_root: Path,
    checkpoint: Path,
    feature_dir: Path,
    trajectory_scorer: torch.nn.Module | None,
    trajectory_scorer_checkpoint: Path | None,
    trajectory_scorer_metadata: dict[str, object],
    trajectory_candidates: list | None,
    trajectory_coverage: CoverageMemory | None,
    start_timestamp_ns: int,
    device: torch.device,
) -> tuple[BrainOutputEvent, str]:
    patch_features, _cls_feature = feature_store.load_for_frame(frame)
    features = torch.from_numpy(np.transpose(patch_features, (2, 0, 1))[None, ...].astype(np.float32)).to(device)
    timestamp_s = torch.tensor(
        [[(frame.timestamp_ns - start_timestamp_ns) / 1_000_000_000.0]],
        dtype=torch.float32,
        device=device,
    )
    sensor_context_dim = max(0, int(getattr(model.config, "sensor_dim", 5)) - 1)
    sensor_mask = torch.zeros((1, sensor_context_dim), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        outputs = model(features, timestamp_s, sensor_mask)
    logits = outputs["bev_logits"][0].detach().cpu().numpy().astype(np.float32)
    probabilities = torch.sigmoid(outputs["bev_logits"])[0].detach().cpu().numpy().astype(np.float32)
    uncertainty_grid = torch.sigmoid(outputs["uncertainty_logits"])[0, 0].detach().cpu().numpy().astype(np.float32)
    pose_delta = tuple(float(value) for value in outputs["pose_delta"][0].detach().cpu().numpy())
    uncertainty_scalar = float(outputs["uncertainty_scalar"][0].detach().cpu())

    relative_artifact = f"brain_outputs/spatial_v0/{frame.camera_id}_{frame.frame_id:06d}.npz"
    artifact_path = output_root / relative_artifact
    arrays = {
        "bev_logits": logits,
        "bev_free_prob": probabilities[0].astype(np.float32),
        "bev_occupied_prob": probabilities[1].astype(np.float32),
        "bev_unknown_prob": probabilities[2].astype(np.float32),
        "bev_traversable_prob": probabilities[3].astype(np.float32),
        "bev_risky_prob": probabilities[4].astype(np.float32),
        "uncertainty_grid": uncertainty_grid,
    }
    local_bev = LocalBev(
        free=arrays["bev_free_prob"],
        occupied=arrays["bev_occupied_prob"],
        unknown=arrays["bev_unknown_prob"],
        traversable=arrays["bev_traversable_prob"],
        risky=arrays["bev_risky_prob"],
        confidence=(np.float32(1.0) - uncertainty_grid).astype(np.float32),
        uncertainty=uncertainty_grid,
        source="model",
    )
    candidate_trajectories = None
    selected_trajectory_id = None
    trajectory_debug: dict[str, object] = {
        "trajectory_scoring": False,
    }
    if trajectory_scorer is not None:
        if trajectory_candidates is None or trajectory_coverage is None or trajectory_scorer_checkpoint is None:
            raise ValueError("trajectory scorer runtime is incomplete")
        trajectory_coverage.align_with_pose_delta(pose_delta)
        trajectory_logits, selected_trajectory_id, trajectory_features = score_local_bev_with_model(
            model=trajectory_scorer,  # type: ignore[arg-type]
            bev=local_bev,
            candidates=trajectory_candidates,
            coverage_memory=trajectory_coverage,
            device=device,
        )
        trajectory_coverage.update_current_frame(local_bev)
        arrays["trajectory_logits"] = trajectory_logits.astype(np.float32)
        arrays["trajectory_selected_index"] = np.asarray(
            [next(index for index, candidate in enumerate(trajectory_candidates) if candidate.id == selected_trajectory_id)],
            dtype=np.int64,
        )
        candidate_trajectories = candidate_score_records(
            candidates=trajectory_candidates,
            logits=trajectory_logits,
            candidate_features=trajectory_features,
            selected_candidate_id=selected_trajectory_id,
        )
        trajectory_debug = {
            "trajectory_scoring": True,
            "trajectory_scorer": TRAJECTORY_SCORER_V0_SOURCE,
            "trajectory_scorer_checkpoint": trajectory_scorer_checkpoint.as_posix(),
            "trajectory_scorer_checkpoint_sha256": scorer_checkpoint_hash(trajectory_scorer_checkpoint),
            "trajectory_scorer_metadata": trajectory_scorer_metadata,
            "selected_candidate_id": selected_trajectory_id,
            "coverage_memory": trajectory_coverage.to_dict(),
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "product_training_approved": False,
        }
    write_deterministic_npz(artifact_path, arrays)
    input_id = event_identity(frame)
    return (
        BrainOutputEvent(
            timestamp_ns=frame.timestamp_ns,
            sequence_id=frame.sequence_id,
            source=SPATIAL_MEMORY_V0_SOURCE,
            input_event_ids=[input_id],
            pose_delta=pose_delta,  # type: ignore[arg-type]
            pose_confidence=max(0.0, min(1.0, 1.0 - uncertainty_scalar)),
            local_bev_ref=relative_artifact,
            candidate_trajectories=candidate_trajectories,
            selected_trajectory_id=selected_trajectory_id,
            cmd_vel=None,
            uncertainty=uncertainty_scalar,
            stop_reason="replay_only_learned_trajectory_scorer_not_executed"
            if selected_trajectory_id is not None
            else "representation_pretraining_only_no_control",
            debug={
                "mock": False,
                "model": SPATIAL_MEMORY_V0_SOURCE,
                "checkpoint": checkpoint.as_posix(),
                "feature_artifacts": feature_dir.as_posix(),
                "camera_id": frame.camera_id,
                "output_channels": list(BEV_OUTPUT_CHANNELS),
                "local_bev_shape": list(probabilities.shape[1:]),
                "representation_pretraining_only": True,
                "control_safe": False,
                "replay_only": True,
                "not_executed": True,
                "product_training_approved": False,
                "input_frame_id": frame.frame_id,
                **trajectory_debug,
            },
        ),
        relative_artifact,
    )


def _metadata_float(metadata: dict[str, object], key: str, default: float) -> float:
    value = metadata.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if np.isfinite(number) and number > 0.0:
            return number
    return default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic dummy modeld.")
    parser.add_argument("--log", required=True, help="Input segment log directory.")
    parser.add_argument("--out", required=True, help="Output segment log directory.")
    parser.add_argument("--checkpoint", default=None, help="Optional SpatialMemoryNet v0 checkpoint.")
    parser.add_argument("--features", default=None, help="Optional DINO feature artifact directory.")
    parser.add_argument("--trajectory-scorer-checkpoint", default=None, help="Optional TrajectoryScorerNet v0 checkpoint.")
    parser.add_argument("--device", default=None, help="Optional torch device for checkpoint inference.")
    args = parser.parse_args(argv)
    if args.checkpoint:
        write_spatial_model_outputs(
            args.log,
            args.out,
            checkpoint=args.checkpoint,
            feature_dir=args.features,
            trajectory_scorer_checkpoint=args.trajectory_scorer_checkpoint,
            device_name=args.device,
        )
    else:
        write_dummy_model_outputs(args.log, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
