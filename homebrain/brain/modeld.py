from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import atan2, cos, sin
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from homebrain.brain.spatial_memory_v0 import SPATIAL_MEMORY_V0_SOURCE, load_checkpoint
from homebrain.brain.spatial_memory_v1 import (
    SPATIAL_MEMORY_V1_SOURCE,
    SpatialMemoryState,
    load_checkpoint as load_v1_checkpoint,
)
from homebrain.data.spatial_dataset import write_deterministic_npz
from homebrain.messages.schema import BrainOutputEvent, Event, FrameEvent, OdomEvent, PoseEvent, event_identity
from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.policies.trajectory_scorer import CoverageMemory, LocalBev
from homebrain.policies.runtime_decision import (
    RuntimeFutureRolloutScorer,
    RuntimeTrajectoryScorer,
    decide_trajectory,
    load_runtime_future_rollout_scorer,
    load_runtime_trajectory_scorer,
    metadata_float,
)
from homebrain.replay.segment_log import load_manifest, read_events, write_segment
from homebrain.train.spatial_dataset import BEV_OUTPUT_CHANNELS, DINOFeatureStore

DUMMY_MODELD_SOURCE = "modeld_dummy_v0"
V1_POSE_WARP_SOURCES = ("route_pose", "predicted_pose", "none")
V1_POLICY_BEV_SOURCES = ("current", "memory")


@dataclass(frozen=True)
class RoutePoseDelta:
    delta: tuple[float, float, float]
    source: str


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
    future_rollout_checkpoint: str | Path | None = None,
    device_name: str | None = None,
    v1_pose_warp_source: str = "route_pose",
    v1_policy_bev_source: str = "memory",
) -> tuple[list[Event], list[str]]:
    ordered_events = list(events)
    outputs, artifacts = spatial_model_outputs(
        ordered_events,
        log_dir=log_dir,
        out_dir=out_dir,
        checkpoint=checkpoint,
        feature_dir=feature_dir,
        trajectory_scorer_checkpoint=trajectory_scorer_checkpoint,
        future_rollout_checkpoint=future_rollout_checkpoint,
        device_name=device_name,
        v1_pose_warp_source=v1_pose_warp_source,
        v1_policy_bev_source=v1_policy_bev_source,
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
    future_rollout_checkpoint: str | Path | None = None,
    device_name: str | None = None,
    v1_pose_warp_source: str = "route_pose",
    v1_policy_bev_source: str = "memory",
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
        future_rollout_checkpoint=future_rollout_checkpoint,
        device_name=device_name,
        v1_pose_warp_source=v1_pose_warp_source,
        v1_policy_bev_source=v1_policy_bev_source,
    )
    suffix = "spatial-v1-model" if outputs and outputs[0].source == SPATIAL_MEMORY_V1_SOURCE else "spatial-v0-model"
    write_segment(
        out_dir,
        outputs,
        segment_id=f"{manifest.segment_id}-{suffix}",
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
    future_rollout_checkpoint: str | Path | None = None,
    device_name: str | None = None,
    v1_pose_warp_source: str = "route_pose",
    v1_policy_bev_source: str = "memory",
) -> tuple[list[BrainOutputEvent], list[str]]:
    if trajectory_scorer_checkpoint is not None and future_rollout_checkpoint is not None:
        raise ValueError("use either --trajectory-scorer-checkpoint or --future-rollout-checkpoint, not both")
    if _checkpoint_model_name(checkpoint) == "SpatialMemoryNetV1":
        return _spatial_v1_model_outputs(
            events,
            out_dir=out_dir,
            checkpoint=checkpoint,
            feature_dir=feature_dir,
            trajectory_scorer_checkpoint=trajectory_scorer_checkpoint,
            future_rollout_checkpoint=future_rollout_checkpoint,
            device_name=device_name,
            pose_warp_source=v1_pose_warp_source,
            policy_bev_source=v1_policy_bev_source,
        )

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model.to(device)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    resolved_feature_dir = feature_dir or metadata.get("feature_artifacts")
    if not isinstance(resolved_feature_dir, (str, Path)):
        raise ValueError("DINO feature artifacts must be supplied with --features or checkpoint metadata")
    feature_store = DINOFeatureStore(resolved_feature_dir)
    trajectory_scorer: RuntimeTrajectoryScorer | None = None
    future_rollout_scorer: RuntimeFutureRolloutScorer | None = None
    trajectory_candidates = None
    trajectory_coverage: CoverageMemory | None = None
    if trajectory_scorer_checkpoint is not None or future_rollout_checkpoint is not None:
        if trajectory_scorer_checkpoint is not None:
            trajectory_scorer = load_runtime_trajectory_scorer(trajectory_scorer_checkpoint, device=device)
            trajectory_metadata = trajectory_scorer.metadata
        else:
            future_rollout_scorer = load_runtime_future_rollout_scorer(future_rollout_checkpoint, device=device)  # type: ignore[arg-type]
            trajectory_metadata = future_rollout_scorer.metadata
        meters_per_cell = metadata_float(trajectory_metadata, "meters_per_cell", 0.05)
        robot_radius_m = metadata_float(trajectory_metadata, "robot_radius_m", 0.18)
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
            future_rollout_scorer=future_rollout_scorer,
            trajectory_candidates=trajectory_candidates,
            trajectory_coverage=trajectory_coverage,
            start_timestamp_ns=start_timestamp_ns,
            device=device,
        )
        brain_outputs.append(output)
        artifact_files.append(artifact)
    return brain_outputs, artifact_files


def _spatial_v1_model_outputs(
    events: Iterable[Event],
    *,
    out_dir: str | Path,
    checkpoint: str | Path,
    feature_dir: str | Path | None = None,
    trajectory_scorer_checkpoint: str | Path | None = None,
    future_rollout_checkpoint: str | Path | None = None,
    device_name: str | None = None,
    pose_warp_source: str = "route_pose",
    policy_bev_source: str = "memory",
) -> tuple[list[BrainOutputEvent], list[str]]:
    if pose_warp_source not in V1_POSE_WARP_SOURCES:
        raise ValueError(f"v1_pose_warp_source must be one of {V1_POSE_WARP_SOURCES}")
    if policy_bev_source not in V1_POLICY_BEV_SOURCES:
        raise ValueError(f"v1_policy_bev_source must be one of {V1_POLICY_BEV_SOURCES}")
    ordered_events = list(events)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_v1_checkpoint(checkpoint, map_location=device)
    model.to(device)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    resolved_feature_dir = feature_dir or metadata.get("feature_artifacts")
    if not isinstance(resolved_feature_dir, (str, Path)):
        raise ValueError("DINO feature artifacts must be supplied with --features or checkpoint metadata")
    feature_store = DINOFeatureStore(resolved_feature_dir)
    output_root = Path(out_dir)
    frames = [event for event in ordered_events if isinstance(event, FrameEvent)]
    start_timestamp_ns = min((frame.timestamp_ns for frame in frames), default=0)
    brain_outputs: list[BrainOutputEvent] = []
    artifact_files: list[str] = []
    trajectory_scorer = (
        load_runtime_trajectory_scorer(trajectory_scorer_checkpoint, device=device)
        if trajectory_scorer_checkpoint is not None
        else None
    )
    future_rollout_scorer = (
        load_runtime_future_rollout_scorer(future_rollout_checkpoint, device=device)
        if future_rollout_checkpoint is not None
        else None
    )
    trajectory_metadata = (
        future_rollout_scorer.metadata
        if future_rollout_scorer is not None
        else trajectory_scorer.metadata
        if trajectory_scorer is not None
        else metadata
    )
    meters_per_cell = metadata_float(trajectory_metadata, "meters_per_cell", metadata_float(metadata, "meters_per_cell", 0.05))
    robot_radius_m = metadata_float(trajectory_metadata, "robot_radius_m", metadata_float(metadata, "robot_radius_m", 0.18))
    trajectory_candidates = generate_default_candidates(
        grid_shape=model.config.bev_shape,
        meters_per_cell=meters_per_cell,
        robot_radius_m=robot_radius_m,
    )
    trajectory_coverage = CoverageMemory(model.config.bev_shape, meters_per_cell=meters_per_cell)
    state: SpatialMemoryState | None = None
    previous_predicted_pose_delta: torch.Tensor | None = None
    previous_sequence_camera: tuple[str, str] | None = None
    route_pose_deltas = _route_pose_deltas(frames, ordered_events)
    for frame in frames:
        current_key = (frame.sequence_id, frame.camera_id)
        reset = previous_sequence_camera is None or current_key != previous_sequence_camera
        if reset:
            trajectory_coverage = CoverageMemory(model.config.bev_shape, meters_per_cell=meters_per_cell)
        selected_pose_delta: torch.Tensor | None = None
        selected_pose_source = pose_warp_source
        route_pose_available = False
        if not reset and pose_warp_source == "route_pose":
            route_delta = route_pose_deltas.get(event_identity(frame))
            if route_delta is not None:
                selected_pose_delta = torch.tensor(route_delta.delta, dtype=torch.float32)
                selected_pose_source = route_delta.source
                route_pose_available = True
            else:
                selected_pose_source = "route_pose_missing"
        elif not reset and pose_warp_source == "predicted_pose":
            selected_pose_delta = previous_predicted_pose_delta
            route_pose_available = selected_pose_delta is not None
            selected_pose_source = "predicted_pose" if selected_pose_delta is not None else "predicted_pose_missing"
        elif reset:
            selected_pose_source = "sequence_reset"
        else:
            selected_pose_source = "none"
        output, artifact, state, previous_pose_delta = _spatial_v1_output_for_frame(
            model=model,
            frame=frame,
            feature_store=feature_store,
            output_root=output_root,
            checkpoint=Path(checkpoint),
            feature_dir=Path(resolved_feature_dir),
            start_timestamp_ns=start_timestamp_ns,
            pose_delta_to_current=selected_pose_delta,
            pose_warp_source=selected_pose_source,
            pose_warp_source_requested=pose_warp_source,
            route_pose_delta_available=route_pose_available,
            predicted_pose_warp_ablation=pose_warp_source == "predicted_pose",
            reset_memory=reset,
            state=None if reset else state,
            trajectory_scorer=trajectory_scorer,
            future_rollout_scorer=future_rollout_scorer,
            trajectory_candidates=trajectory_candidates,
            trajectory_coverage=trajectory_coverage,
            policy_bev_source=policy_bev_source,
            device=device,
        )
        previous_predicted_pose_delta = previous_pose_delta
        previous_sequence_camera = current_key
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
    trajectory_scorer: RuntimeTrajectoryScorer | None,
    future_rollout_scorer: RuntimeFutureRolloutScorer | None,
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
    if trajectory_scorer is not None or future_rollout_scorer is not None:
        if trajectory_candidates is None or trajectory_coverage is None:
            raise ValueError("trajectory scorer runtime is incomplete")
        decision = decide_trajectory(
            bev=local_bev,
            candidates=trajectory_candidates,
            coverage_memory=trajectory_coverage,
            pose_delta=pose_delta,  # type: ignore[arg-type]
            learned_scorer=trajectory_scorer,
            future_rollout_scorer=future_rollout_scorer,
            patch_features=patch_features,
            sensor_mask=sensor_mask.detach().cpu().numpy()[0],
            policy_bev_source="model",
        )
        arrays.update(decision.artifact_arrays)
        candidate_trajectories = decision.candidate_trajectories
        selected_trajectory_id = decision.selected_candidate_id
        trajectory_debug = decision.debug
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


def _spatial_v1_output_for_frame(
    *,
    model: torch.nn.Module,
    frame: FrameEvent,
    feature_store: DINOFeatureStore,
    output_root: Path,
    checkpoint: Path,
    feature_dir: Path,
    start_timestamp_ns: int,
    pose_delta_to_current: torch.Tensor | None,
    pose_warp_source: str,
    pose_warp_source_requested: str,
    route_pose_delta_available: bool,
    predicted_pose_warp_ablation: bool,
    reset_memory: bool,
    state: SpatialMemoryState | None,
    trajectory_scorer: RuntimeTrajectoryScorer | None,
    future_rollout_scorer: RuntimeFutureRolloutScorer | None,
    trajectory_candidates: list,
    trajectory_coverage: CoverageMemory,
    policy_bev_source: str,
    device: torch.device,
) -> tuple[BrainOutputEvent, str, SpatialMemoryState, torch.Tensor]:
    patch_features, _cls_feature = feature_store.load_for_frame(frame)
    features = torch.from_numpy(np.transpose(patch_features, (2, 0, 1))[None, ...].astype(np.float32)).to(device)
    timestamp_s = torch.tensor(
        [[(frame.timestamp_ns - start_timestamp_ns) / 1_000_000_000.0]],
        dtype=torch.float32,
        device=device,
    )
    sensor_context_dim = max(0, int(getattr(model.config, "sensor_dim", 5)) - 1)
    sensor_mask = torch.zeros((1, sensor_context_dim), dtype=torch.float32, device=device)
    pose_to_current = pose_delta_to_current.view(1, 3).to(device) if pose_delta_to_current is not None else None
    pose_mask = torch.ones((1, 1), dtype=torch.float32, device=device) if pose_delta_to_current is not None else None
    model.eval()
    with torch.no_grad():
        outputs = model.step(
            features,
            timestamp_s,
            sensor_mask,
            memory_state=state,
            pose_delta_to_current=pose_to_current,
            pose_delta_to_current_mask=pose_mask,
            force_reset=torch.tensor([reset_memory], dtype=torch.bool, device=device),
        )
    current_logits = outputs["current_bev_logits"][0].detach().cpu().numpy().astype(np.float32)
    memory_logits = outputs["fused_memory_bev_logits"][0].detach().cpu().numpy().astype(np.float32)
    current_probabilities = torch.sigmoid(outputs["current_bev_logits"])[0].detach().cpu().numpy().astype(np.float32)
    memory_probabilities = torch.sigmoid(outputs["fused_memory_bev_logits"])[0].detach().cpu().numpy().astype(np.float32)
    uncertainty_grid = outputs["uncertainty_grid"][0, 0].detach().cpu().numpy().astype(np.float32)
    uncertainty_scalar = float(outputs["uncertainty_scalar"][0].detach().cpu())
    pose_delta = tuple(float(value) for value in outputs["pose_delta"][0].detach().cpu().numpy())
    next_pose_delta = outputs["pose_delta"][0].detach()
    next_state = outputs["memory_state"].detach()
    debug = outputs["debug"]
    memory_used = bool(debug["memory_used"][0].detach().cpu())
    pose_warp_used = bool(debug["pose_warp_used"][0].detach().cpu())
    pose_warp_valid = bool(debug["pose_warp_valid"][0].detach().cpu())
    memory_reset = bool(debug["memory_reset"][0].detach().cpu())
    update_mask_coverage = float(debug["update_mask_coverage"][0].detach().cpu())
    memory_overwrite_fraction = float(debug["memory_overwrite_fraction"][0].detach().cpu())
    observation_mask_source = str(debug.get("observation_mask_source", "unknown"))

    relative_artifact = f"brain_outputs/spatial_v1/{frame.camera_id}_{frame.frame_id:06d}.npz"
    artifact_path = output_root / relative_artifact
    arrays = {
        "current_bev_logits": current_logits,
        "memory_bev_logits": memory_logits,
        "current_bev_free_prob": current_probabilities[0].astype(np.float32),
        "current_bev_occupied_prob": current_probabilities[1].astype(np.float32),
        "current_bev_unknown_prob": current_probabilities[2].astype(np.float32),
        "current_bev_traversable_prob": current_probabilities[3].astype(np.float32),
        "current_bev_risky_prob": current_probabilities[4].astype(np.float32),
        "memory_bev_free_prob": memory_probabilities[0].astype(np.float32),
        "memory_bev_occupied_prob": memory_probabilities[1].astype(np.float32),
        "memory_bev_unknown_prob": memory_probabilities[2].astype(np.float32),
        "memory_bev_traversable_prob": memory_probabilities[3].astype(np.float32),
        "memory_bev_risky_prob": memory_probabilities[4].astype(np.float32),
        "uncertainty_grid": uncertainty_grid,
        "update_mask": outputs["update_mask"][0, 0].detach().cpu().numpy().astype(np.float32),
        "memory_observed_mask": next_state.observed_mask[0, 0].detach().cpu().numpy().astype(np.float32),
        "memory_used": np.asarray([memory_used], dtype=np.bool_),
        "pose_warp_used": np.asarray([pose_warp_used], dtype=np.bool_),
        "pose_warp_valid": np.asarray([pose_warp_valid], dtype=np.bool_),
        "memory_reset": np.asarray([memory_reset], dtype=np.bool_),
        "update_mask_coverage": np.asarray([update_mask_coverage], dtype=np.float32),
        "memory_overwrite_fraction": np.asarray([memory_overwrite_fraction], dtype=np.float32),
    }
    policy_probabilities = memory_probabilities if policy_bev_source == "memory" else current_probabilities
    local_bev = LocalBev(
        free=policy_probabilities[0].astype(np.float32),
        occupied=policy_probabilities[1].astype(np.float32),
        unknown=policy_probabilities[2].astype(np.float32),
        traversable=policy_probabilities[3].astype(np.float32),
        risky=policy_probabilities[4].astype(np.float32),
        confidence=(np.float32(1.0) - uncertainty_grid).astype(np.float32),
        uncertainty=uncertainty_grid,
        source=f"v1_{policy_bev_source}_bev",
    )
    coverage_pose_delta = (
        tuple(float(value) for value in pose_delta_to_current.detach().cpu().numpy())
        if pose_delta_to_current is not None
        else None
    )
    decision = decide_trajectory(
        bev=local_bev,
        candidates=trajectory_candidates,
        coverage_memory=trajectory_coverage,
        pose_delta=coverage_pose_delta,  # type: ignore[arg-type]
        learned_scorer=trajectory_scorer,
        future_rollout_scorer=future_rollout_scorer,
        patch_features=patch_features,
        sensor_mask=sensor_mask.detach().cpu().numpy()[0],
        policy_bev_source=policy_bev_source,
        coverage_memory_reset=reset_memory,
    )
    arrays.update(decision.artifact_arrays)
    write_deterministic_npz(artifact_path, arrays)
    input_id = event_identity(frame)
    return (
        BrainOutputEvent(
            timestamp_ns=frame.timestamp_ns,
            sequence_id=frame.sequence_id,
            source=SPATIAL_MEMORY_V1_SOURCE,
            input_event_ids=[input_id],
            pose_delta=pose_delta,  # type: ignore[arg-type]
            pose_confidence=max(0.0, min(1.0, 1.0 - uncertainty_scalar)),
            local_bev_ref=relative_artifact,
            candidate_trajectories=decision.candidate_trajectories,
            selected_trajectory_id=decision.selected_candidate_id,
            cmd_vel=None,
            uncertainty=uncertainty_scalar,
            stop_reason="spatial_memory_v1_replay_only_trajectory_decision_not_executed",
            debug={
                "mock": False,
                "model": SPATIAL_MEMORY_V1_SOURCE,
                "checkpoint": checkpoint.as_posix(),
                "feature_artifacts": feature_dir.as_posix(),
                "camera_id": frame.camera_id,
                "output_channels": list(BEV_OUTPUT_CHANNELS),
                "local_bev_shape": list(memory_probabilities.shape[1:]),
                "memory_used": memory_used,
                "pose_warp_used": pose_warp_used,
                "pose_warp_valid": pose_warp_valid,
                "valid_warp_fraction": 1.0 if pose_warp_valid else 0.0,
                "pose_warp_source_requested": pose_warp_source_requested,
                "pose_warp_source": pose_warp_source,
                "pose_warp_source_available": bool(route_pose_delta_available),
                "predicted_pose_warp_ablation": bool(predicted_pose_warp_ablation),
                "memory_reset": memory_reset,
                "update_mask_coverage": update_mask_coverage,
                "memory_overwrite_fraction": memory_overwrite_fraction,
                "observation_mask_source": observation_mask_source,
                "missing_pose_behavior": str(getattr(model.config, "missing_pose_behavior", "unknown")),
                "representation_pretraining_only": True,
                "control_safe": False,
                "replay_only": True,
                "not_executed": True,
                "product_training_approved": False,
                "cmd_vel_emitted": False,
                "input_frame_id": frame.frame_id,
                **decision.debug,
            },
        ),
        relative_artifact,
        next_state,
        next_pose_delta,
    )


def _route_pose_deltas(frames: list[FrameEvent], events: Iterable[Event]) -> dict[str, RoutePoseDelta]:
    pose_samples = _route_pose_samples(events)
    deltas: dict[str, RoutePoseDelta] = {}
    previous_frame: FrameEvent | None = None
    for frame in frames:
        if (
            previous_frame is None
            or previous_frame.sequence_id != frame.sequence_id
            or previous_frame.camera_id != frame.camera_id
        ):
            previous_frame = frame
            continue
        previous_sample = _sample_for_frame(previous_frame, pose_samples)
        current_sample = _sample_for_frame(frame, pose_samples)
        if previous_sample is None or current_sample is None:
            previous_frame = frame
            continue
        deltas[event_identity(frame)] = RoutePoseDelta(
            delta=_relative_planar_delta(previous_sample, current_sample),
            source=current_sample["source"],
        )
        previous_frame = frame
    return deltas


def _route_pose_samples(events: Iterable[Event]) -> dict[tuple[str, int], dict[str, object]]:
    samples: dict[tuple[str, int], dict[str, object]] = {}
    for event in events:
        if isinstance(event, OdomEvent):
            samples[(event.sequence_id, event.timestamp_ns)] = {
                "x": float(event.position_m[0]),
                "y": float(event.position_m[1]),
                "yaw": _yaw_from_quaternion_xyzw(event.orientation_xyzw),
                "source": "route_odom",
            }
    for event in events:
        if isinstance(event, PoseEvent):
            samples[(event.sequence_id, event.timestamp_ns)] = {
                "x": float(event.position_m[0]),
                "y": float(event.position_m[1]),
                "yaw": _yaw_from_quaternion_xyzw(event.orientation_xyzw),
                "source": "route_pose",
            }
    return samples


def _sample_for_frame(frame: FrameEvent, samples: dict[tuple[str, int], dict[str, object]]) -> dict[str, object] | None:
    exact = samples.get((frame.sequence_id, frame.timestamp_ns))
    if exact is not None:
        return exact
    candidates = [
        (abs(timestamp_ns - frame.timestamp_ns), sample)
        for (sequence_id, timestamp_ns), sample in samples.items()
        if sequence_id == frame.sequence_id
    ]
    if not candidates:
        return None
    distance_ns, sample = min(candidates, key=lambda item: item[0])
    if distance_ns > 100_000_000:
        return None
    return sample


def _relative_planar_delta(previous: dict[str, object], current: dict[str, object]) -> tuple[float, float, float]:
    dx_world = float(current["x"]) - float(previous["x"])
    dy_world = float(current["y"]) - float(previous["y"])
    yaw_prev = float(previous["yaw"])
    dx = cos(yaw_prev) * dx_world + sin(yaw_prev) * dy_world
    dy = -sin(yaw_prev) * dx_world + cos(yaw_prev) * dy_world
    dyaw = _wrap_angle(float(current["yaw"]) - yaw_prev)
    return (float(dx), float(dy), float(dyaw))


def _yaw_from_quaternion_xyzw(quaternion: tuple[float, float, float, float]) -> float:
    qx, qy, qz, qw = (float(value) for value in quaternion)
    return atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _wrap_angle(value: float) -> float:
    while value > np.pi:
        value -= 2.0 * float(np.pi)
    while value < -np.pi:
        value += 2.0 * float(np.pi)
    return float(value)


def _checkpoint_model_name(checkpoint: str | Path) -> str:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    value = payload.get("model_name") if isinstance(payload, dict) else None
    return str(value) if isinstance(value, str) else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic dummy modeld.")
    parser.add_argument("--log", required=True, help="Input segment log directory.")
    parser.add_argument("--out", required=True, help="Output segment log directory.")
    parser.add_argument("--checkpoint", default=None, help="Optional SpatialMemoryNet v0 checkpoint.")
    parser.add_argument("--features", default=None, help="Optional DINO feature artifact directory.")
    parser.add_argument("--trajectory-scorer-checkpoint", default=None, help="Optional TrajectoryScorerNet v0 checkpoint.")
    parser.add_argument("--future-rollout-checkpoint", default=None, help="Optional Future BEV Rollout v1 checkpoint.")
    parser.add_argument(
        "--v1-pose-warp-source",
        choices=V1_POSE_WARP_SOURCES,
        default="route_pose",
        help="Pose source for SpatialMemoryNet v1 memory warp; predicted_pose is an explicit ablation.",
    )
    parser.add_argument(
        "--v1-policy-bev-source",
        choices=V1_POLICY_BEV_SOURCES,
        default="memory",
        help="BEV source used for SpatialMemoryNet v1 replay-only trajectory decisions.",
    )
    parser.add_argument("--device", default=None, help="Optional torch device for checkpoint inference.")
    args = parser.parse_args(argv)
    if args.checkpoint:
        write_spatial_model_outputs(
            args.log,
            args.out,
            checkpoint=args.checkpoint,
            feature_dir=args.features,
            trajectory_scorer_checkpoint=args.trajectory_scorer_checkpoint,
            future_rollout_checkpoint=args.future_rollout_checkpoint,
            device_name=args.device,
            v1_pose_warp_source=args.v1_pose_warp_source,
            v1_policy_bev_source=args.v1_policy_bev_source,
        )
    else:
        write_dummy_model_outputs(args.log, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
