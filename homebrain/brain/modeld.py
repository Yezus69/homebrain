from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import atan2, cos, sin
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import torch

from homebrain.brain.direct_rgbd_features import (
    depth_valid_observation_mask,
    load_depth_for_frame,
    load_depth_refs,
    load_rgb_image,
    rgbd_patch_features,
)
from homebrain.data.spatial_dataset import read_json
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
V1_POSE_WARP_SOURCES = ("odom", "odom_or_route_pose", "route_pose", "route_pose_ablation", "predicted_pose", "none")
V1_POLICY_BEV_SOURCES = ("current", "memory")
RUNTIME_FEATURE_SOURCES = ("dino", "direct_rgbd")


@dataclass(frozen=True)
class RoutePoseDelta:
    delta: tuple[float, float, float]
    source: str


@dataclass(frozen=True)
class SceneState:
    pose_estimate: tuple[float, float, float]
    local_bev_ref: str | None
    selected_trajectory_id: str | None
    policy_bev_source: str
    future_prediction_horizon_count: int
    step_index: int

    def to_debug(self) -> dict[str, Any]:
        return {
            "schema_version": "homebrain.runtime.scene_state.v0",
            "pose_estimate_in_scene": _pose_estimate_dict(self.pose_estimate),
            "local_bev_ref": self.local_bev_ref,
            "selected_trajectory_id": self.selected_trajectory_id,
            "policy_bev_source": self.policy_bev_source,
            "future_prediction_horizon_count": int(self.future_prediction_horizon_count),
            "updated_online_inside_brain_step": True,
            "step_index": int(self.step_index),
        }


class DirectRGBDFeatureStore:
    """Runtime-only first slice: derive model input tensors directly from current RGB-D.

    This intentionally does not try to mimic DINO quality. It is a bounded,
    transparent adapter from current real route sensors into the existing
    SpatialMemoryNet feature interface so replay can exercise a no-precomputed-
    DINO runtime path.
    """

    def __init__(self, log_dir: str | Path, *, feature_dim: int, patch_shape: tuple[int, int] = (16, 16)) -> None:
        if feature_dim <= 0:
            raise ValueError("direct RGB-D feature_dim must be positive")
        self.log_dir = Path(log_dir)
        self.feature_dim = int(feature_dim)
        self.patch_shape = patch_shape
        self.depth_by_key = load_depth_refs(self.log_dir)
        self._last_depth_key: tuple[str, str, int] | None = None
        self._last_depth: np.ndarray | None = None

    def load_for_frame(self, frame: FrameEvent) -> tuple[np.ndarray, np.ndarray]:
        rgb = load_rgb_image(self.log_dir / frame.data_ref)
        depth = self._load_depth(frame)
        patch_features = rgbd_patch_features(
            rgb=rgb,
            depth_m=depth,
            feature_dim=self.feature_dim,
            patch_shape=self.patch_shape,
        )
        cls_feature = patch_features.mean(axis=(0, 1)).astype(np.float32)
        return patch_features, cls_feature

    def _load_depth(self, frame: FrameEvent) -> np.ndarray | None:
        key = (frame.sequence_id, frame.camera_id, int(frame.frame_id))
        if self._last_depth_key == key:
            return self._last_depth
        depth = load_depth_for_frame(self.log_dir, frame, self.depth_by_key)
        self._last_depth_key = key
        self._last_depth = depth
        return depth

    def load_observation_mask_for_frame(self, frame: FrameEvent, *, bev_shape: tuple[int, int]) -> np.ndarray | None:
        return depth_valid_observation_mask(self._load_depth(frame), bev_shape=bev_shape)


class Brain:
    """Online-style runtime owner for SpatialMemoryNetV1 replay ticks.

    The replay CLI still supplies synchronized sensor events, but this object
    owns the persistent model memory, coverage memory, pose estimate, and action
    history across calls to ``step``.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        feature_store: Any,
        output_root: Path,
        checkpoint: Path,
        feature_dir: Path | None,
        runtime_feature_source: str,
        start_timestamp_ns: int,
        pose_warp_source_requested: str,
        policy_bev_source: str,
        trajectory_scorer: RuntimeTrajectoryScorer | None,
        future_rollout_scorer: RuntimeFutureRolloutScorer | None,
        trajectory_candidates: list,
        meters_per_cell: float,
        device: torch.device,
        future_rollout_selection_mode: str = "guided_transparent",
    ) -> None:
        self.model = model
        self.feature_store = feature_store
        self.output_root = output_root
        self.checkpoint = checkpoint
        self.feature_dir = feature_dir
        self.runtime_feature_source = runtime_feature_source
        self.start_timestamp_ns = int(start_timestamp_ns)
        self.pose_warp_source_requested = pose_warp_source_requested
        self.policy_bev_source = policy_bev_source
        self.trajectory_scorer = trajectory_scorer
        self.future_rollout_scorer = future_rollout_scorer
        self.trajectory_candidates = trajectory_candidates
        self.meters_per_cell = float(meters_per_cell)
        self.device = device
        self.future_rollout_selection_mode = future_rollout_selection_mode
        self.state: SpatialMemoryState | None = None
        self.previous_predicted_pose_delta: torch.Tensor | None = None
        self.previous_sequence_camera: tuple[str, str] | None = None
        self.trajectory_coverage = CoverageMemory(model.config.bev_shape, meters_per_cell=self.meters_per_cell)
        self.pose_estimate = (0.0, 0.0, 0.0)
        self.step_count = 0
        self.selection_history: list[str] = []
        self.scene_state: SceneState | None = None

    def should_reset(self, frame: FrameEvent) -> bool:
        key = (frame.sequence_id, frame.camera_id)
        return self.previous_sequence_camera is None or key != self.previous_sequence_camera

    def step(
        self,
        frame: FrameEvent,
        *,
        pose_delta_to_current: torch.Tensor | None,
        pose_warp_source: str,
        route_pose_delta_available: bool,
        predicted_pose_warp_ablation: bool,
        route_pose_leakage_ablation: bool,
        reset_memory: bool,
    ) -> tuple[BrainOutputEvent, str]:
        if reset_memory:
            self.state = None
            self.previous_predicted_pose_delta = None
            self.pose_estimate = (0.0, 0.0, 0.0)
            self.selection_history = []
            self.scene_state = None
            self.trajectory_coverage = CoverageMemory(
                self.model.config.bev_shape,
                meters_per_cell=self.meters_per_cell,
            )
        else:
            self.pose_estimate = _integrate_pose_estimate(self.pose_estimate, pose_delta_to_current)

        output, artifact, state, predicted_pose_delta = _spatial_v1_output_for_frame(
            model=self.model,
            frame=frame,
            feature_store=self.feature_store,
            output_root=self.output_root,
            checkpoint=self.checkpoint,
            feature_dir=self.feature_dir,
            runtime_feature_source=self.runtime_feature_source,
            start_timestamp_ns=self.start_timestamp_ns,
            pose_delta_to_current=pose_delta_to_current,
            pose_warp_source=pose_warp_source,
            pose_warp_source_requested=self.pose_warp_source_requested,
            route_pose_delta_available=route_pose_delta_available,
            predicted_pose_warp_ablation=predicted_pose_warp_ablation,
            route_pose_leakage_ablation=route_pose_leakage_ablation,
            reset_memory=reset_memory,
            state=self.state,
            trajectory_scorer=self.trajectory_scorer,
            future_rollout_scorer=self.future_rollout_scorer,
            trajectory_candidates=self.trajectory_candidates,
            trajectory_coverage=self.trajectory_coverage,
            policy_bev_source=self.policy_bev_source,
            device=self.device,
            future_rollout_selection_mode=self.future_rollout_selection_mode,
            selection_history=list(self.selection_history),
        )
        self.state = state
        self.previous_predicted_pose_delta = predicted_pose_delta
        self.previous_sequence_camera = (frame.sequence_id, frame.camera_id)
        self.step_count += 1
        if output.selected_trajectory_id is not None:
            self.selection_history.append(str(output.selected_trajectory_id))
            self.selection_history = self.selection_history[-24:]
        future_horizon_count = _future_horizon_count_from_artifact(artifact, self.output_root)
        self.scene_state = SceneState(
            pose_estimate=self.pose_estimate,
            local_bev_ref=output.local_bev_ref,
            selected_trajectory_id=output.selected_trajectory_id,
            policy_bev_source=self.policy_bev_source,
            future_prediction_horizon_count=future_horizon_count,
            step_index=self.step_count - 1,
        )
        output.debug.update(
            {
                "runtime_api": "Brain.step",
                "runtime_api_step_index": self.step_count - 1,
                "runtime_api_owns_persistent_memory": True,
                "scene_state": self.scene_state.to_debug(),
                "scene_memory_used_for_policy": self.policy_bev_source == "memory",
                "scene_pose_estimate": _pose_estimate_dict(self.pose_estimate),
                "pose_estimate_source": pose_warp_source,
                "teacher_runtime_dependency": self.runtime_feature_source == "dino",
                "future_or_groundtruth_runtime_dependency": bool(route_pose_leakage_ablation),
                "accepted_runtime_student_path": self.runtime_feature_source == "direct_rgbd",
            }
        )
        return output, artifact


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
    v1_pose_warp_source: str = "odom",
    v1_policy_bev_source: str = "memory",
    runtime_feature_source: str = "dino",
    future_rollout_selection_mode: str = "guided_transparent",
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
        runtime_feature_source=runtime_feature_source,
        future_rollout_selection_mode=future_rollout_selection_mode,
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
    v1_pose_warp_source: str = "odom",
    v1_policy_bev_source: str = "memory",
    runtime_feature_source: str = "dino",
    future_rollout_selection_mode: str = "guided_transparent",
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
        runtime_feature_source=runtime_feature_source,
        future_rollout_selection_mode=future_rollout_selection_mode,
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
    v1_pose_warp_source: str = "odom",
    v1_policy_bev_source: str = "memory",
    runtime_feature_source: str = "dino",
    future_rollout_selection_mode: str = "guided_transparent",
) -> tuple[list[BrainOutputEvent], list[str]]:
    if runtime_feature_source not in RUNTIME_FEATURE_SOURCES:
        raise ValueError(f"runtime_feature_source must be one of {RUNTIME_FEATURE_SOURCES}")
    if _checkpoint_model_name(checkpoint) == "SpatialMemoryNetV1":
        return _spatial_v1_model_outputs(
            events,
            log_dir=log_dir,
            out_dir=out_dir,
            checkpoint=checkpoint,
            feature_dir=feature_dir,
            trajectory_scorer_checkpoint=trajectory_scorer_checkpoint,
            future_rollout_checkpoint=future_rollout_checkpoint,
            device_name=device_name,
            pose_warp_source=v1_pose_warp_source,
            policy_bev_source=v1_policy_bev_source,
            runtime_feature_source=runtime_feature_source,
            future_rollout_selection_mode=future_rollout_selection_mode,
        )

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model.to(device)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    resolved_feature_dir = feature_dir or metadata.get("feature_artifacts")
    if runtime_feature_source == "dino" and not isinstance(resolved_feature_dir, (str, Path)):
        raise ValueError("DINO feature artifacts must be supplied with --features or checkpoint metadata")
    feature_store = (
        DINOFeatureStore(resolved_feature_dir)
        if runtime_feature_source == "dino"
        else DirectRGBDFeatureStore(log_dir, feature_dim=int(model.config.feature_dim))
    )
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
            feature_dir=Path(resolved_feature_dir) if isinstance(resolved_feature_dir, (str, Path)) else None,
            runtime_feature_source=runtime_feature_source,
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
    log_dir: str | Path,
    out_dir: str | Path,
    checkpoint: str | Path,
    feature_dir: str | Path | None = None,
    trajectory_scorer_checkpoint: str | Path | None = None,
    future_rollout_checkpoint: str | Path | None = None,
    device_name: str | None = None,
    pose_warp_source: str = "odom",
    policy_bev_source: str = "memory",
    runtime_feature_source: str = "dino",
    future_rollout_selection_mode: str = "guided_transparent",
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
    if runtime_feature_source not in RUNTIME_FEATURE_SOURCES:
        raise ValueError(f"runtime_feature_source must be one of {RUNTIME_FEATURE_SOURCES}")
    resolved_feature_dir = feature_dir or metadata.get("feature_artifacts")
    if runtime_feature_source == "dino" and not isinstance(resolved_feature_dir, (str, Path)):
        raise ValueError("DINO feature artifacts must be supplied with --features or checkpoint metadata")
    feature_store = (
        DINOFeatureStore(resolved_feature_dir)
        if runtime_feature_source == "dino"
        else DirectRGBDFeatureStore(log_dir=log_dir, feature_dim=int(model.config.feature_dim))
    )
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
    brain = Brain(
        model=model,
        feature_store=feature_store,
        output_root=output_root,
        checkpoint=Path(checkpoint),
        feature_dir=Path(resolved_feature_dir) if isinstance(resolved_feature_dir, (str, Path)) else None,
        runtime_feature_source=runtime_feature_source,
        start_timestamp_ns=start_timestamp_ns,
        pose_warp_source_requested=pose_warp_source,
        policy_bev_source=policy_bev_source,
        trajectory_scorer=trajectory_scorer,
        future_rollout_scorer=future_rollout_scorer,
        trajectory_candidates=trajectory_candidates,
        meters_per_cell=meters_per_cell,
        device=device,
        future_rollout_selection_mode=future_rollout_selection_mode,
    )
    route_pose_deltas = _route_pose_deltas(frames, ordered_events, pose_warp_source=pose_warp_source)
    for frame in frames:
        reset = brain.should_reset(frame)
        selected_pose_delta: torch.Tensor | None = None
        selected_pose_source = pose_warp_source
        route_pose_available = False
        measured_pose_sources = {"odom", "odom_or_route_pose", "route_pose", "route_pose_ablation"}
        if not reset and pose_warp_source in measured_pose_sources:
            route_delta = route_pose_deltas.get(event_identity(frame))
            if route_delta is not None:
                selected_pose_delta = torch.tensor(route_delta.delta, dtype=torch.float32)
                selected_pose_source = route_delta.source
                route_pose_available = True
            else:
                selected_pose_source = f"{pose_warp_source}_missing"
        elif not reset and pose_warp_source == "predicted_pose":
            selected_pose_delta = brain.previous_predicted_pose_delta
            route_pose_available = selected_pose_delta is not None
            selected_pose_source = "predicted_pose" if selected_pose_delta is not None else "predicted_pose_missing"
        elif reset:
            selected_pose_source = "sequence_reset"
        else:
            selected_pose_source = "none"
        output, artifact = brain.step(
            frame,
            pose_delta_to_current=selected_pose_delta,
            pose_warp_source=selected_pose_source,
            route_pose_delta_available=route_pose_available,
            predicted_pose_warp_ablation=pose_warp_source == "predicted_pose",
            route_pose_leakage_ablation=pose_warp_source in {"route_pose", "route_pose_ablation"}
            or selected_pose_source == "route_pose",
            reset_memory=reset,
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
    feature_dir: Path | None,
    runtime_feature_source: str,
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
                "feature_artifacts": feature_dir.as_posix() if feature_dir is not None else None,
                "runtime_feature_source": runtime_feature_source,
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
    feature_dir: Path | None,
    runtime_feature_source: str,
    start_timestamp_ns: int,
    pose_delta_to_current: torch.Tensor | None,
    pose_warp_source: str,
    pose_warp_source_requested: str,
    route_pose_delta_available: bool,
    predicted_pose_warp_ablation: bool,
    route_pose_leakage_ablation: bool,
    reset_memory: bool,
    state: SpatialMemoryState | None,
    trajectory_scorer: RuntimeTrajectoryScorer | None,
    future_rollout_scorer: RuntimeFutureRolloutScorer | None,
    trajectory_candidates: list,
    trajectory_coverage: CoverageMemory,
    policy_bev_source: str,
    device: torch.device,
    future_rollout_selection_mode: str = "guided_transparent",
    selection_history: list[str] | None = None,
) -> tuple[BrainOutputEvent, str, SpatialMemoryState, torch.Tensor]:
    total_started = time.perf_counter()
    feature_started = time.perf_counter()
    patch_features, _cls_feature = feature_store.load_for_frame(frame)
    feature_load_latency_ms = _elapsed_ms(feature_started)
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
    runtime_observation_mask: torch.Tensor | None = None
    runtime_observation_mask_source = "model_predicted_current_bev_confidence"
    load_observation = getattr(feature_store, "load_observation_mask_for_frame", None)
    if callable(load_observation):
        observation = load_observation(frame, bev_shape=tuple(model.config.bev_shape))
        if observation is not None:
            runtime_observation_mask = torch.from_numpy(observation[None, None, ...].astype(np.float32)).to(device)
            runtime_observation_mask_source = "current_depth_validity_mask"
    model.eval()
    model_started = time.perf_counter()
    _sync_device(device)
    with torch.no_grad():
        outputs = model.step(
            features,
            timestamp_s,
            sensor_mask,
            memory_state=state,
            pose_delta_to_current=pose_to_current,
            pose_delta_to_current_mask=pose_mask,
            observation_mask=runtime_observation_mask,
            force_reset=torch.tensor([reset_memory], dtype=torch.bool, device=device),
        )
    _sync_device(device)
    model_latency_ms = _elapsed_ms(model_started)
    postprocess_started = time.perf_counter()
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
    postprocess_latency_ms = _elapsed_ms(postprocess_started)
    decision_started = time.perf_counter()
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
        future_rollout_selection_mode=future_rollout_selection_mode,
        selection_history=selection_history,
    )
    _sync_device(device)
    decision_latency_ms = _elapsed_ms(decision_started)
    arrays.update(decision.artifact_arrays)
    artifact_started = time.perf_counter()
    write_deterministic_npz(artifact_path, arrays)
    artifact_write_latency_ms = _elapsed_ms(artifact_started)
    total_latency_ms = _elapsed_ms(total_started)
    latency_debug = {
        "feature_load_latency_ms": round(feature_load_latency_ms, 6),
        "model_latency_ms": round(model_latency_ms, 6),
        "memory_update_latency_ms": round(model_latency_ms, 6),
        "postprocess_latency_ms": round(postprocess_latency_ms, 6),
        "decision_latency_ms": round(decision_latency_ms, 6),
        "artifact_write_latency_ms": round(artifact_write_latency_ms, 6),
        "end_to_end_latency_ms": round(total_latency_ms, 6),
        "target_hz": 10.0,
        "target_period_ms": 100.0,
        "meets_10hz_budget": bool(total_latency_ms <= 100.0),
        "device": str(device),
    }
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
                "feature_artifacts": feature_dir.as_posix() if feature_dir is not None else None,
                "runtime_feature_source": runtime_feature_source,
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
                "route_pose_leakage_ablation": bool(route_pose_leakage_ablation),
                "memory_reset": memory_reset,
                "update_mask_coverage": update_mask_coverage,
                "memory_overwrite_fraction": memory_overwrite_fraction,
                "observation_mask_source": observation_mask_source,
                "runtime_observation_mask_source": runtime_observation_mask_source,
                "missing_pose_behavior": str(getattr(model.config, "missing_pose_behavior", "unknown")),
                "representation_pretraining_only": True,
                "control_safe": False,
                "replay_only": True,
                "not_executed": True,
                "product_training_approved": False,
                "cmd_vel_emitted": False,
                "input_frame_id": frame.frame_id,
                "latency": latency_debug,
                **decision.debug,
            },
        ),
        relative_artifact,
        next_state,
        next_pose_delta,
    )


def _route_pose_deltas(
    frames: list[FrameEvent],
    events: Iterable[Event],
    *,
    pose_warp_source: str,
) -> dict[str, RoutePoseDelta]:
    pose_samples = _route_pose_samples(events, pose_warp_source=pose_warp_source)
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


def _route_pose_samples(events: Iterable[Event], *, pose_warp_source: str) -> dict[tuple[str, int], dict[str, object]]:
    if pose_warp_source == "none" or pose_warp_source == "predicted_pose":
        return {}
    use_odom = pose_warp_source in {"odom", "odom_or_route_pose"}
    use_route_pose = pose_warp_source in {"odom_or_route_pose", "route_pose", "route_pose_ablation"}
    samples: dict[tuple[str, int], dict[str, object]] = {}
    event_list = list(events)
    if use_odom:
        for event in event_list:
            if isinstance(event, OdomEvent):
                samples[(event.sequence_id, event.timestamp_ns)] = {
                    "x": float(event.position_m[0]),
                    "y": float(event.position_m[1]),
                    "yaw": _yaw_from_quaternion_xyzw(event.orientation_xyzw),
                    "source": "route_odom",
                }
    if use_route_pose:
        for event in event_list:
            if not isinstance(event, PoseEvent):
                continue
            key = (event.sequence_id, event.timestamp_ns)
            if key in samples and pose_warp_source == "odom_or_route_pose":
                continue
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


def _integrate_pose_estimate(
    pose: tuple[float, float, float],
    delta: torch.Tensor | None,
) -> tuple[float, float, float]:
    if delta is None:
        return pose
    values = delta.detach().cpu().numpy().reshape(-1)
    if values.shape[0] < 3 or not np.all(np.isfinite(values[:3])):
        return pose
    x_m, y_m, yaw_rad = pose
    dx, dy, dyaw = (float(values[0]), float(values[1]), float(values[2]))
    x_m += cos(yaw_rad) * dx - sin(yaw_rad) * dy
    y_m += sin(yaw_rad) * dx + cos(yaw_rad) * dy
    yaw_rad = _wrap_angle(yaw_rad + dyaw)
    return (float(x_m), float(y_m), float(yaw_rad))


def _pose_estimate_dict(pose: tuple[float, float, float]) -> dict[str, float]:
    return {
        "x_m": round(float(pose[0]), 6),
        "y_m": round(float(pose[1]), 6),
        "yaw_rad": round(float(pose[2]), 6),
    }


def _future_horizon_count_from_artifact(relative_artifact: str, output_root: Path) -> int:
    path = output_root / relative_artifact
    if not path.exists():
        return 0
    try:
        with np.load(path, allow_pickle=False) as data:
            if "future_rollout_future_bev_prob" not in data:
                return 0
            future = np.asarray(data["future_rollout_future_bev_prob"])
            return int(future.shape[0]) if future.ndim >= 1 else 0
    except Exception:  # noqa: BLE001
        return 0


def _load_runtime_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"runtime RGB frame is missing: {path}")
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Pillow is required for direct RGB-D runtime features") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _rgbd_patch_features(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray | None,
    feature_dim: int,
    patch_shape: tuple[int, int],
) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("runtime RGB must be HxWx3")
    patch_h, patch_w = patch_shape
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("patch_shape must be positive")
    if depth_m is not None and depth_m.shape[:2] != rgb.shape[:2]:
        depth = _resize_nearest_float(depth_m, rgb.shape[:2])
    else:
        depth = depth_m
    rows = _bin_edges(rgb.shape[0], patch_h)
    cols = _bin_edges(rgb.shape[1], patch_w)
    base = np.zeros((patch_h, patch_w, 10), dtype=np.float32)
    rgb_float = rgb.astype(np.float32) / np.float32(255.0)
    for row_index in range(patch_h):
        row_slice = slice(rows[row_index], rows[row_index + 1])
        for col_index in range(patch_w):
            col_slice = slice(cols[col_index], cols[col_index + 1])
            rgb_patch = rgb_float[row_slice, col_slice]
            rgb_mean = rgb_patch.reshape(-1, 3).mean(axis=0) if rgb_patch.size else np.zeros((3,), dtype=np.float32)
            if depth is None:
                depth_valid_ratio = np.float32(0.0)
                depth_mean = np.float32(0.0)
                depth_std = np.float32(0.0)
                depth_near = np.float32(0.0)
                depth_far = np.float32(0.0)
            else:
                depth_patch = np.asarray(depth[row_slice, col_slice], dtype=np.float32)
                valid = np.isfinite(depth_patch) & (depth_patch > np.float32(0.0))
                depth_valid_ratio = np.float32(np.count_nonzero(valid) / max(depth_patch.size, 1))
                if np.any(valid):
                    values = np.clip(depth_patch[valid], 0.0, 6.0).astype(np.float32)
                    depth_mean = np.float32(np.mean(values) / 6.0)
                    depth_std = np.float32(np.std(values) / 3.0)
                    depth_near = np.float32(np.min(values) / 6.0)
                    depth_far = np.float32(np.max(values) / 6.0)
                else:
                    depth_mean = depth_std = depth_near = depth_far = np.float32(0.0)
            base[row_index, col_index] = np.asarray(
                [
                    float(rgb_mean[0]),
                    float(rgb_mean[1]),
                    float(rgb_mean[2]),
                    float(depth_mean),
                    float(depth_std),
                    float(depth_near),
                    float(depth_far),
                    float(depth_valid_ratio),
                    row_index / max(patch_h - 1, 1),
                    col_index / max(patch_w - 1, 1),
                ],
                dtype=np.float32,
            )
    repeats = int(np.ceil(feature_dim / base.shape[-1]))
    tiled = np.tile(base, (1, 1, repeats))[..., :feature_dim]
    return tiled.astype(np.float32)


def _bin_edges(size: int, bins: int) -> np.ndarray:
    edges = np.linspace(0, size, bins + 1).round().astype(np.int64)
    edges[0] = 0
    edges[-1] = size
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = min(size, edges[index - 1] + 1)
    return np.clip(edges, 0, size)


def _resize_nearest_float(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = np.linspace(0, array.shape[0] - 1, shape[0]).round().astype(np.int64)
    cols = np.linspace(0, array.shape[1] - 1, shape[1]).round().astype(np.int64)
    return np.asarray(array, dtype=np.float32)[rows[:, None], cols[None, :]]


def _yaw_from_quaternion_xyzw(quaternion: tuple[float, float, float, float]) -> float:
    qx, qy, qz, qw = (float(value) for value in quaternion)
    return atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _wrap_angle(value: float) -> float:
    while value > np.pi:
        value -= 2.0 * float(np.pi)
    while value < -np.pi:
        value += 2.0 * float(np.pi)
    return float(value)


def _elapsed_ms(started: float) -> float:
    return float((time.perf_counter() - started) * 1000.0)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


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
        default="odom",
        help=(
            "Pose source for SpatialMemoryNet v1 memory warp. route_pose and "
            "route_pose_ablation are explicit ground-truth leakage ablations."
        ),
    )
    parser.add_argument(
        "--v1-policy-bev-source",
        choices=V1_POLICY_BEV_SOURCES,
        default="memory",
        help="BEV source used for SpatialMemoryNet v1 replay-only trajectory decisions.",
    )
    parser.add_argument(
        "--runtime-feature-source",
        choices=RUNTIME_FEATURE_SOURCES,
        default="dino",
        help="Use precomputed DINO features or direct current RGB-D runtime features.",
    )
    parser.add_argument(
        "--future-rollout-selection-mode",
        choices=("argmin", "guided_transparent"),
        default="guided_transparent",
        help="Select from raw FutureBEV argmin or a safety/coverage-guided FutureBEV score.",
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
            runtime_feature_source=args.runtime_feature_source,
            future_rollout_selection_mode=args.future_rollout_selection_mode,
        )
    else:
        write_dummy_model_outputs(args.log, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
