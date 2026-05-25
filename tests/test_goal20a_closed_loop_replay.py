from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from homebrain.brain.spatial_memory_v1 import SpatialMemoryNetV1, SpatialMemoryNetV1Config, save_checkpoint
from homebrain.eval.closed_loop_replay_report import build_closed_loop_replay_report
from homebrain.messages.schema import BrainOutputEvent, FrameEvent, PoseEvent
from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.policies.runtime_decision import decide_trajectory
from homebrain.policies.trajectory_scorer import CoverageMemory, LocalBev
from homebrain.policies.trajectory_scorer_net_v0 import (
    TrajectoryScorerNetConfig,
    TrajectoryScorerNetV0,
    save_trajectory_scorer_checkpoint,
)
from homebrain.replay.replayd import replay_log
from homebrain.replay.segment_log import read_events, write_segment
from homebrain.teachers.dino_teacher import run_dino_teacher


def test_runtime_decision_emits_candidates_selection_and_coverage_debug() -> None:
    free = np.ones((6, 6), dtype=np.float32)
    occupied = np.zeros((6, 6), dtype=np.float32)
    unknown = np.zeros((6, 6), dtype=np.float32)
    occupied[0, :] = 0.8
    bev = LocalBev(
        free=free,
        occupied=occupied,
        unknown=unknown,
        traversable=free,
        risky=occupied,
        confidence=np.ones((6, 6), dtype=np.float32),
        uncertainty=np.zeros((6, 6), dtype=np.float32),
        source="v1_memory_bev",
    )
    candidates = generate_default_candidates(grid_shape=bev.shape, meters_per_cell=0.5, robot_radius_m=0.18)
    coverage = CoverageMemory(bev.shape, meters_per_cell=0.5)

    decision = decide_trajectory(
        bev=bev,
        candidates=candidates,
        coverage_memory=coverage,
        pose_delta=(0.1, 0.0, 0.0),
        policy_bev_source="memory",
    )

    assert len(decision.candidate_trajectories) == 9
    assert decision.selected_candidate_id
    assert decision.debug["trajectory_scoring"] is True
    assert decision.debug["policy_bev_source"] == "memory"
    assert decision.debug["control_safe"] is False
    assert decision.debug["cmd_vel_emitted"] is False
    assert decision.debug["coverage_memory"]["coverage_memory_cells_seen"] > 0
    assert "trajectory_selected_index" in decision.artifact_arrays


def test_v1_replay_outputs_candidates_selected_id_and_report_no_cmd_vel(tmp_path: Path) -> None:
    route = tmp_path / "route"
    features = tmp_path / "dino_fake"
    replay_out = tmp_path / "replayed"
    checkpoint = tmp_path / "v1.pt"
    _write_route_with_pose(route, count=4)
    run_dino_teacher(route, features, backend_name="fake")
    model = SpatialMemoryNetV1(SpatialMemoryNetV1Config(feature_dim=32, bev_shape=(4, 4), hidden_channels=16, sensor_dim=5))
    save_checkpoint(checkpoint, model, metadata={"control_safe": False, "replay_only": True}, metrics={})

    replay_log(
        route,
        replay_out,
        checkpoint=checkpoint,
        feature_dir=features,
        device_name="cpu",
        v1_policy_bev_source="memory",
    )

    outputs = [event for event in read_events(replay_out) if isinstance(event, BrainOutputEvent)]
    assert outputs
    assert all(event.cmd_vel is None for event in outputs)
    assert all(event.selected_trajectory_id is not None for event in outputs)
    assert all(event.candidate_trajectories for event in outputs)
    assert outputs[0].debug["trajectory_scoring"] is True
    assert outputs[0].debug["policy_bev_source"] == "memory"
    assert outputs[0].debug["control_safe"] is False
    assert outputs[0].debug["replay_only"] is True
    with np.load(replay_out / outputs[0].local_bev_ref, allow_pickle=False) as data:
        assert "memory_bev_free_prob" in data.files
        assert "trajectory_selected_index" in data.files

    report = build_closed_loop_replay_report(log_dir=replay_out)
    assert report["decision_count"] == len(outputs)
    assert report["cmd_vel_non_null_count"] == 0
    assert report["control_safe"] is False
    assert report["replay_only"] is True
    assert report["policy_bev_source"] == "memory"


def test_v1_replay_accepts_optional_learned_trajectory_scorer(tmp_path: Path) -> None:
    route = tmp_path / "route"
    features = tmp_path / "dino_fake"
    replay_out = tmp_path / "replayed"
    checkpoint = tmp_path / "v1.pt"
    scorer_checkpoint = tmp_path / "scorer" / "checkpoint.pt"
    _write_route_with_pose(route, count=3)
    run_dino_teacher(route, features, backend_name="fake")
    model = SpatialMemoryNetV1(SpatialMemoryNetV1Config(feature_dim=32, bev_shape=(4, 4), hidden_channels=16, sensor_dim=5))
    save_checkpoint(checkpoint, model, metadata={"control_safe": False, "replay_only": True}, metrics={})
    scorer = TrajectoryScorerNetV0(TrajectoryScorerNetConfig())
    save_trajectory_scorer_checkpoint(
        scorer_checkpoint,
        scorer,
        metadata={
            "meters_per_cell": 0.5,
            "robot_radius_m": 0.18,
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "product_training_approved": False,
        },
        metrics={"control_safe": False, "replay_only": True},
    )

    replay_log(
        route,
        replay_out,
        checkpoint=checkpoint,
        feature_dir=features,
        trajectory_scorer_checkpoint=scorer_checkpoint,
        device_name="cpu",
        v1_policy_bev_source="current",
    )

    outputs = [event for event in read_events(replay_out) if isinstance(event, BrainOutputEvent)]
    assert outputs[0].cmd_vel is None
    assert outputs[0].selected_trajectory_id is not None
    assert len(outputs[0].candidate_trajectories or []) == 9
    assert outputs[0].debug["trajectory_scorer_mode"] == "learned"
    assert outputs[0].debug["policy_bev_source"] == "current"
    assert outputs[0].debug["control_safe"] is False
    assert outputs[0].candidate_trajectories[0]["trajectory_score"]["control_safe"] is False
    assert "transparent_trajectory_score" in outputs[0].candidate_trajectories[0]


def _write_route_with_pose(route: Path, *, count: int) -> None:
    events = []
    artifacts = []
    for index in range(count):
        timestamp_ns = 1_700_000_000_000_000_000 + index * 100_000_000
        frame = FrameEvent(
            timestamp_ns=timestamp_ns,
            sequence_id=route.name,
            source="test",
            camera_id="front_rgb",
            frame_id=index,
            width=8,
            height=8,
            format="encoded_ppm",
            data_ref=f"frames/frame_{index:06d}.ppm",
            intrinsics={"available": False},
        )
        pose = PoseEvent(
            timestamp_ns=timestamp_ns,
            sequence_id=route.name,
            source="test_pose",
            position_m=(index * 0.1, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="map",
            child_frame_id="base_link",
            pose_kind="base_pose",
        )
        target = route / frame.data_ref
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
        events.extend([frame, pose])
        artifacts.append(frame.data_ref)
    write_segment(route, events, segment_id=route.name, artifact_files=artifacts)
