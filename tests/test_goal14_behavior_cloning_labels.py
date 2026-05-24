import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from homebrain.data.spatial_dataset import load_example_npz, scalar_bool, scalar_int, scalar_str, write_deterministic_npz, write_json
from homebrain.datasets.openloris_scene import OPENLORIS_ROUTE_ASSOCIATIONS_FILE
from homebrain.policies.build_action_label_pack_v5 import build_action_label_pack_v5
from homebrain.policies.candidate_trajectories import CandidateTrajectory, Pose2D, generate_default_candidates
from homebrain.policies.future_motion_action_labels import TimedPose2D, match_future_motion_to_candidate
from homebrain.policies.qa_action_label_pack import qa_action_label_pack
from homebrain.policies.train_trajectory_scorer_v1 import train_trajectory_scorer_v1
from homebrain.policies.eval_trajectory_scorer_v1 import eval_trajectory_scorer_v1


def test_future_motion_matching_exact_and_invalid_reasons() -> None:
    candidates = _candidates()
    straight = _candidate(candidates, "straight_short")
    sequence = _pose_sequence_for_candidate(straight, horizon_s=straight.duration_s)

    label = match_future_motion_to_candidate(
        sequence=sequence,
        current_frame_id=0,
        current_timestamp_ns=sequence[0].timestamp_ns,
        candidates=candidates,
        horizon_s=straight.duration_s,
    )

    assert label.bc_label_valid is True
    assert label.candidate_id == "straight_short"
    assert label.margin > 0.0
    assert len(label.top3) == 3

    missing = match_future_motion_to_candidate(
        sequence=[],
        current_frame_id=0,
        current_timestamp_ns=0,
        candidates=candidates,
        horizon_s=1.0,
    )
    assert missing.bc_label_valid is False
    assert missing.invalid_reason == "pose_missing"

    truncated = match_future_motion_to_candidate(
        sequence=[TimedPose2D(timestamp_ns=0, frame_id=0, x_m=0.0, y_m=0.0, yaw_rad=0.0)],
        current_frame_id=0,
        current_timestamp_ns=0,
        candidates=candidates,
        horizon_s=1.0,
    )
    assert truncated.invalid_reason == "future_horizon_truncated"

    stationary = match_future_motion_to_candidate(
        sequence=[
            TimedPose2D(timestamp_ns=0, frame_id=0, x_m=0.0, y_m=0.0, yaw_rad=0.0),
            TimedPose2D(timestamp_ns=1_000_000_000, frame_id=1, x_m=0.0, y_m=0.0, yaw_rad=0.0),
        ],
        current_frame_id=0,
        current_timestamp_ns=0,
        candidates=candidates,
        horizon_s=1.0,
    )
    assert stationary.invalid_reason == "stationary_below_threshold"


def test_action_label_pack_v5_is_noncollapsed_deterministic_and_disagrees_with_synthetic(tmp_path: Path) -> None:
    source = _write_bc_spatial_pack(
        tmp_path / "spatial",
        action_ids=("straight_short", "arc_left_small", "arc_right_small", "rotate_left"),
        horizon_s=1.2,
    )
    pack_a = tmp_path / "actions_a"
    pack_b = tmp_path / "actions_b"

    build_action_label_pack_v5(sources=[source], out_dir=pack_a, horizon_s=1.2)
    build_action_label_pack_v5(sources=[source], out_dir=pack_b, horizon_s=1.2)

    qa = qa_action_label_pack(pack_a)
    manifest_a = json.loads((pack_a / "manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((pack_b / "manifest.json").read_text(encoding="utf-8"))
    first = load_example_npz(pack_a / manifest_a["examples"][0]["example_path"])

    assert qa["action_label_pack_qa_pass"] is True
    assert qa["v5_noncollapse_gate_pass"] is True
    assert qa["action_entropy"] >= 1.5
    assert qa["dominant_action_fraction"] <= 0.5
    assert qa["v5_synthetic_oracle_agreement_fraction"] < 1.0
    assert manifest_a["pack_version"] == 5
    assert manifest_a["label_source"] == "future_motion_behavior_cloning"
    assert manifest_a["not_synthetic_expert"] is True
    assert manifest_a["selected_distribution"] == manifest_b["selected_distribution"]
    assert qa_action_label_pack(pack_a)["deterministic_hash"] == qa_action_label_pack(pack_b)["deterministic_hash"]
    assert str(np.asarray(first["selected_candidate_id"]).item()) == str(np.asarray(first["bc_label_candidate_id"]).item())
    assert str(np.asarray(first["selected_candidate_id"]).item()) != str(
        np.asarray(first["coverage_expert_selected_candidate_id"]).item()
    )
    assert bool(np.asarray(first["replay_only"]).item()) is True
    assert bool(np.asarray(first["not_executed"]).item()) is True
    assert bool(np.asarray(first["control_safe"]).item()) is False


def test_trajectory_scorer_v1_preserves_replay_only_safety_flags(tmp_path: Path) -> None:
    source = _write_bc_spatial_pack(
        tmp_path / "spatial",
        action_ids=("straight_short", "arc_left_small", "arc_right_small", "rotate_left", "straight_medium"),
        horizon_s=1.2,
    )
    pack = tmp_path / "actions"
    scorer = tmp_path / "scorer"
    eval_json = tmp_path / "eval.json"
    build_action_label_pack_v5(sources=[source], out_dir=pack, horizon_s=1.2)

    train_metrics = train_trajectory_scorer_v1(
        action_pack=pack,
        out_dir=scorer,
        max_steps=2,
        batch_size=2,
        device_name="cpu",
    )
    eval_metrics = eval_trajectory_scorer_v1(
        checkpoint=scorer / "checkpoint.pt",
        action_pack=pack,
        out_path=eval_json,
        split="val",
        device_name="cpu",
    )

    assert (scorer / "checkpoint.pt").exists()
    assert train_metrics["replay_only"] is True
    assert train_metrics["not_executed"] is True
    assert train_metrics["control_safe"] is False
    assert train_metrics["product_training_approved"] is False
    assert eval_metrics["replay_only"] is True
    assert eval_metrics["not_executed"] is True
    assert eval_metrics["control_safe"] is False
    assert eval_metrics["product_training_approved"] is False
    assert "agreement_with_future_motion_label" in eval_metrics
    assert "agreement_with_synthetic_oracle_label" in eval_metrics


def _write_bc_spatial_pack(root: Path, *, action_ids: tuple[str, ...], horizon_s: float) -> Path:
    candidates = _candidates()
    examples_dir = root / "examples"
    route_dir = root / "route"
    examples_dir.mkdir(parents=True, exist_ok=True)
    route_dir.mkdir(parents=True, exist_ok=True)
    records = []
    association_frames = []
    frame_id = 0
    for action_index, action_id in enumerate(action_ids):
        candidate = _candidate(candidates, action_id)
        sequence_id = f"seq_{action_index:02d}_{action_id}"
        pose_samples = _pose_sequence_for_candidate(candidate, horizon_s=horizon_s)
        for sample_index, sample in enumerate(pose_samples):
            current = sample_index == 0
            arrays = _open_arrays() if current else _blocked_arrays()
            timestamp_ns = sample.timestamp_ns + action_index * 10_000_000_000
            example_path = examples_dir / f"frame_{frame_id:06d}.npz"
            write_deterministic_npz(
                example_path,
                {
                    "frame_id": scalar_int(frame_id),
                    "timestamp_ns": scalar_int(timestamp_ns),
                    "bev_free": arrays["free"],
                    "bev_obstacle": arrays["obstacle"],
                    "bev_unknown": arrays["unknown"],
                    "bev_confidence": arrays["confidence"],
                    "weak_label": scalar_bool(True),
                    "control_safe": scalar_bool(False),
                    "not_robot_frame_truth": scalar_bool(False),
                    "source_name": scalar_str("unit_route"),
                    "source_family": scalar_str("public_robot_mounted"),
                    "scenario_name": scalar_str(action_id),
                    "supervision_grade": scalar_str("public_robot_frame_geometry"),
                    "replay_only": scalar_bool(True),
                    "not_executed": scalar_bool(True),
                },
            )
            records.append(
                {
                    "sequence_id": sequence_id,
                    "camera_id": "front_rgb",
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "source_name": "unit_route",
                    "source_family": "public_robot_mounted",
                    "scenario_name": action_id,
                    "supervision_grade": "public_robot_frame_geometry",
                    "dataset_frame_type": "public_robot_mounted",
                    "robot_frame_truth": True,
                    "not_robot_frame_truth": False,
                    "example_path": f"examples/frame_{frame_id:06d}.npz",
                }
            )
            association_frames.append(
                {
                    "frame_id": frame_id,
                    "base_pose": _base_pose(sample.x_m, sample.y_m, sample.yaw_rad),
                }
            )
            frame_id += 1
    write_json(route_dir / OPENLORIS_ROUTE_ASSOCIATIONS_FILE, {"frames": association_frames}, pretty=True)
    write_json(
        root / "manifest.json",
        {
            "schema_version": "homebrain.spatial_train_pack.v0",
            "package_type": "SpatialTrainPack",
            "source_log": route_dir.as_posix(),
            "source_name": "unit_route",
            "source_family": "public_robot_mounted",
            "dataset_frame_type": "public_robot_mounted",
            "supervision_grade": "public_robot_frame_geometry",
            "robot_supervision_grade": "public_robot_frame_geometry",
            "robot_frame_truth": True,
            "not_robot_frame_truth": False,
            "weak_label": True,
            "control_safe": False,
            "replay_only": True,
            "not_executed": True,
            "camera_config": {"meters_per_cell": 0.1, "robot_radius_m": 0.15},
            "grid_shape": [24, 24],
            "examples": records,
            "frames": records,
        },
        pretty=True,
    )
    return root


def _pose_sequence_for_candidate(candidate: CandidateTrajectory, *, horizon_s: float) -> list[TimedPose2D]:
    samples = []
    for index in range(13):
        t_s = horizon_s * index / 12.0
        pose = _candidate_pose_at(candidate, t_s)
        samples.append(
            TimedPose2D(
                timestamp_ns=int(round(t_s * 1_000_000_000)),
                frame_id=index,
                x_m=pose.x_m,
                y_m=pose.y_m,
                yaw_rad=pose.yaw_rad,
            )
        )
    return samples


def _candidate_pose_at(candidate: CandidateTrajectory, t_s: float) -> Pose2D:
    if t_s >= candidate.duration_s:
        return candidate.poses[-1]
    scaled = (t_s / candidate.duration_s) * (len(candidate.poses) - 1)
    left_index = int(math.floor(scaled))
    right_index = min(left_index + 1, len(candidate.poses) - 1)
    alpha = scaled - left_index
    left = candidate.poses[left_index]
    right = candidate.poses[right_index]
    return Pose2D(
        x_m=(1.0 - alpha) * left.x_m + alpha * right.x_m,
        y_m=(1.0 - alpha) * left.y_m + alpha * right.y_m,
        yaw_rad=left.yaw_rad + alpha * (right.yaw_rad - left.yaw_rad),
    )


def _base_pose(x_m: float, y_m: float, yaw_rad: float) -> dict:
    half = yaw_rad / 2.0
    return {
        "tx": float(x_m),
        "ty": float(y_m),
        "tz": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": float(math.sin(half)),
        "qw": float(math.cos(half)),
    }


def _open_arrays() -> dict[str, np.ndarray]:
    shape = (24, 24)
    return {
        "free": np.ones(shape, dtype=np.float32),
        "obstacle": np.zeros(shape, dtype=np.float32),
        "unknown": np.zeros(shape, dtype=np.float32),
        "confidence": np.ones(shape, dtype=np.float32),
    }


def _blocked_arrays() -> dict[str, np.ndarray]:
    arrays = _open_arrays()
    arrays["obstacle"][-1, 12] = 1.0
    arrays["free"][-1, 12] = 0.0
    return arrays


def _candidates() -> list[CandidateTrajectory]:
    return generate_default_candidates(grid_shape=(24, 24), meters_per_cell=0.1, robot_radius_m=0.15)


def _candidate(candidates: list[CandidateTrajectory], candidate_id: str) -> CandidateTrajectory:
    return next(candidate for candidate in candidates if candidate.id == candidate_id)
