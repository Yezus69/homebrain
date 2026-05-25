from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from homebrain.data.spatial_dataset import (
    load_example_npz,
    scalar_bool,
    scalar_float,
    scalar_int,
    scalar_str,
    write_deterministic_npz,
    write_json,
)
from homebrain.datasets.openloris_scene import OPENLORIS_ROUTE_ASSOCIATIONS_FILE
from homebrain.eval.eval_future_bev_rollout_v1 import eval_future_bev_rollout_v1
from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.policies.runtime_decision import decide_trajectory, load_runtime_future_rollout_scorer
from homebrain.policies.trajectory_scorer import CoverageMemory, LocalBev
from homebrain.train.build_future_bev_rollout_pack import build_future_bev_rollout_pack
from homebrain.train.future_bev_rollout_dataset import (
    candidate_outcome_labels,
    load_spatial_rollout_frames,
    warp_future_bev_to_current,
)
from homebrain.train.train_future_bev_rollout_v1 import train_future_bev_rollout_v1


def test_future_bev_warp_is_deterministic() -> None:
    future = np.zeros((3, 7, 7), dtype=np.float32)
    future[1, 3, 3] = 1.0
    first, first_mask = warp_future_bev_to_current(future, (0.0, 0.0, 0.0), meters_per_cell=1.0)
    second, second_mask = warp_future_bev_to_current(future, (0.0, 0.0, 0.0), meters_per_cell=1.0)
    assert np.array_equal(first, second)
    assert np.array_equal(first_mask, second_mask)
    assert first[1, 3, 3] == 1.0


def test_missing_future_pose_marks_rollout_labels_invalid(tmp_path: Path) -> None:
    spatial = _write_spatial_pack(tmp_path / "spatial", count=3, missing_pose_frame_ids={1})
    out = tmp_path / "future_pack"
    build_future_bev_rollout_pack(sources=[spatial], out_dir=out, horizons_s=(1.0,))
    manifest = (out / "manifest.json").read_text(encoding="utf-8")
    assert '"control_safe": false' in manifest
    first = load_example_npz(out / "examples" / "future_rollout_000000.npz")
    assert float(first["horizon_valid"][0]) == 0.0
    assert np.count_nonzero(first["future_valid_mask"]) == 0
    assert bool(first["control_safe"].item()) is False
    assert bool(first["replay_only"].item()) is True


def test_non_robot_frame_rollout_never_becomes_control_safe(tmp_path: Path) -> None:
    spatial = _write_spatial_pack(tmp_path / "spatial", count=4, robot_frame_truth=False)
    out = tmp_path / "future_pack"
    build_future_bev_rollout_pack(sources=[spatial], out_dir=out, horizons_s=(1.0,))
    manifest = load_example_npz(out / "examples" / "future_rollout_000000.npz")
    frames, _source_manifest = load_spatial_rollout_frames(spatial)
    assert frames[0].robot_frame_truth is False
    assert bool(manifest["control_safe"].item()) is False
    assert bool(manifest["not_executed"].item()) is True


def test_candidate_collision_and_new_area_labels_on_tiny_bev() -> None:
    candidates = generate_default_candidates(grid_shape=(12, 12), meters_per_cell=0.1, robot_radius_m=0.05)
    straight = next(candidate for candidate in candidates if candidate.id == "straight_short")
    future_occupied = np.zeros((1, 12, 12), dtype=np.float32)
    future_unknown = np.zeros((1, 12, 12), dtype=np.float32)
    future_free = np.ones((1, 12, 12), dtype=np.float32)
    newly_observed = np.zeros((1, 12, 12), dtype=np.float32)
    for row, col in straight.footprint_cells:
        future_occupied[0, row, col] = 1.0
        if row < 11:
            newly_observed[0, row, col] = 1.0
    labels = candidate_outcome_labels(
        candidates=candidates,
        future_occupied=future_occupied,
        future_unknown=future_unknown,
        newly_observed=newly_observed,
        future_free=future_free,
        horizon_valid=np.asarray([1.0], dtype=np.float32),
    )
    straight_index = [candidate.id for candidate in candidates].index("straight_short")
    rotate_index = [candidate.id for candidate in candidates].index("rotate_left")
    assert labels["candidate_collision"][straight_index] == 1.0
    assert labels["candidate_new_area_gain"][straight_index] > labels["candidate_new_area_gain"][rotate_index]
    assert labels["candidate_valid_mask"][straight_index] == 1.0


def test_candidate_labels_separate_unsafe_now_and_make_rotate_meaningful() -> None:
    candidates = generate_default_candidates(grid_shape=(12, 12), meters_per_cell=0.1, robot_radius_m=0.05)
    straight_index = [candidate.id for candidate in candidates].index("straight_short")
    stop_index = [candidate.id for candidate in candidates].index("stop")
    rotate_index = [candidate.id for candidate in candidates].index("rotate_left")
    straight = candidates[straight_index]
    future_occupied = np.zeros((1, 12, 12), dtype=np.float32)
    future_unknown = np.zeros((1, 12, 12), dtype=np.float32)
    future_free = np.ones((1, 12, 12), dtype=np.float32)
    newly_observed = np.zeros((1, 12, 12), dtype=np.float32)
    current_occupied = np.zeros((12, 12), dtype=np.float32)
    current_risky = np.zeros((12, 12), dtype=np.float32)
    current_unknown = np.zeros((12, 12), dtype=np.float32)
    current_free = np.ones((12, 12), dtype=np.float32)
    for row, col in straight.footprint_cells:
        if row < 10:
            current_occupied[row, col] = 1.0
    labels = candidate_outcome_labels(
        candidates=candidates,
        current_occupied=current_occupied,
        current_risky=current_risky,
        current_unknown=current_unknown,
        current_free=current_free,
        current_traversable=current_free,
        future_occupied=future_occupied,
        future_unknown=future_unknown,
        newly_observed=newly_observed,
        future_free=future_free,
        horizon_valid=np.asarray([1.0], dtype=np.float32),
    )
    assert labels["candidate_future_collision"][straight_index] == 0.0
    assert labels["candidate_unsafe_now"][straight_index] > 0.0
    assert labels["candidate_collision"][straight_index] >= labels["candidate_unsafe_now"][straight_index]
    assert labels["candidate_progress"][rotate_index] > labels["candidate_progress"][stop_index]


def test_route_balanced_sampling_preserves_routes_splits_and_writes_qa(tmp_path: Path) -> None:
    sources = [
        _write_spatial_pack(tmp_path / "spatial_a", count=12, source_name="route_a", include_review=True),
        _write_spatial_pack(tmp_path / "spatial_b", count=12, source_name="route_b", include_review=True),
        _write_spatial_pack(tmp_path / "spatial_c", count=12, source_name="route_c", include_review=True),
    ]
    out = tmp_path / "future_pack"
    build_future_bev_rollout_pack(
        sources=sources,
        out_dir=out,
        horizons_s=(1.0,),
        max_examples=18,
        sampling_strategy="route_balanced",
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    included = {record["source_name"]: int(record["included_example_count"]) for record in manifest["sources"]}
    assert set(included) == {"route_a", "route_b", "route_c"}
    assert all(count > 0 for count in included.values())
    examples = manifest["examples"]
    assert {record["source_name"] for record in examples} == {"route_a", "route_b", "route_c"}
    assert {"train", "val", "review"}.issubset({record["split"] for record in examples})
    label_qa = json.loads((out / "label_qa.json").read_text(encoding="utf-8"))
    assert "route_a::train" in label_qa["by_route_split"]
    assert label_qa["by_route_split"]["route_a::train"]["future_horizon_valid_fraction"] >= 0.0
    assert manifest["sampling_strategy"] == "route_balanced"
    assert manifest["label_qa_path"] == "label_qa.json"


def test_future_bev_rollout_train_eval_smoke(tmp_path: Path) -> None:
    spatial = _write_spatial_pack(tmp_path / "spatial", count=7)
    pack = tmp_path / "future_pack"
    train_out = tmp_path / "train"
    eval_out = tmp_path / "eval.json"
    build_future_bev_rollout_pack(sources=[spatial], out_dir=pack, horizons_s=(1.0,), max_examples=6)
    train_metrics = train_future_bev_rollout_v1(
        rollout_pack=pack,
        out_dir=train_out,
        max_steps=2,
        batch_size=2,
        hidden_channels=12,
        tiny_overfit=True,
        device_name="cpu",
    )
    eval_metrics = eval_future_bev_rollout_v1(
        checkpoint=train_out / "checkpoint.pt",
        rollout_pack=pack,
        out_path=eval_out,
        split="train",
        batch_size=2,
        device_name="cpu",
    )
    assert (train_out / "checkpoint.pt").exists()
    assert train_metrics["control_safe"] is False
    assert eval_metrics["control_safe"] is False
    assert "future_occupied_iou_or_proxy" in eval_metrics
    assert "candidate_new_area_gain_ranking_quality" in eval_metrics

    scorer = load_runtime_future_rollout_scorer(train_out / "checkpoint.pt", device=torch.device("cpu"))
    bev_arrays = _bev_arrays(0)
    local_bev = LocalBev(
        free=bev_arrays["free"],
        occupied=bev_arrays["obstacle"],
        unknown=bev_arrays["unknown"],
        traversable=bev_arrays["free"],
        risky=bev_arrays["obstacle"],
        confidence=bev_arrays["confidence"],
        source="fixture",
    )
    candidates = generate_default_candidates(grid_shape=(12, 12), meters_per_cell=0.1, robot_radius_m=0.05)
    decision = decide_trajectory(
        bev=local_bev,
        candidates=candidates,
        coverage_memory=CoverageMemory((12, 12), meters_per_cell=0.1),
        pose_delta=None,
        future_rollout_scorer=scorer,
        sensor_mask=np.zeros((4,), dtype=np.float32),
        policy_bev_source="fixture",
    )
    assert decision.debug["trajectory_scorer_mode"] == "future_rollout"
    assert decision.debug["cmd_vel_emitted"] is False
    assert decision.debug["control_safe"] is False
    assert all(candidate["trajectory_score"]["control_safe"] is False for candidate in decision.candidate_trajectories)


def _write_spatial_pack(
    root: Path,
    *,
    count: int,
    missing_pose_frame_ids: set[int] | None = None,
    robot_frame_truth: bool = True,
    source_name: str = "unit_route",
    include_review: bool = False,
) -> Path:
    missing_pose_frame_ids = missing_pose_frame_ids or set()
    examples_dir = root / "examples"
    route_dir = root / "route"
    examples_dir.mkdir(parents=True, exist_ok=True)
    route_dir.mkdir(parents=True, exist_ok=True)
    records = []
    association_frames = []
    for frame_id in range(count):
        if include_review and count >= 6:
            split = "train" if frame_id < count - 4 else ("val" if frame_id < count - 2 else "review")
        else:
            split = "train" if frame_id < max(2, count - 2) else "val"
        arrays = _bev_arrays(frame_id)
        example_path = examples_dir / f"frame_{frame_id:06d}.npz"
        timestamp_ns = frame_id * 1_000_000_000
        write_deterministic_npz(
            example_path,
            {
                "frame_id": scalar_int(frame_id),
                "timestamp_ns": scalar_int(timestamp_ns),
                "timestamp": scalar_int(timestamp_ns),
                "rgb_ref": scalar_str(f"frames/frame_{frame_id:06d}.jpg"),
                "rgb_path": scalar_str(f"frames/frame_{frame_id:06d}.jpg"),
                "bev_free": arrays["free"],
                "bev_obstacle": arrays["obstacle"],
                "bev_unknown": arrays["unknown"],
                "bev_confidence": arrays["confidence"],
                "weak_label": scalar_bool(True),
                "control_safe": scalar_bool(False),
                "not_robot_frame_truth": scalar_bool(not robot_frame_truth),
                "split": scalar_str(split),
                "split_unit_id": scalar_str(source_name),
                "pose_delta_mask": scalar_float(1.0),
                "action_label_mask": scalar_float(0.0),
                "imu_label_mask": scalar_float(0.0),
                "wheel_label_mask": scalar_float(0.0),
                "source_name": scalar_str(source_name),
                "source_family": scalar_str("public_robot_mounted" if robot_frame_truth else "public_rgbd_camera_pose"),
                "supervision_grade": scalar_str("public_robot_frame_geometry" if robot_frame_truth else "public_rgbd_anchor"),
                "dataset_frame_type": scalar_str("public_robot_mounted" if robot_frame_truth else "public_rgbd_camera_pose_geometry"),
                "robot_frame_truth": scalar_bool(robot_frame_truth),
            },
        )
        records.append(
            {
                "sequence_id": "seq",
                "camera_id": "front_rgb",
                "frame_id": frame_id,
                "timestamp_ns": timestamp_ns,
                "split": split,
                "split_unit_id": source_name,
                "source_name": source_name,
                "source_family": "public_robot_mounted" if robot_frame_truth else "public_rgbd_camera_pose",
                "supervision_grade": "public_robot_frame_geometry" if robot_frame_truth else "public_rgbd_anchor",
                "dataset_frame_type": "public_robot_mounted" if robot_frame_truth else "public_rgbd_camera_pose_geometry",
                "robot_frame_truth": robot_frame_truth,
                "not_robot_frame_truth": not robot_frame_truth,
                "example_path": f"examples/frame_{frame_id:06d}.npz",
            }
        )
        association: dict[str, object] = {"frame_id": frame_id, "timestamp_ns": timestamp_ns}
        if frame_id not in missing_pose_frame_ids:
            association["base_pose"] = _base_pose(0.10 * frame_id, 0.0, 0.0)
        association_frames.append(association)
    write_json(
        route_dir / OPENLORIS_ROUTE_ASSOCIATIONS_FILE,
        {"source_type": "openloris_scene_associations", "frames": association_frames},
        pretty=True,
    )
    write_json(
        root / "manifest.json",
        {
            "schema_version": "homebrain.spatial_train_pack.v0",
            "package_type": "SpatialTrainPack",
            "source_log": route_dir.as_posix(),
            "source_name": source_name,
            "source_family": "public_robot_mounted" if robot_frame_truth else "public_rgbd_camera_pose",
            "dataset_frame_type": "public_robot_mounted" if robot_frame_truth else "public_rgbd_camera_pose_geometry",
            "robot_supervision_grade": "public_robot_frame_geometry" if robot_frame_truth else "public_rgbd_anchor",
            "robot_frame_truth": robot_frame_truth,
            "not_robot_frame_truth": not robot_frame_truth,
            "weak_label": True,
            "control_safe": False,
            "replay_only": True,
            "not_executed": True,
            "camera_config": {"meters_per_cell": 0.1, "robot_radius_m": 0.05},
            "grid_shape": [12, 12],
            "examples": records,
            "frames": records,
        },
        pretty=True,
    )
    return root


def _bev_arrays(frame_id: int) -> dict[str, np.ndarray]:
    free = np.zeros((12, 12), dtype=np.float32)
    obstacle = np.zeros((12, 12), dtype=np.float32)
    unknown = np.ones((12, 12), dtype=np.float32)
    free[7:12, 4:8] = 1.0
    unknown[7:12, 4:8] = 0.0
    reveal_row = max(1, 7 - frame_id)
    free[reveal_row:7, 5:7] = 1.0
    unknown[reveal_row:7, 5:7] = 0.0
    obstacle[5, min(10, 4 + frame_id)] = 1.0
    unknown[5, min(10, 4 + frame_id)] = 0.0
    confidence = np.where(unknown > 0.5, 0.2, 0.9).astype(np.float32)
    return {"free": free, "obstacle": obstacle, "unknown": unknown, "confidence": confidence}


def _base_pose(x_m: float, y_m: float, yaw_rad: float) -> dict[str, float]:
    return {
        "tx": float(x_m),
        "ty": float(y_m),
        "tz": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": math.sin(yaw_rad / 2.0),
        "qw": math.cos(yaw_rad / 2.0),
    }
