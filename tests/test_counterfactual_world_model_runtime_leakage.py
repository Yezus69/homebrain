from pathlib import Path
import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from homebrain.brain.counterfactual_dynamic_bev_world_model_v0 import (
    CounterfactualDynamicBEVWorldModelV0,
    CounterfactualDynamicBEVWorldModelV0Config,
    save_checkpoint,
)
from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.policies.trajectory_scorer import LocalBev
from homebrain.runtime.counterfactual_world_model_scorer import (
    load_counterfactual_world_model,
    predict_future_bev_from_scene_state,
    score_candidates_with_world_model,
)


FORBIDDEN = {
    "target_bev",
    "future_bev",
    "future_labels",
    "teacher_masks",
    "candidate_oracle_cost",
    "route_ground_truth",
    "ground_truth_global_trajectory",
    "future_frames",
}


def test_runtime_signatures_reject_forbidden_fields(tmp_path: Path) -> None:
    assert FORBIDDEN.isdisjoint(inspect.signature(predict_future_bev_from_scene_state).parameters)
    assert FORBIDDEN.isdisjoint(inspect.signature(score_candidates_with_world_model).parameters)
    runtime = _runtime(tmp_path)
    history = np.zeros((2, 7, 8, 8), dtype=np.float32)
    pose = np.zeros((2, 3), dtype=np.float32)
    trajectories = np.zeros((3, 4, 3), dtype=np.float32)
    for name in FORBIDDEN:
        with pytest.raises(TypeError):
            predict_future_bev_from_scene_state(runtime, history, pose, trajectories, **{name: object()})  # type: ignore[arg-type]


def test_runtime_scores_only_from_allowed_scene_inputs(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    candidates = generate_default_candidates(grid_shape=(8, 8), meters_per_cell=0.1, robot_radius_m=0.05)
    free = np.ones((8, 8), dtype=np.float32)
    local = LocalBev(
        free=free,
        occupied=np.zeros_like(free),
        unknown=np.zeros_like(free),
        traversable=free,
        risky=np.zeros_like(free),
        uncertainty=np.zeros_like(free),
        confidence=np.ones_like(free),
        source="runtime_leakage_fixture",
    )
    result = score_candidates_with_world_model(
        runtime,
        local_bev_history=[local, local],
        pose_delta_history=np.zeros((2, 3), dtype=np.float32),
        previous_action_history=np.zeros((2, 2), dtype=np.float32),
        sensor_mask=np.ones((2, 4), dtype=np.float32),
        candidates=candidates,
    )
    assert "future_risk_debug_maps" in result
    assert "candidate_scores" in result
    debug = result["debug"]
    assert debug["no_future_labels_used_at_runtime"] is True
    assert debug["no_teacher_fields_at_runtime"] is True
    assert debug["replay_only"] is True
    assert debug["control_safe"] is False
    assert "future_labels" not in debug["runtime_inputs"]


def _runtime(tmp_path: Path):
    model = CounterfactualDynamicBEVWorldModelV0(
        CounterfactualDynamicBEVWorldModelV0Config(
            bev_shape=(8, 8),
            history_frames=2,
            hidden_channels=8,
            future_horizons_sec=(0.5,),
            meters_per_cell=0.1,
            robot_radius_m=0.05,
        )
    )
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model,
        metadata={
            "no_future_labels_used_at_runtime": True,
            "no_teacher_fields_at_runtime": True,
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "raw_pwm_emitted": False,
            "hardware_validated": False,
        },
        metrics={},
    )
    return load_counterfactual_world_model(checkpoint, device="cpu")
