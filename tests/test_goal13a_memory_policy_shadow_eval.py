from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from homebrain.data.spatial_dataset import write_deterministic_npz
from homebrain.messages.schema import BrainOutputEvent
from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.policies.run_trajectory_scorer import _record_from_model_event
from homebrain.policies.trajectory_scorer import LocalBev
from homebrain.tools.goal13a_memory_policy_shadow_eval import (
    COLLAPSE_DOMINANT_MAX,
    ShadowFrame,
    _include_hidden_steps,
    _include_occluded_steps,
    deterministic_occlusion_keep_mask,
    memory_action_benefit_gate,
    score_shadow_frames,
    summarize_decision_records,
)


def test_loads_v1_current_and_memory_bev_explicitly(tmp_path) -> None:
    artifact = tmp_path / "brain_outputs" / "spatial_v1" / "front_000001.npz"
    shape = (4, 4)
    write_deterministic_npz(
        artifact,
        {
            "current_bev_free_prob": np.ones(shape, dtype=np.float32),
            "current_bev_occupied_prob": np.zeros(shape, dtype=np.float32),
            "current_bev_unknown_prob": np.zeros(shape, dtype=np.float32),
            "current_bev_traversable_prob": np.ones(shape, dtype=np.float32),
            "current_bev_risky_prob": np.zeros(shape, dtype=np.float32),
            "memory_bev_free_prob": np.zeros(shape, dtype=np.float32),
            "memory_bev_occupied_prob": np.ones(shape, dtype=np.float32),
            "memory_bev_unknown_prob": np.zeros(shape, dtype=np.float32),
            "memory_bev_traversable_prob": np.zeros(shape, dtype=np.float32),
            "memory_bev_risky_prob": np.ones(shape, dtype=np.float32),
            "uncertainty_grid": np.full(shape, 0.25, dtype=np.float32),
        },
    )
    event = BrainOutputEvent(
        timestamp_ns=10,
        sequence_id="seq",
        source="spatial_memory_net_v1",
        input_event_ids=["frame"],
        pose_confidence=1.0,
        uncertainty=0.25,
        local_bev_ref="brain_outputs/spatial_v1/front_000001.npz",
        debug={"input_frame_id": 1, "camera_id": "front"},
    )

    current = _record_from_model_event(event, tmp_path, bev_source="v1_current_bev")
    memory = _record_from_model_event(event, tmp_path, bev_source="v1_memory_bev")

    assert current.bev.source == "v1_current_bev"
    assert memory.bev.source == "v1_memory_bev"
    assert float(current.bev.free.mean()) == 1.0
    assert float(memory.bev.occupied.mean()) == 1.0
    with pytest.raises(ValueError, match="explicit"):
        _record_from_model_event(event, tmp_path, bev_source="model")


def test_shadow_eval_records_have_report_metrics_and_no_cmd_vel() -> None:
    frames = [_shadow_frame(_open_bev(), _open_bev(), _open_bev())]
    candidates = _candidates()

    records = score_shadow_frames(frames, candidates=candidates, meters_per_cell=0.1, mode="unit")
    metrics = summarize_decision_records(records)

    assert records[0]["cmd_vel"] is None
    assert records[0]["control_safe"] is False
    assert records[0]["not_executed"] is True
    assert metrics["schema_version"] == "homebrain.goal13a_mode_shadow_metrics.v0"
    assert "selected_action_distribution" in metrics["per_source_metrics"]["v1_memory_bev"]
    assert "memory_vs_current_action_changed_fraction" in metrics["memory_vs_current"]


def test_memory_action_benefit_gate_passes_only_on_real_memory_benefit() -> None:
    current = _source_metrics(unsafe=0.2, unknown=0.4, oracle_agree=0.5, future_agree=0.5, collision=0.0)
    memory = _source_metrics(unsafe=0.1, unknown=0.4, oracle_agree=0.5, future_agree=0.5, collision=0.0)
    gate = memory_action_benefit_gate(
        normal_metrics=_mode_metrics(current=current, memory=memory),
        hidden_metrics=_mode_metrics(current=current, memory=memory),
        occlusion_metrics={"aggregate": _mode_metrics(current=current, memory=memory)},
        route_out_metrics={"available": True, "blocked": False, "folds": [{"memory_quality_delta": 0.1}]},
    )

    assert gate["memory_action_benefit_pass"] is True

    no_benefit = memory_action_benefit_gate(
        normal_metrics=_mode_metrics(current=current, memory=current),
        hidden_metrics=_mode_metrics(current=current, memory=current),
        occlusion_metrics={"aggregate": _mode_metrics(current=current, memory=current)},
        route_out_metrics={"available": True, "blocked": False, "folds": [{"memory_quality_delta": 0.1}]},
    )
    assert no_benefit["memory_action_benefit_pass"] is False
    assert "did not reduce" in no_benefit["failure_reasons"][0]


def test_memory_source_cannot_be_treated_as_oracle() -> None:
    frame = _shadow_frame(_open_bev(), _open_bev(), _open_bev(source="oracle_bev"))
    with pytest.raises(ValueError, match="cannot be treated as oracle"):
        score_shadow_frames([frame], candidates=_candidates(), meters_per_cell=0.1, mode="unit")


def test_goal11b_collapse_detection_blocks_gate() -> None:
    current = _source_metrics(unsafe=0.2, unknown=0.4, oracle_agree=0.5, future_agree=0.5, collision=0.0)
    memory = _source_metrics(unsafe=0.1, unknown=0.3, oracle_agree=0.6, future_agree=0.6, collision=0.0)
    memory["dominant_action_fraction"] = COLLAPSE_DOMINANT_MAX + 0.01
    memory["goal11b_distribution_collapse_flag"] = True

    gate = memory_action_benefit_gate(
        normal_metrics=_mode_metrics(current=current, memory=memory),
        hidden_metrics=_mode_metrics(current=current, memory=memory),
        occlusion_metrics={"aggregate": _mode_metrics(current=current, memory=memory)},
        route_out_metrics={"available": True, "blocked": False, "folds": [{"memory_quality_delta": 0.1}]},
    )

    assert gate["memory_action_benefit_pass"] is False
    assert any("collapse" in reason for reason in gate["failure_reasons"])


def test_hidden_and_occlusion_subset_selection_is_deterministic() -> None:
    batch = _hidden_batch()
    outputs = {"update_mask": torch.zeros((1, 2, 1, 4, 4), dtype=torch.bool)}

    first_hidden = _include_hidden_steps(batch, outputs, future_horizon=1, meters_per_cell=1.0)
    second_hidden = _include_hidden_steps(batch, outputs, future_horizon=1, meters_per_cell=1.0)
    update = torch.ones((1, 2, 1, 4, 4), dtype=torch.float32)
    frame_ids = torch.tensor([[3, 4]], dtype=torch.int64)
    keep_a = deterministic_occlusion_keep_mask(update, mode="random_block_dropout", frame_ids=frame_ids)
    keep_b = deterministic_occlusion_keep_mask(update, mode="random_block_dropout", frame_ids=frame_ids)
    first_occ = _include_occluded_steps(update, update * keep_a)
    second_occ = _include_occluded_steps(update, update * keep_b)

    assert first_hidden == second_hidden
    assert first_hidden == [(0, 0)]
    assert torch.equal(keep_a, keep_b)
    assert first_occ == second_occ
    assert first_occ


def _open_bev(source: str = "unit") -> LocalBev:
    shape = (24, 24)
    free = np.ones(shape, dtype=np.float32)
    occupied = np.zeros(shape, dtype=np.float32)
    unknown = np.zeros(shape, dtype=np.float32)
    return LocalBev(
        free=free,
        occupied=occupied,
        unknown=unknown,
        traversable=free.copy(),
        risky=occupied.copy(),
        confidence=np.ones(shape, dtype=np.float32),
        source=source,
    )


def _shadow_frame(oracle: LocalBev, current: LocalBev, memory: LocalBev) -> ShadowFrame:
    return ShadowFrame(
        source_name="route",
        sequence_id="seq",
        camera_id="front",
        frame_id=1,
        timestamp_ns=1,
        pose_delta=None,
        oracle_bev=oracle,
        v1_current_bev=current,
        v1_memory_bev=memory,
        v0_current_bev=None,
        future_motion_label="straight_short",
        subset_reason="unit",
    )


def _candidates() -> list:
    return generate_default_candidates(grid_shape=(24, 24), meters_per_cell=0.1, robot_radius_m=0.15)


def _source_metrics(
    *,
    unsafe: float,
    unknown: float,
    oracle_agree: float,
    future_agree: float,
    collision: float,
) -> dict[str, object]:
    return {
        "unsafe_selected_rate": unsafe,
        "unknown_penalty_mean": unknown,
        "agreement_with_oracle_action": oracle_agree,
        "agreement_with_future_motion_label": future_agree,
        "future_motion_label_valid_count": 3,
        "collision_proxy_rate": collision,
        "stop_fraction": 0.1,
        "all_motion_candidates_risky_fraction": 0.0,
        "goal11b_distribution_collapse_flag": False,
        "dominant_action_fraction": 0.4,
        "action_entropy": 2.0,
    }


def _mode_metrics(*, current: dict[str, object], memory: dict[str, object]) -> dict[str, object]:
    return {
        "per_source_metrics": {
            "v1_current_bev": dict(current),
            "v1_memory_bev": dict(memory),
        }
    }


def _hidden_batch() -> dict[str, torch.Tensor]:
    labels = torch.zeros((1, 2, 5, 4, 4), dtype=torch.float32)
    labels[:, :, 2] = 1.0
    labels[:, 1, 2, 2, 2] = 0.0
    labels[:, 1, 0, 2, 2] = 1.0
    pose_mask = torch.zeros((1, 2, 1), dtype=torch.float32)
    pose_mask[:, 1] = 1.0
    return {
        "bev_labels": labels,
        "pose_delta_to_current": torch.zeros((1, 2, 3), dtype=torch.float32),
        "pose_delta_to_current_mask": pose_mask,
        "frame_id": torch.tensor([[0, 1]], dtype=torch.int64),
    }
