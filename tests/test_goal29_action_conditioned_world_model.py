from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from homebrain.brain.future_bev_rollout_v1 import FutureBEVRolloutV1, FutureBEVRolloutV1Config, score_local_bev_with_future_rollout
from homebrain.data.spatial_dataset import load_example_npz
from homebrain.eval.goal29_fixture_eval import run_goal29_fixture_eval
from homebrain.policies.candidate_trajectories import CandidateTrajectory, generate_default_candidates
from homebrain.policies.runtime_decision import RuntimeFutureRolloutScorer, decide_trajectory
from homebrain.policies.trajectory_scorer import CoverageMemory, LocalBev
from homebrain.train.future_bev_rollout_dataset import FutureBEVRolloutDataset
from homebrain.train.goal29_fixtures import GOAL29_FIXTURE_NAMES, write_goal29_fixture_rollout_pack


class _ContextEchoFutureRollout(torch.nn.Module):
    def __init__(self, *, bev_shape: tuple[int, int] = (8, 8), history_steps: int = 2) -> None:
        super().__init__()
        self.config = FutureBEVRolloutV1Config(
            bev_shape=bev_shape,
            feature_dim=1,
            sensor_dim=4,
            hidden_channels=4,
            history_steps=history_steps,
            horizons_s=(0.5, 1.0),
            meters_per_cell=0.1,
            robot_radius_m=0.05,
        )
        self.candidates = generate_default_candidates(
            grid_shape=bev_shape,
            meters_per_cell=0.1,
            robot_radius_m=0.05,
        )
        self.candidate_ids = tuple(candidate.id for candidate in self.candidates)

    def forward(
        self,
        current_bev: torch.Tensor,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        sensor_mask: torch.Tensor,
        memory_bev: torch.Tensor | None = None,
        bev_history: torch.Tensor | None = None,
        uncertainty_map: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = current_bev.shape[0]
        height, width = self.config.bev_shape
        horizon_count = len(self.config.horizons_s)
        candidate_count = len(self.candidate_ids)
        context = torch.zeros((batch,), dtype=current_bev.dtype, device=current_bev.device)
        if memory_bev is not None:
            context = context + memory_bev.mean(dim=(1, 2, 3))
        if bev_history is not None:
            context = context + 2.0 * bev_history.mean(dim=(1, 2, 3, 4))
        if uncertainty_map is not None:
            context = context + 3.0 * uncertainty_map.mean(dim=(1, 2, 3))
        candidate_offsets = torch.linspace(0.0, 0.2, candidate_count, dtype=current_bev.dtype, device=current_bev.device)
        horizon_offsets = torch.linspace(0.0, 0.1, horizon_count, dtype=current_bev.dtype, device=current_bev.device)
        candidate_logits = context[:, None] + candidate_offsets[None, :]
        horizon_logits = context[:, None, None] + horizon_offsets[None, None, :]
        future_logits = context.view(batch, 1, 1, 1, 1).expand(batch, horizon_count, 3, height, width)
        risk_prob = torch.sigmoid(context.view(batch, 1, 1, 1, 1).expand(batch, horizon_count, 1, height, width))
        uncertainty = torch.sigmoid(
            (context + 0.25).view(batch, 1, 1, 1, 1).expand(batch, horizon_count, 1, height, width)
        )
        return {
            "future_bev_logits": future_logits,
            "future_risk_prob": risk_prob,
            "uncertainty_grid": uncertainty,
            "candidate_outcome_logits": torch.zeros(
                (batch, candidate_count, 6),
                dtype=current_bev.dtype,
                device=current_bev.device,
            ),
            "candidate_collision_logit": candidate_logits,
            "candidate_future_collision": torch.sigmoid(candidate_logits + 0.05),
            "candidate_unsafe_now": torch.sigmoid(candidate_logits - 0.25),
            "candidate_unknown_exposure": torch.sigmoid(candidate_logits - 0.5),
            "candidate_new_area_gain": torch.sigmoid(-candidate_logits),
            "candidate_progress": torch.sigmoid(-candidate_logits - 0.1),
            "candidate_horizon_future_risk": torch.sigmoid(horizon_logits.expand(batch, candidate_count, horizon_count)),
        }


def test_goal29_fixture_pack_shapes_determinism_and_moving_obstacle(tmp_path: Path) -> None:
    first = tmp_path / "pack_a"
    second = tmp_path / "pack_b"
    write_goal29_fixture_rollout_pack(first)
    write_goal29_fixture_rollout_pack(second)
    assert (first / "manifest.json").read_text(encoding="utf-8") == (second / "manifest.json").read_text(encoding="utf-8")

    dataset = FutureBEVRolloutDataset(first)
    sample = dataset[0]
    assert sample["bev_history"].shape == (3, 5, 16, 16)
    assert sample["memory_bev"].shape == (5, 16, 16)
    assert sample["uncertainty_map"].shape == (1, 16, 16)
    assert sample["future_risk"].shape == (3, 1, 16, 16)
    assert sample["candidate_horizon_future_risk"].shape == (9, 3)
    assert sample["candidate_footprint_masks"].shape == (9, 16, 16)

    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    moving = next(record for record in manifest["examples"] if record["fixture_name"] == "moving_obstacle_crossing_path")
    example = load_example_npz(first / moving["example_path"])
    candidate_ids = [str(value) for value in example["candidate_ids"]]
    straight_index = candidate_ids.index("straight_medium")
    stop_index = candidate_ids.index("stop")
    assert float(example["candidate_future_collision"][straight_index]) > float(example["candidate_future_collision"][stop_index])
    assert float(example["candidate_horizon_future_risk"][straight_index].max()) >= 0.5
    assert float(example["candidate_horizon_future_risk"][stop_index].max()) < 0.5


def test_goal29_model_forward_consumes_history_memory_and_risk_targets(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    write_goal29_fixture_rollout_pack(pack)
    dataset = FutureBEVRolloutDataset(pack)
    sample = dataset[0]
    batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else value for key, value in sample.items()}
    model = FutureBEVRolloutV1(
        FutureBEVRolloutV1Config(
            bev_shape=dataset.grid_shape,
            feature_dim=1,
            sensor_dim=4,
            hidden_channels=8,
            history_steps=dataset.history_steps,
            horizons_s=dataset.horizons_s,
            meters_per_cell=0.1,
            robot_radius_m=0.05,
        )
    )
    outputs = model(
        batch["current_bev"],
        batch["features"],
        batch["feature_mask"],
        batch["sensor_mask"],
        memory_bev=batch["memory_bev"],
        bev_history=batch["bev_history"],
        uncertainty_map=batch["uncertainty_map"],
    )
    assert outputs["future_bev_logits"].shape == (1, 3, 3, 16, 16)
    assert outputs["future_risk_logits"].shape == (1, 3, 1, 16, 16)
    assert outputs["candidate_future_risk_logits"].shape == (1, 9, 3)
    assert torch.isfinite(outputs["candidate_horizon_future_risk"]).all()


def test_goal29_runtime_helper_output_changes_with_memory_history_and_uncertainty() -> None:
    model = _ContextEchoFutureRollout(bev_shape=(8, 8), history_steps=2)
    free = np.ones((8, 8), dtype=np.float32)
    local = LocalBev(
        free=free,
        occupied=np.zeros_like(free),
        unknown=np.zeros_like(free),
        traversable=free,
        risky=np.zeros_like(free),
        confidence=np.ones_like(free),
        uncertainty=np.zeros_like(free),
        source="goal29a_runtime_context_fixture",
    )
    empty_context = score_local_bev_with_future_rollout(
        model=model,  # type: ignore[arg-type]
        bev=local,
        patch_features=None,
        sensor_mask=np.ones((4,), dtype=np.float32),
        device=torch.device("cpu"),
        memory_bev=np.zeros((5, 8, 8), dtype=np.float32),
        bev_history=np.zeros((2, 5, 8, 8), dtype=np.float32),
        uncertainty_map=np.zeros((8, 8), dtype=np.float32),
    )
    changed_context = score_local_bev_with_future_rollout(
        model=model,  # type: ignore[arg-type]
        bev=local,
        patch_features=None,
        sensor_mask=np.ones((4,), dtype=np.float32),
        device=torch.device("cpu"),
        memory_bev=np.ones((5, 8, 8), dtype=np.float32),
        bev_history=np.ones((2, 5, 8, 8), dtype=np.float32),
        uncertainty_map=np.full((8, 8), 0.9, dtype=np.float32),
    )
    assert not np.allclose(
        empty_context["candidate_lower_is_better_score"],
        changed_context["candidate_lower_is_better_score"],
    )
    assert float(np.mean(changed_context["candidate_future_risk"])) > float(
        np.mean(empty_context["candidate_future_risk"])
    )


def test_goal29_runtime_scores_have_no_future_label_keys_and_safety_envelope_stops() -> None:
    model = FutureBEVRolloutV1(
        FutureBEVRolloutV1Config(
            bev_shape=(16, 16),
            feature_dim=1,
            sensor_dim=4,
            hidden_channels=8,
            history_steps=1,
            horizons_s=(0.5, 1.0, 2.0),
            meters_per_cell=0.1,
            robot_radius_m=0.05,
        )
    )
    free = np.ones((16, 16), dtype=np.float32)
    occupied = np.zeros((16, 16), dtype=np.float32)
    unknown = np.zeros((16, 16), dtype=np.float32)
    uncertainty = np.full((16, 16), 0.95, dtype=np.float32)
    local = LocalBev(
        free=free,
        occupied=occupied,
        unknown=unknown,
        traversable=free,
        risky=occupied,
        confidence=1.0 - uncertainty,
        uncertainty=uncertainty,
        source="goal29_runtime_fixture",
    )
    scores = score_local_bev_with_future_rollout(
        model=model,
        bev=local,
        patch_features=None,
        sensor_mask=np.ones((4,), dtype=np.float32),
        device=torch.device("cpu"),
    )
    assert "candidate_future_risk" in scores
    assert "future_labels" not in scores
    assert "teacher_artifacts" not in scores

    candidates = generate_default_candidates(grid_shape=(16, 16), meters_per_cell=0.1, robot_radius_m=0.05)
    decision = decide_trajectory(
        bev=local,
        candidates=candidates,
        coverage_memory=CoverageMemory((16, 16), meters_per_cell=0.1),
        pose_delta=None,
        policy_bev_source="fixture",
        sensor_age_s=2.0,
    )
    assert decision.selected_candidate_id == "stop"
    envelope = decision.debug["replay_safety_envelope"]
    assert envelope["stale_sensor_stop"] is True
    assert envelope["high_uncertainty_stop"] is True
    assert decision.debug["control_safe"] is False
    assert decision.debug["raw_pwm_emitted"] is False
    assert decision.debug["hardware_validated"] is False
    assert decision.debug["cmd_vel_proposal"]["linear_velocity_mps"] == 0.0


def test_goal29_runtime_decision_rejects_and_does_not_forward_future_labels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    forbidden_inputs = {
        "future_bev": np.full((2, 3, 8, 8), 7.0, dtype=np.float32),
        "future_risk": np.full((2, 1, 8, 8), 8.0, dtype=np.float32),
        "candidate_oracle_cost": np.full((9,), 9.0, dtype=np.float32),
        "teacher_artifacts": {"teacher_only": True},
        "route_ground_truth": {"pose": [1.0, 2.0, 3.0]},
    }
    forbidden_names = set(forbidden_inputs)
    assert forbidden_names.isdisjoint(inspect.signature(decide_trajectory).parameters)
    assert forbidden_names.isdisjoint(inspect.signature(score_local_bev_with_future_rollout).parameters)

    free = np.ones((8, 8), dtype=np.float32)
    local = LocalBev(
        free=free,
        occupied=np.zeros_like(free),
        unknown=np.zeros_like(free),
        traversable=free,
        risky=np.zeros_like(free),
        confidence=np.ones_like(free),
        uncertainty=np.zeros_like(free),
        source="goal29a_no_leakage_fixture",
    )
    candidates = generate_default_candidates(grid_shape=(8, 8), meters_per_cell=0.1, robot_radius_m=0.05)
    base_kwargs = {
        "bev": local,
        "candidates": candidates,
        "coverage_memory": CoverageMemory((8, 8), meters_per_cell=0.1),
        "pose_delta": None,
        "policy_bev_source": "fixture",
    }
    for name, value in forbidden_inputs.items():
        with pytest.raises(TypeError):
            decide_trajectory(**base_kwargs, **{name: value})  # type: ignore[arg-type]

    checkpoint = tmp_path / "fake_future_rollout.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    model = _ContextEchoFutureRollout(bev_shape=(8, 8), history_steps=2)
    scorer = RuntimeFutureRolloutScorer(
        model=model,
        checkpoint=checkpoint,
        metadata={},
        device=torch.device("cpu"),
    )
    forbidden_object_ids = {id(value) for value in forbidden_inputs.values()}
    captured: dict[str, object] = {}

    def fake_rollout_score(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        assert forbidden_names.isdisjoint(kwargs)
        assert not any(id(value) in forbidden_object_ids for value in kwargs.values())
        candidate_count = len(model.candidate_ids)
        horizon_count = len(model.config.horizons_s)
        return {
            "candidate_ids": list(model.candidate_ids),
            "candidate_collision": np.zeros((candidate_count,), dtype=np.float32),
            "candidate_future_collision": np.zeros((candidate_count,), dtype=np.float32),
            "candidate_future_risk": np.zeros((candidate_count,), dtype=np.float32),
            "candidate_horizon_future_risk": np.zeros((candidate_count, horizon_count), dtype=np.float32),
            "candidate_unsafe_now": np.zeros((candidate_count,), dtype=np.float32),
            "candidate_unknown_exposure": np.zeros((candidate_count,), dtype=np.float32),
            "candidate_new_area_gain": np.zeros((candidate_count,), dtype=np.float32),
            "candidate_progress": np.zeros((candidate_count,), dtype=np.float32),
            "candidate_lower_is_better_score": np.arange(candidate_count, dtype=np.float32),
            "future_bev_prob": np.zeros((horizon_count, 3, 8, 8), dtype=np.float32),
            "future_risk_prob": np.zeros((horizon_count, 8, 8), dtype=np.float32),
            "future_uncertainty_grid": np.zeros((horizon_count, 8, 8), dtype=np.float32),
            "future_horizons_s": np.asarray(model.config.horizons_s, dtype=np.float32),
            "scoring_formula": "fixture",
        }

    monkeypatch.setattr(
        "homebrain.policies.runtime_decision.score_local_bev_with_future_rollout",
        fake_rollout_score,
    )
    memory = np.ones((5, 8, 8), dtype=np.float32)
    history = np.ones((2, 5, 8, 8), dtype=np.float32)
    uncertainty = np.full((8, 8), 0.2, dtype=np.float32)
    decision = decide_trajectory(
        **base_kwargs,
        future_rollout_scorer=scorer,
        memory_bev=memory,
        bev_history=history,
        uncertainty_map=uncertainty,
    )
    assert decision.debug["trajectory_scorer_mode"] == "future_rollout"
    assert captured["memory_bev"] is memory
    assert captured["bev_history"] is history
    assert captured["uncertainty_map"] is uncertainty
    assert forbidden_names.isdisjoint(captured)


def test_goal29_invalid_candidate_is_rejected_and_stop_remains_available() -> None:
    free = np.ones((8, 8), dtype=np.float32)
    local = LocalBev(
        free=free,
        occupied=np.zeros_like(free),
        unknown=np.zeros_like(free),
        traversable=free,
        risky=np.zeros_like(free),
        confidence=np.ones_like(free),
        uncertainty=np.zeros_like(free),
        source="goal29_invalid_candidate_fixture",
    )
    candidates = generate_default_candidates(grid_shape=(8, 8), meters_per_cell=0.2, robot_radius_m=0.05)
    bad = CandidateTrajectory(
        id="bad_forward",
        cmd_vel_proxy={"linear_velocity_mps": float("nan"), "angular_velocity_radps": 0.0, "not_hardware_control": True},
        duration_s=1.0,
        poses=candidates[1].poses,
        footprint_cells=candidates[1].footprint_cells,
    )
    decision = decide_trajectory(
        bev=local,
        candidates=[candidates[0], bad, *candidates[2:]],
        coverage_memory=CoverageMemory((8, 8), meters_per_cell=0.2),
        pose_delta=None,
        policy_bev_source="fixture",
    )
    assert any(record["id"] == "stop" for record in decision.candidate_trajectories)
    rejected = next(record for record in decision.candidate_trajectories if record["id"] == "bad_forward")
    assert rejected["runtime_candidate_valid"] is False
    assert decision.debug["replay_safety_envelope"]["invalid_candidate_rejection_count"] == 1


def test_goal29_report_json_is_written_with_baselines_and_safety(tmp_path: Path) -> None:
    report = run_goal29_fixture_eval(
        out_dir=tmp_path / "goal29_eval",
        max_steps=2,
        hidden_channels=8,
        device_name="cpu",
    )
    report_path = tmp_path / "goal29_eval" / "goal29_report.json"
    assert report_path.exists()
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["goal"] == "Goal29a fixture probe"
    assert loaded["accepted"] is False
    assert isinstance(loaded["accepted_fixture_probe"], bool)
    assert loaded["accepted_robot_brain_milestone"] is False
    assert loaded["accepted_goal29_route_heldout"] is False
    assert loaded["accepted_goal29_robot_brain"] is False
    assert set(GOAL29_FIXTURE_NAMES).issubset(set(loaded["fixtures"]))
    assert "current_bev_copy_forward" in loaded["baselines"]
    assert "transparent_scorer" in loaded["baselines"]
    assert "Gate H" in loaded["gates_exercised"]
    assert "Gate H" not in loaded["gates_improved"]
    assert loaded["safety"]["replay_only"] is True
    assert loaded["safety"]["control_safe"] is False
    assert loaded["safety"]["raw_pwm_emitted"] is False
    assert loaded["safety"]["hardware_validated"] is False
    assert isinstance(report["accepted"], bool)
