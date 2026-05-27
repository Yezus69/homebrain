from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from homebrain.data.spatial_dataset import load_example_npz
from homebrain.eval.eval_passive_dynamic_future_risk import run_passive_dynamic_future_risk_eval
from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.policies.runtime_decision import RuntimeFutureRolloutScorer, decide_trajectory
from homebrain.policies.trajectory_scorer import CoverageMemory, LocalBev
from homebrain.teachers.passive_dynamic_teacher import write_passive_dynamic_fixture_sequences
from homebrain.train.future_bev_rollout_dataset import FutureBEVRolloutDataset
from homebrain.train.passive_dynamic_future_risk_pack import build_passive_dynamic_future_risk_pack


def test_passive_dynamic_teacher_sidecars_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "fixtures_a"
    second = tmp_path / "fixtures_b"
    write_passive_dynamic_fixture_sequences(first, frame_count=6, shape=(16, 16), fps=2.0)
    write_passive_dynamic_fixture_sequences(second, frame_count=6, shape=(16, 16), fps=2.0)

    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["sequences"] == second_manifest["sequences"]
    first_sidecar = first / "train_moving_blob_crossing_path" / "sidecars" / "frame_000002.npz"
    second_sidecar = second / "train_moving_blob_crossing_path" / "sidecars" / "frame_000002.npz"
    assert first_sidecar.read_bytes() == second_sidecar.read_bytes()


def test_passive_dynamic_pack_loads_and_marks_future_crossing_risk(tmp_path: Path) -> None:
    sequences = tmp_path / "fixtures"
    pack_a = tmp_path / "pack_a"
    pack_b = tmp_path / "pack_b"
    write_passive_dynamic_fixture_sequences(sequences, frame_count=7, shape=(16, 16), fps=2.0)
    build_passive_dynamic_future_risk_pack(input_dir=sequences, out_dir=pack_a)
    build_passive_dynamic_future_risk_pack(input_dir=sequences, out_dir=pack_b)
    assert (pack_a / "manifest.json").read_text(encoding="utf-8") == (pack_b / "manifest.json").read_text(encoding="utf-8")

    dataset = FutureBEVRolloutDataset(pack_a, split="val")
    sample = dataset[0]
    assert sample["current_bev"].shape == (5, 16, 16)
    assert sample["memory_bev"].shape == (5, 16, 16)
    assert sample["bev_history"].shape == (3, 5, 16, 16)
    assert sample["future_risk"].shape == (3, 1, 16, 16)
    assert sample["candidate_footprint_masks"].shape == (9, 16, 16)

    manifest = json.loads((pack_a / "manifest.json").read_text(encoding="utf-8"))
    moving = next(record for record in manifest["examples"] if record["scenario"] == "moving_blob_crossing_path")
    example = load_example_npz(pack_a / moving["example_path"])
    candidate_ids = [str(value) for value in example["candidate_ids"]]
    straight_index = candidate_ids.index("straight_medium")
    stop_index = candidate_ids.index("stop")
    turn_index = candidate_ids.index("arc_left_small")
    straight_risk = float(np.max(example["candidate_horizon_future_risk"][straight_index]))
    stop_risk = float(np.max(example["candidate_horizon_future_risk"][stop_index]))
    turn_risk = float(np.max(example["candidate_horizon_future_risk"][turn_index]))
    assert straight_risk > stop_risk
    assert straight_risk >= turn_risk
    assert bool(example["replay_only"].item()) is True
    assert bool(example["control_safe"].item()) is False
    assert bool(example["raw_pwm_emitted"].item()) is False
    assert bool(example["hardware_validated"].item()) is False


def test_passive_dynamic_eval_report_accepts_only_when_heldout_beats_baselines(tmp_path: Path) -> None:
    report = run_passive_dynamic_future_risk_eval(
        out_dir=tmp_path / "eval",
        max_steps=400,
        hidden_channels=16,
        batch_size=8,
        device_name="cpu",
    )
    report_path = tmp_path / "eval" / "passive_dynamic_future_risk_report.json"
    assert report_path.exists()
    assert report["accepted_robot_brain"] is False
    assert report["accepted_passive_dynamic_future_risk_probe"] is True
    metrics = report["metrics"]
    assert metrics["candidate_future_risk_mse"] < report["baselines"]["current_bev_copy_forward"]["candidate_future_risk_mse"]
    assert metrics["candidate_future_risk_mse"] < report["baselines"]["static_memory_copy_forward"]["candidate_future_risk_mse"]
    assert metrics["future_collision_recall_on_moving_obstacles"] > report["baselines"]["current_bev_copy_forward"][
        "moving_obstacle_collision_recall"
    ]
    assert metrics["unsafe_selected_rate"] == 0.0
    assert metrics["dominant_action_fraction"] < 0.95
    assert report["safety"]["replay_only"] is True
    assert report["safety"]["control_safe"] is False
    assert report["safety"]["raw_pwm_emitted"] is False
    assert report["safety"]["hardware_validated"] is False


def test_runtime_future_risk_explanation_fields_and_leakage_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    forbidden = {"future_bev", "future_risk", "candidate_oracle_cost", "teacher_artifacts", "route_ground_truth"}
    assert forbidden.isdisjoint(inspect.signature(decide_trajectory).parameters)

    free = np.ones((16, 16), dtype=np.float32)
    local = LocalBev(
        free=free,
        occupied=np.zeros_like(free),
        unknown=np.zeros_like(free),
        traversable=free,
        risky=np.zeros_like(free),
        confidence=np.ones_like(free),
        uncertainty=np.zeros_like(free),
        source="passive_dynamic_runtime_test",
    )
    candidates = generate_default_candidates(grid_shape=(16, 16), meters_per_cell=0.1, robot_radius_m=0.05)
    candidate_ids = [candidate.id for candidate in candidates]
    turn_index = candidate_ids.index("arc_left_small")

    class _NoopFutureRollout(torch.nn.Module):
        pass

    checkpoint = tmp_path / "future.pt"
    checkpoint.write_bytes(b"runtime fixture checkpoint")
    scorer = RuntimeFutureRolloutScorer(
        model=_NoopFutureRollout(),
        checkpoint=checkpoint,
        metadata={},
        device=torch.device("cpu"),
    )

    def fake_score(**kwargs: object) -> dict[str, object]:
        count = len(candidate_ids)
        horizon_count = 3
        dynamic_risk = np.full((count,), 0.95, dtype=np.float32)
        dynamic_risk[candidate_ids.index("stop")] = 0.0
        dynamic_risk[turn_index] = 0.0
        lower = np.full((count,), 4.0, dtype=np.float32)
        lower[turn_index] = 0.1
        return {
            "candidate_ids": candidate_ids,
            "candidate_collision": dynamic_risk.copy(),
            "candidate_future_collision": dynamic_risk.copy(),
            "candidate_future_risk": dynamic_risk,
            "candidate_horizon_future_risk": np.repeat(dynamic_risk[:, None], horizon_count, axis=1),
            "candidate_unsafe_now": np.zeros((count,), dtype=np.float32),
            "candidate_unknown_exposure": np.zeros((count,), dtype=np.float32),
            "candidate_new_area_gain": np.zeros((count,), dtype=np.float32),
            "candidate_progress": np.ones((count,), dtype=np.float32),
            "candidate_lower_is_better_score": lower,
            "future_bev_prob": np.zeros((horizon_count, 3, 16, 16), dtype=np.float32),
            "future_risk_prob": np.zeros((horizon_count, 16, 16), dtype=np.float32),
            "future_uncertainty_grid": np.zeros((horizon_count, 16, 16), dtype=np.float32),
            "future_horizons_s": np.asarray([0.5, 1.0, 2.0], dtype=np.float32),
            "scoring_formula": "runtime fixture",
        }

    monkeypatch.setattr("homebrain.policies.runtime_decision.score_local_bev_with_future_rollout", fake_score)
    decision = decide_trajectory(
        bev=local,
        candidates=candidates,
        coverage_memory=CoverageMemory((16, 16), meters_per_cell=0.1),
        pose_delta=None,
        future_rollout_scorer=scorer,
        policy_bev_source="fixture",
    )
    debug = decision.debug
    assert debug["selected_candidate_id_before_future_risk"] != debug["selected_candidate_id_after_future_risk"]
    assert debug["selected_candidate_id_after_future_risk"] == "arc_left_small"
    assert debug["future_risk_avoidance_reason"] in {
        "future_dynamic_risk_blocks_transparent_candidate",
        "future_rollout_score_changes_candidate",
    }
    assert "candidate_dynamic_future_risk" in debug
    assert "candidate_static_risk" in debug
    assert "candidate_dynamic_future_risk" in decision.artifact_arrays
    assert "candidate_static_risk" in decision.artifact_arrays
    assert debug["replay_only"] is True
    assert debug["not_executed"] is True
    assert debug["control_safe"] is False
    assert debug["raw_pwm_emitted"] is False
    assert debug["hardware_validated"] is False
