import hashlib
import json

import numpy as np

from homebrain.messages.schema import deterministic_json
from homebrain.policies.candidate_trajectories import candidates_hash, generate_default_candidates
from homebrain.policies.run_trajectory_scorer import run_trajectory_scorer
from homebrain.policies.synthetic_fixture import write_synthetic_policy_fixture
from homebrain.policies.trajectory_scorer import CoverageMemory, LocalBev, score_trajectories


def _open_bev(shape: tuple[int, int] = (24, 24)) -> LocalBev:
    free = np.ones(shape, dtype=np.float32)
    occupied = np.zeros(shape, dtype=np.float32)
    unknown = np.zeros(shape, dtype=np.float32)
    confidence = np.ones(shape, dtype=np.float32)
    return LocalBev(
        free=free,
        occupied=occupied,
        unknown=unknown,
        traversable=free.copy(),
        risky=occupied.copy(),
        confidence=confidence,
        source="synthetic",
    )


def _candidates() -> list:
    return generate_default_candidates(grid_shape=(24, 24), meters_per_cell=0.1, robot_radius_m=0.15)


def _score_by_id(decision):
    return {score.candidate_id: score for score in decision.scores}


def test_candidate_generation_is_deterministic() -> None:
    first = _candidates()
    second = _candidates()

    assert [candidate.id for candidate in first] == [
        "stop",
        "straight_short",
        "straight_medium",
        "arc_left_small",
        "arc_right_small",
        "arc_left_medium",
        "arc_right_medium",
        "rotate_left",
        "rotate_right",
    ]
    assert candidates_hash(first) == candidates_hash(second)
    assert first[2].cmd_vel_proxy["not_hardware_control"] is True
    assert first[2].footprint_cells == second[2].footprint_cells


def test_obstacle_candidates_are_penalized() -> None:
    candidates = _candidates()
    bev = _open_bev()
    straight = next(candidate for candidate in candidates if candidate.id == "straight_medium")
    stop = next(candidate for candidate in candidates if candidate.id == "stop")
    stop_cells = set(stop.footprint_cells)
    for row, col in straight.footprint_cells:
        if (row, col) not in stop_cells:
            bev.occupied[row, col] = 1.0
            bev.risky[row, col] = 1.0
            bev.free[row, col] = 0.0
            break

    decision = score_trajectories(bev=bev, candidates=candidates)
    scores = _score_by_id(decision)

    assert scores["straight_medium"].risk_score > scores["arc_left_small"].risk_score
    assert scores["straight_medium"].total_score > scores["arc_left_small"].total_score


def test_open_straight_path_is_preferred_when_coverage_gain_is_higher() -> None:
    candidates = _candidates()
    shape = (24, 24)
    free = np.zeros(shape, dtype=np.float32)
    occupied = np.zeros(shape, dtype=np.float32)
    unknown = np.ones(shape, dtype=np.float32)
    straight = next(candidate for candidate in candidates if candidate.id == "straight_medium")
    for row, col in straight.footprint_cells:
        free[row, col] = 1.0
        unknown[row, col] = 0.0
    bev = LocalBev(
        free=free,
        occupied=occupied,
        unknown=unknown,
        traversable=free.copy(),
        risky=occupied.copy(),
        confidence=np.where(unknown > 0, 0.0, 1.0).astype(np.float32),
        source="synthetic",
    )
    memory = CoverageMemory(shape, meters_per_cell=0.1)

    decision = score_trajectories(bev=bev, candidates=candidates, coverage_memory=memory)

    assert decision.selected_candidate_id == "straight_medium"
    assert decision.selected_score().coverage_gain_proxy > 0.0


def test_stop_selected_only_when_all_motion_candidates_are_risky() -> None:
    candidates = _candidates()
    open_decision = score_trajectories(bev=_open_bev(), candidates=candidates)

    risky_bev = _open_bev()
    for candidate in candidates:
        if candidate.id == "stop":
            continue
        for row, col in candidate.footprint_cells:
            risky_bev.occupied[row, col] = 1.0
            risky_bev.risky[row, col] = 1.0
            risky_bev.free[row, col] = 0.0
    risky_decision = score_trajectories(bev=risky_bev, candidates=candidates)

    assert open_decision.selected_candidate_id != "stop"
    assert risky_decision.selected_candidate_id == "stop"
    assert risky_decision.reason == "all_motion_candidates_risky_select_stop"


def test_policy_decision_artifact_hash_is_deterministic(tmp_path) -> None:
    fixture = tmp_path / "fixture"
    out_a = tmp_path / "policy_a"
    out_b = tmp_path / "policy_b"
    write_synthetic_policy_fixture(fixture, count=5)

    run_trajectory_scorer(log_dir=fixture, out_dir=out_a, bev_source="labels")
    run_trajectory_scorer(log_dir=fixture, out_dir=out_b, bev_source="labels")

    decisions_a = (out_a / "trajectory_decisions.jsonl").read_text(encoding="utf-8")
    decisions_b = (out_b / "trajectory_decisions.jsonl").read_text(encoding="utf-8")
    assert hashlib.sha256(decisions_a.encode("utf-8")).hexdigest() == hashlib.sha256(
        decisions_b.encode("utf-8")
    ).hexdigest()
    assert decisions_a == decisions_b

    first = json.loads(decisions_a.splitlines()[0])
    assert first["replay_only"] is True
    assert first["control_safe"] is False
    assert first["not_executed"] is True
    assert first["candidates"][0]["cmd_vel_proxy"]["not_hardware_control"] is True

    manifest = json.loads((out_a / "trajectory_policy_manifest.json").read_text(encoding="utf-8"))
    eval_metrics = json.loads((out_a / "trajectory_eval.json").read_text(encoding="utf-8"))
    assert manifest["replay_only"] is True
    assert manifest["control_safe"] is False
    assert manifest["not_executed"] is True
    assert eval_metrics["candidate_count"] == 9
    assert deterministic_json({"selected": eval_metrics["selected_candidate_id"]})
