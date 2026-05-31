from types import SimpleNamespace
import inspect

import numpy as np
import pytest

from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.policies.trajectory_scorer import LocalBev, score_trajectories
from homebrain.runtime.counterfactual_world_model_scorer import score_candidates_with_world_model
from homebrain.runtime.direct_bev_runtime import predict_local_bev_from_rgbd


def test_thin_cable_free_cell_is_penalized_by_hazard_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    shape = (16, 16)
    candidates = generate_default_candidates(grid_shape=shape, meters_per_cell=0.1, robot_radius_m=0.05)
    straight = next(candidate for candidate in candidates if candidate.id == "straight_medium")
    clear = next(candidate for candidate in candidates if candidate.id == "arc_left_small")
    stop_cells = set(next(candidate for candidate in candidates if candidate.id == "stop").footprint_cells)
    hazard = np.zeros(shape, dtype=np.float32)
    for row, col in straight.footprint_cells:
        if (row, col) not in stop_cells:
            hazard[row, col] = 1.0
            break
    free = np.ones(shape, dtype=np.float32)
    occupied = np.zeros(shape, dtype=np.float32)
    unknown = np.zeros(shape, dtype=np.float32)
    bev = LocalBev(
        free=free,
        occupied=occupied,
        unknown=unknown,
        traversable=free,
        risky=occupied,
        hazard=hazard,
        confidence=np.ones(shape, dtype=np.float32),
        uncertainty=np.zeros(shape, dtype=np.float32),
        source="thin_cable_fixture",
    )

    transparent = score_trajectories(bev=bev, candidates=candidates)
    scores = {score.candidate_id: score for score in transparent.scores}
    assert scores["straight_medium"].hazard_exposure > scores["arc_left_small"].hazard_exposure
    assert scores["straight_medium"].total_score > scores["arc_left_small"].total_score

    def fake_predict(*_args: object, **_kwargs: object) -> dict[str, object]:
        count = len(candidates)
        return {
            "future": {
                "candidate_score": np.zeros((count,), dtype=np.float32),
                "candidate_risk": np.zeros((count,), dtype=np.float32),
                "candidate_dynamic_risk": np.zeros((count,), dtype=np.float32),
                "candidate_unknown_exposure": np.zeros((count,), dtype=np.float32),
                "future_occupied": np.zeros((1, *shape), dtype=np.float32),
                "future_dynamic_risk": np.zeros((1, *shape), dtype=np.float32),
                "future_uncertainty": np.zeros((1, *shape), dtype=np.float32),
                "future_flow_xy": np.zeros((1, 2, *shape), dtype=np.float32),
            },
            "debug": {"runtime_inputs": [], "no_teacher_fields_at_runtime": True},
        }

    monkeypatch.setattr("homebrain.runtime.counterfactual_world_model_scorer.predict_future_bev_from_scene_state", fake_predict)
    runtime = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(history_frames=2, bev_channels=7, bev_shape=shape, pose_delta_dim=3)
        )
    )
    world = score_candidates_with_world_model(
        runtime,  # type: ignore[arg-type]
        local_bev_history=[bev, bev],
        pose_delta_history=np.zeros((2, 3), dtype=np.float32),
        candidates=candidates,
    )
    straight_index = [candidate.id for candidate in candidates].index("straight_medium")
    clear_index = [candidate.id for candidate in candidates].index("arc_left_small")
    assert world["candidate_hazard_exposure"][straight_index] > world["candidate_hazard_exposure"][clear_index]
    assert world["candidate_scores"][straight_index] > world["candidate_scores"][clear_index]


def test_runtime_signatures_reject_teacher_hazard_fields() -> None:
    forbidden = {"hazard_boxes", "hazard_masks", "open_vocab_detector", "teacher_hazard_artifacts", "detector"}
    assert forbidden.isdisjoint(inspect.signature(predict_local_bev_from_rgbd).parameters)
    assert forbidden.isdisjoint(inspect.signature(score_candidates_with_world_model).parameters)
    with pytest.raises(TypeError):
        predict_local_bev_from_rgbd(object(), np.zeros((4, 4, 3), dtype=np.uint8), hazard_boxes=[])  # type: ignore[arg-type,call-arg]
