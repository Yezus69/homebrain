import inspect

import numpy as np

from homebrain.brain.direct_bev_student_v0 import DirectBEVStudentV0, DirectBEVStudentV0Config, save_checkpoint
from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.policies.trajectory_scorer import CoverageMemory, score_trajectories
from homebrain.runtime.direct_bev_runtime import (
    direct_bev_to_local_bev,
    load_runtime_direct_bev_student,
    predict_local_bev_from_rgbd,
)


def test_runtime_signature_rejects_teacher_or_future_leakage_fields() -> None:
    forbidden = {
        "target_bev",
        "future_labels",
        "candidate_oracle_cost",
        "teacher_masks",
        "ground_truth_global_trajectory",
        "future_frames",
        "route_level_oracle_data",
    }
    assert not (set(inspect.signature(predict_local_bev_from_rgbd).parameters) & forbidden)


def test_direct_bev_runtime_outputs_replay_safe_local_bev_and_scores_candidates(tmp_path) -> None:
    model = DirectBEVStudentV0(
        DirectBEVStudentV0Config(bev_shape=(16, 16), image_size=(32, 24), hidden_channels=8)
    )
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model,
        metadata={
            "source_dataset_name": "tiny_schema_smoke",
            "no_future_labels_used": True,
            "no_teacher_fields_at_runtime": True,
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "raw_pwm_emitted": False,
            "hardware_validated": False,
        },
        metrics={},
    )
    runtime = load_runtime_direct_bev_student(checkpoint, device="cpu")
    result = predict_local_bev_from_rgbd(
        runtime,
        np.zeros((24, 32, 3), dtype=np.uint8),
        depth=np.ones((24, 32), dtype=np.float32),
        pose_delta_prev=(0.0, 0.0, 0.0),
        previous_action=(0.0, 0.0),
    )

    result["local_bev"].validate()
    assert result["debug"]["direct_bev_teacher_fields_used_at_runtime"] is False
    assert result["debug"]["direct_bev_future_labels_used_at_runtime"] is False
    assert result["debug"]["replay_only"] is True
    assert result["debug"]["not_executed"] is True
    assert result["debug"]["control_safe"] is False
    assert result["debug"]["raw_pwm_emitted"] is False
    assert result["debug"]["hardware_validated"] is False

    local_bev = direct_bev_to_local_bev(result["arrays"])
    candidates = generate_default_candidates(grid_shape=(16, 16), meters_per_cell=0.1, robot_radius_m=0.15)
    decision = score_trajectories(
        bev=local_bev,
        candidates=candidates,
        coverage_memory=CoverageMemory((16, 16), meters_per_cell=0.1),
    )
    assert decision.selected_candidate_id in {candidate.id for candidate in candidates}
