from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from homebrain.messages.schema import BrainOutputEvent, deterministic_json
from homebrain.policies.audit_closed_loop_policy_collapse import audit_closed_loop_policy_collapse
from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.replay.segment_log import write_segment


def test_goal21a_detects_learned_collapse_when_transparent_does_not(tmp_path: Path) -> None:
    learned_log = tmp_path / "learned"
    transparent_log = tmp_path / "transparent"
    labels = tmp_path / "action_v5"
    _write_label_pack(labels, ["straight_short", "arc_left_medium", "rotate_left"])
    _write_log(
        learned_log,
        selected=["arc_right_medium", "arc_right_medium", "arc_right_medium"],
        transparent_selected=["straight_short", "arc_left_medium", "rotate_left"],
        scorer_mode="learned",
        policy_source="memory",
    )
    _write_log(
        transparent_log,
        selected=["straight_short", "arc_left_medium", "rotate_left"],
        transparent_selected=["straight_short", "arc_left_medium", "rotate_left"],
        scorer_mode="transparent",
        policy_source="memory",
    )

    report = audit_closed_loop_policy_collapse(
        logs=[f"learned={learned_log}", f"transparent={transparent_log}"],
        action_label_pack=labels,
        out_json=tmp_path / "audit.json",
        out_md=tmp_path / "audit.md",
        command="unit",
    )

    assert report["root_cause_classification"]["buckets"]["learned_scorer_prior_bias"]["present"] is True
    assert report["root_cause_classification"]["buckets"]["transparent_scorer_collapse"]["present"] is False
    learned = next(item for item in report["logs"] if item["name"] == "learned")
    assert learned["selected_candidate_distribution"] == {"arc_right_medium": 3}
    assert learned["learned_vs_transparent_selected_agreement"] == 0.0
    assert learned["action_label_v5_comparison"]["learned_selected_vs_v5_label_agreement"] == 0.0
    assert report["safety_flags"]["replay_only"] is True
    assert report["safety_flags"]["control_safe"] is False
    assert (tmp_path / "audit.json").exists()
    assert "# Goal 21A" in (tmp_path / "audit.md").read_text(encoding="utf-8")


def test_goal21a_non_collapsed_fixture_does_not_raise_collapse_bucket(tmp_path: Path) -> None:
    log = tmp_path / "healthy"
    selected = ["straight_short", "arc_left_medium", "rotate_left", "arc_right_small"]
    _write_log(
        log,
        selected=selected,
        transparent_selected=selected,
        scorer_mode="learned",
        policy_source="current",
    )

    report = audit_closed_loop_policy_collapse(
        logs=[log],
        out_json=tmp_path / "audit.json",
        out_md=tmp_path / "audit.md",
        command="unit",
    )

    root = report["root_cause_classification"]
    assert report["logs"][0]["selected_actions_collapsed"] is False
    assert root["buckets"]["learned_scorer_prior_bias"]["present"] is False
    assert root["buckets"]["transparent_scorer_collapse"]["present"] is False
    assert root["buckets"]["implementation_bug_suspected"]["present"] is False


def test_goal21a_reports_memory_delta_without_action_change(tmp_path: Path) -> None:
    memory_log = tmp_path / "memory"
    current_log = tmp_path / "current"
    selected = ["arc_right_medium", "arc_right_medium", "arc_right_medium"]
    _write_log(
        memory_log,
        selected=selected,
        transparent_selected=["straight_short", "straight_short", "straight_short"],
        scorer_mode="learned",
        policy_source="memory",
        memory_offset=0.08,
    )
    _write_log(
        current_log,
        selected=selected,
        transparent_selected=["straight_short", "straight_short", "straight_short"],
        scorer_mode="learned",
        policy_source="current",
        memory_offset=0.08,
    )

    report = audit_closed_loop_policy_collapse(
        logs=[f"memory={memory_log}", f"current={current_log}"],
        out_json=tmp_path / "audit.json",
        out_md=tmp_path / "audit.md",
        command="unit",
    )

    assert report["root_cause_classification"]["buckets"]["memory_delta_too_small_for_policy"]["present"] is True
    comparison = report["comparisons"][0]
    assert comparison["selected_difference_fraction"] == 0.0
    assert report["logs"][0]["current_vs_memory_bev_delta_statistics"]["mean_abs_delta_over_channels"] > 0.0


def _write_log(
    root: Path,
    *,
    selected: list[str],
    transparent_selected: list[str],
    scorer_mode: str,
    policy_source: str,
    memory_offset: float = 0.02,
) -> None:
    candidates = generate_default_candidates(grid_shape=(8, 8), meters_per_cell=0.25, robot_radius_m=0.18)
    events = []
    artifacts = []
    for index, selected_id in enumerate(selected):
        local_bev_ref = f"brain_outputs/spatial_v1/d400_color_{index:06d}.npz"
        _write_artifact(root / local_bev_ref, memory_offset=memory_offset)
        transparent_id = transparent_selected[index]
        candidate_records = []
        for candidate_index, candidate in enumerate(candidates):
            learned_logit = 5.0 if candidate.id == selected_id else 1.0 - candidate_index * 0.05
            if candidate.id == "arc_right_medium":
                learned_logit += 0.5
            transparent_total = 0.0 if candidate.id == transparent_id else 3.0 + candidate_index * 0.01
            transparent_score = {
                "schema_version": "homebrain.transparent_trajectory_score.v0",
                "scorer": "transparent_coverage_risk_v0",
                "candidate_id": candidate.id,
                "risk_score": 0.0,
                "unknown_penalty": 0.2,
                "uncertainty_penalty": 0.1,
                "coverage_gain_proxy": 2.0,
                "smoothness_penalty": 0.0,
                "total_score": transparent_total,
                "risky": False,
                "selected_by_transparent_scorer": candidate.id == transparent_id,
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
                "product_training_approved": False,
            }
            record = {
                **candidate.to_dict(),
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
                "product_training_approved": False,
            }
            if scorer_mode == "learned":
                record["trajectory_score"] = {
                    "schema_version": "homebrain.learned_trajectory_score.v0",
                    "scorer": "trajectory_scorer_net_v0",
                    "candidate_id": candidate.id,
                    "learned_logit": learned_logit,
                    "higher_is_better": True,
                    "selected_by_learned_scorer": candidate.id == selected_id,
                    "candidate_features": _candidate_features(candidate_index, candidate.id),
                    "replay_only": True,
                    "not_executed": True,
                    "control_safe": False,
                    "product_training_approved": False,
                }
                record["transparent_trajectory_score"] = transparent_score
            else:
                transparent_score["selected_by_runtime_policy"] = candidate.id == selected_id
                record["trajectory_score"] = transparent_score
            candidate_records.append(record)
        timestamp_ns = 1_700_000_000_000_000_000 + index
        events.append(
            BrainOutputEvent(
                timestamp_ns=timestamp_ns,
                sequence_id="openloris_cafe1_1_2_route",
                source="spatial_memory_v1",
                input_event_ids=[f"frame:{index}"],
                pose_delta=(0.0, 0.0, 0.0),
                pose_confidence=1.0,
                local_bev_ref=local_bev_ref,
                candidate_trajectories=candidate_records,
                selected_trajectory_id=selected_id,
                cmd_vel=None,
                uncertainty=0.0,
                stop_reason="test",
                debug={
                    "camera_id": "d400_color",
                    "input_frame_id": index,
                    "policy_bev_source": policy_source,
                    "trajectory_scorer_mode": scorer_mode,
                    "selected_candidate_id": selected_id,
                    "transparent_decision": {"selected_candidate_id": transparent_id},
                    "coverage_memory": {
                        "coverage_memory_cells_seen": 10 + index,
                        "coverage_memory_cells_covered": 3 + index,
                        "pose_alignment_attempt_count": index,
                        "missing_pose_delta_count": 0,
                    },
                    "coverage_memory_before": {
                        "coverage_memory_cells_seen": 9 + index,
                        "coverage_memory_cells_covered": 2 + index,
                    },
                    "coverage_memory_update": {"cells_seen_delta": 1, "cells_covered_delta": 1},
                    "replay_only": True,
                    "not_executed": True,
                    "control_safe": False,
                    "product_training_approved": False,
                    "cmd_vel_emitted": False,
                    "raw_pwm_emitted": False,
                },
            )
        )
        artifacts.append(local_bev_ref)
    write_segment(root, events, segment_id=root.name, artifact_files=artifacts)


def _write_artifact(path: Path, *, memory_offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = np.full((8, 8), 0.30, dtype=np.float32)
    memory = np.clip(current + np.float32(memory_offset), 0.0, 1.0)
    np.savez_compressed(
        path,
        current_bev_free_prob=current,
        current_bev_occupied_prob=np.zeros_like(current),
        current_bev_unknown_prob=np.full_like(current, 0.50),
        current_bev_traversable_prob=current,
        current_bev_risky_prob=np.zeros_like(current),
        memory_bev_free_prob=memory,
        memory_bev_occupied_prob=np.zeros_like(current),
        memory_bev_unknown_prob=np.full_like(current, 0.45),
        memory_bev_traversable_prob=memory,
        memory_bev_risky_prob=np.zeros_like(current),
    )


def _candidate_features(candidate_index: int, candidate_id: str) -> dict[str, float]:
    return {
        name: round(0.01 * candidate_index + feature_index * 0.001, 6)
        for feature_index, name in enumerate(
            [
                "linear_velocity_norm",
                "angular_velocity_norm",
                "duration_norm",
                "is_stop",
                "is_rotation_only",
                "footprint_fraction",
                "occupied_max",
                "occupied_mean",
                "unknown_max",
                "unknown_mean",
                "uncertainty_mean",
                "free_mean",
                "traversable_mean",
                "coverage_gain_fraction",
                "covered_fraction",
                "smoothness_norm",
                "terminal_forward_norm",
                "terminal_left_norm",
                "terminal_yaw_norm",
            ]
        )
    } | {"is_stop": 1.0 if candidate_id == "stop" else 0.0}


def _write_label_pack(root: Path, labels: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    examples = []
    for index, label in enumerate(labels):
        examples.append(
            {
                "action_supervision_ok": True,
                "bc_label_valid": True,
                "camera_id": "d400_color",
                "control_safe": False,
                "frame_id": index,
                "future_motion_primary_candidate_id": label,
                "not_executed": True,
                "replay_only": True,
                "selected_candidate_id": label,
                "sequence_id": "openloris_cafe1_1_2_route",
            }
        )
    (root / "manifest.json").write_text(
        deterministic_json(
            {
                "package_type": "ActionLabelPack",
                "version": 5,
                "control_safe": False,
                "source_dirs": [],
                "examples": examples,
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    assert json.loads((root / "manifest.json").read_text(encoding="utf-8"))["version"] == 5
