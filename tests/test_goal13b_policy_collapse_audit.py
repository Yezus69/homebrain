from __future__ import annotations

import json
from pathlib import Path

from homebrain.messages.schema import deterministic_json
from homebrain.policies.audit_goal13a_collapse import (
    audit_goal13a_collapse,
)


def test_goal13b_detects_oracle_collapse_and_writes_safety_flags(tmp_path: Path) -> None:
    records = [
        _record(frame_id=index, oracle_action="straight_short", current_action="straight_short", memory_action="straight_short")
        for index in range(6)
    ]

    report = _run_audit(tmp_path, records)

    root = report["root_cause_classification"]
    assert root["primary"] == "oracle_labels_collapsed"
    assert root["buckets"]["oracle_labels_collapsed"]["present"] is True
    oracle = report["mode_route_audit"]["normal"]["all_routes"]["per_source_metrics"]["oracle_bev"]
    assert oracle["collapse_flag"] is True
    assert report["safety_flags"]["replay_only"] is True
    assert report["safety_flags"]["not_executed"] is True
    assert report["safety_flags"]["control_safe"] is False
    assert report["safety_flags"]["product_training_approved"] is False
    assert report["safety_flags"]["cmd_vel_non_null_count"] == 0


def test_goal13b_reports_top1_top2_margin_from_full_scores(tmp_path: Path) -> None:
    records = [
        _record(
            frame_id=1,
            oracle_action="straight_short",
            current_action="straight_short",
            memory_action="straight_short",
            full_scores=[
                _score("straight_short", total=1.0),
                _score("rotate_left", total=1.0005),
                _score("stop", total=4.0),
            ],
        )
    ]

    report = _run_audit(tmp_path, records)

    margin = report["mode_route_audit"]["normal"]["all_routes"]["per_source_metrics"]["v1_current_bev"][
        "top1_vs_top2_margin"
    ]
    assert margin["available"] is True
    assert margin["near_tie_fraction"] == 1.0
    assert margin["min"] == 0.0004999999999999449


def test_goal13b_detects_candidate_order_sensitivity(tmp_path: Path) -> None:
    records = [
        _record(
            frame_id=1,
            oracle_action="straight_short",
            current_action="straight_short",
            memory_action="straight_short",
            full_scores=[
                _score("straight_short", total=1.0),
                _score("rotate_left", total=1.0),
                _score("stop", total=4.0),
            ],
        )
    ]

    report = _run_audit(tmp_path, records)

    source = report["mode_route_audit"]["normal"]["all_routes"]["per_source_metrics"]["v1_current_bev"]
    assert source["candidate_order_dominance"]["candidate_order_dominates_flag"] is True
    ablation = report["mode_route_audit"]["normal"]["all_routes"]["ablation_metrics"]["v1_current_bev"]
    assert ablation["available"] is True
    assert ablation["modes"]["reversed_candidate_order"]["changed_fraction"] == 1.0
    assert report["root_cause_classification"]["buckets"]["scorer_tie_break_dominates"]["present"] is True


def test_goal13b_classifies_memory_delta_too_small_for_action(tmp_path: Path) -> None:
    records = [
        _record(
            frame_id=1,
            oracle_action="straight_short",
            current_action="straight_short",
            memory_action="straight_short",
            future_label="straight_short",
            current_score=_score("straight_short", total=1.0, unknown=0.20),
            memory_score=_score("straight_short", total=0.9, unknown=0.05),
        ),
        _record(
            frame_id=2,
            oracle_action="rotate_left",
            current_action="rotate_left",
            memory_action="rotate_left",
            future_label="rotate_left",
            current_score=_score("rotate_left", total=1.0, unknown=0.20),
            memory_score=_score("rotate_left", total=0.9, unknown=0.05),
        ),
    ]

    report = _run_audit(tmp_path, records)

    memory = report["mode_route_audit"]["normal"]["all_routes"]["current_vs_memory"]
    assert memory["memory_delta_too_small_for_action_flag"] is True
    assert memory["memory_vs_current_action_changed_fraction"] == 0.0
    assert report["root_cause_classification"]["primary"] == "memory_delta_too_small_for_action"
    assert report["root_cause_classification"]["buckets"]["memory_delta_too_small_for_action"]["present"] is True


def _run_audit(tmp_path: Path, records: list[dict]) -> dict:
    decisions = tmp_path / "decisions.jsonl"
    with decisions.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(deterministic_json(record))
            handle.write("\n")
    report = audit_goal13a_collapse(
        goal13a_decisions=decisions,
        out_json=tmp_path / "audit.json",
        out_md=tmp_path / "audit.md",
        contact_sheet=tmp_path / "audit.ppm",
        command="unit",
    )
    assert (tmp_path / "audit.json").exists()
    assert (tmp_path / "audit.md").exists()
    assert (tmp_path / "audit.ppm").read_bytes().startswith(b"P6\n")
    with (tmp_path / "audit.json").open("r", encoding="utf-8") as handle:
        assert json.load(handle)["schema_version"] == report["schema_version"]
    return report


def _record(
    *,
    frame_id: int,
    oracle_action: str,
    current_action: str,
    memory_action: str,
    future_label: str = "arc_right_medium",
    full_scores: list[dict] | None = None,
    current_score: dict | None = None,
    memory_score: dict | None = None,
) -> dict:
    oracle_score = _score(oracle_action, total=0.5)
    current_score = current_score or _score(current_action, total=0.5)
    memory_score = memory_score or _score(memory_action, total=0.5)
    v0_score = _score(current_action, total=0.5)
    return {
        "schema_version": "homebrain.goal13a_shadow_trajectory_decision.v0",
        "mode": "normal",
        "source_name": "route",
        "sequence_id": "seq",
        "camera_id": "front",
        "frame_id": frame_id,
        "timestamp_ns": frame_id,
        "candidate_count": 3,
        "future_motion_label": future_label,
        "decisions": {
            "oracle_bev": _decision(oracle_action, oracle_score, future_label, full_scores=full_scores),
            "v0_current_bev": _decision(current_action, v0_score, future_label, full_scores=full_scores),
            "v1_current_bev": _decision(current_action, current_score, future_label, full_scores=full_scores),
            "v1_memory_bev": _decision(memory_action, memory_score, future_label, full_scores=full_scores),
        },
        "memory_vs_current": {
            "available": True,
            "current_action": current_action,
            "memory_action": memory_action,
            "action_changed": current_action != memory_action,
            "memory_oracle_score_delta": 0.0,
            "memory_improved": False,
            "memory_worsened": False,
        },
        "cmd_vel": None,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "product_training_approved": False,
        "raw_pwm_emitted": False,
    }


def _decision(action: str, selected_score: dict, future_label: str, *, full_scores: list[dict] | None) -> dict:
    data = {
        "selected_candidate_id": action,
        "selected_score": selected_score,
        "oracle_collision_proxy": 0.0,
        "oracle_unknown_penalty": 0.0,
        "oracle_coverage_gain": 1.0,
        "oracle_total_expert_score": 0.0,
        "collision_proxy_positive": False,
        "unsafe_selected": False,
        "all_motion_candidates_risky": False,
        "future_motion_label_valid": True,
        "agreement_with_future_motion_label": action == future_label,
        "agreement_with_oracle_action": True,
    }
    if full_scores is not None:
        data["scores"] = full_scores
    return data


def _score(
    candidate_id: str,
    *,
    total: float,
    risk: float = 0.0,
    unknown: float = 0.0,
    uncertainty: float = 0.0,
    coverage: float = 1.0,
    smoothness: float = 0.0,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "risk_score": risk,
        "unknown_penalty": unknown,
        "uncertainty_penalty": uncertainty,
        "coverage_gain_proxy": coverage,
        "smoothness_penalty": smoothness,
        "total_score": total,
        "risky": False,
    }
