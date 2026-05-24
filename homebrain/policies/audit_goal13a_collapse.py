from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from homebrain.data.spatial_dataset import write_json
from homebrain.messages.schema import JsonDict, deterministic_json
from homebrain.policies.candidate_trajectories import DEFAULT_CANDIDATE_SPECS

GOAL13B_AUDIT_SCHEMA_VERSION = "homebrain.goal13b_policy_collapse_audit.v0"

BEV_DECISION_SOURCES: tuple[str, ...] = (
    "oracle_bev",
    "v0_current_bev",
    "v1_current_bev",
    "v1_memory_bev",
)
ROOT_CAUSE_BUCKETS: tuple[str, ...] = (
    "oracle_labels_collapsed",
    "model_bev_collapsed",
    "candidate_set_too_weak",
    "scorer_tie_break_dominates",
    "coverage_gain_not_discriminative",
    "risk_or_unknown_dominates",
    "memory_delta_too_small_for_action",
    "route_out_data_too_small",
    "implementation_bug_suspected",
    "inconclusive",
)

CANDIDATE_ORDER: tuple[str, ...] = tuple(spec.candidate_id for spec in DEFAULT_CANDIDATE_SPECS)
COLLAPSE_ENTROPY_MIN = 1.0
COLLAPSE_DOMINANT_MAX = 0.65
NEAR_TIE_MARGIN = 1.0e-3
EXACT_TIE_MARGIN = 1.0e-9
MEMORY_ACTION_CHANGE_MAX_FOR_SMALL_DELTA = 0.05
MEMORY_SCORE_DELTA_EPS = 1.0e-4
ROUTE_OUT_MIN_ROUTE_COUNT = 3
ROUTE_OUT_MIN_FRAMES_PER_ROUTE = 50


def audit_goal13a_collapse(
    *,
    goal13a_decisions: str | Path,
    out_json: str | Path,
    out_md: str | Path,
    contact_sheet: str | Path,
    command: str | None = None,
) -> JsonDict:
    started = time.perf_counter()
    decisions_path = Path(goal13a_decisions)
    records = _read_decisions_jsonl(decisions_path)
    if not records:
        raise ValueError(f"no Goal 13A decision records found: {decisions_path}")

    mode_route_audit: dict[str, JsonDict] = {}
    for mode in sorted({str(record.get("mode", "unknown")) for record in records}):
        mode_records = [record for record in records if str(record.get("mode", "unknown")) == mode]
        routes = {
            route: _summarize_scope(route_records)
            for route, route_records in _group_by(mode_records, lambda item: str(item.get("source_name", "unknown"))).items()
        }
        mode_route_audit[mode] = {
            "frame_count": len(mode_records),
            "all_routes": _summarize_scope(mode_records),
            "routes": routes,
        }

    normal_records = [record for record in records if str(record.get("mode")) == "normal"]
    classification_scope = _summarize_scope(normal_records if normal_records else records)
    root_cause = _classify_root_cause(
        classification_scope,
        route_frame_counts=_route_frame_counts(normal_records if normal_records else records),
    )
    recommendations = _recommendations(root_cause)
    input_limitations = _input_limitations(records)
    safety = _safety_summary(records)

    worst_records = _worst_records(records, limit=24)
    _write_contact_sheet(Path(contact_sheet), worst_records)

    report: JsonDict = {
        "schema_version": GOAL13B_AUDIT_SCHEMA_VERSION,
        "goal": "13B diagnose Goal 13A action collapse without training",
        "input": {
            "goal13a_decisions": decisions_path.as_posix(),
            "record_count": len(records),
            "modes": sorted({str(record.get("mode", "unknown")) for record in records}),
            "routes": sorted({str(record.get("source_name", "unknown")) for record in records}),
        },
        "collapse_thresholds": {
            "action_entropy_min": COLLAPSE_ENTROPY_MIN,
            "dominant_action_fraction_max": COLLAPSE_DOMINANT_MAX,
            "near_tie_margin": NEAR_TIE_MARGIN,
            "exact_tie_margin": EXACT_TIE_MARGIN,
        },
        "mode_route_audit": mode_route_audit,
        "root_cause_classification": root_cause,
        "recommendations": recommendations,
        "input_limitations": input_limitations,
        "artifacts_created": {
            "json_report": Path(out_json).as_posix(),
            "markdown_report": Path(out_md).as_posix(),
            "contact_sheet": Path(contact_sheet).as_posix(),
        },
        "safety_flags": safety,
        "hard_constraints": {
            "trained_model": False,
            "teachers_added": False,
            "ros_or_sim_or_hardware_control_added": False,
            "cmd_vel_emitted": False,
            "raw_pwm_emitted": False,
        },
        "commands_run": [command] if command else [],
        "runtime_sec": float(time.perf_counter() - started),
    }
    write_json(out_json, report, pretty=True)
    _write_markdown(Path(out_md), report)
    return report


def _summarize_scope(records: list[JsonDict]) -> JsonDict:
    per_source = {source: _source_metrics(records, source) for source in BEV_DECISION_SOURCES}
    return {
        "schema_version": "homebrain.goal13b_policy_collapse_scope.v0",
        "frame_count": len(records),
        "per_source_metrics": per_source,
        "future_motion_labels": _future_motion_metrics(records),
        "current_vs_memory": _memory_delta_metrics(records),
        "ablation_metrics": {source: _ablation_metrics(records, source) for source in BEV_DECISION_SOURCES},
        "input_score_availability": {
            source: _score_availability(records, source) for source in BEV_DECISION_SOURCES
        },
        "safety_flags": _safety_summary(records),
    }


def _source_metrics(records: list[JsonDict], source: str) -> JsonDict:
    items = _source_items(records, source)
    selected = [str(item.get("selected_candidate_id", "")) for item in items]
    distribution = _distribution(selected)
    dominant = max(distribution.values(), default=0) / max(len(selected), 1)
    entropy = _entropy(distribution)
    collapse_reasons = _collapse_reasons(entropy=entropy, dominant_action_fraction=dominant)
    future_valid = [item for item in items if item.get("future_motion_label_valid") is True]
    return {
        "frame_count": len(items),
        "selected_action_distribution": distribution,
        "action_entropy": entropy,
        "dominant_action_fraction": float(dominant),
        "collapse_flag": bool(collapse_reasons),
        "collapse_reasons": collapse_reasons,
        "stop_fraction": _fraction(value == "stop" for value in selected),
        "future_motion_label_valid_count": len(future_valid),
        "agreement_with_future_motion_label": _fraction(
            bool(item.get("agreement_with_future_motion_label")) for item in future_valid
        ),
        "agreement_with_oracle_action": _fraction(
            bool(item.get("agreement_with_oracle_action")) for item in items
        ),
        "unsafe_selected_rate": _fraction(bool(item.get("unsafe_selected")) for item in items),
        "collision_proxy_rate": _fraction(bool(item.get("collision_proxy_positive")) for item in items),
        "top1_vs_top2_margin": _margin_metrics(items),
        "candidate_order_dominance": _candidate_order_metrics(items),
        "selected_score_component_means": _component_stats(
            [item.get("selected_score") for item in items if isinstance(item.get("selected_score"), dict)]
        ),
        "per_action_score_component_means": _per_action_component_means(items),
    }


def _future_motion_metrics(records: list[JsonDict]) -> JsonDict:
    labels = [
        str(record.get("future_motion_label"))
        for record in records
        if isinstance(record.get("future_motion_label"), str)
    ]
    distribution = _distribution(labels)
    per_source_agreement: dict[str, float] = {}
    for source in BEV_DECISION_SOURCES:
        items = _source_items(records, source)
        valid = [item for item in items if item.get("future_motion_label_valid") is True]
        per_source_agreement[source] = _fraction(
            bool(item.get("agreement_with_future_motion_label")) for item in valid
        )
    return {
        "valid_count": len(labels),
        "distribution": distribution,
        "entropy": _entropy(distribution),
        "dominant_label_fraction": max(distribution.values(), default=0) / max(len(labels), 1),
        "agreement_by_source": per_source_agreement,
    }


def _memory_delta_metrics(records: list[JsonDict]) -> JsonDict:
    comparisons: list[JsonDict] = []
    component_deltas: dict[str, list[float]] = {name: [] for name in _component_names()}
    changed_with_score_change = 0
    unchanged_with_score_change = 0
    available = 0
    oracle_score_deltas: list[float] = []

    for record in records:
        decisions = record.get("decisions")
        if not isinstance(decisions, dict):
            continue
        current = decisions.get("v1_current_bev")
        memory = decisions.get("v1_memory_bev")
        if not isinstance(current, dict) or not isinstance(memory, dict):
            continue
        available += 1
        current_action = str(current.get("selected_candidate_id", ""))
        memory_action = str(memory.get("selected_candidate_id", ""))
        action_changed = current_action != memory_action
        current_score = current.get("selected_score") if isinstance(current.get("selected_score"), dict) else {}
        memory_score = memory.get("selected_score") if isinstance(memory.get("selected_score"), dict) else {}
        score_changed = False
        for component in _component_names():
            delta = _component_value(memory_score, component) - _component_value(current_score, component)
            component_deltas[component].append(delta)
            score_changed = score_changed or abs(delta) > MEMORY_SCORE_DELTA_EPS
        if action_changed and score_changed:
            changed_with_score_change += 1
        if (not action_changed) and score_changed:
            unchanged_with_score_change += 1
        comparison = record.get("memory_vs_current")
        if isinstance(comparison, dict) and isinstance(comparison.get("memory_oracle_score_delta"), (int, float)):
            oracle_score_deltas.append(float(comparison["memory_oracle_score_delta"]))
        comparisons.append(
            {
                "current_action": current_action,
                "memory_action": memory_action,
                "action_changed": action_changed,
                "score_changed": score_changed,
            }
        )

    action_changed_fraction = _fraction(bool(item["action_changed"]) for item in comparisons)
    return {
        "comparison_count": available,
        "memory_vs_current_action_changed_fraction": action_changed_fraction,
        "memory_changes_scores_but_not_actions_fraction": float(unchanged_with_score_change / max(available, 1)),
        "memory_changes_scores_and_actions_fraction": float(changed_with_score_change / max(available, 1)),
        "component_delta_means_memory_minus_current": {
            component: _mean(values) for component, values in component_deltas.items()
        },
        "component_delta_abs_means": {
            component: _mean(abs(value) for value in values) for component, values in component_deltas.items()
        },
        "memory_oracle_score_mean_delta": _mean(oracle_score_deltas),
        "memory_delta_too_small_for_action_flag": bool(
            available
            and action_changed_fraction < MEMORY_ACTION_CHANGE_MAX_FOR_SMALL_DELTA
            and (
                _mean(abs(value) for value in component_deltas["total"]) > MEMORY_SCORE_DELTA_EPS
                or _mean(abs(value) for value in component_deltas["unknown"]) > MEMORY_SCORE_DELTA_EPS
            )
        ),
    }


def _ablation_metrics(records: list[JsonDict], source: str) -> JsonDict:
    modes = (
        "risk_only",
        "risk_unknown",
        "risk_coverage",
        "no_smoothness",
        "no_coverage_memory",
        "reversed_candidate_order",
    )
    evaluated = 0
    production: list[str] = []
    selections: dict[str, list[str]] = {mode: [] for mode in modes}
    skipped = 0
    for item in _source_items(records, source):
        scores = _all_candidate_scores(item)
        if len(scores) < 2:
            skipped += 1
            continue
        evaluated += 1
        production_id = _select_from_scores(scores, "production")
        production.append(production_id)
        for mode in modes:
            selections[mode].append(_select_from_scores(scores, mode))
    if evaluated == 0:
        return {
            "available": False,
            "evaluated_count": 0,
            "skipped_count": skipped,
            "unavailable_reason": "Goal 13A decisions do not include full per-candidate scores",
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
        }
    by_mode: dict[str, JsonDict] = {}
    for mode in modes:
        values = selections[mode]
        distribution = _distribution(values)
        dominant = max(distribution.values(), default=0) / max(len(values), 1)
        entropy = _entropy(distribution)
        by_mode[mode] = {
            "selected_action_distribution": distribution,
            "agreement_with_production_fraction": _fraction(
                value == production_value for value, production_value in zip(values, production)
            ),
            "changed_fraction": _fraction(
                value != production_value for value, production_value in zip(values, production)
            ),
            "action_entropy": entropy,
            "dominant_action_fraction": float(dominant),
            "collapse_flag": bool(
                _collapse_reasons(entropy=entropy, dominant_action_fraction=dominant)
            ),
        }
    return {
        "available": True,
        "evaluated_count": evaluated,
        "skipped_count": skipped,
        "modes": by_mode,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
    }


def _margin_metrics(items: list[JsonDict]) -> JsonDict:
    margins: list[float] = []
    skipped = 0
    exact_ties = 0
    near_ties = 0
    for item in items:
        scores = _all_candidate_scores(item)
        if len(scores) < 2:
            skipped += 1
            continue
        ranked = _rank_scores(scores, mode="production")
        margin = _score_value(ranked[1], "production") - _score_value(ranked[0], "production")
        margins.append(float(margin))
        exact_ties += int(abs(margin) <= EXACT_TIE_MARGIN)
        near_ties += int(margin <= NEAR_TIE_MARGIN)
    if not margins:
        return {
            "available": False,
            "evaluated_count": 0,
            "skipped_count": skipped,
            "unavailable_reason": "Goal 13A decisions do not include full per-candidate scores",
        }
    return {
        "available": True,
        "evaluated_count": len(margins),
        "skipped_count": skipped,
        "mean": _mean(margins),
        "min": min(margins),
        "p10": _quantile(margins, 0.10),
        "p50": _quantile(margins, 0.50),
        "near_tie_fraction": float(near_ties / max(len(margins), 1)),
        "exact_tie_fraction": float(exact_ties / max(len(margins), 1)),
    }


def _candidate_order_metrics(items: list[JsonDict]) -> JsonDict:
    evaluated = 0
    near_tie_order_wins = 0
    reversed_changes = 0
    skipped = 0
    for item in items:
        scores = _all_candidate_scores(item)
        if len(scores) < 2:
            skipped += 1
            continue
        evaluated += 1
        ranked = _rank_scores(scores, mode="production")
        margin = _score_value(ranked[1], "production") - _score_value(ranked[0], "production")
        if margin <= NEAR_TIE_MARGIN and _order_index(_score_candidate_id(ranked[0])) < _order_index(_score_candidate_id(ranked[1])):
            near_tie_order_wins += 1
        production = _score_candidate_id(ranked[0])
        reversed_order = _select_from_scores(scores, "reversed_candidate_order")
        reversed_changes += int(reversed_order != production)
    if evaluated == 0:
        return {
            "available": False,
            "evaluated_count": 0,
            "skipped_count": skipped,
            "unavailable_reason": "Goal 13A decisions do not include full per-candidate scores",
        }
    return {
        "available": True,
        "evaluated_count": evaluated,
        "skipped_count": skipped,
        "near_tie_order_win_fraction": float(near_tie_order_wins / max(evaluated, 1)),
        "reversed_candidate_order_changed_fraction": float(reversed_changes / max(evaluated, 1)),
        "candidate_order_dominates_flag": bool(
            near_tie_order_wins / max(evaluated, 1) >= 0.25
            or reversed_changes / max(evaluated, 1) >= 0.05
        ),
    }


def _per_action_component_means(items: list[JsonDict]) -> JsonDict:
    all_scores: list[JsonDict] = []
    for item in items:
        scores = _all_candidate_scores(item)
        if len(scores) >= 2:
            all_scores.extend(scores)
    basis = "all_candidate_scores"
    if not all_scores:
        basis = "selected_scores_only"
        all_scores = [
            item.get("selected_score")
            for item in items
            if isinstance(item.get("selected_score"), dict)
        ]
    by_action: dict[str, list[JsonDict]] = {}
    for score in all_scores:
        if not isinstance(score, dict):
            continue
        candidate_id = _score_candidate_id(score)
        by_action.setdefault(candidate_id, []).append(score)
    return {
        "basis": basis,
        "actions": {
            action: _component_stats(scores) for action, scores in sorted(by_action.items())
        },
    }


def _classify_root_cause(scope: JsonDict, *, route_frame_counts: dict[str, int]) -> JsonDict:
    per_source = scope.get("per_source_metrics", {}) if isinstance(scope.get("per_source_metrics"), dict) else {}
    future = scope.get("future_motion_labels", {}) if isinstance(scope.get("future_motion_labels"), dict) else {}
    memory = scope.get("current_vs_memory", {}) if isinstance(scope.get("current_vs_memory"), dict) else {}
    safety = scope.get("safety_flags", {}) if isinstance(scope.get("safety_flags"), dict) else {}
    buckets: dict[str, JsonDict] = {
        bucket: {"present": False, "evidence": []} for bucket in ROOT_CAUSE_BUCKETS if bucket != "inconclusive"
    }

    oracle = per_source.get("oracle_bev") if isinstance(per_source.get("oracle_bev"), dict) else {}
    if oracle.get("collapse_flag") is True:
        buckets["oracle_labels_collapsed"]["present"] = True
        buckets["oracle_labels_collapsed"]["evidence"].append(
            f"oracle action_entropy={oracle.get('action_entropy')} dominant_action_fraction={oracle.get('dominant_action_fraction')}"
        )

    for source in ("v0_current_bev", "v1_current_bev", "v1_memory_bev"):
        metrics = per_source.get(source) if isinstance(per_source.get(source), dict) else {}
        if metrics.get("collapse_flag") is True:
            buckets["model_bev_collapsed"]["present"] = True
            buckets["model_bev_collapsed"]["evidence"].append(
                f"{source} collapsed with distribution={metrics.get('selected_action_distribution')}"
            )

    future_entropy = float(future.get("entropy", 0.0)) if isinstance(future.get("entropy"), (int, float)) else 0.0
    oracle_future_agree = float(oracle.get("agreement_with_future_motion_label", 0.0)) if isinstance(oracle.get("agreement_with_future_motion_label"), (int, float)) else 0.0
    if buckets["oracle_labels_collapsed"]["present"] and (future_entropy >= COLLAPSE_ENTROPY_MIN or oracle_future_agree < 0.25):
        buckets["candidate_set_too_weak"]["present"] = True
        buckets["candidate_set_too_weak"]["evidence"].append(
            f"oracle labels collapsed while future labels entropy={future_entropy} and oracle/future agreement={oracle_future_agree}"
        )

    for source, metrics in per_source.items():
        if not isinstance(metrics, dict):
            continue
        order_metrics = metrics.get("candidate_order_dominance")
        if isinstance(order_metrics, dict) and order_metrics.get("candidate_order_dominates_flag") is True:
            buckets["scorer_tie_break_dominates"]["present"] = True
            buckets["scorer_tie_break_dominates"]["evidence"].append(f"{source} order metrics={order_metrics}")

    memory_metrics = per_source.get("v1_memory_bev") if isinstance(per_source.get("v1_memory_bev"), dict) else {}
    selected_components = memory_metrics.get("selected_score_component_means") if isinstance(memory_metrics, dict) else {}
    coverage_std = _stats_std(selected_components, "coverage")
    if memory_metrics.get("collapse_flag") is True and coverage_std <= 0.10:
        buckets["coverage_gain_not_discriminative"]["present"] = True
        buckets["coverage_gain_not_discriminative"]["evidence"].append(
            f"v1_memory selected coverage std={coverage_std}"
        )

    for source, ablation in (scope.get("ablation_metrics") or {}).items():
        if not isinstance(ablation, dict) or ablation.get("available") is not True:
            continue
        modes = ablation.get("modes") if isinstance(ablation.get("modes"), dict) else {}
        risk_unknown = modes.get("risk_unknown") if isinstance(modes.get("risk_unknown"), dict) else {}
        if float(risk_unknown.get("agreement_with_production_fraction", 0.0)) >= 0.90:
            buckets["risk_or_unknown_dominates"]["present"] = True
            buckets["risk_or_unknown_dominates"]["evidence"].append(f"{source} risk+unknown ablation matched production")
        reverse = modes.get("reversed_candidate_order") if isinstance(modes.get("reversed_candidate_order"), dict) else {}
        if float(reverse.get("changed_fraction", 0.0)) >= 0.05:
            buckets["scorer_tie_break_dominates"]["present"] = True
            buckets["scorer_tie_break_dominates"]["evidence"].append(f"{source} reversed candidate order changed selections")

    if memory.get("memory_delta_too_small_for_action_flag") is True:
        buckets["memory_delta_too_small_for_action"]["present"] = True
        buckets["memory_delta_too_small_for_action"]["evidence"].append(
            f"memory action_changed_fraction={memory.get('memory_vs_current_action_changed_fraction')} with score-change/no-action fraction={memory.get('memory_changes_scores_but_not_actions_fraction')}"
        )

    if len(route_frame_counts) < ROUTE_OUT_MIN_ROUTE_COUNT or any(
        count < ROUTE_OUT_MIN_FRAMES_PER_ROUTE for count in route_frame_counts.values()
    ):
        buckets["route_out_data_too_small"]["present"] = True
        buckets["route_out_data_too_small"]["evidence"].append(f"route_frame_counts={route_frame_counts}")

    if (
        int(safety.get("input_bad_replay_only_count", 0)) > 0
        or int(safety.get("input_bad_not_executed_count", 0)) > 0
        or int(safety.get("input_bad_control_safe_count", 0)) > 0
        or int(safety.get("input_bad_product_training_count", 0)) > 0
        or int(safety.get("cmd_vel_non_null_count", 0)) > 0
        or int(safety.get("raw_pwm_emitted_count", 0)) > 0
    ):
        buckets["implementation_bug_suspected"]["present"] = True
        buckets["implementation_bug_suspected"]["evidence"].append(f"safety violations={safety}")

    present = [bucket for bucket, detail in buckets.items() if detail["present"]]
    if not present:
        return {
            "primary": "inconclusive",
            "buckets": {**buckets, "inconclusive": {"present": True, "evidence": ["no root-cause bucket crossed deterministic thresholds"]}},
        }

    priority = (
        "implementation_bug_suspected",
        "oracle_labels_collapsed",
        "scorer_tie_break_dominates",
        "candidate_set_too_weak",
        "memory_delta_too_small_for_action",
        "model_bev_collapsed",
        "coverage_gain_not_discriminative",
        "risk_or_unknown_dominates",
        "route_out_data_too_small",
    )
    primary = next(bucket for bucket in priority if bucket in present)
    return {
        "primary": primary,
        "buckets": {**buckets, "inconclusive": {"present": False, "evidence": []}},
    }


def _recommendations(root_cause: JsonDict) -> list[str]:
    buckets = root_cause.get("buckets") if isinstance(root_cause.get("buckets"), dict) else {}
    recommendations: list[str] = []
    if _bucket_present(buckets, "oracle_labels_collapsed"):
        recommendations.append(
            "Repair data/action-label generation before training: oracle labels are collapsed, so a memory-aware scorer would inherit a bad target."
        )
    if _bucket_present(buckets, "scorer_tie_break_dominates"):
        recommendations.append(
            "Repair candidate/scorer tie behavior: add discriminative candidate geometry or score terms before using these decisions as action evidence."
        )
    if _bucket_present(buckets, "memory_delta_too_small_for_action"):
        recommendations.append(
            "Repair SpatialMemory/data signal if action labels are fixed: memory changes scores but usually not selected actions."
        )
    if _bucket_present(buckets, "route_out_data_too_small"):
        recommendations.append(
            "Add route diversity and rerun route-out diagnostics before claiming generalization."
        )
    if not recommendations:
        recommendations.append(
            "Keep this as replay-only diagnosis and collect full candidate-score artifacts for the next audit pass."
        )
    return recommendations


def _input_limitations(records: list[JsonDict]) -> list[str]:
    limitations: list[str] = []
    any_full_scores = any(
        len(_all_candidate_scores(item)) >= 2
        for record in records
        for item in _source_items([record], source=None)
    )
    if not any_full_scores:
        limitations.append(
            "Goal 13A JSONL contains selected scores only, so top1/top2 margins and score ablations are marked unavailable for the existing artifact."
        )
    if not any(str(record.get("mode", "")).startswith("route_out") for record in records):
        limitations.append(
            "Goal 13A decision JSONL does not serialize true route-out fold records; this audit uses per-route normal/hidden/occlusion records."
        )
    return limitations


def _safety_summary(records: list[JsonDict]) -> JsonDict:
    return {
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "product_training_approved": False,
        "cmd_vel_emitted": False,
        "raw_pwm_emitted": False,
        "input_bad_replay_only_count": sum(1 for record in records if record.get("replay_only") is not True),
        "input_bad_not_executed_count": sum(1 for record in records if record.get("not_executed") is not True),
        "input_bad_control_safe_count": sum(1 for record in records if record.get("control_safe") is not False),
        "input_bad_product_training_count": sum(
            1 for record in records if record.get("product_training_approved") is not False
        ),
        "cmd_vel_non_null_count": sum(1 for record in records if record.get("cmd_vel") is not None),
        "raw_pwm_emitted_count": sum(1 for record in records if record.get("raw_pwm_emitted") is True),
    }


def _read_decisions_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        raise FileNotFoundError(f"missing Goal 13A decisions JSONL: {path}")
    records: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            if not isinstance(data, dict):
                raise ValueError(f"decision line {line_number} is not a JSON object: {path}")
            records.append(data)
    return records


def _source_items(records: list[JsonDict], source: str | None) -> list[JsonDict]:
    items: list[JsonDict] = []
    for record in records:
        decisions = record.get("decisions")
        if not isinstance(decisions, dict):
            continue
        if source is None:
            items.extend(item for item in decisions.values() if isinstance(item, dict))
        else:
            item = decisions.get(source)
            if isinstance(item, dict):
                items.append(item)
    return items


def _all_candidate_scores(item: JsonDict) -> list[JsonDict]:
    scores = item.get("scores")
    if isinstance(scores, list):
        return [dict(score) for score in scores if isinstance(score, dict) and _score_candidate_id(score)]
    candidates = item.get("candidates")
    if isinstance(candidates, list):
        parsed: list[JsonDict] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            score = candidate.get("score")
            if not isinstance(score, dict):
                continue
            merged = dict(score)
            if "candidate_id" not in merged and isinstance(candidate.get("id"), str):
                merged["candidate_id"] = str(candidate["id"])
            parsed.append(merged)
        return parsed
    return []


def _score_availability(records: list[JsonDict], source: str) -> JsonDict:
    items = _source_items(records, source)
    full_count = sum(1 for item in items if len(_all_candidate_scores(item)) >= 2)
    selected_count = sum(1 for item in items if isinstance(item.get("selected_score"), dict))
    return {
        "decision_count": len(items),
        "full_candidate_score_count": full_count,
        "selected_score_count": selected_count,
        "full_candidate_score_fraction": float(full_count / max(len(items), 1)),
    }


def _select_from_scores(scores: list[JsonDict], mode: str) -> str:
    ranked = _rank_scores(scores, mode=mode)
    return _score_candidate_id(ranked[0]) if ranked else ""


def _rank_scores(scores: list[JsonDict], *, mode: str) -> list[JsonDict]:
    reverse_order = mode == "reversed_candidate_order"
    return sorted(
        scores,
        key=lambda score: (
            _score_value(score, mode),
            _reverse_order_index(_score_candidate_id(score)) if reverse_order else _order_index(_score_candidate_id(score)),
            _score_candidate_id(score),
        ),
    )


def _score_value(score: JsonDict, mode: str) -> float:
    if mode in ("production", "reversed_candidate_order"):
        return _component_value(score, "total")
    if mode == "risk_only":
        return 12.0 * _component_value(score, "risk")
    if mode == "risk_unknown":
        return 12.0 * _component_value(score, "risk") + 1.6 * _component_value(score, "unknown")
    if mode == "risk_coverage":
        return 12.0 * _component_value(score, "risk") - 0.18 * _component_value(score, "coverage")
    if mode == "no_smoothness":
        return _component_value(score, "total") - 0.25 * _component_value(score, "smoothness")
    if mode == "no_coverage_memory":
        return _component_value(score, "total") + 0.18 * _component_value(score, "coverage")
    raise ValueError(f"unknown ablation mode: {mode}")


def _score_candidate_id(score: JsonDict) -> str:
    value = score.get("candidate_id")
    if isinstance(value, str):
        return value
    value = score.get("id")
    return str(value) if isinstance(value, str) else ""


def _component_names() -> tuple[str, ...]:
    return ("risk", "unknown", "uncertainty", "coverage", "smoothness", "total")


def _component_value(score: Any, component: str) -> float:
    if not isinstance(score, dict):
        return 0.0
    keys = {
        "risk": ("risk_score", "collision_proxy", "risk"),
        "unknown": ("unknown_penalty", "unknown"),
        "uncertainty": ("uncertainty_penalty", "uncertainty"),
        "coverage": ("coverage_gain_proxy", "coverage_gain", "coverage"),
        "smoothness": ("smoothness_penalty", "smoothness"),
        "total": ("total_score", "total_expert_score", "total"),
    }[component]
    for key in keys:
        value = score.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 0.0


def _component_stats(scores: Iterable[Any]) -> JsonDict:
    parsed = [score for score in scores if isinstance(score, dict)]
    result: JsonDict = {"count": len(parsed)}
    for component in _component_names():
        values = [_component_value(score, component) for score in parsed]
        result[component] = {
            "mean": _mean(values),
            "std": _std(values),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
    return result


def _stats_std(stats: Any, component: str) -> float:
    if not isinstance(stats, dict):
        return 0.0
    value = stats.get(component)
    if not isinstance(value, dict):
        return 0.0
    std = value.get("std")
    return float(std) if isinstance(std, (int, float)) else 0.0


def _collapse_reasons(*, entropy: float, dominant_action_fraction: float) -> list[str]:
    reasons: list[str] = []
    if entropy < COLLAPSE_ENTROPY_MIN:
        reasons.append(f"action_entropy<{COLLAPSE_ENTROPY_MIN}")
    if dominant_action_fraction > COLLAPSE_DOMINANT_MAX:
        reasons.append(f"dominant_action_fraction>{COLLAPSE_DOMINANT_MAX}")
    return reasons


def _route_frame_counts(records: list[JsonDict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        route = str(record.get("source_name", "unknown"))
        counts[route] = counts.get(route, 0) + 1
    return dict(sorted(counts.items()))


def _worst_records(records: list[JsonDict], *, limit: int) -> list[JsonDict]:
    ranked = sorted(records, key=_record_severity, reverse=True)
    return ranked[:limit]


def _record_severity(record: JsonDict) -> tuple[float, float, float, float]:
    comparison = record.get("memory_vs_current")
    memory_delta = 0.0
    action_changed = 0.0
    if isinstance(comparison, dict):
        memory_delta = abs(float(comparison.get("memory_oracle_score_delta", 0.0)))
        action_changed = 1.0 if comparison.get("action_changed") is True else 0.0
    worst_margin = 0.0
    decisions = record.get("decisions")
    if isinstance(decisions, dict):
        for item in decisions.values():
            if not isinstance(item, dict):
                continue
            margins = _margin_metrics([item])
            if margins.get("available") is True:
                worst_margin = max(worst_margin, 1.0 / max(float(margins.get("min", 0.0)), EXACT_TIE_MARGIN))
    return (action_changed, memory_delta, worst_margin, float(record.get("frame_id", 0)))


def _write_contact_sheet(path: Path, records: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles = [_decision_tile(record) for record in records]
    if not tiles:
        path.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
        return
    tile_h, tile_w, _channels = tiles[0].shape
    cols = min(6, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // cols
        col = index % cols
        sheet[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    path.write_bytes(f"P6\n{sheet.shape[1]} {sheet.shape[0]}\n255\n".encode("ascii") + sheet.tobytes())


def _decision_tile(record: JsonDict) -> np.ndarray:
    candidate_ids = list(CANDIDATE_ORDER)
    height = 12 * len(candidate_ids)
    width = 96
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    decisions = record.get("decisions") if isinstance(record.get("decisions"), dict) else {}
    selected_by_source = {
        source: str(decisions.get(source, {}).get("selected_candidate_id", ""))
        for source in BEV_DECISION_SOURCES
        if isinstance(decisions.get(source), dict)
    }
    future = str(record.get("future_motion_label", ""))
    colors = {
        "oracle_bev": np.asarray([20, 180, 60], dtype=np.uint8),
        "v0_current_bev": np.asarray([180, 180, 20], dtype=np.uint8),
        "v1_current_bev": np.asarray([220, 100, 20], dtype=np.uint8),
        "v1_memory_bev": np.asarray([40, 130, 220], dtype=np.uint8),
        "future": np.asarray([160, 60, 190], dtype=np.uint8),
    }
    for index, candidate_id in enumerate(candidate_ids):
        row0 = index * 12
        row1 = row0 + 12
        tile[row0:row1, :, :] = np.asarray([35, 35, 35], dtype=np.uint8)
        x0 = 0
        for source in BEV_DECISION_SOURCES:
            if selected_by_source.get(source) == candidate_id:
                tile[row0:row1, x0 : x0 + 18, :] = colors[source]
            x0 += 19
        if future == candidate_id:
            tile[row0:row1, 82:96, :] = colors["future"]
    return tile


def _write_markdown(path: Path, report: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = report["root_cause_classification"]
    normal = report["mode_route_audit"].get("normal", {}).get("all_routes", {})
    per_source = normal.get("per_source_metrics", {}) if isinstance(normal, dict) else {}
    oracle = per_source.get("oracle_bev", {}) if isinstance(per_source, dict) else {}
    memory = per_source.get("v1_memory_bev", {}) if isinstance(per_source, dict) else {}
    current_memory = normal.get("current_vs_memory", {}) if isinstance(normal, dict) else {}
    lines = [
        "# Goal 13B Policy Collapse Audit",
        "",
        "## Result",
        f"- primary_root_cause: `{root['primary']}`",
        f"- records: `{report['input']['record_count']}`",
        f"- modes: `{', '.join(report['input']['modes'])}`",
        f"- routes: `{', '.join(report['input']['routes'])}`",
        "",
        "## Normal Mode Snapshot",
        f"- oracle distribution: `{oracle.get('selected_action_distribution', {})}`",
        f"- oracle entropy/dominance: `{oracle.get('action_entropy')}` / `{oracle.get('dominant_action_fraction')}`",
        f"- v1_memory distribution: `{memory.get('selected_action_distribution', {})}`",
        f"- v1_memory entropy/dominance: `{memory.get('action_entropy')}` / `{memory.get('dominant_action_fraction')}`",
        f"- memory action changed fraction: `{current_memory.get('memory_vs_current_action_changed_fraction')}`",
        f"- memory changes scores but not actions: `{current_memory.get('memory_changes_scores_but_not_actions_fraction')}`",
        "",
        "## Buckets",
    ]
    buckets = root.get("buckets", {}) if isinstance(root.get("buckets"), dict) else {}
    for bucket in ROOT_CAUSE_BUCKETS:
        detail = buckets.get(bucket, {}) if isinstance(buckets.get(bucket), dict) else {}
        lines.append(f"- {bucket}: `{str(bool(detail.get('present'))).lower()}`")
    lines.extend(
        [
            "",
            "## Recommendations",
            *[f"- {item}" for item in report["recommendations"]],
            "",
            "## Safety",
            "- replay_only=`true`, not_executed=`true`, control_safe=`false`, product_training_approved=`false`.",
            "- No model was trained, no teacher was added, and no cmd_vel/raw PWM was emitted.",
        ]
    )
    if report.get("input_limitations"):
        lines.extend(["", "## Input Limitations", *[f"- {item}" for item in report["input_limitations"]]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _bucket_present(buckets: JsonDict, bucket: str) -> bool:
    detail = buckets.get(bucket)
    return isinstance(detail, dict) and detail.get("present") is True


def _group_by(items: list[JsonDict], key_fn: Any) -> dict[str, list[JsonDict]]:
    groups: dict[str, list[JsonDict]] = {}
    for item in items:
        key = str(key_fn(item))
        groups.setdefault(key, []).append(item)
    return dict(sorted(groups.items()))


def _distribution(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        probability = count / total
        if probability > 0.0:
            value -= probability * math.log2(probability)
    return float(value)


def _fraction(values: Iterable[bool]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(sum(1 for value in values_list if value) / len(values_list))


def _mean(values: Iterable[float]) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))


def _std(values: Iterable[float]) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    mean = _mean(values_list)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values_list) / len(values_list)))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[index])


def _order_index(candidate_id: str) -> int:
    try:
        return CANDIDATE_ORDER.index(candidate_id)
    except ValueError:
        return 9999


def _reverse_order_index(candidate_id: str) -> int:
    try:
        return len(CANDIDATE_ORDER) - 1 - CANDIDATE_ORDER.index(candidate_id)
    except ValueError:
        return -1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Goal 13A replay-only action collapse decisions.")
    parser.add_argument("--goal13a-decisions", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--contact-sheet", required=True)
    args = parser.parse_args(argv)
    command = "python -m homebrain.policies.audit_goal13a_collapse " + " ".join(sys.argv[1:])
    report = audit_goal13a_collapse(
        goal13a_decisions=args.goal13a_decisions,
        out_json=args.out_json,
        out_md=args.out_md,
        contact_sheet=args.contact_sheet,
        command=command,
    )
    print(deterministic_json(report["root_cause_classification"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
