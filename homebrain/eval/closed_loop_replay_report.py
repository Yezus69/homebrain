from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.messages.schema import BrainOutputEvent, Event, FrameEvent, JsonDict
from homebrain.replay.segment_log import read_events

CLOSED_LOOP_REPLAY_REPORT_SCHEMA_VERSION = "homebrain.closed_loop_replay_report.v0"


def build_closed_loop_replay_report(
    *,
    log_dir: str | Path,
    compare_log_dir: str | Path | None = None,
) -> JsonDict:
    log_path = Path(log_dir)
    events = read_events(log_path)
    metrics = _summarize_events(events)
    metrics.update(
        {
            "schema_version": CLOSED_LOOP_REPLAY_REPORT_SCHEMA_VERSION,
            "log": log_path.as_posix(),
            "comparison": None,
        }
    )
    if compare_log_dir is not None:
        compare_path = Path(compare_log_dir)
        compare_events = read_events(compare_path)
        metrics["comparison"] = _compare_decisions(events, compare_events, compare_path)
    metrics["answers"] = _answers(metrics)
    metrics["next_blocker"] = _next_blocker(metrics)
    return metrics


def write_report(report: JsonDict, *, out_json: str | Path, out_md: str | Path) -> None:
    json_path = Path(out_json)
    md_path = Path(out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write("\n")
    md_path.write_text(_markdown_report(report), encoding="utf-8", newline="\n")


def _summarize_events(events: list[Event]) -> JsonDict:
    frames = [event for event in events if isinstance(event, FrameEvent)]
    outputs = [event for event in events if isinstance(event, BrainOutputEvent)]
    decisions = [
        event
        for event in outputs
        if event.selected_trajectory_id is not None and bool(event.candidate_trajectories)
    ]
    selected_ids = [str(event.selected_trajectory_id) for event in decisions if event.selected_trajectory_id is not None]
    selected_distribution = dict(sorted(Counter(selected_ids).items()))
    candidate_counts = [len(event.candidate_trajectories or []) for event in decisions]
    selected_scores = [_selected_score(event) for event in decisions]
    policy_sources = [
        str(event.debug.get("policy_bev_source"))
        for event in decisions
        if isinstance(event.debug, dict) and event.debug.get("policy_bev_source") is not None
    ]
    coverage_records = [
        event.debug.get("coverage_memory")
        for event in decisions
        if isinstance(event.debug, dict) and isinstance(event.debug.get("coverage_memory"), dict)
    ]
    latency_records = [
        event.debug.get("latency")
        for event in outputs
        if isinstance(event.debug, dict) and isinstance(event.debug.get("latency"), dict)
    ]
    pose_warp_values = [_pose_warp_value(event) for event in outputs]
    route_pose_leakage_values = [_route_pose_leakage_value(event) for event in outputs]
    risky_fractions = [_risky_candidate_fraction(event) for event in decisions]
    unsafe_selected_values = [_unsafe_selected_value(event) for event in decisions]
    cmd_vel_non_null_count = sum(1 for event in outputs if event.cmd_vel is not None)
    cmd_vel_proposals = [_cmd_vel_proposal(event) for event in outputs]
    control_safe = any(_event_or_candidate_flag(event, "control_safe") for event in outputs)
    replay_only = bool(outputs) and all(_debug_flag(event, "replay_only") for event in outputs)
    not_executed = bool(outputs) and all(_debug_flag(event, "not_executed") for event in outputs)
    product_training_approved = any(_event_or_candidate_flag(event, "product_training_approved") for event in outputs)

    return {
        "frame_count": len(frames),
        "brain_output_count": len(outputs),
        "decision_count": len(decisions),
        "candidate_count": _mode_int(candidate_counts),
        "candidate_record_count": int(sum(candidate_counts)),
        "candidate_count_mean": _mean(candidate_counts),
        "selected_candidate_distribution": selected_distribution,
        "selected_candidate_entropy": _entropy(selected_distribution),
        "selected_candidate_unique_count": len(selected_distribution),
        "selected_candidate_dominant_fraction": _dominant_fraction(selected_distribution),
        "selected_actions_collapsed": len(selected_distribution) <= 1 and bool(selected_distribution),
        "stop_selected_fraction": _fraction(selected_ids, "stop"),
        "unsafe_selected_rate": _mean([value for value in unsafe_selected_values if value is not None]),
        "risky_candidate_fraction_mean": _mean([value for value in risky_fractions if value is not None]),
        "selected_risk_score_mean": _mean_score(selected_scores, "risk_score"),
        "selected_unknown_penalty_mean": _mean_score(selected_scores, "unknown_penalty"),
        "selected_uncertainty_penalty_mean": _mean_score(selected_scores, "uncertainty_penalty"),
        "selected_coverage_gain_mean": _mean_score(selected_scores, "coverage_gain_proxy"),
        "selected_collision_probability_mean": _mean_score(selected_scores, "future_collision_probability"),
        "selected_unknown_exposure_mean": _mean_score(selected_scores, "future_unknown_exposure"),
        "selected_new_area_gain_mean": _mean_score(selected_scores, "future_new_area_gain"),
        "selected_progress_mean": _mean_score(selected_scores, "future_progress"),
        "coverage_memory_cells_seen": _max_coverage_value(coverage_records, "coverage_memory_cells_seen"),
        "coverage_memory_cells_covered": _max_coverage_value(coverage_records, "coverage_memory_cells_covered"),
        "pose_warp_valid_fraction": _mean([value for value in pose_warp_values if value is not None]),
        "route_pose_leakage_ablation_fraction": _mean([value for value in route_pose_leakage_values if value is not None]),
        "policy_bev_source": _single_or_mixed(policy_sources),
        "policy_bev_source_distribution": dict(sorted(Counter(policy_sources).items())),
        "cmd_vel_non_null_count": int(cmd_vel_non_null_count),
        "cmd_vel_proposal_count": len(cmd_vel_proposals),
        "cmd_vel_proposal_nonzero_count": sum(1 for item in cmd_vel_proposals if _proposal_nonzero(item)),
        "route_progress_proxy_mean": _mean(
            [
                item.get("linear_velocity_mps")
                for item in cmd_vel_proposals
                if isinstance(item, dict)
            ]
        ),
        "latency_end_to_end_p50_ms": _latency_percentile(latency_records, "end_to_end_latency_ms", 50),
        "latency_end_to_end_p95_ms": _latency_percentile(latency_records, "end_to_end_latency_ms", 95),
        "latency_model_p50_ms": _latency_percentile(latency_records, "model_latency_ms", 50),
        "latency_model_p95_ms": _latency_percentile(latency_records, "model_latency_ms", 95),
        "latency_memory_update_p50_ms": _latency_percentile(latency_records, "memory_update_latency_ms", 50),
        "latency_memory_update_p95_ms": _latency_percentile(latency_records, "memory_update_latency_ms", 95),
        "latency_decision_p50_ms": _latency_percentile(latency_records, "decision_latency_ms", 50),
        "latency_decision_p95_ms": _latency_percentile(latency_records, "decision_latency_ms", 95),
        "realtime_10hz_pass_fraction": _mean(
            [
                1.0 if record.get("meets_10hz_budget") is True else 0.0
                for record in latency_records
                if isinstance(record, dict)
            ]
        ),
        "control_safe": bool(control_safe),
        "replay_only": bool(replay_only),
        "not_executed": bool(not_executed),
        "product_training_approved": bool(product_training_approved),
    }


def _compare_decisions(primary_events: list[Event], compare_events: list[Event], compare_path: Path) -> JsonDict:
    primary = _decision_map(primary_events)
    compare = _decision_map(compare_events)
    matched = sorted(set(primary) & set(compare))
    differing = [key for key in matched if primary[key] != compare[key]]
    primary_metrics = _summarize_events(primary_events)
    compare_metrics = _summarize_events(compare_events)
    return {
        "compare_log": compare_path.as_posix(),
        "comparison_kind": "primary_vs_transparent_or_alternate_replay_log",
        "matched_decision_count": len(matched),
        "differing_selected_count": len(differing),
        "differing_selected_fraction": float(len(differing) / max(len(matched), 1)),
        "decisions_differ": bool(differing),
        "primary_decision_count": int(primary_metrics.get("decision_count", 0)),
        "compare_decision_count": int(compare_metrics.get("decision_count", 0)),
        "primary_selected_candidate_entropy": primary_metrics.get("selected_candidate_entropy"),
        "compare_selected_candidate_entropy": compare_metrics.get("selected_candidate_entropy"),
        "primary_stop_selected_fraction": primary_metrics.get("stop_selected_fraction"),
        "compare_stop_selected_fraction": compare_metrics.get("stop_selected_fraction"),
        "primary_selected_risk_score_mean": primary_metrics.get("selected_risk_score_mean"),
        "compare_selected_risk_score_mean": compare_metrics.get("selected_risk_score_mean"),
        "primary_selected_unknown_penalty_mean": primary_metrics.get("selected_unknown_penalty_mean"),
        "compare_selected_unknown_penalty_mean": compare_metrics.get("selected_unknown_penalty_mean"),
        "primary_selected_coverage_gain_mean": primary_metrics.get("selected_coverage_gain_mean"),
        "compare_selected_coverage_gain_mean": compare_metrics.get("selected_coverage_gain_mean"),
        "primary_selected_collision_probability_mean": primary_metrics.get("selected_collision_probability_mean"),
        "compare_selected_collision_probability_mean": compare_metrics.get("selected_collision_probability_mean"),
        "primary_unsafe_selected_rate": primary_metrics.get("unsafe_selected_rate"),
        "compare_unsafe_selected_rate": compare_metrics.get("unsafe_selected_rate"),
        "primary_selected_unknown_exposure_mean": primary_metrics.get("selected_unknown_exposure_mean"),
        "compare_selected_unknown_exposure_mean": compare_metrics.get("selected_unknown_exposure_mean"),
        "primary_selected_new_area_gain_mean": primary_metrics.get("selected_new_area_gain_mean"),
        "compare_selected_new_area_gain_mean": compare_metrics.get("selected_new_area_gain_mean"),
        "primary_route_progress_proxy_mean": primary_metrics.get("route_progress_proxy_mean"),
        "compare_route_progress_proxy_mean": compare_metrics.get("route_progress_proxy_mean"),
        "selected_risk_score_delta_primary_minus_compare": _numeric_delta(
            primary_metrics.get("selected_risk_score_mean"),
            compare_metrics.get("selected_risk_score_mean"),
        ),
        "selected_unknown_penalty_delta_primary_minus_compare": _numeric_delta(
            primary_metrics.get("selected_unknown_penalty_mean"),
            compare_metrics.get("selected_unknown_penalty_mean"),
        ),
        "selected_coverage_gain_delta_primary_minus_compare": _numeric_delta(
            primary_metrics.get("selected_coverage_gain_mean"),
            compare_metrics.get("selected_coverage_gain_mean"),
        ),
        "selected_collision_probability_delta_primary_minus_compare": _numeric_delta(
            primary_metrics.get("selected_collision_probability_mean"),
            compare_metrics.get("selected_collision_probability_mean"),
        ),
        "unsafe_selected_rate_delta_primary_minus_compare": _numeric_delta(
            primary_metrics.get("unsafe_selected_rate"),
            compare_metrics.get("unsafe_selected_rate"),
        ),
        "selected_unknown_exposure_delta_primary_minus_compare": _numeric_delta(
            primary_metrics.get("selected_unknown_exposure_mean"),
            compare_metrics.get("selected_unknown_exposure_mean"),
        ),
        "selected_new_area_gain_delta_primary_minus_compare": _numeric_delta(
            primary_metrics.get("selected_new_area_gain_mean"),
            compare_metrics.get("selected_new_area_gain_mean"),
        ),
        "route_progress_proxy_delta_primary_minus_compare": _numeric_delta(
            primary_metrics.get("route_progress_proxy_mean"),
            compare_metrics.get("route_progress_proxy_mean"),
        ),
    }


def _decision_map(events: list[Event]) -> dict[tuple[str, str, int, int], str]:
    result: dict[tuple[str, str, int, int], str] = {}
    for event in events:
        if not isinstance(event, BrainOutputEvent) or event.selected_trajectory_id is None:
            continue
        camera_id = str(event.debug.get("camera_id", "unknown")) if isinstance(event.debug, dict) else "unknown"
        frame_id = int(event.debug.get("input_frame_id", -1)) if isinstance(event.debug, dict) else -1
        result[(event.sequence_id, camera_id, frame_id, event.timestamp_ns)] = str(event.selected_trajectory_id)
    return result


def _selected_score(event: BrainOutputEvent) -> JsonDict:
    result: JsonDict = {}
    if isinstance(event.debug, dict) and isinstance(event.debug.get("selected_candidate_metrics"), dict):
        result.update(dict(event.debug["selected_candidate_metrics"]))
    selected = event.selected_trajectory_id
    for candidate in event.candidate_trajectories or []:
        candidate_id = str(candidate.get("id", candidate.get("trajectory_id", "")))
        if candidate_id != selected:
            continue
        transparent = candidate.get("transparent_trajectory_score")
        if isinstance(transparent, dict):
            for key, value in transparent.items():
                result.setdefault(key, value)
        score = candidate.get("trajectory_score")
        if isinstance(score, dict):
            result.update(dict(score))
        return result
    return result


def _risky_candidate_fraction(event: BrainOutputEvent) -> float | None:
    if isinstance(event.debug, dict) and isinstance(event.debug.get("risky_candidate_fraction"), (int, float)):
        return float(event.debug["risky_candidate_fraction"])
    risky: list[bool] = []
    for candidate in event.candidate_trajectories or []:
        score = candidate.get("transparent_trajectory_score")
        if not isinstance(score, dict):
            score = candidate.get("trajectory_score")
        if isinstance(score, dict) and isinstance(score.get("risky"), bool):
            risky.append(bool(score["risky"]))
    if not risky:
        return None
    return float(sum(1 for value in risky if value) / len(risky))


def _pose_warp_value(event: BrainOutputEvent) -> float | None:
    if not isinstance(event.debug, dict):
        return None
    if isinstance(event.debug.get("pose_warp_valid"), bool):
        return 1.0 if bool(event.debug["pose_warp_valid"]) else 0.0
    value = event.debug.get("valid_warp_fraction")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _route_pose_leakage_value(event: BrainOutputEvent) -> float | None:
    if not isinstance(event.debug, dict):
        return None
    value = event.debug.get("route_pose_leakage_ablation")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    source = event.debug.get("pose_warp_source")
    if isinstance(source, str):
        return 1.0 if source == "route_pose" else 0.0
    return None


def _unsafe_selected_value(event: BrainOutputEvent) -> float | None:
    score = _selected_score(event)
    if not score:
        return None
    risky = score.get("risky")
    if isinstance(risky, bool):
        return 1.0 if risky else 0.0
    for key in ("unsafe_now_probability", "future_collision_probability", "collision_probability"):
        value = score.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return 1.0 if float(value) >= 0.5 else 0.0
    risk_score = score.get("risk_score")
    if isinstance(risk_score, (int, float)) and not isinstance(risk_score, bool):
        return 1.0 if float(risk_score) > 0.0 else 0.0
    return None


def _event_or_candidate_flag(event: BrainOutputEvent, key: str) -> bool:
    if _debug_flag(event, key):
        return True
    for candidate in event.candidate_trajectories or []:
        value = candidate.get(key)
        if isinstance(value, bool) and value:
            return True
        score = candidate.get("trajectory_score")
        if isinstance(score, dict) and isinstance(score.get(key), bool) and bool(score[key]):
            return True
    return False


def _debug_flag(event: BrainOutputEvent, key: str) -> bool:
    return bool(event.debug.get(key)) if isinstance(event.debug, dict) else False


def _answers(metrics: JsonDict) -> JsonDict:
    comparison = metrics.get("comparison") if isinstance(metrics.get("comparison"), dict) else {}
    return {
        "v1_memory_replay_produced_non_empty_trajectory_decisions": int(metrics.get("decision_count", 0)) > 0
        and str(metrics.get("policy_bev_source")) == "memory",
        "selected_actions_collapsed_to_one_candidate": bool(metrics.get("selected_actions_collapsed", False)),
        "memory_bev_decisions_differed_from_current_bev_decisions": comparison.get("decisions_differ")
        if comparison
        else None,
        "primary_decisions_differed_from_compare_log": comparison.get("decisions_differ") if comparison else None,
        "cmd_vel_emitted_count": int(metrics.get("cmd_vel_non_null_count", 0)),
        "cmd_vel_proposal_count": int(metrics.get("cmd_vel_proposal_count", 0)),
        "latency_p95_under_100ms": float(metrics.get("latency_end_to_end_p95_ms", 0.0)) <= 100.0
        if int(metrics.get("brain_output_count", 0)) > 0
        else None,
        "product_control_safe": bool(metrics.get("control_safe", False)),
    }


def _next_blocker(metrics: JsonDict) -> str:
    if int(metrics.get("decision_count", 0)) == 0:
        return "v1_replay_did_not_emit_trajectory_decisions"
    if int(metrics.get("cmd_vel_non_null_count", 0)) != 0 or bool(metrics.get("control_safe", False)):
        return "replay_safety_flags_regressed"
    if float(metrics.get("route_pose_leakage_ablation_fraction", 0.0)) > 0.0:
        return "runtime_used_route_pose_leakage_ablation"
    if bool(metrics.get("selected_actions_collapsed", False)):
        return "selected_actions_collapsed_to_single_candidate"
    comparison = metrics.get("comparison")
    if not isinstance(comparison, dict):
        return "run_comparison_replay_log"
    if int(comparison.get("matched_decision_count", 0)) == 0:
        return "comparison_replay_had_no_matched_decisions"
    if not bool(comparison.get("decisions_differ", False)):
        return "primary_and_compare_policy_decisions_were_identical"
    return "needs_route_out_policy_quality_and_control_safety_gate"


def _markdown_report(report: JsonDict) -> str:
    answers = report.get("answers") if isinstance(report.get("answers"), dict) else {}
    comparison = report.get("comparison") if isinstance(report.get("comparison"), dict) else {}
    lines = [
        "# Goal 20A Closed-Loop Replay Report",
        "",
        f"- Log: `{report.get('log')}`",
        f"- Frame count: `{report.get('frame_count')}`",
        f"- Brain outputs: `{report.get('brain_output_count')}`",
        f"- Decisions: `{report.get('decision_count')}`",
        f"- Policy BEV source: `{report.get('policy_bev_source')}`",
        f"- Candidate count per decision: `{report.get('candidate_count')}`",
        f"- Selected distribution: `{report.get('selected_candidate_distribution')}`",
        f"- Selected entropy: `{report.get('selected_candidate_entropy')}`",
        f"- Stop selected fraction: `{report.get('stop_selected_fraction')}`",
        f"- Unsafe selected rate: `{report.get('unsafe_selected_rate')}`",
        f"- Mean selected risk/unknown/uncertainty/coverage: `{report.get('selected_risk_score_mean')}` / `{report.get('selected_unknown_penalty_mean')}` / `{report.get('selected_uncertainty_penalty_mean')}` / `{report.get('selected_coverage_gain_mean')}`",
        f"- Coverage cells seen/covered: `{report.get('coverage_memory_cells_seen')}` / `{report.get('coverage_memory_cells_covered')}`",
        f"- Pose warp valid fraction: `{report.get('pose_warp_valid_fraction')}`",
        f"- Route-pose leakage ablation fraction: `{report.get('route_pose_leakage_ablation_fraction')}`",
        "",
        "## Required Answers",
        "",
        f"- Did v1 memory replay produce non-empty trajectory decisions? `{answers.get('v1_memory_replay_produced_non_empty_trajectory_decisions')}`",
        f"- Did selected actions collapse to one candidate? `{answers.get('selected_actions_collapsed_to_one_candidate')}`",
        f"- Did memory-BEV decisions differ from current-BEV decisions? `{answers.get('memory_bev_decisions_differed_from_current_bev_decisions')}`",
        f"- Did primary decisions differ from compare log? `{answers.get('primary_decisions_differed_from_compare_log')}`",
        f"- Comparison matched/different: `{comparison.get('matched_decision_count')}` / `{comparison.get('differing_selected_count')}`",
        f"- Comparison primary/compare risk mean: `{comparison.get('primary_selected_risk_score_mean')}` / `{comparison.get('compare_selected_risk_score_mean')}`",
        f"- Comparison primary/compare unknown mean: `{comparison.get('primary_selected_unknown_penalty_mean')}` / `{comparison.get('compare_selected_unknown_penalty_mean')}`",
        f"- Comparison primary/compare coverage gain: `{comparison.get('primary_selected_coverage_gain_mean')}` / `{comparison.get('compare_selected_coverage_gain_mean')}`",
        f"- Was any cmd_vel emitted? `{answers.get('cmd_vel_emitted_count')}`",
        f"- Bounded cmd_vel proposals: `{answers.get('cmd_vel_proposal_count')}`",
        f"- Latency p50/p95 end-to-end ms: `{report.get('latency_end_to_end_p50_ms')}` / `{report.get('latency_end_to_end_p95_ms')}`",
        f"- Latency p95 under 100ms? `{answers.get('latency_p95_under_100ms')}`",
        f"- Is this product/control safe? `{answers.get('product_control_safe')}`",
        f"- Next blocker: `{report.get('next_blocker')}`",
    ]
    return "\n".join(lines) + "\n"


def _mean(values: list[Any]) -> float:
    numbers = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not numbers:
        return 0.0
    return float(np.mean(numbers))


def _numeric_delta(primary: Any, compare: Any) -> float | None:
    if (
        isinstance(primary, (int, float))
        and not isinstance(primary, bool)
        and isinstance(compare, (int, float))
        and not isinstance(compare, bool)
    ):
        return float(primary) - float(compare)
    return None


def _mean_score(scores: list[JsonDict], key: str) -> float:
    return _mean([score.get(key) for score in scores if isinstance(score, dict)])


def _latency_percentile(records: list[Any], key: str, percentile_value: float) -> float:
    values = [
        float(record[key])
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get(key), (int, float))
        and not isinstance(record.get(key), bool)
    ]
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile_value))


def _cmd_vel_proposal(event: BrainOutputEvent) -> JsonDict | None:
    if not isinstance(event.debug, dict):
        return None
    proposal = event.debug.get("cmd_vel_proposal")
    return dict(proposal) if isinstance(proposal, dict) else None


def _proposal_nonzero(proposal: JsonDict | None) -> bool:
    if not isinstance(proposal, dict):
        return False
    linear = proposal.get("linear_velocity_mps")
    angular = proposal.get("angular_velocity_radps")
    return (
        isinstance(linear, (int, float))
        and not isinstance(linear, bool)
        and abs(float(linear)) > 1.0e-6
    ) or (
        isinstance(angular, (int, float))
        and not isinstance(angular, bool)
        and abs(float(angular)) > 1.0e-6
    )


def _max_coverage_value(records: list[Any], key: str) -> int:
    values = [int(record.get(key, 0)) for record in records if isinstance(record, dict)]
    return max(values) if values else 0


def _mode_int(values: list[int]) -> int:
    if not values:
        return 0
    counts = Counter(values)
    return int(min(counts, key=lambda value: (-counts[value], value)))


def _single_or_mixed(values: list[str]) -> str | None:
    if not values:
        return None
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else "mixed"


def _fraction(values: list[str], target: str) -> float:
    if not values:
        return 0.0
    return float(sum(1 for value in values if value == target) / len(values))


def _dominant_fraction(distribution: dict[str, int]) -> float:
    total = sum(int(value) for value in distribution.values())
    if total <= 0:
        return 0.0
    return float(max(int(value) for value in distribution.values()) / total)


def _entropy(distribution: dict[str, int]) -> float:
    total = sum(int(value) for value in distribution.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in distribution.values():
        probability = float(count) / total
        if probability > 0.0:
            entropy -= probability * math.log2(probability)
    return float(entropy)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize replay-only closed-loop BrainOutput trajectory decisions.")
    parser.add_argument("--log", required=True, help="Primary replay/modeld output log.")
    parser.add_argument("--compare-log", default=None, help="Optional current-vs-memory comparison log.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args(argv)
    report = build_closed_loop_replay_report(log_dir=args.log, compare_log_dir=args.compare_log)
    write_report(report, out_json=args.out_json, out_md=args.out_md)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
