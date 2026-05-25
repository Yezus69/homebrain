from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from homebrain.artifacts.io import read_json_object, write_json_object
from homebrain.messages.schema import BrainOutputEvent, JsonDict, deterministic_json
from homebrain.policies.trajectory_scorer_net_v0 import CANDIDATE_FEATURE_NAMES
from homebrain.replay.segment_log import read_events

AUDIT_SCHEMA_VERSION = "homebrain.closed_loop_policy_collapse_audit.v0"
PAIR_IDS: tuple[tuple[str, str, str], ...] = (
    ("arc_left_small", "arc_right_small", "arc_left_small_vs_arc_right_small"),
    ("arc_left_medium", "arc_right_medium", "arc_left_medium_vs_arc_right_medium"),
    ("rotate_left", "rotate_right", "rotate_left_vs_rotate_right"),
)
TRANSPARENT_SCORE_FIELDS: tuple[tuple[str, str], ...] = (
    ("total_score", "total"),
    ("risk_score", "risk"),
    ("unknown_penalty", "unknown"),
    ("uncertainty_penalty", "uncertainty"),
    ("coverage_gain_proxy", "coverage"),
)
BEV_CHANNELS: tuple[str, ...] = ("free", "occupied", "unknown", "traversable", "risky")


@dataclass(frozen=True)
class LogSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class DecisionRecord:
    log_name: str
    log_path: Path
    key: tuple[str, str, int, int]
    sequence_id: str
    camera_id: str
    frame_id: int
    timestamp_ns: int
    policy_bev_source: str
    scorer_mode: str
    selected_id: str
    transparent_selected_id: str | None
    candidate_ids: tuple[str, ...]
    learned_logits: dict[str, float]
    transparent_scores: dict[str, JsonDict]
    candidate_features: dict[str, dict[str, float]]
    coverage_memory: JsonDict
    coverage_memory_before: JsonDict
    coverage_memory_update: JsonDict
    local_bev_ref: str | None
    debug: JsonDict
    cmd_vel_non_null: bool
    control_safe: bool
    replay_only: bool
    not_executed: bool
    product_training_approved: bool
    raw_pwm_emitted: bool
    candidate_base_records: tuple[JsonDict, ...]


def audit_closed_loop_policy_collapse(
    *,
    logs: list[str | Path],
    out_json: str | Path,
    out_md: str | Path,
    compare_logs: list[str | Path] | None = None,
    action_label_pack: str | Path | None = None,
    spatial_packs: list[str | Path] | None = None,
    trajectory_scorer_checkpoint: str | Path | None = None,
    summary_csv: str | Path | None = None,
    summary_jsonl: str | Path | None = None,
    command: str | None = None,
) -> JsonDict:
    log_specs = [_parse_log_spec(item) for item in logs]
    compare_specs = [_parse_log_spec(item) for item in (compare_logs or [])]
    all_specs = _dedupe_log_specs([*log_specs, *compare_specs])
    if not all_specs:
        raise ValueError("at least one --log is required")

    label_index = _load_action_label_index(action_label_pack) if action_label_pack is not None else {}
    oracle_index = _load_oracle_bev_index(action_label_pack=action_label_pack, spatial_packs=spatial_packs or [])
    checkpoint_metadata = _load_checkpoint_metadata(trajectory_scorer_checkpoint)

    log_reports: list[JsonDict] = []
    records_by_log: dict[str, list[DecisionRecord]] = {}
    for spec in all_specs:
        records = _load_decision_records(spec)
        records_by_log[spec.name] = records
        log_reports.append(
            _summarize_log(
                spec=spec,
                records=records,
                label_index=label_index,
                oracle_index=oracle_index,
                checkpoint_metadata=checkpoint_metadata,
            )
        )

    comparisons = _build_comparisons(records_by_log, log_reports)
    aggregate = _aggregate_report(log_reports, comparisons, label_index, checkpoint_metadata)
    root_cause = _classify_root_cause(log_reports=log_reports, comparisons=comparisons, aggregate=aggregate)

    report: JsonDict = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "command": command,
        "logs": log_reports,
        "comparisons": comparisons,
        "aggregate": aggregate,
        "root_cause_classification": root_cause,
        "safety_flags": _aggregate_safety(log_reports),
        "inputs": {
            "logs": [spec.path.as_posix() for spec in log_specs],
            "compare_logs": [spec.path.as_posix() for spec in compare_specs],
            "action_label_pack": Path(action_label_pack).as_posix() if action_label_pack is not None else None,
            "spatial_packs": [Path(item).as_posix() for item in (spatial_packs or [])],
            "trajectory_scorer_checkpoint": Path(trajectory_scorer_checkpoint).as_posix()
            if trajectory_scorer_checkpoint is not None
            else None,
        },
    }
    write_json_object(out_json, report)
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text(_markdown_report(report), encoding="utf-8", newline="\n")
    if summary_csv is not None:
        _write_summary_csv(summary_csv, log_reports)
    if summary_jsonl is not None:
        _write_summary_jsonl(summary_jsonl, log_reports)
    return report


def _parse_log_spec(value: str | Path) -> LogSpec:
    text = str(value)
    if "=" in text:
        name, path = text.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"log alias is empty: {text!r}")
        return LogSpec(name=name, path=Path(path))
    path = Path(text)
    return LogSpec(name=path.name, path=path)


def _dedupe_log_specs(specs: list[LogSpec]) -> list[LogSpec]:
    result: list[LogSpec] = []
    seen: set[tuple[str, str]] = set()
    for spec in specs:
        key = (spec.name, spec.path.as_posix())
        if key in seen:
            continue
        seen.add(key)
        result.append(spec)
    return result


def _load_decision_records(spec: LogSpec) -> list[DecisionRecord]:
    events = read_events(spec.path)
    records: list[DecisionRecord] = []
    for event in events:
        if not isinstance(event, BrainOutputEvent):
            continue
        if event.selected_trajectory_id is None or not event.candidate_trajectories:
            continue
        debug = dict(event.debug) if isinstance(event.debug, dict) else {}
        camera_id = str(debug.get("camera_id", "unknown"))
        frame_id = _int_or_default(debug.get("input_frame_id"), -1)
        selected_id = str(event.selected_trajectory_id)
        candidate_ids: list[str] = []
        learned_logits: dict[str, float] = {}
        transparent_scores: dict[str, JsonDict] = {}
        candidate_features: dict[str, dict[str, float]] = {}
        candidate_base_records: list[JsonDict] = []
        for candidate in event.candidate_trajectories:
            candidate_id = _candidate_id(candidate)
            if not candidate_id:
                continue
            candidate_ids.append(candidate_id)
            candidate_base_records.append(_candidate_base_record(candidate))
            learned_score = candidate.get("trajectory_score")
            if isinstance(learned_score, dict):
                logit = _float_or_none(learned_score.get("learned_logit"))
                if logit is not None:
                    learned_logits[candidate_id] = logit
                features = learned_score.get("candidate_features")
                if isinstance(features, dict):
                    candidate_features[candidate_id] = {
                        str(name): float(value)
                        for name, value in features.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                    }
                if "total_score" in learned_score:
                    transparent_scores[candidate_id] = dict(learned_score)
            transparent = candidate.get("transparent_trajectory_score")
            if isinstance(transparent, dict):
                transparent_scores[candidate_id] = dict(transparent)
        transparent_selected_id = _transparent_selected_id(debug, transparent_scores)
        scorer_mode = str(debug.get("trajectory_scorer_mode", "learned" if learned_logits else "transparent"))
        records.append(
            DecisionRecord(
                log_name=spec.name,
                log_path=spec.path,
                key=(event.sequence_id, camera_id, frame_id, event.timestamp_ns),
                sequence_id=event.sequence_id,
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp_ns=event.timestamp_ns,
                policy_bev_source=str(debug.get("policy_bev_source", "unknown")),
                scorer_mode=scorer_mode,
                selected_id=selected_id,
                transparent_selected_id=transparent_selected_id,
                candidate_ids=tuple(candidate_ids),
                learned_logits=learned_logits,
                transparent_scores=transparent_scores,
                candidate_features=candidate_features,
                coverage_memory=dict(debug.get("coverage_memory", {})) if isinstance(debug.get("coverage_memory"), dict) else {},
                coverage_memory_before=dict(debug.get("coverage_memory_before", {}))
                if isinstance(debug.get("coverage_memory_before"), dict)
                else {},
                coverage_memory_update=dict(debug.get("coverage_memory_update", {}))
                if isinstance(debug.get("coverage_memory_update"), dict)
                else {},
                local_bev_ref=event.local_bev_ref,
                debug=debug,
                cmd_vel_non_null=event.cmd_vel is not None,
                control_safe=_event_flag(event, "control_safe"),
                replay_only=_event_flag(event, "replay_only"),
                not_executed=_event_flag(event, "not_executed"),
                product_training_approved=_event_flag(event, "product_training_approved"),
                raw_pwm_emitted=_event_flag(event, "raw_pwm_emitted"),
                candidate_base_records=tuple(candidate_base_records),
            )
        )
    return records


def _candidate_id(candidate: JsonDict) -> str:
    value = candidate.get("id", candidate.get("trajectory_id", ""))
    return str(value) if value is not None else ""


def _candidate_base_record(candidate: JsonDict) -> JsonDict:
    return {
        "id": str(candidate.get("id", candidate.get("trajectory_id", ""))),
        "cmd_vel_proxy": dict(candidate.get("cmd_vel_proxy", {})) if isinstance(candidate.get("cmd_vel_proxy"), dict) else {},
        "duration_s": candidate.get("duration_s", candidate.get("duration_sec", 0.0)),
        "poses": candidate.get("poses", []),
        "footprint_cells": candidate.get("footprint_cells", []),
    }


def _transparent_selected_id(debug: JsonDict, scores: dict[str, JsonDict]) -> str | None:
    transparent = debug.get("transparent_decision")
    if isinstance(transparent, dict) and isinstance(transparent.get("selected_candidate_id"), str):
        return str(transparent["selected_candidate_id"])
    for candidate_id, score in scores.items():
        if score.get("selected_by_transparent_scorer") is True or score.get("selected_by_runtime_policy") is True:
            return candidate_id
    totals = [
        (candidate_id, _float_or_none(score.get("total_score")))
        for candidate_id, score in scores.items()
        if _float_or_none(score.get("total_score")) is not None
    ]
    if not totals:
        return None
    return min(totals, key=lambda item: float(item[1]))[0]


def _event_flag(event: BrainOutputEvent, field: str) -> bool:
    if isinstance(event.debug, dict) and isinstance(event.debug.get(field), bool):
        return bool(event.debug[field])
    for candidate in event.candidate_trajectories or []:
        if isinstance(candidate.get(field), bool) and bool(candidate[field]):
            return True
        for score_key in ("trajectory_score", "transparent_trajectory_score"):
            score = candidate.get(score_key)
            if isinstance(score, dict) and isinstance(score.get(field), bool) and bool(score[field]):
                return True
    return False


def _summarize_log(
    *,
    spec: LogSpec,
    records: list[DecisionRecord],
    label_index: dict[tuple[str, str, int], JsonDict],
    oracle_index: dict[tuple[str, str, int], dict[str, np.ndarray]],
    checkpoint_metadata: JsonDict,
) -> JsonDict:
    selected_distribution = _distribution(record.selected_id for record in records)
    transparent_selected = [record.transparent_selected_id for record in records if record.transparent_selected_id is not None]
    transparent_distribution = _distribution(str(value) for value in transparent_selected)
    logit_stats = _learned_logit_stats(records)
    transparent_stats = _transparent_score_stats(records)
    feature_stats = _candidate_feature_stats(records)
    selected_feature_means = _selected_feature_means(records)
    label_metrics = _label_agreement(records, label_index)
    bev_delta = _current_memory_bev_delta(spec.path, records)
    oracle_delta = _oracle_bev_delta(spec.path, records, oracle_index)
    candidate_hash = _candidate_hash(records)
    candidate_hash_check = _candidate_hash_check(candidate_hash, checkpoint_metadata)
    near_ties = _near_tie_stats(records)
    paired_margins = _paired_logit_margins(records)
    route_names = sorted({record.sequence_id for record in records})
    scorer_modes = _distribution(record.scorer_mode for record in records)
    policy_sources = _distribution(record.policy_bev_source for record in records)
    learned_transparent_agreements = [
        record.selected_id == record.transparent_selected_id
        for record in records
        if record.transparent_selected_id is not None and record.learned_logits
    ]
    safety = _log_safety(records)
    return {
        "name": spec.name,
        "log": spec.path.as_posix(),
        "route_names": route_names,
        "decision_count": len(records),
        "candidate_count": _mode_int([len(record.candidate_ids) for record in records]),
        "candidate_hash": candidate_hash,
        "candidate_hash_check": candidate_hash_check,
        "policy_bev_source_distribution": policy_sources,
        "scorer_mode_distribution": scorer_modes,
        "inferred_policy_bev_source": _single_or_mixed(policy_sources),
        "inferred_scorer_mode": _single_or_mixed(scorer_modes),
        "selected_candidate_distribution": selected_distribution,
        "selected_candidate_entropy": _entropy(selected_distribution),
        "selected_candidate_dominant_fraction": _dominant_fraction(selected_distribution),
        "selected_actions_collapsed": bool(selected_distribution) and _dominant_fraction(selected_distribution) >= 0.95,
        "transparent_selected_candidate_distribution": transparent_distribution,
        "transparent_selected_entropy": _entropy(transparent_distribution),
        "transparent_selected_dominant_fraction": _dominant_fraction(transparent_distribution),
        "transparent_selected_collapsed": bool(transparent_distribution)
        and _dominant_fraction(transparent_distribution) >= 0.95,
        "learned_vs_transparent_selected_agreement": _bool_mean(learned_transparent_agreements),
        "learned_vs_transparent_selected_count": len(learned_transparent_agreements),
        "learned_logit_stats_per_candidate": logit_stats,
        "transparent_score_means_per_candidate": transparent_stats,
        "left_right_paired_logit_margins": paired_margins,
        "near_tie_fractions": near_ties,
        "candidate_feature_stats_per_candidate": feature_stats,
        "selected_candidate_feature_means": selected_feature_means,
        "coverage_memory_growth": _coverage_growth(records),
        "current_vs_memory_bev_delta_statistics": bev_delta,
        "model_vs_oracle_bev_delta_statistics": oracle_delta,
        "action_label_v5_comparison": label_metrics,
        "safety_flags": safety,
    }


def _learned_logit_stats(records: list[DecisionRecord]) -> JsonDict:
    by_candidate: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for candidate_id, value in record.learned_logits.items():
            by_candidate[candidate_id].append(value)
    return {candidate_id: _number_stats(values) for candidate_id, values in sorted(by_candidate.items())}


def _transparent_score_stats(records: list[DecisionRecord]) -> JsonDict:
    accum: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        for candidate_id, score in record.transparent_scores.items():
            for source_key, report_key in TRANSPARENT_SCORE_FIELDS:
                value = _float_or_none(score.get(source_key))
                if value is not None:
                    accum[candidate_id][report_key].append(value)
    return {
        candidate_id: {key: _mean(values) for key, values in sorted(fields.items())}
        for candidate_id, fields in sorted(accum.items())
    }


def _candidate_feature_stats(records: list[DecisionRecord]) -> JsonDict:
    accum: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        for candidate_id, features in record.candidate_features.items():
            for feature_name, value in features.items():
                accum[candidate_id][feature_name].append(value)
    return {
        candidate_id: {
            feature_name: _number_stats(values)
            for feature_name, values in sorted(features.items())
        }
        for candidate_id, features in sorted(accum.items())
    }


def _selected_feature_means(records: list[DecisionRecord]) -> JsonDict:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        features = record.candidate_features.get(record.selected_id)
        if not features:
            continue
        for feature_name, value in features.items():
            values[feature_name].append(value)
    return {feature_name: _mean(items) for feature_name, items in sorted(values.items())}


def _paired_logit_margins(records: list[DecisionRecord]) -> JsonDict:
    report: JsonDict = {}
    for left_id, right_id, name in PAIR_IDS:
        margins = [
            record.learned_logits[left_id] - record.learned_logits[right_id]
            for record in records
            if left_id in record.learned_logits and right_id in record.learned_logits
        ]
        report[name] = {
            **_number_stats(margins),
            "left_id": left_id,
            "right_id": right_id,
            "positive_left_higher_fraction": _fraction_numbers(margins, lambda value: value > 0.0),
            "negative_right_higher_fraction": _fraction_numbers(margins, lambda value: value < 0.0),
            "near_tie_fraction_abs_le_0_001": _fraction_numbers(margins, lambda value: abs(value) <= 0.001),
            "near_tie_fraction_abs_le_0_05": _fraction_numbers(margins, lambda value: abs(value) <= 0.05),
        }
    return report


def _near_tie_stats(records: list[DecisionRecord]) -> JsonDict:
    learned_margins: list[float] = []
    transparent_margins: list[float] = []
    for record in records:
        if len(record.learned_logits) >= 2:
            values = sorted(record.learned_logits.values(), reverse=True)
            learned_margins.append(float(values[0] - values[1]))
        totals = [
            _float_or_none(score.get("total_score"))
            for score in record.transparent_scores.values()
            if _float_or_none(score.get("total_score")) is not None
        ]
        if len(totals) >= 2:
            ordered = sorted(float(value) for value in totals)
            transparent_margins.append(float(ordered[1] - ordered[0]))
    return {
        "learned_top1_top2_logit_margin": {
            **_number_stats(learned_margins),
            "near_tie_fraction_abs_le_0_001": _fraction_numbers(learned_margins, lambda value: abs(value) <= 0.001),
            "near_tie_fraction_abs_le_0_05": _fraction_numbers(learned_margins, lambda value: abs(value) <= 0.05),
        },
        "transparent_top1_top2_total_margin": {
            **_number_stats(transparent_margins),
            "near_tie_fraction_abs_le_0_001": _fraction_numbers(
                transparent_margins, lambda value: abs(value) <= 0.001
            ),
            "near_tie_fraction_abs_le_0_05": _fraction_numbers(transparent_margins, lambda value: abs(value) <= 0.05),
        },
    }


def _coverage_growth(records: list[DecisionRecord]) -> JsonDict:
    seen = [_float_or_none(record.coverage_memory.get("coverage_memory_cells_seen")) for record in records]
    covered = [_float_or_none(record.coverage_memory.get("coverage_memory_cells_covered")) for record in records]
    seen_values = [float(value) for value in seen if value is not None]
    covered_values = [float(value) for value in covered if value is not None]
    deltas = [
        _float_or_none(record.coverage_memory_update.get("cells_seen_delta"))
        for record in records
        if _float_or_none(record.coverage_memory_update.get("cells_seen_delta")) is not None
    ]
    return {
        "available": bool(seen_values or covered_values),
        "seen_start": seen_values[0] if seen_values else 0.0,
        "seen_end": seen_values[-1] if seen_values else 0.0,
        "seen_growth": (seen_values[-1] - seen_values[0]) if seen_values else 0.0,
        "covered_start": covered_values[0] if covered_values else 0.0,
        "covered_end": covered_values[-1] if covered_values else 0.0,
        "covered_growth": (covered_values[-1] - covered_values[0]) if covered_values else 0.0,
        "cells_seen_delta_mean": _mean([float(value) for value in deltas if value is not None]),
        "pose_alignment_attempt_count_end": int(records[-1].coverage_memory.get("pose_alignment_attempt_count", 0))
        if records
        else 0,
        "missing_pose_delta_count_end": int(records[-1].coverage_memory.get("missing_pose_delta_count", 0)) if records else 0,
    }


def _current_memory_bev_delta(log_path: Path, records: list[DecisionRecord]) -> JsonDict:
    deltas: dict[str, list[float]] = defaultdict(list)
    nonzero_counts: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.local_bev_ref is None:
            continue
        path = log_path / record.local_bev_ref
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            for channel in BEV_CHANNELS:
                current_key = f"current_bev_{channel}_prob"
                memory_key = f"memory_bev_{channel}_prob"
                if current_key not in data.files or memory_key not in data.files:
                    continue
                current = np.asarray(data[current_key], dtype=np.float32)
                memory = np.asarray(data[memory_key], dtype=np.float32)
                abs_delta = np.abs(memory - current)
                deltas[channel].append(float(np.mean(abs_delta)))
                nonzero_counts[channel].append(float(np.count_nonzero(abs_delta > 1.0e-6) / max(abs_delta.size, 1)))
    per_channel: JsonDict = {}
    for channel in BEV_CHANNELS:
        per_channel[channel] = {
            **_number_stats(deltas.get(channel, [])),
            "nonzero_cell_fraction_mean": _mean(nonzero_counts.get(channel, [])),
        }
    all_means = [value for values in deltas.values() for value in values]
    return {
        "available": bool(all_means),
        "matched_artifact_count": max((len(values) for values in deltas.values()), default=0),
        "mean_abs_delta_over_channels": _mean(all_means),
        "max_channel_mean_abs_delta": max((_mean(values) for values in deltas.values()), default=0.0),
        "per_channel": per_channel,
    }


def _load_action_label_index(action_label_pack: str | Path | None) -> dict[tuple[str, str, int], JsonDict]:
    if action_label_pack is None:
        return {}
    root = Path(action_label_pack)
    manifest = read_json_object(root / "manifest.json")
    records: dict[tuple[str, str, int], JsonDict] = {}
    for item in manifest.get("examples", []):
        if not isinstance(item, dict):
            continue
        if item.get("bc_label_valid") is not True or item.get("action_supervision_ok") is not True:
            continue
        sequence_id = str(item.get("sequence_id", ""))
        camera_id = str(item.get("camera_id", ""))
        frame_id = _int_or_default(item.get("frame_id"), -1)
        if sequence_id and camera_id and frame_id >= 0:
            records[(sequence_id, camera_id, frame_id)] = dict(item)
    return records


def _label_agreement(records: list[DecisionRecord], label_index: dict[tuple[str, str, int], JsonDict]) -> JsonDict:
    if not label_index:
        return {"available": False, "matched_decision_count": 0}
    labels: list[str] = []
    learned_matches: list[bool] = []
    transparent_matches: list[bool] = []
    collapsed_mismatches: Counter[str] = Counter()
    for record in records:
        label = label_index.get((record.sequence_id, record.camera_id, record.frame_id))
        if label is None:
            continue
        label_id = str(label.get("selected_candidate_id", label.get("future_motion_primary_candidate_id", "")))
        if not label_id:
            continue
        labels.append(label_id)
        learned_matches.append(record.selected_id == label_id)
        if record.transparent_selected_id is not None:
            transparent_matches.append(record.transparent_selected_id == label_id)
        if record.selected_id != label_id:
            collapsed_mismatches[f"{record.selected_id}!=label:{label_id}"] += 1
    return {
        "available": True,
        "matched_decision_count": len(labels),
        "future_motion_label_distribution": _distribution(labels),
        "future_motion_label_entropy": _entropy(_distribution(labels)),
        "future_motion_label_dominant_fraction": _dominant_fraction(_distribution(labels)),
        "learned_selected_vs_v5_label_agreement": _bool_mean(learned_matches),
        "transparent_selected_vs_v5_label_agreement": _bool_mean(transparent_matches),
        "transparent_selected_vs_v5_label_count": len(transparent_matches),
        "collapse_mismatch_summary": dict(collapsed_mismatches.most_common(12)),
    }


def _load_oracle_bev_index(
    *,
    action_label_pack: str | Path | None,
    spatial_packs: list[str | Path],
) -> dict[tuple[str, str, int], dict[str, np.ndarray]]:
    index: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    if action_label_pack is not None:
        root = Path(action_label_pack)
        manifest = read_json_object(root / "manifest.json")
        source_roots = _resolve_action_source_roots(root, manifest)
        for item in manifest.get("examples", []):
            if not isinstance(item, dict) or item.get("action_supervision_ok") is not True:
                continue
            key = (
                str(item.get("sequence_id", "")),
                str(item.get("camera_id", "")),
                _int_or_default(item.get("frame_id"), -1),
            )
            if key[2] < 0 or key in index:
                continue
            source_ref = item.get("source_ref")
            if not isinstance(source_ref, str):
                continue
            source_path = _first_existing(root_candidate / source_ref for root_candidate in source_roots)
            if source_path is None:
                continue
            arrays = _load_oracle_arrays(source_path)
            if arrays is not None:
                index[key] = arrays
    for spatial_pack in spatial_packs:
        root = Path(spatial_pack)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json_object(manifest_path)
        for item in manifest.get("examples", []):
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("sequence_id", "")),
                str(item.get("camera_id", "")),
                _int_or_default(item.get("frame_id"), -1),
            )
            if key[2] < 0 or key in index:
                continue
            example_path = item.get("example_path")
            if not isinstance(example_path, str):
                continue
            arrays = _load_oracle_arrays(root / example_path)
            if arrays is not None:
                index[key] = arrays
    return index


def _resolve_action_source_roots(action_pack_root: Path, manifest: JsonDict) -> list[Path]:
    roots: list[Path] = []
    raw = manifest.get("source_dirs")
    if not isinstance(raw, list):
        return roots
    for item in raw:
        if not isinstance(item, str):
            continue
        path = Path(item)
        roots.append(path if path.exists() else action_pack_root.parent / path)
    return roots


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _load_oracle_arrays(path: Path) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            free = np.asarray(data["bev_free"], dtype=np.float32)
            return {
                "free": free,
                "occupied": np.asarray(data["bev_obstacle"], dtype=np.float32),
                "unknown": np.asarray(data["bev_unknown"], dtype=np.float32),
                "traversable": np.asarray(data["bev_traversable"], dtype=np.float32)
                if "bev_traversable" in data.files
                else free,
                "risky": np.asarray(data["bev_risky"], dtype=np.float32)
                if "bev_risky" in data.files
                else np.asarray(data["bev_obstacle"], dtype=np.float32),
                "confidence": np.asarray(data["bev_confidence"], dtype=np.float32)
                if "bev_confidence" in data.files
                else np.ones_like(free, dtype=np.float32),
            }
    except (FileNotFoundError, KeyError, ValueError):
        return None


def _oracle_bev_delta(
    log_path: Path,
    records: list[DecisionRecord],
    oracle_index: dict[tuple[str, str, int], dict[str, np.ndarray]],
) -> JsonDict:
    if not oracle_index:
        return {"available": False, "matched_decision_count": 0}
    deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    matched = 0
    for record in records:
        oracle = oracle_index.get((record.sequence_id, record.camera_id, record.frame_id))
        if oracle is None or record.local_bev_ref is None:
            continue
        path = log_path / record.local_bev_ref
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            matched += 1
            for source in ("current", "memory"):
                for channel in BEV_CHANNELS:
                    key = f"{source}_bev_{channel}_prob"
                    if key not in data.files or channel not in oracle:
                        continue
                    model = np.asarray(data[key], dtype=np.float32)
                    label = np.asarray(oracle[channel], dtype=np.float32)
                    if model.shape != label.shape:
                        continue
                    deltas[source][channel].append(float(np.mean(np.abs(model - label))))
    per_source: JsonDict = {}
    for source, by_channel in sorted(deltas.items()):
        all_values = [value for values in by_channel.values() for value in values]
        per_source[source] = {
            "mean_abs_delta_over_channels": _mean(all_values),
            "per_channel": {channel: _number_stats(values) for channel, values in sorted(by_channel.items())},
        }
    return {
        "available": bool(per_source),
        "matched_decision_count": matched,
        "per_source": per_source,
    }


def _load_checkpoint_metadata(path: str | Path | None) -> JsonDict:
    if path is None:
        return {}
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    model_config = payload.get("model_config") if isinstance(payload.get("model_config"), dict) else {}
    return {
        "path": Path(path).as_posix(),
        "checkpoint_version": payload.get("checkpoint_version"),
        "model_name": payload.get("model_name"),
        "metadata": dict(metadata),
        "metrics_summary": {
            "bev_source": metadata.get("bev_source", metrics.get("bev_source")),
            "modeld_dir": metadata.get("modeld_dir", metrics.get("modeld_dir")),
            "candidate_hash": metadata.get("candidate_hash", metrics.get("candidate_hash")),
            "candidate_ids": metadata.get("candidate_ids", metrics.get("candidate_ids")),
            "val_top1_action_agreement": metrics.get("val_top1_action_agreement"),
            "val_rank_correlation_or_proxy": metrics.get("val_rank_correlation_or_proxy"),
            "train_top1_action_agreement": metrics.get("train_top1_action_agreement"),
            "source_distribution": metrics.get("source_distribution"),
            "val_source_distribution": metrics.get("val_source_distribution"),
            "control_safe": metadata.get("control_safe", metrics.get("control_safe")),
            "replay_only": metadata.get("replay_only", metrics.get("replay_only")),
            "not_executed": metadata.get("not_executed", metrics.get("not_executed")),
        },
        "model_config": model_config,
    }


def _candidate_hash(records: list[DecisionRecord]) -> str | None:
    if not records or not records[0].candidate_base_records:
        return None
    text = deterministic_json({"candidates": list(records[0].candidate_base_records)})
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _candidate_hash_check(candidate_hash: str | None, checkpoint_metadata: JsonDict) -> JsonDict:
    if candidate_hash is None or not checkpoint_metadata:
        return {"available": False}
    summary = checkpoint_metadata.get("metrics_summary")
    expected = summary.get("candidate_hash") if isinstance(summary, dict) else None
    return {
        "available": isinstance(expected, str),
        "runtime_candidate_hash": candidate_hash,
        "checkpoint_candidate_hash": expected,
        "matches": expected == candidate_hash if isinstance(expected, str) else None,
    }


def _build_comparisons(records_by_log: dict[str, list[DecisionRecord]], log_reports: list[JsonDict]) -> list[JsonDict]:
    reports_by_name = {str(report["name"]): report for report in log_reports}
    comparisons: list[JsonDict] = []
    names = sorted(records_by_log)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left_report = reports_by_name[left_name]
            right_report = reports_by_name[right_name]
            same_route = set(left_report.get("route_names", [])) & set(right_report.get("route_names", []))
            if not same_route:
                continue
            left_source = str(left_report.get("inferred_policy_bev_source"))
            right_source = str(right_report.get("inferred_policy_bev_source"))
            left_mode = str(left_report.get("inferred_scorer_mode"))
            right_mode = str(right_report.get("inferred_scorer_mode"))
            interesting = (
                left_source != right_source
                or left_mode != right_mode
                or "current" in {left_source, right_source}
                or "memory" in {left_source, right_source}
            )
            if not interesting:
                continue
            comparisons.append(
                _compare_record_sets(
                    left_name=left_name,
                    left_records=records_by_log[left_name],
                    right_name=right_name,
                    right_records=records_by_log[right_name],
                )
            )
    return comparisons


def _compare_record_sets(
    *,
    left_name: str,
    left_records: list[DecisionRecord],
    right_name: str,
    right_records: list[DecisionRecord],
) -> JsonDict:
    left_by_key = {record.key: record for record in left_records}
    right_by_key = {record.key: record for record in right_records}
    keys = sorted(set(left_by_key) & set(right_by_key))
    selected_diff = [left_by_key[key].selected_id != right_by_key[key].selected_id for key in keys]
    transparent_diff = [
        left_by_key[key].transparent_selected_id != right_by_key[key].transparent_selected_id
        for key in keys
        if left_by_key[key].transparent_selected_id is not None and right_by_key[key].transparent_selected_id is not None
    ]
    logit_deltas: list[float] = []
    feature_deltas: list[float] = []
    for key in keys:
        left = left_by_key[key]
        right = right_by_key[key]
        for candidate_id in set(left.learned_logits) & set(right.learned_logits):
            logit_deltas.append(abs(left.learned_logits[candidate_id] - right.learned_logits[candidate_id]))
        for candidate_id in set(left.candidate_features) & set(right.candidate_features):
            for feature_name in set(left.candidate_features[candidate_id]) & set(right.candidate_features[candidate_id]):
                feature_deltas.append(
                    abs(left.candidate_features[candidate_id][feature_name] - right.candidate_features[candidate_id][feature_name])
                )
    return {
        "left_log": left_name,
        "right_log": right_name,
        "matched_decision_count": len(keys),
        "selected_difference_count": int(sum(1 for value in selected_diff if value)),
        "selected_difference_fraction": _bool_mean(selected_diff),
        "transparent_selected_difference_fraction": _bool_mean(transparent_diff),
        "transparent_selected_difference_count": int(sum(1 for value in transparent_diff if value)),
        "learned_logit_abs_delta": _number_stats(logit_deltas),
        "candidate_feature_abs_delta": _number_stats(feature_deltas),
    }


def _aggregate_report(
    log_reports: list[JsonDict],
    comparisons: list[JsonDict],
    label_index: dict[tuple[str, str, int], JsonDict],
    checkpoint_metadata: JsonDict,
) -> JsonDict:
    learned = [report for report in log_reports if str(report.get("inferred_scorer_mode")) == "learned"]
    transparent = [report for report in log_reports if str(report.get("inferred_scorer_mode")) == "transparent"]
    route_names = sorted({route for report in log_reports for route in report.get("route_names", [])})
    selected_by_route_mode = {
        str(report["name"]): {
            "route_names": report.get("route_names", []),
            "policy_bev_source": report.get("inferred_policy_bev_source"),
            "scorer_mode": report.get("inferred_scorer_mode"),
            "selected_candidate_distribution": report.get("selected_candidate_distribution", {}),
            "dominant_fraction": report.get("selected_candidate_dominant_fraction", 0.0),
            "entropy": report.get("selected_candidate_entropy", 0.0),
        }
        for report in log_reports
    }
    label_distribution = _distribution(
        str(record.get("selected_candidate_id", record.get("future_motion_primary_candidate_id", "")))
        for record in label_index.values()
    )
    memory_current_comparisons = [
        comparison
        for comparison in comparisons
        if comparison.get("matched_decision_count", 0)
        and _comparison_looks_current_memory(comparison, log_reports)
    ]
    return {
        "log_count": len(log_reports),
        "route_count": len(route_names),
        "route_names": route_names,
        "learned_log_count": len(learned),
        "transparent_log_count": len(transparent),
        "learned_collapsed_log_count": sum(1 for report in learned if bool(report.get("selected_actions_collapsed"))),
        "transparent_collapsed_log_count": sum(1 for report in transparent if bool(report.get("selected_actions_collapsed"))),
        "transparent_selected_collapsed_log_count": sum(
            1 for report in log_reports if bool(report.get("transparent_selected_collapsed"))
        ),
        "learned_dominant_fraction_mean": _mean(
            [float(report.get("selected_candidate_dominant_fraction", 0.0)) for report in learned]
        ),
        "transparent_dominant_fraction_mean": _mean(
            [float(report.get("selected_candidate_dominant_fraction", 0.0)) for report in transparent]
        ),
        "selected_by_route_mode": selected_by_route_mode,
        "current_vs_memory_selected_difference_fraction_mean": _mean(
            [float(item.get("selected_difference_fraction", 0.0)) for item in memory_current_comparisons]
        ),
        "current_vs_memory_comparison_count": len(memory_current_comparisons),
        "action_label_v5_distribution": label_distribution,
        "action_label_v5_entropy": _entropy(label_distribution),
        "action_label_v5_dominant_fraction": _dominant_fraction(label_distribution),
        "checkpoint_metadata": checkpoint_metadata,
    }


def _comparison_looks_current_memory(comparison: JsonDict, log_reports: list[JsonDict]) -> bool:
    by_name = {str(report["name"]): report for report in log_reports}
    left = by_name.get(str(comparison.get("left_log")))
    right = by_name.get(str(comparison.get("right_log")))
    if left is None or right is None:
        return False
    return {left.get("inferred_policy_bev_source"), right.get("inferred_policy_bev_source")} == {"current", "memory"} and (
        left.get("inferred_scorer_mode") == right.get("inferred_scorer_mode")
    )


def _classify_root_cause(*, log_reports: list[JsonDict], comparisons: list[JsonDict], aggregate: JsonDict) -> JsonDict:
    buckets: JsonDict = {
        "learned_scorer_prior_bias": _bucket(False, "not enough evidence"),
        "model_bev_distribution_shift": _bucket(False, "not enough evidence"),
        "candidate_feature_scaling_bug": _bucket(False, "candidate feature stats are unavailable or within rough bounds"),
        "left_right_signed_feature_bias": _bucket(False, "paired learned margins do not show a consistent right-side preference"),
        "transparent_scorer_collapse": _bucket(False, "transparent scorer decisions did not collapse"),
        "memory_delta_too_small_for_policy": _bucket(False, "current-vs-memory comparison unavailable or decisions changed"),
        "route_data_too_narrow": _bucket(False, "at least three route logs or label sources are present"),
        "implementation_bug_suspected": _bucket(False, "no hard metadata or wiring inconsistency detected"),
    }

    learned = [report for report in log_reports if str(report.get("inferred_scorer_mode")) == "learned"]
    transparent = [report for report in log_reports if str(report.get("inferred_scorer_mode")) == "transparent"]
    learned_collapsed = bool(learned) and all(bool(report.get("selected_actions_collapsed")) for report in learned)
    transparent_not_collapsed = bool(transparent) and any(not bool(report.get("selected_actions_collapsed")) for report in transparent)
    low_agreement = [
        float(report.get("learned_vs_transparent_selected_agreement", 0.0))
        for report in learned
        if int(report.get("learned_vs_transparent_selected_count", 0)) > 0
    ]
    if learned_collapsed and (transparent_not_collapsed or _mean(low_agreement) < 0.5):
        buckets["learned_scorer_prior_bias"] = _bucket(
            True,
            "learned replay collapsed while transparent selected actions were more diverse or disagreed",
        )

    oracle_deltas = [
        float(
            report.get("model_vs_oracle_bev_delta_statistics", {})
            .get("per_source", {})
            .get(str(report.get("inferred_policy_bev_source")), {})
            .get("mean_abs_delta_over_channels", 0.0)
        )
        for report in log_reports
        if report.get("model_vs_oracle_bev_delta_statistics", {}).get("available")
    ]
    label_dominant = float(aggregate.get("action_label_v5_dominant_fraction", 0.0))
    learned_label_agreement = [
        float(report.get("action_label_v5_comparison", {}).get("learned_selected_vs_v5_label_agreement", 0.0))
        for report in learned
        if report.get("action_label_v5_comparison", {}).get("available")
    ]
    checkpoint = aggregate.get("checkpoint_metadata", {})
    trained_on_oracle = (
        isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("metrics_summary"), dict)
        and checkpoint["metrics_summary"].get("bev_source") == "oracle"
    )
    if learned_collapsed and trained_on_oracle and oracle_deltas and _mean(oracle_deltas) >= 0.10 and label_dominant < 0.80:
        buckets["model_bev_distribution_shift"] = _bucket(
            True,
            "learned scorer was trained on oracle BEV and model-vs-oracle BEV deltas are large on matched frames",
        )
    elif learned_collapsed and trained_on_oracle and learned_label_agreement and _mean(learned_label_agreement) < 0.35 and label_dominant < 0.80:
        buckets["model_bev_distribution_shift"] = _bucket(
            True,
            "learned scorer was trained on oracle BEV and disagrees strongly with non-collapsed v5 labels on model BEV",
        )

    feature_abs_max = _max_feature_abs(log_reports)
    feature_nan = _feature_has_nan_or_inf(log_reports)
    if feature_nan or feature_abs_max > 5.0:
        buckets["candidate_feature_scaling_bug"] = _bucket(
            True,
            f"candidate feature values exceeded rough normalized bounds or were non-finite; max_abs={feature_abs_max}",
        )

    if _right_margin_bias_fraction(learned) >= 0.95:
        buckets["left_right_signed_feature_bias"] = _bucket(
            True,
            "left-vs-right learned logit margins consistently favor right-side candidates",
        )

    if transparent and all(bool(report.get("selected_actions_collapsed")) for report in transparent):
        buckets["transparent_scorer_collapse"] = _bucket(True, "transparent scorer selected one candidate in every route/mode")

    memory_delta_evidence = []
    for report in log_reports:
        delta = report.get("current_vs_memory_bev_delta_statistics", {})
        if isinstance(delta, dict) and delta.get("available"):
            memory_delta_evidence.append(float(delta.get("mean_abs_delta_over_channels", 0.0)))
    memory_comparisons = [
        comparison
        for comparison in comparisons
        if int(comparison.get("matched_decision_count", 0)) > 0
        and float(comparison.get("selected_difference_fraction", 0.0)) == 0.0
    ]
    if memory_comparisons and _mean(memory_delta_evidence) > 1.0e-4:
        buckets["memory_delta_too_small_for_policy"] = _bucket(
            True,
            "current and memory BEV arrays differ, but matched current-vs-memory selected actions did not change",
        )

    if int(aggregate.get("route_count", 0)) < 3:
        buckets["route_data_too_narrow"] = _bucket(True, "audit did not cover cafe, office, and corridor replay logs")

    bug_reasons: list[str] = []
    for report in log_reports:
        check = report.get("candidate_hash_check", {})
        if isinstance(check, dict) and check.get("available") and check.get("matches") is False:
            bug_reasons.append(f"{report.get('name')}: checkpoint candidate_hash mismatch")
        safety = report.get("safety_flags", {})
        if isinstance(safety, dict) and (
            int(safety.get("cmd_vel_non_null_count", 0)) > 0
            or bool(safety.get("control_safe_any", False))
            or bool(safety.get("product_training_approved_any", False))
            or bool(safety.get("raw_pwm_emitted_any", False))
            or not bool(safety.get("replay_only_all", False))
            or not bool(safety.get("not_executed_all", False))
        ):
            bug_reasons.append(f"{report.get('name')}: replay safety flags regressed")
    for comparison in comparisons:
        logit_delta = comparison.get("learned_logit_abs_delta", {})
        feature_delta = comparison.get("candidate_feature_abs_delta", {})
        if (
            int(comparison.get("matched_decision_count", 0)) > 0
            and isinstance(logit_delta, dict)
            and isinstance(feature_delta, dict)
            and int(logit_delta.get("count", 0)) > 0
            and int(feature_delta.get("count", 0)) > 0
            and float(feature_delta.get("max", 0.0)) == 0.0
            and float(logit_delta.get("max", 0.0)) == 0.0
            and _comparison_looks_current_memory(comparison, log_reports)
            and _mean(memory_delta_evidence) > 1.0e-4
        ):
            bug_reasons.append(
                f"{comparison.get('left_log')} vs {comparison.get('right_log')}: BEV changed but learned features/logits did not"
            )
    if bug_reasons:
        buckets["implementation_bug_suspected"] = _bucket(True, "; ".join(bug_reasons))

    priority = [
        "implementation_bug_suspected",
        "candidate_feature_scaling_bug",
        "transparent_scorer_collapse",
        "model_bev_distribution_shift",
        "left_right_signed_feature_bias",
        "learned_scorer_prior_bias",
        "memory_delta_too_small_for_policy",
        "route_data_too_narrow",
    ]
    present = [name for name in priority if buckets[name]["present"]]
    return {
        "buckets": buckets,
        "primary": present[0] if present else "no_single_root_cause_identified",
        "present_buckets": present,
        "recommended_next_repair": _recommended_next_repair(present),
    }


def _bucket(present: bool, reason: str) -> JsonDict:
    return {"present": bool(present), "reason": reason}


def _recommended_next_repair(present: list[str]) -> str:
    if "implementation_bug_suspected" in present:
        return "fix_runtime_or_audit_wiring_before_any_training"
    if "candidate_feature_scaling_bug" in present or "left_right_signed_feature_bias" in present:
        if "model_bev_distribution_shift" in present and "learned_scorer_prior_bias" in present:
            return "scorer_retraining_on_model_bev_with_left_right_feature_ablation"
        if "candidate_feature_scaling_bug" in present:
            return "candidate_feature_fix"
        return "candidate_left_right_feature_ablation_before_retraining"
    if "transparent_scorer_collapse" in present:
        return "transparent_scorer_or_candidate_set_repair"
    if "model_bev_distribution_shift" in present and "learned_scorer_prior_bias" in present:
        return "scorer_retraining_on_model_bev_after_bev_calibration_check"
    if "model_bev_distribution_shift" in present:
        return "bev_calibration_then_model_bev_scorer_eval"
    if "learned_scorer_prior_bias" in present:
        return "route_balanced_scorer_retraining_on_runtime_model_bev"
    if "memory_delta_too_small_for_policy" in present:
        return "memory_to_policy_sensitivity_or_bev_calibration_repair"
    if "route_data_too_narrow" in present:
        return "more_route_data"
    return "inspect_per_route_outliers_before_repair"


def _max_feature_abs(log_reports: list[JsonDict]) -> float:
    max_value = 0.0
    for report in log_reports:
        stats = report.get("candidate_feature_stats_per_candidate", {})
        if not isinstance(stats, dict):
            continue
        for feature_stats in stats.values():
            if not isinstance(feature_stats, dict):
                continue
            for values in feature_stats.values():
                if isinstance(values, dict):
                    max_value = max(max_value, abs(float(values.get("min", 0.0))), abs(float(values.get("max", 0.0))))
    return float(max_value)


def _feature_has_nan_or_inf(log_reports: list[JsonDict]) -> bool:
    for report in log_reports:
        stats = report.get("candidate_feature_stats_per_candidate", {})
        if not isinstance(stats, dict):
            continue
        for feature_stats in stats.values():
            if not isinstance(feature_stats, dict):
                continue
            for values in feature_stats.values():
                if isinstance(values, dict) and int(values.get("non_finite_count", 0)) > 0:
                    return True
    return False


def _right_margin_bias_fraction(log_reports: list[JsonDict]) -> float:
    fractions: list[float] = []
    for report in log_reports:
        margins = report.get("left_right_paired_logit_margins", {})
        if not isinstance(margins, dict):
            continue
        for item in margins.values():
            if isinstance(item, dict) and int(item.get("count", 0)) > 0:
                fractions.append(float(item.get("negative_right_higher_fraction", 0.0)))
    return _mean(fractions)


def _log_safety(records: list[DecisionRecord]) -> JsonDict:
    return {
        "cmd_vel_non_null_count": sum(1 for record in records if record.cmd_vel_non_null),
        "control_safe_any": any(record.control_safe for record in records),
        "replay_only_all": bool(records) and all(record.replay_only for record in records),
        "not_executed_all": bool(records) and all(record.not_executed for record in records),
        "product_training_approved_any": any(record.product_training_approved for record in records),
        "raw_pwm_emitted_any": any(record.raw_pwm_emitted for record in records),
    }


def _aggregate_safety(log_reports: list[JsonDict]) -> JsonDict:
    safety_records = [report.get("safety_flags", {}) for report in log_reports if isinstance(report.get("safety_flags"), dict)]
    return {
        "replay_only": bool(safety_records) and all(bool(item.get("replay_only_all", False)) for item in safety_records),
        "not_executed": bool(safety_records) and all(bool(item.get("not_executed_all", False)) for item in safety_records),
        "control_safe": any(bool(item.get("control_safe_any", False)) for item in safety_records),
        "product_training_approved": any(
            bool(item.get("product_training_approved_any", False)) for item in safety_records
        ),
        "cmd_vel_non_null_count": int(sum(int(item.get("cmd_vel_non_null_count", 0)) for item in safety_records)),
        "raw_pwm_emitted": any(bool(item.get("raw_pwm_emitted_any", False)) for item in safety_records),
    }


def _write_summary_csv(path: str | Path, log_reports: list[JsonDict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "route_names",
                "scorer_mode",
                "policy_bev_source",
                "decision_count",
                "dominant_fraction",
                "entropy",
                "selected_distribution",
                "transparent_dominant_fraction",
                "learned_vs_transparent_agreement",
                "v5_agreement",
            ],
        )
        writer.writeheader()
        for report in log_reports:
            writer.writerow(
                {
                    "name": report.get("name"),
                    "route_names": ",".join(str(item) for item in report.get("route_names", [])),
                    "scorer_mode": report.get("inferred_scorer_mode"),
                    "policy_bev_source": report.get("inferred_policy_bev_source"),
                    "decision_count": report.get("decision_count"),
                    "dominant_fraction": report.get("selected_candidate_dominant_fraction"),
                    "entropy": report.get("selected_candidate_entropy"),
                    "selected_distribution": json.dumps(report.get("selected_candidate_distribution", {}), sort_keys=True),
                    "transparent_dominant_fraction": report.get("transparent_selected_dominant_fraction"),
                    "learned_vs_transparent_agreement": report.get("learned_vs_transparent_selected_agreement"),
                    "v5_agreement": report.get("action_label_v5_comparison", {}).get(
                        "learned_selected_vs_v5_label_agreement"
                    )
                    if isinstance(report.get("action_label_v5_comparison"), dict)
                    else None,
                }
            )


def _write_summary_jsonl(path: str | Path, log_reports: list[JsonDict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for report in log_reports:
            handle.write(deterministic_json(report))
            handle.write("\n")


def _markdown_report(report: JsonDict) -> str:
    aggregate = report.get("aggregate", {}) if isinstance(report.get("aggregate"), dict) else {}
    root = report.get("root_cause_classification", {}) if isinstance(report.get("root_cause_classification"), dict) else {}
    safety = report.get("safety_flags", {}) if isinstance(report.get("safety_flags"), dict) else {}
    lines = [
        "# Goal 21A Closed-Loop Policy Collapse Audit",
        "",
        "## Summary",
        "",
        f"- Logs audited: `{aggregate.get('log_count')}`",
        f"- Routes audited: `{aggregate.get('route_names')}`",
        f"- Learned collapsed logs: `{aggregate.get('learned_collapsed_log_count')}` / `{aggregate.get('learned_log_count')}`",
        f"- Transparent collapsed logs: `{aggregate.get('transparent_collapsed_log_count')}` / `{aggregate.get('transparent_log_count')}`",
        f"- Current-vs-memory selected difference mean: `{aggregate.get('current_vs_memory_selected_difference_fraction_mean')}`",
        f"- v5 label dominant fraction: `{aggregate.get('action_label_v5_dominant_fraction')}`",
        f"- Primary root cause: `{root.get('primary')}`",
        f"- Present buckets: `{root.get('present_buckets')}`",
        f"- Recommended next repair: `{root.get('recommended_next_repair')}`",
        "",
        "## Safety Flags",
        "",
        f"- replay_only: `{safety.get('replay_only')}`",
        f"- not_executed: `{safety.get('not_executed')}`",
        f"- control_safe: `{safety.get('control_safe')}`",
        f"- cmd_vel_non_null_count: `{safety.get('cmd_vel_non_null_count')}`",
        f"- raw_pwm_emitted: `{safety.get('raw_pwm_emitted')}`",
        "",
        "## Per-Log Results",
        "",
    ]
    for item in report.get("logs", []):
        if not isinstance(item, dict):
            continue
        label = item.get("action_label_v5_comparison", {}) if isinstance(item.get("action_label_v5_comparison"), dict) else {}
        bev_delta = (
            item.get("current_vs_memory_bev_delta_statistics", {})
            if isinstance(item.get("current_vs_memory_bev_delta_statistics"), dict)
            else {}
        )
        oracle_delta = (
            item.get("model_vs_oracle_bev_delta_statistics", {})
            if isinstance(item.get("model_vs_oracle_bev_delta_statistics"), dict)
            else {}
        )
        lines.extend(
            [
                f"### {item.get('name')}",
                "",
                f"- route: `{item.get('route_names')}`",
                f"- mode/source: `{item.get('inferred_scorer_mode')}` / `{item.get('inferred_policy_bev_source')}`",
                f"- decisions: `{item.get('decision_count')}`",
                f"- selected distribution: `{item.get('selected_candidate_distribution')}`",
                f"- selected entropy/dominant: `{item.get('selected_candidate_entropy')}` / `{item.get('selected_candidate_dominant_fraction')}`",
                f"- transparent selected distribution: `{item.get('transparent_selected_candidate_distribution')}`",
                f"- learned-vs-transparent agreement: `{item.get('learned_vs_transparent_selected_agreement')}`",
                f"- v5 agreement learned/transparent: `{label.get('learned_selected_vs_v5_label_agreement')}` / `{label.get('transparent_selected_vs_v5_label_agreement')}`",
                f"- current-vs-memory BEV mean abs delta: `{bev_delta.get('mean_abs_delta_over_channels')}`",
                f"- model-vs-oracle BEV matched: `{oracle_delta.get('matched_decision_count')}`",
                "",
            ]
        )
    lines.extend(["## Bucket Classification", ""])
    buckets = root.get("buckets", {}) if isinstance(root.get("buckets"), dict) else {}
    for name, bucket in buckets.items():
        if not isinstance(bucket, dict):
            continue
        lines.append(f"- `{name}`: `{bucket.get('present')}` - {bucket.get('reason')}")
    return "\n".join(lines) + "\n"


def _distribution(values: Iterable[str]) -> dict[str, int]:
    counts: Counter[str] = Counter(value for value in values if value)
    return dict(sorted(counts.items()))


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


def _dominant_fraction(distribution: dict[str, int]) -> float:
    total = sum(int(value) for value in distribution.values())
    if total <= 0:
        return 0.0
    return float(max(int(value) for value in distribution.values()) / total)


def _number_stats(values: Iterable[float]) -> JsonDict:
    finite: list[float] = []
    non_finite_count = 0
    for value in values:
        number = float(value)
        if np.isfinite(number):
            finite.append(number)
        else:
            non_finite_count += 1
    if not finite:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "non_finite_count": non_finite_count,
        }
    array = np.asarray(finite, dtype=np.float64)
    return {
        "count": len(finite),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "non_finite_count": non_finite_count,
    }


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values if np.isfinite(float(value))]
    if not items:
        return 0.0
    return float(np.mean(np.asarray(items, dtype=np.float64)))


def _bool_mean(values: Iterable[bool]) -> float:
    items = [bool(value) for value in values]
    if not items:
        return 0.0
    return float(sum(1 for value in items if value) / len(items))


def _fraction_numbers(values: Iterable[float], predicate: Any) -> float:
    items = [float(value) for value in values if np.isfinite(float(value))]
    if not items:
        return 0.0
    return float(sum(1 for value in items if predicate(value)) / len(items))


def _mode_int(values: list[int]) -> int:
    if not values:
        return 0
    counts = Counter(values)
    return int(min(counts, key=lambda value: (-counts[value], value)))


def _single_or_mixed(distribution: dict[str, int]) -> str | None:
    if not distribution:
        return None
    values = sorted(distribution)
    return values[0] if len(values) == 1 else "mixed"


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _int_or_default(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit closed-loop replay policy collapse without training.")
    parser.add_argument("--log", action="append", required=True, help="Replay log path, optionally alias=path.")
    parser.add_argument("--compare-log", action="append", default=None, help="Additional log path, optionally alias=path.")
    parser.add_argument("--action-label-pack", default=None, help="Optional ActionLabelPack v5 root.")
    parser.add_argument("--spatial-pack", action="append", default=None, help="Optional SpatialTrainPack root.")
    parser.add_argument("--trajectory-scorer-checkpoint", default=None, help="Optional scorer checkpoint metadata.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--summary-jsonl", default=None)
    args = parser.parse_args(argv)
    report = audit_closed_loop_policy_collapse(
        logs=args.log,
        compare_logs=args.compare_log,
        action_label_pack=args.action_label_pack,
        spatial_packs=args.spatial_pack,
        trajectory_scorer_checkpoint=args.trajectory_scorer_checkpoint,
        out_json=args.out_json,
        out_md=args.out_md,
        summary_csv=args.summary_csv,
        summary_jsonl=args.summary_jsonl,
        command=" ".join(["python", "-m", "homebrain.policies.audit_closed_loop_policy_collapse", *(argv or [])]),
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
