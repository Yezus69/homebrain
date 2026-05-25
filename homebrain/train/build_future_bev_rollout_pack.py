from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from homebrain.data.spatial_dataset import DETERMINISTIC_CREATED_AT_UTC, write_deterministic_npz, write_json
from homebrain.messages.schema import JsonDict
from homebrain.teachers.artifacts import file_sha256, relative_to_root
from homebrain.train.future_bev_rollout_dataset import (
    DEFAULT_FUTURE_HORIZONS_S,
    FUTURE_BEV_CHANNELS,
    FUTURE_DERIVED_CHANNELS,
    FUTURE_ROLLOUT_EXAMPLE_SCHEMA_VERSION,
    FUTURE_ROLLOUT_PACK_SCHEMA_VERSION,
    ROLLOUT_PROVENANCE_FLAGS,
    SpatialRolloutFrame,
    build_future_rollout_targets_for_frame,
    default_candidates_for_manifest,
    group_frames_by_sequence,
    load_spatial_rollout_frames,
    meters_per_cell_from_manifest,
    robot_radius_m_from_manifest,
)
from homebrain.train.spatial_dataset import DINOFeatureStore
from homebrain.policies.candidate_trajectories import CandidateTrajectory, candidates_hash

SAMPLING_STRATEGIES: tuple[str, ...] = ("source_order", "route_balanced")


@dataclass(frozen=True)
class _SourceContext:
    root: Path
    feature_root: Path | None
    feature_store: DINOFeatureStore | None
    frames: list[SpatialRolloutFrame]
    manifest: JsonDict
    grouped: dict[tuple[str, str, str], list[SpatialRolloutFrame]]
    candidates: list[CandidateTrajectory]


def build_future_bev_rollout_pack(
    *,
    sources: list[str | Path],
    out_dir: str | Path,
    feature_dirs: list[str | Path] | None = None,
    horizons_s: tuple[float, ...] = DEFAULT_FUTURE_HORIZONS_S,
    max_examples: int | None = None,
    sampling_strategy: str = "source_order",
    fail_on_degenerate_labels: bool = False,
    reject_degenerate_groups: bool = False,
) -> Path:
    if not sources:
        raise ValueError("at least one SpatialTrainPack source is required")
    if not horizons_s or any(float(value) <= 0.0 for value in horizons_s):
        raise ValueError("horizons_s must contain positive values")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive when supplied")
    if sampling_strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"sampling_strategy must be one of {SAMPLING_STRATEGIES}")
    output = Path(out_dir)
    examples_dir = output / "examples"
    _clear_generated_outputs(output)
    examples_dir.mkdir(parents=True, exist_ok=True)

    source_roots = [Path(source) for source in sources]
    feature_roots = _feature_roots(feature_dirs, len(source_roots))
    all_examples: list[JsonDict] = []
    candidate_ids: list[str] | None = None
    candidate_hash_value: str | None = None
    grid_shape: tuple[int, int] | None = None
    meters_per_cell: float | None = None
    robot_radius_m: float | None = None
    horizon_valid_values: list[float] = []
    candidate_valid_values: list[float] = []
    invalid_reasons: dict[str, int] = {}
    examples_written = 0
    contexts: list[_SourceContext] = []

    for source_index, source_root in enumerate(source_roots):
        frames, manifest = load_spatial_rollout_frames(source_root)
        if not frames:
            continue
        source_grid = _grid_shape(manifest)
        if grid_shape is None:
            grid_shape = source_grid
            meters_per_cell = meters_per_cell_from_manifest(manifest)
            robot_radius_m = robot_radius_m_from_manifest(manifest)
            candidates = default_candidates_for_manifest(manifest)
            candidate_ids = [candidate.id for candidate in candidates]
            candidate_hash_value = candidates_hash(candidates)
        else:
            if source_grid != grid_shape:
                raise ValueError(f"mixed rollout grid shapes are not supported: {source_grid} != {grid_shape}")
            candidates = default_candidates_for_manifest(manifest)
            if [candidate.id for candidate in candidates] != candidate_ids:
                raise ValueError("mixed candidate sets are not supported")
        feature_store = DINOFeatureStore(feature_roots[source_index]) if feature_roots[source_index] is not None else None
        contexts.append(
            _SourceContext(
                root=source_root,
                feature_root=feature_roots[source_index],
                feature_store=feature_store,
                frames=frames,
                manifest=manifest,
                grouped=group_frames_by_sequence(frames),
                candidates=candidates,
            )
        )

    if grid_shape is None or candidate_ids is None or candidate_hash_value is None or meters_per_cell is None:
        raise ValueError("no rollout examples were available")
    frame_plan = _frame_plan(contexts, max_examples=max_examples, sampling_strategy=sampling_strategy)
    if not frame_plan:
        raise ValueError("no rollout examples were selected")
    included_by_context = [0 for _context in contexts]
    valid_horizons_by_context = [0 for _context in contexts]
    label_qa_acc = _new_label_qa_acc(candidate_ids)

    for context_index, frame in frame_plan:
        context = contexts[context_index]
        sequence = context.grouped[(frame.source_name, frame.sequence_id, frame.camera_id)]
        targets = build_future_rollout_targets_for_frame(
            frame=frame,
            sequence_frames=sequence,
            horizons_s=tuple(float(value) for value in horizons_s),
            candidates=context.candidates,
            meters_per_cell=meters_per_cell,
            feature_store=context.feature_store,
        )
        example_path = examples_dir / f"future_rollout_{examples_written:06d}.npz"
        write_deterministic_npz(example_path, targets.arrays)
        horizon_valid = [float(value) for value in targets.metadata["horizon_valid"]]
        candidate_valid = np.asarray(targets.arrays["candidate_valid_mask"], dtype=np.float32)
        horizon_valid_values.extend(horizon_valid)
        candidate_valid_values.extend(candidate_valid.tolist())
        valid_horizons_by_context[context_index] += sum(1 for value in horizon_valid if value > 0.0)
        for reason in targets.metadata["invalid_reasons"]:
            invalid_reasons[str(reason)] = invalid_reasons.get(str(reason), 0) + 1
        _update_label_qa(label_qa_acc, metadata=targets.metadata, arrays=targets.arrays)
        all_examples.append(
            {
                **targets.metadata,
                "example_path": relative_to_root(example_path, output),
                "example_sha256": file_sha256(example_path),
                "candidate_count": len(context.candidates),
                "valid_horizon_count": int(sum(1 for value in horizon_valid if value > 0.0)),
                "feature_mask": float(np.asarray(targets.arrays["feature_mask"]).item()),
            }
        )
        examples_written += 1
        included_by_context[context_index] += 1

    if not all_examples:
        raise ValueError("no rollout examples were written")
    label_qa = _finalize_label_qa(label_qa_acc)
    pre_rejection_label_qa: JsonDict | None = None
    rejected_degenerate_label_groups: list[JsonDict] = []
    rejected_degenerate_example_count = 0
    if reject_degenerate_groups and label_qa["warnings"]:
        pre_rejection_label_qa = label_qa
        rejected_group_keys = _warning_group_keys(label_qa["warnings"])
        rejected_degenerate_label_groups = _rejection_group_records(label_qa, rejected_group_keys)
        kept_examples: list[JsonDict] = []
        rejected_examples: list[JsonDict] = []
        for example in all_examples:
            group_key = f"{example.get('source_name')}::{example.get('split')}"
            if group_key in rejected_group_keys:
                rejected_examples.append(example)
            else:
                kept_examples.append(example)
        if not kept_examples:
            raise ValueError(
                "rollout label QA rejected every example as degenerate; "
                f"groups={sorted(rejected_group_keys)}"
            )
        _remove_rejected_example_files(output, rejected_examples)
        all_examples = kept_examples
        rejected_degenerate_example_count = len(rejected_examples)
        label_qa, horizon_valid_values, candidate_valid_values, invalid_reasons = _pack_stats_for_examples(
            output=output,
            examples=all_examples,
            candidate_ids=candidate_ids,
        )
    if fail_on_degenerate_labels and label_qa["warnings"]:
        raise ValueError(f"rollout label QA found degenerate labels: {label_qa['warnings'][:5]}")
    source_records = _source_records(contexts=contexts, examples=all_examples)
    valid_fraction = float(sum(1 for value in horizon_valid_values if value > 0.0) / max(len(horizon_valid_values), 1))
    candidate_valid_fraction = float(
        sum(1 for value in candidate_valid_values if value > 0.0) / max(len(candidate_valid_values), 1)
    )
    manifest: JsonDict = {
        "schema_version": FUTURE_ROLLOUT_PACK_SCHEMA_VERSION,
        "example_schema_version": FUTURE_ROLLOUT_EXAMPLE_SCHEMA_VERSION,
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "package_type": "FutureBEVRolloutPack",
        "version": 1,
        "sources": source_records,
        "source_dirs": [source.as_posix() for source in source_roots],
        "example_count": len(all_examples),
        "max_examples": int(max_examples) if max_examples is not None else None,
        "sampling_strategy": sampling_strategy,
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "meters_per_cell": round(float(meters_per_cell or 0.05), 6),
        "robot_radius_m": round(float(robot_radius_m or 0.18), 6),
        "horizons_s": [round(float(value), 6) for value in horizons_s],
        "future_bev_channels": list(FUTURE_BEV_CHANNELS),
        "future_derived_channels": list(FUTURE_DERIVED_CHANNELS),
        "candidate_count": len(candidate_ids),
        "candidate_ids": candidate_ids,
        "candidate_hash": candidate_hash_value,
        "target_generation": {
            "future_labels_warped_to_current_frame": True,
            "warp_source": "route_pose_delta_future_to_current",
            "warp_mode": "nearest_se2_grid_sample",
            "candidate_label_source": (
                "fixed_candidate_footprints_over_current_bev_and_warped_future_bev_with_"
                "separate_unsafe_now_and_future_collision"
            ),
            "uses_model_predictions_as_labels": False,
            "weak_label": True,
            "control_safe": False,
        },
        "valid_horizon_fraction": valid_fraction,
        "candidate_valid_fraction": candidate_valid_fraction,
        "invalid_reason_distribution": dict(sorted(invalid_reasons.items())),
        "label_qa": label_qa,
        "label_qa_path": "label_qa.json",
        "label_qa_status": "warn" if label_qa["warnings"] else "pass",
        "label_qa_warning_count": len(label_qa["warnings"]),
        "reject_degenerate_groups": bool(reject_degenerate_groups),
        "rejected_degenerate_example_count": int(rejected_degenerate_example_count),
        "rejected_degenerate_label_groups": rejected_degenerate_label_groups,
        "pre_rejection_label_qa_path": "label_qa_pre_rejection.json" if pre_rejection_label_qa is not None else None,
        "route_held_out_split_basis": "split_unit_id",
        "examples": all_examples,
        **ROLLOUT_PROVENANCE_FLAGS,
    }
    if pre_rejection_label_qa is not None:
        write_json(output / "label_qa_pre_rejection.json", pre_rejection_label_qa, pretty=True)
    write_json(output / "label_qa.json", label_qa, pretty=True)
    write_json(output / "manifest.json", manifest, pretty=True)
    return output / "manifest.json"


def _frame_plan(
    contexts: list[_SourceContext],
    *,
    max_examples: int | None,
    sampling_strategy: str,
) -> list[tuple[int, SpatialRolloutFrame]]:
    if sampling_strategy == "source_order" or max_examples is None:
        plan = [
            (context_index, frame)
            for context_index, context in enumerate(contexts)
            for frame in context.frames
        ]
        return plan[:max_examples] if max_examples is not None else plan
    return _route_balanced_frame_plan(contexts, max_examples=max_examples)


def _route_balanced_frame_plan(
    contexts: list[_SourceContext],
    *,
    max_examples: int,
) -> list[tuple[int, SpatialRolloutFrame]]:
    split_counts: dict[str, int] = {}
    for context in contexts:
        for frame in context.frames:
            split_counts[str(frame.split)] = split_counts.get(str(frame.split), 0) + 1
    split_order = sorted(split_counts, key=_split_sort_key)
    split_quotas = _apportion_quota(
        total=max_examples,
        keys=split_order,
        weights={split: float(split_counts[split]) for split in split_order},
        capacities=split_counts,
    )
    selected: list[tuple[int, SpatialRolloutFrame]] = []
    for split in split_order:
        quota = split_quotas.get(split, 0)
        if quota <= 0:
            continue
        groups: list[tuple[int, list[SpatialRolloutFrame]]] = []
        for context_index, context in enumerate(contexts):
            frames = [frame for frame in context.frames if str(frame.split) == split]
            if frames:
                groups.append((context_index, frames))
        capacities = {str(context_index): len(frames) for context_index, frames in groups}
        route_quotas = _apportion_quota(
            total=quota,
            keys=[str(context_index) for context_index, _frames in groups],
            weights={str(context_index): 1.0 for context_index, _frames in groups},
            capacities=capacities,
        )
        for context_index, frames in groups:
            count = route_quotas.get(str(context_index), 0)
            selected.extend((context_index, frame) for frame in _evenly_spaced(frames, count))
    return sorted(selected, key=lambda item: (item[0], _split_sort_key(str(item[1].split)), item[1].timestamp_ns, item[1].frame_id))


def _apportion_quota(
    *,
    total: int,
    keys: list[str],
    weights: dict[str, float],
    capacities: dict[str, int],
) -> dict[str, int]:
    if total <= 0 or not keys:
        return {key: 0 for key in keys}
    total = min(total, sum(max(0, int(capacities.get(key, 0))) for key in keys))
    weight_sum = sum(max(0.0, float(weights.get(key, 0.0))) for key in keys)
    if weight_sum <= 0.0:
        weights = {key: 1.0 for key in keys}
        weight_sum = float(len(keys))
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    used = 0
    for key in keys:
        capacity = max(0, int(capacities.get(key, 0)))
        exact = total * max(0.0, float(weights.get(key, 0.0))) / weight_sum
        count = min(capacity, int(np.floor(exact)))
        quotas[key] = count
        used += count
        remainders.append((exact - count, key))
    while used < total:
        progressed = False
        for _remainder, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if quotas[key] >= int(capacities.get(key, 0)):
                continue
            quotas[key] += 1
            used += 1
            progressed = True
            if used >= total:
                break
        if not progressed:
            break
    return quotas


def _evenly_spaced(frames: list[SpatialRolloutFrame], count: int) -> list[SpatialRolloutFrame]:
    if count <= 0:
        return []
    if count >= len(frames):
        return list(frames)
    if count == 1:
        return [frames[len(frames) // 2]]
    used: set[int] = set()
    for index in range(count):
        raw = index * (len(frames) - 1) / float(count - 1)
        used.add(int(round(raw)))
    if len(used) < count:
        for index in range(len(frames)):
            used.add(index)
            if len(used) >= count:
                break
    return [frames[index] for index in sorted(used)[:count]]


def _split_sort_key(split: str) -> tuple[int, str]:
    order = {"train": 0, "val": 1, "review": 2}
    return (order.get(split, 99), split)


def _new_label_qa_acc(candidate_ids: list[str]) -> JsonDict:
    return {
        "schema_version": "homebrain.future_bev_rollout_label_qa.v1",
        "candidate_ids": list(candidate_ids),
        "candidate_positive_threshold": 0.05,
        "groups": {},
    }


def _update_label_qa(label_qa_acc: JsonDict, *, metadata: JsonDict, arrays: dict[str, np.ndarray]) -> None:
    candidate_ids = [str(value) for value in label_qa_acc["candidate_ids"]]
    groups = label_qa_acc["groups"]
    key = f"{metadata.get('source_name')}::{metadata.get('split')}"
    group = groups.setdefault(
        key,
        {
            "route": str(metadata.get("source_name")),
            "split": str(metadata.get("split")),
            "examples": 0,
            "horizon_valid": [],
            "free": [],
            "occupied": [],
            "unknown": [],
            "newly_observed": [],
            "persistent_obstacle": [],
            "candidate_collision_positive": [0 for _candidate in candidate_ids],
            "candidate_collision_total": [0 for _candidate in candidate_ids],
            "candidate_unknown_exposure": [],
            "candidate_new_area_gain": [],
            "candidate_progress": [],
            "oracle_distribution": {},
        },
    )
    group["examples"] += 1
    horizon_valid = np.asarray(arrays["horizon_valid"], dtype=np.float32)
    group["horizon_valid"].extend(float(value) for value in horizon_valid.tolist())
    valid = np.asarray(arrays["future_valid_mask"], dtype=np.float32) > 0.5
    if np.any(valid):
        for field, name in (
            ("future_free", "free"),
            ("future_occupied", "occupied"),
            ("future_unknown", "unknown"),
            ("newly_observed", "newly_observed"),
            ("persistent_obstacle", "persistent_obstacle"),
        ):
            values = np.asarray(arrays[field], dtype=np.float32)
            group[name].append(float(values[valid].mean()))
    candidate_valid = np.asarray(arrays["candidate_valid_mask"], dtype=np.float32) > 0.0
    collision = np.asarray(arrays["candidate_collision"], dtype=np.float32)
    threshold = float(label_qa_acc["candidate_positive_threshold"])
    for index, valid_value in enumerate(candidate_valid.tolist()):
        if index >= len(candidate_ids) or not bool(valid_value):
            continue
        group["candidate_collision_total"][index] += 1
        group["candidate_collision_positive"][index] += int(float(collision[index]) >= threshold)
    for field, name in (
        ("candidate_unknown_exposure", "candidate_unknown_exposure"),
        ("candidate_new_area_gain", "candidate_new_area_gain"),
        ("candidate_progress", "candidate_progress"),
    ):
        values = np.asarray(arrays[field], dtype=np.float32)
        group[name].extend(float(values[index]) for index in np.where(candidate_valid)[0].tolist())
    oracle = np.asarray(arrays["candidate_oracle_cost"], dtype=np.float32)
    valid_indices = np.where(candidate_valid)[0]
    if valid_indices.size > 0:
        best_index = int(valid_indices[np.argmin(oracle[valid_indices])])
        candidate_id = candidate_ids[best_index] if best_index < len(candidate_ids) else str(best_index)
        group["oracle_distribution"][candidate_id] = int(group["oracle_distribution"].get(candidate_id, 0)) + 1


def _finalize_label_qa(label_qa_acc: JsonDict) -> JsonDict:
    candidate_ids = [str(value) for value in label_qa_acc["candidate_ids"]]
    groups_out: JsonDict = {}
    warnings: list[str] = []
    for key, group in sorted(label_qa_acc["groups"].items()):
        collision_rates: JsonDict = {}
        for index, candidate_id in enumerate(candidate_ids):
            total = int(group["candidate_collision_total"][index])
            positives = int(group["candidate_collision_positive"][index])
            collision_rates[candidate_id] = float(positives / max(total, 1))
        oracle_distribution = dict(sorted(group["oracle_distribution"].items()))
        oracle_total = sum(int(value) for value in oracle_distribution.values())
        oracle_dominant = max((int(value) for value in oracle_distribution.values()), default=0) / max(oracle_total, 1)
        record: JsonDict = {
            "route": group["route"],
            "split": group["split"],
            "examples": int(group["examples"]),
            "future_horizon_valid_fraction": _mean(group["horizon_valid"]),
            "free_target_ratio": _mean(group["free"]),
            "occupied_target_ratio": _mean(group["occupied"]),
            "unknown_target_ratio": _mean(group["unknown"]),
            "newly_observed_ratio": _mean(group["newly_observed"]),
            "persistent_obstacle_ratio": _mean(group["persistent_obstacle"]),
            "candidate_collision_positive_rate_per_candidate": collision_rates,
            "candidate_unknown_exposure_distribution": _distribution(group["candidate_unknown_exposure"]),
            "candidate_new_area_gain_distribution": _distribution(group["candidate_new_area_gain"]),
            "candidate_progress_distribution": _distribution(group["candidate_progress"]),
            "selected_oracle_lower_is_better_candidate_distribution": oracle_distribution,
            "selected_oracle_dominant_fraction": float(oracle_dominant),
        }
        groups_out[key] = record
        warnings.extend(_label_qa_warnings(key, record))
    return {
        "schema_version": label_qa_acc["schema_version"],
        "candidate_ids": candidate_ids,
        "candidate_positive_threshold": float(label_qa_acc["candidate_positive_threshold"]),
        "by_route_split": groups_out,
        "warnings": warnings,
        "status": "warn" if warnings else "pass",
    }


def _label_qa_warnings(key: str, record: JsonDict) -> list[str]:
    warnings: list[str] = []
    if float(record["future_horizon_valid_fraction"]) < 0.25:
        warnings.append(f"{key}:future_horizon_valid_fraction_below_0.25")
    if float(record["free_target_ratio"]) <= 1.0e-6 and float(record["occupied_target_ratio"]) <= 1.0e-6:
        warnings.append(f"{key}:future_observed_targets_degenerate")
    if float(record["newly_observed_ratio"]) <= 1.0e-6:
        warnings.append(f"{key}:newly_observed_ratio_zero")
    collision_rates = record["candidate_collision_positive_rate_per_candidate"]
    if isinstance(collision_rates, dict):
        rates = [float(value) for value in collision_rates.values()]
        if rates and max(rates) <= 0.0:
            warnings.append(f"{key}:candidate_collision_all_negative")
        if rates and min(rates) >= 1.0:
            warnings.append(f"{key}:candidate_collision_all_positive")
    if float(record["candidate_new_area_gain_distribution"]["max"]) <= 1.0e-6:
        warnings.append(f"{key}:candidate_new_area_gain_zero")
    if float(record["candidate_progress_distribution"]["max"]) <= 1.0e-6:
        warnings.append(f"{key}:candidate_progress_zero")
    if int(record["examples"]) > 1 and float(record["selected_oracle_dominant_fraction"]) > 0.95:
        warnings.append(f"{key}:oracle_candidate_distribution_dominant_above_0.95")
    return warnings


def _distribution(values: list[float]) -> JsonDict:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _warning_group_keys(warnings: list[str]) -> set[str]:
    groups: set[str] = set()
    for warning in warnings:
        if "::" not in warning or ":" not in warning:
            continue
        groups.add(warning.rsplit(":", 1)[0])
    return groups


def _rejection_group_records(label_qa: JsonDict, group_keys: set[str]) -> list[JsonDict]:
    by_group = label_qa.get("by_route_split", {})
    warnings = [str(value) for value in label_qa.get("warnings", [])]
    records: list[JsonDict] = []
    for key in sorted(group_keys):
        group_record = by_group.get(key, {}) if isinstance(by_group, dict) else {}
        records.append(
            {
                "group": key,
                "warnings": [warning for warning in warnings if warning.startswith(f"{key}:")],
                "examples": int(group_record.get("examples", 0)) if isinstance(group_record, dict) else 0,
                "rejection_reason": "degenerate_future_rollout_labels",
            }
        )
    return records


def _remove_rejected_example_files(output: Path, rejected_examples: list[JsonDict]) -> None:
    for example in rejected_examples:
        relative = example.get("example_path")
        if not isinstance(relative, str):
            continue
        path = output / relative
        if path.exists():
            path.unlink()


def _pack_stats_for_examples(
    *,
    output: Path,
    examples: list[JsonDict],
    candidate_ids: list[str],
) -> tuple[JsonDict, list[float], list[float], dict[str, int]]:
    label_qa_acc = _new_label_qa_acc(candidate_ids)
    horizon_valid_values: list[float] = []
    candidate_valid_values: list[float] = []
    invalid_reasons: dict[str, int] = {}
    for example in examples:
        relative = example.get("example_path")
        if not isinstance(relative, str):
            raise ValueError(f"rollout example is missing example_path: {example}")
        with np.load(output / relative, allow_pickle=False) as loaded:
            arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
        horizon_valid = [float(value) for value in arrays["horizon_valid"].tolist()]
        candidate_valid = np.asarray(arrays["candidate_valid_mask"], dtype=np.float32)
        horizon_valid_values.extend(horizon_valid)
        candidate_valid_values.extend(candidate_valid.tolist())
        for reason in example.get("invalid_reasons", []):
            invalid_reasons[str(reason)] = invalid_reasons.get(str(reason), 0) + 1
        _update_label_qa(label_qa_acc, metadata=example, arrays=arrays)
    return _finalize_label_qa(label_qa_acc), horizon_valid_values, candidate_valid_values, invalid_reasons


def _source_records(*, contexts: list[_SourceContext], examples: list[JsonDict]) -> list[JsonDict]:
    included_counts: dict[str, int] = {}
    valid_horizon_counts: dict[str, int] = {}
    for example in examples:
        source_name = str(example.get("source_name", ""))
        included_counts[source_name] = included_counts.get(source_name, 0) + 1
        valid_horizon_counts[source_name] = valid_horizon_counts.get(source_name, 0) + int(
            example.get("valid_horizon_count", 0)
        )
    return [
        {
            "source": context.root.as_posix(),
            "source_manifest_sha256": file_sha256(context.root / "manifest.json"),
            "features": context.feature_root.as_posix() if context.feature_root is not None else None,
            "feature_manifest_sha256": file_sha256(context.feature_root / "teacher_manifest.json")
            if context.feature_root is not None
            else None,
            "source_example_count": len(context.frames),
            "included_example_count": int(included_counts.get(str(context.manifest.get("source_name", "")), 0)),
            "valid_horizon_count": int(valid_horizon_counts.get(str(context.manifest.get("source_name", "")), 0)),
            "source_name": str(context.manifest.get("source_name", context.root.name)),
            "robot_supervision_grade": str(context.manifest.get("robot_supervision_grade", "unknown")),
            "dataset_frame_type": str(context.manifest.get("dataset_frame_type", "unknown")),
            "robot_frame_truth": bool(context.manifest.get("robot_frame_truth", False)),
            "control_safe": False,
        }
        for context in contexts
    ]


def _feature_roots(feature_dirs: list[str | Path] | None, source_count: int) -> list[Path | None]:
    if not feature_dirs:
        return [None for _ in range(source_count)]
    roots = [Path(value) for value in feature_dirs]
    if len(roots) == 1 and source_count > 1:
        return roots * source_count
    if len(roots) != source_count:
        raise ValueError("--features must be supplied once or once per --source")
    return roots


def _grid_shape(manifest: JsonDict) -> tuple[int, int]:
    value = manifest.get("grid_shape")
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("source manifest must include grid_shape")
    return (int(value[0]), int(value[1]))


def _clear_generated_outputs(output: Path) -> None:
    if (output / "examples").exists():
        shutil.rmtree(output / "examples")
    for filename in ("manifest.json", "label_qa.json", "label_qa_pre_rejection.json"):
        path = output / filename
        if path.exists():
            path.unlink()


def _parse_horizons(value: str) -> tuple[float, ...]:
    horizons = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not horizons:
        raise argparse.ArgumentTypeError("at least one horizon is required")
    if any(item <= 0.0 for item in horizons):
        raise argparse.ArgumentTypeError("horizons must be positive")
    return horizons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic replay-only FutureBEVRolloutPack v1.")
    parser.add_argument("--source", action="append", required=True, help="Input SpatialTrainPack directory.")
    parser.add_argument("--features", action="append", default=None, help="Optional DINO feature artifact directory.")
    parser.add_argument("--out", required=True, help="Output FutureBEVRolloutPack directory.")
    parser.add_argument("--horizons-s", type=_parse_horizons, default=DEFAULT_FUTURE_HORIZONS_S)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--sampling-strategy", choices=SAMPLING_STRATEGIES, default="source_order")
    parser.add_argument("--route-balanced", action="store_true", help="Shortcut for --sampling-strategy route_balanced.")
    parser.add_argument("--fail-on-degenerate-labels", action="store_true")
    parser.add_argument(
        "--reject-degenerate-groups",
        action="store_true",
        help="Drop route/split groups with degenerate rollout label QA, then fail if residual labels are still degenerate.",
    )
    args = parser.parse_args(argv)
    sampling_strategy = "route_balanced" if args.route_balanced else args.sampling_strategy
    manifest = build_future_bev_rollout_pack(
        sources=[Path(source) for source in args.source],
        out_dir=args.out,
        feature_dirs=[Path(item) for item in args.features] if args.features else None,
        horizons_s=args.horizons_s,
        max_examples=args.max_examples,
        sampling_strategy=sampling_strategy,
        fail_on_degenerate_labels=args.fail_on_degenerate_labels,
        reject_degenerate_groups=args.reject_degenerate_groups,
    )
    print(json.dumps({"manifest": manifest.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
