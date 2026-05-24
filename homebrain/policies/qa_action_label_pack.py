from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.data.spatial_dataset import load_example_npz, mean, np_scalar_to_bool, np_scalar_to_str, read_json, write_json
from homebrain.policies.build_action_label_pack import (
    ACTION_LABEL_PACK_SCHEMA_VERSION,
    ACTION_LABEL_PACK_SCHEMA_VERSION_V1,
    ACTION_LABEL_PACK_SCHEMA_VERSION_V2,
    ACTION_LABEL_PACK_SCHEMA_VERSION_V3,
)

ACTION_LABEL_QA_SCHEMA_VERSION = "homebrain.action_label_pack_qa.v0"
DOMINANT_ACTION_FRACTION = 0.90


def qa_action_label_pack(pack_dir: str | Path) -> dict[str, Any]:
    root = Path(pack_dir)
    errors: list[str] = []
    try:
        manifest = read_json(root / "manifest.json")
    except Exception as exc:  # noqa: BLE001 - QA reports malformed packs.
        return _failed_metrics(root, f"missing_or_malformed_manifest: {exc}")

    schema_version = manifest.get("schema_version")
    if schema_version not in {
        ACTION_LABEL_PACK_SCHEMA_VERSION,
        ACTION_LABEL_PACK_SCHEMA_VERSION_V1,
        ACTION_LABEL_PACK_SCHEMA_VERSION_V2,
        ACTION_LABEL_PACK_SCHEMA_VERSION_V3,
    }:
        errors.append(f"unsupported schema_version={manifest.get('schema_version')!r}")
    is_v1 = schema_version in {
        ACTION_LABEL_PACK_SCHEMA_VERSION_V1,
        ACTION_LABEL_PACK_SCHEMA_VERSION_V2,
        ACTION_LABEL_PACK_SCHEMA_VERSION_V3,
    }
    example_records = manifest.get("examples")
    if not isinstance(example_records, list):
        return _failed_metrics(root, "manifest examples must be a list")

    candidate_counts: list[float] = []
    selected_ids: list[str] = []
    source_names: list[str] = []
    selected_by_source: dict[str, list[str]] = {}
    action_ok_by_source: dict[str, list[bool]] = {}
    weight_by_source: dict[str, list[float]] = {}
    coverage_gains: list[float] = []
    collision_positive_count = 0
    collision_total_count = 0
    replay_only_false_count = 0
    not_executed_false_count = 0
    control_safe_true_count = 0
    selected_count_error_count = 0
    missing_count = 0
    shape_error_count = 0
    nan_count = 0

    for record in example_records:
        if not isinstance(record, dict) or not isinstance(record.get("example_path"), str):
            missing_count += 1
            errors.append("example record missing example_path")
            continue
        example_path = root / str(record["example_path"])
        if not example_path.exists():
            missing_count += 1
            errors.append(f"missing example: {record['example_path']}")
            continue
        try:
            example = load_example_npz(example_path)
        except Exception as exc:  # noqa: BLE001
            shape_error_count += 1
            errors.append(f"failed to load example {record['example_path']}: {exc}")
            continue

        required = (
            "candidate_ids",
            "collision_proxy",
            "coverage_gain",
            "selected_by_expert",
            "selected_candidate_id",
            "source_name",
            "action_supervision_ok" if is_v1 else "source_name",
            "source_weight" if is_v1 else "source_name",
            "replay_only",
            "not_executed",
            "control_safe",
        )
        missing_fields = [field for field in required if field not in example]
        if missing_fields:
            missing_count += len(missing_fields)
            errors.append(f"example {record['example_path']} missing fields: {','.join(missing_fields)}")
            continue

        if np_scalar_to_bool(example["replay_only"]) is not True:
            replay_only_false_count += 1
        if np_scalar_to_bool(example["not_executed"]) is not True:
            not_executed_false_count += 1
        if np_scalar_to_bool(example["control_safe"]) is not False:
            control_safe_true_count += 1

        candidate_ids = np.asarray(example["candidate_ids"])
        selected = np.asarray(example["selected_by_expert"])
        collision = np.asarray(example["collision_proxy"], dtype=np.float32)
        coverage = np.asarray(example["coverage_gain"], dtype=np.float32)
        arrays = [candidate_ids, selected, collision, coverage]
        lengths = {int(array.shape[0]) for array in arrays if array.ndim >= 1}
        if len(lengths) != 1:
            shape_error_count += 1
            errors.append(f"candidate field length mismatch: {record['example_path']}")
            continue
        if any(np.issubdtype(array.dtype, np.number) and np.count_nonzero(~np.isfinite(array)) for array in arrays):
            nan_count += 1
            continue

        selected_count = int(np.count_nonzero(selected))
        if selected_count != 1:
            selected_count_error_count += 1
            errors.append(f"example {record['example_path']} has {selected_count} selected candidates")
            continue

        candidate_counts.append(float(next(iter(lengths))))
        selected_ids.append(np_scalar_to_str(example["selected_candidate_id"]))
        source_name = np_scalar_to_str(example["source_name"])
        source_names.append(source_name)
        selected_by_source.setdefault(source_name, []).append(selected_ids[-1])
        if "action_supervision_ok" in example:
            action_ok_by_source.setdefault(source_name, []).append(np_scalar_to_bool(example["action_supervision_ok"]))
        if "source_weight" in example:
            weight_by_source.setdefault(source_name, []).append(float(np.asarray(example["source_weight"], dtype=np.float32).item()))
        coverage_gains.extend(float(value) for value in coverage.tolist())
        collision_positive_count += int(np.count_nonzero(collision >= np.float32(0.35)))
        collision_total_count += int(collision.size)

    example_count = len(selected_ids)
    selected_stop_count = sum(1 for value in selected_ids if value == "stop")
    selected_distribution = _distribution(selected_ids)
    source_distribution = _distribution(source_names)
    source_selected_distribution = {
        source_name: _distribution(values)
        for source_name, values in sorted(selected_by_source.items())
    }
    source_action_entropy = {
        source_name: _entropy(_distribution(values))
        for source_name, values in sorted(selected_by_source.items())
    }
    source_selected_stop_fraction = {
        source_name: float(sum(1 for value in values if value == "stop") / max(len(values), 1))
        for source_name, values in sorted(selected_by_source.items())
    }
    source_selected_motion_fraction = {
        source_name: float(sum(1 for value in values if value != "stop") / max(len(values), 1))
        for source_name, values in sorted(selected_by_source.items())
    }
    source_action_supervision_ok_fraction = {
        source_name: float(sum(1 for value in values if value) / max(len(values), 1))
        for source_name, values in sorted(action_ok_by_source.items())
    }
    source_weight_mean = {
        source_name: mean([float(value) for value in values])
        for source_name, values in sorted(weight_by_source.items())
    }
    excluded_by_source = {}
    action_filter = manifest.get("action_sanity_filter")
    if isinstance(action_filter, dict) and isinstance(action_filter.get("excluded_by_source"), dict):
        excluded_by_source = {str(key): int(value) for key, value in action_filter["excluded_by_source"].items()}
    per_source = {
        source_name: {
            "example_count": len(selected_by_source.get(source_name, [])),
            "excluded_frame_count": int(excluded_by_source.get(source_name, 0)),
            "selected_stop_fraction": source_selected_stop_fraction.get(source_name, 0.0),
            "selected_motion_fraction": source_selected_motion_fraction.get(source_name, 0.0),
            "action_supervision_ok_fraction": source_action_supervision_ok_fraction.get(source_name, 0.0),
            "source_weight_mean": source_weight_mean.get(source_name, 0.0),
            "action_entropy": source_action_entropy.get(source_name, 0.0),
        }
        for source_name in sorted(set(source_distribution) | set(excluded_by_source))
    }
    dominant_fraction = max(selected_distribution.values(), default=0) / max(example_count, 1)
    flags: list[str] = []
    if example_count and selected_stop_count == example_count:
        flags.append("all_labels_are_stop")
    if dominant_fraction >= DOMINANT_ACTION_FRACTION:
        flags.append("one_action_dominates")
    if replay_only_false_count or not_executed_false_count or control_safe_true_count:
        flags.append("safety_flags_invalid")
    if selected_count_error_count:
        flags.append("selected_count_errors")

    metrics: dict[str, Any] = {
        "schema_version": ACTION_LABEL_QA_SCHEMA_VERSION,
        "pack": root.as_posix(),
        "source_manifest_schema_version": manifest.get("schema_version"),
        "example_count": example_count,
        "manifest_example_count": len(example_records),
        "candidate_count_mean": mean(candidate_counts),
        "selected_stop_fraction": float(selected_stop_count / max(example_count, 1)),
        "selected_motion_fraction": float((example_count - selected_stop_count) / max(example_count, 1)),
        "collision_positive_rate": float(collision_positive_count / max(collision_total_count, 1)),
        "coverage_gain_mean": mean(coverage_gains),
        "action_entropy": _entropy(selected_distribution),
        "source_distribution": source_distribution,
        "selected_distribution": selected_distribution,
        "source_selected_distribution": source_selected_distribution,
        "source_selected_stop_fraction": source_selected_stop_fraction,
        "source_selected_motion_fraction": source_selected_motion_fraction,
        "source_action_supervision_ok_fraction": source_action_supervision_ok_fraction,
        "source_action_entropy": source_action_entropy,
        "source_weight_mean": source_weight_mean,
        "per_source": per_source,
        "excluded_by_source": excluded_by_source,
        "excluded_frame_count": int(manifest.get("action_sanity_filter", {}).get("excluded_frame_count", 0))
        if isinstance(manifest.get("action_sanity_filter"), dict)
        else 0,
        "dominant_action_fraction": float(dominant_fraction),
        "deterministic_hash": _pack_hash(root, example_records),
        "missing_count": missing_count,
        "shape_error_count": shape_error_count,
        "nan_count": nan_count,
        "replay_only_false_count": replay_only_false_count,
        "not_executed_false_count": not_executed_false_count,
        "control_safe_true_count": control_safe_true_count,
        "selected_count_error_count": selected_count_error_count,
        "flags": flags,
        "action_label_pack_qa_pass": not flags and missing_count == 0 and shape_error_count == 0 and nan_count == 0,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "errors": errors[:50],
    }
    return metrics


def _failed_metrics(root: Path, error: str) -> dict[str, Any]:
    return {
        "schema_version": ACTION_LABEL_QA_SCHEMA_VERSION,
        "pack": root.as_posix(),
        "example_count": 0,
        "candidate_count_mean": 0.0,
        "selected_stop_fraction": 0.0,
        "selected_motion_fraction": 0.0,
        "collision_positive_rate": 0.0,
        "coverage_gain_mean": 0.0,
        "action_entropy": 0.0,
        "source_distribution": {},
        "deterministic_hash": "",
        "flags": ["missing_or_malformed_manifest"],
        "action_label_pack_qa_pass": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "errors": [error],
    }


def _distribution(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        if probability > 0.0:
            entropy -= probability * math.log2(probability)
    return float(entropy)


def _pack_hash(root: Path, example_records: list[Any]) -> str:
    digest = hashlib.sha256()
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        digest.update(b"manifest.json\0")
        digest.update(manifest_path.read_bytes())
    for record in example_records:
        if not isinstance(record, dict) or not isinstance(record.get("example_path"), str):
            continue
        relpath = str(record["example_path"])
        path = root / relpath
        if not path.exists():
            continue
        digest.update(relpath.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QA deterministic replay-only ActionLabelPack v0.")
    parser.add_argument("--pack", required=True, help="Input ActionLabelPack directory.")
    parser.add_argument("--out", required=True, help="Output QA JSON path.")
    args = parser.parse_args(argv)
    metrics = qa_action_label_pack(args.pack)
    write_json(args.out, metrics, pretty=True)
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
