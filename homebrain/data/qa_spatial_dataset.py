from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.data.spatial_dataset import (
    BEV_LABEL_FIELDS,
    REQUIRED_EXAMPLE_FIELDS,
    SPATIAL_DATASET_SCHEMA_VERSION,
    SPATIAL_MANIFEST_FILE,
    SPATIAL_QA_SCHEMA_VERSION,
    as_float,
    load_example_npz,
    mean,
    np_scalar_to_bool,
    np_scalar_to_str,
    percentile,
    read_json,
    write_json,
)

LOW_CONFIDENCE_THRESHOLD = 0.20
MIN_LABEL_DENSITY = 0.10
MIN_OBSERVED_RATIO = 0.10
MAX_LOW_CONFIDENCE_FRAME_FRACTION = 0.20
MAX_TEMPORAL_VISIBLE_JITTER = 0.30
MIN_EXAMPLES_FOR_TRAINING_GATE = 10
ROBOT_SUPERVISION_GRADES = {
    "weak_visual_geometry",
    "public_rgbd_anchor",
    "robot_frame_metric",
    "unknown",
}


@dataclass(frozen=True)
class FrameQuality:
    frame_id: int
    split: str
    free_ratio: float
    obstacle_ratio: float
    unknown_ratio: float
    confidence_mean: float
    confidence_nonzero_ratio: float
    label_density: float
    observed_ratio: float
    source_observed_ratio: float
    empty_label: bool
    class_grid: np.ndarray
    observed_mask: np.ndarray
    labeled_mask: np.ndarray


def qa_spatial_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest_path = root / SPATIAL_MANIFEST_FILE
    errors: list[str] = []
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:  # noqa: BLE001 - QA should report malformed packages.
        return _failed_manifest_metrics(str(exc))

    if manifest.get("schema_version") != SPATIAL_DATASET_SCHEMA_VERSION:
        errors.append(
            f"unsupported dataset schema {manifest.get('schema_version')!r}; "
            f"expected {SPATIAL_DATASET_SCHEMA_VERSION!r}"
        )
    frame_records = manifest.get("examples") or manifest.get("frames")
    if not isinstance(frame_records, list):
        frame_records = []
        errors.append("dataset manifest examples must be a list")

    expected_shape = _expected_shape(manifest)
    example_count = 0
    missing_count = 0
    shape_error_count = 0
    nan_count = 0
    weak_label_false_count = 0
    control_safe_true_count = 0
    not_robot_frame_truth_false_count = 0
    pose_label_count = 0
    pose_label_missing_mask_count = 0
    label_overlap_cell_count = 0
    label_sum_error_cell_count = 0
    frame_qualities: list[FrameQuality] = []
    split_counts: dict[str, int] = {"train": 0, "val": 0, "review": 0}

    for record in frame_records:
        if not isinstance(record, dict):
            shape_error_count += 1
            errors.append("example manifest record is not an object")
            continue
        example_relpath = record.get("example_path")
        if not isinstance(example_relpath, str):
            missing_count += 1
            errors.append(f"frame {record.get('frame_id')} missing example_path")
            continue
        example_path = root / example_relpath
        if not example_path.exists():
            missing_count += 1
            errors.append(f"missing example file: {example_relpath}")
            continue

        example_count += 1
        try:
            example = load_example_npz(example_path)
        except Exception as exc:  # noqa: BLE001
            shape_error_count += 1
            errors.append(f"failed to load example {example_relpath}: {exc}")
            continue

        missing_fields = [field for field in REQUIRED_EXAMPLE_FIELDS if field not in example]
        if missing_fields:
            missing_count += len(missing_fields)
            errors.append(f"example {example_relpath} missing fields: {','.join(missing_fields)}")
            continue

        split = _safe_scalar_str(example, "split", "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1
        if _safe_scalar_bool(example, "weak_label", False) is not True:
            weak_label_false_count += 1
        if _safe_scalar_bool(example, "control_safe", True) is not False:
            control_safe_true_count += 1
        if "not_robot_frame_truth" in example and _safe_scalar_bool(example, "not_robot_frame_truth", False) is not True:
            not_robot_frame_truth_false_count += 1
        if "pose_delta_mask" in example:
            pose_mask = _safe_scalar_float(example, "pose_delta_mask", 0.0)
            if pose_mask > 0.0:
                pose_label_count += 1
            else:
                pose_label_missing_mask_count += 1

        arrays = {field: np.asarray(example[field]) for field in BEV_LABEL_FIELDS + ("bev_confidence",)}
        if "bev_floor_candidate" in example:
            arrays["bev_floor_candidate"] = np.asarray(example["bev_floor_candidate"])
        if "bev_height" in example:
            arrays["bev_height"] = np.asarray(example["bev_height"])

        shape_error_count += _count_shape_errors(arrays, expected_shape, example_relpath, errors)
        nan_count += _count_nan_values(arrays)
        if any(array.shape != expected_shape for array in arrays.values() if array.ndim >= 2):
            continue

        free = arrays["bev_free"] > 0
        obstacle = arrays["bev_obstacle"] > 0
        unknown = arrays["bev_unknown"] > 0
        confidence = arrays["bev_confidence"].astype(np.float32)
        floor = arrays.get("bev_floor_candidate", np.zeros(expected_shape, dtype=np.uint8)) > 0
        total = max(int(free.size), 1)
        label_overlap_cell_count += int(np.count_nonzero((free.astype(np.uint8) + obstacle.astype(np.uint8) + unknown.astype(np.uint8)) > 1))
        label_sum_error_cell_count += int(np.count_nonzero((free.astype(np.uint8) + obstacle.astype(np.uint8) + unknown.astype(np.uint8)) != 1))

        labeled = free | obstacle
        observed = labeled | floor | (confidence > np.float32(0.0))
        source_observed_ratio = _source_observed_ratio(record, total)
        frame_id = _safe_scalar_int(example, "frame_id", int(record.get("frame_id", -1)))
        frame_qualities.append(
            FrameQuality(
                frame_id=frame_id,
                split=split,
                free_ratio=float(np.count_nonzero(free) / total),
                obstacle_ratio=float(np.count_nonzero(obstacle) / total),
                unknown_ratio=float(np.count_nonzero(unknown) / total),
                confidence_mean=float(np.mean(confidence)),
                confidence_nonzero_ratio=float(np.count_nonzero(confidence > np.float32(0.0)) / total),
                label_density=float(np.count_nonzero(labeled) / total),
                observed_ratio=float(np.count_nonzero(observed) / total),
                source_observed_ratio=source_observed_ratio,
                empty_label=not bool(np.any(labeled)),
                class_grid=np.where(obstacle, 2, np.where(free, 1, 0)).astype(np.uint8),
                observed_mask=observed,
                labeled_mask=labeled,
            )
        )

    free_ratios = [item.free_ratio for item in frame_qualities]
    obstacle_ratios = [item.obstacle_ratio for item in frame_qualities]
    unknown_ratios = [item.unknown_ratio for item in frame_qualities]
    confidence_means = [item.confidence_mean for item in frame_qualities]
    confidence_nonzero_ratios = [item.confidence_nonzero_ratio for item in frame_qualities]
    label_densities = [item.label_density for item in frame_qualities]
    observed_ratios = [item.observed_ratio for item in frame_qualities]
    source_observed_ratios = [item.source_observed_ratio for item in frame_qualities if item.source_observed_ratio > 0.0]
    low_confidence_frame_count = sum(1 for item in frame_qualities if item.confidence_mean < LOW_CONFIDENCE_THRESHOLD)
    empty_label_frame_count = sum(1 for item in frame_qualities if item.empty_label)
    temporal = _temporal_metrics(frame_qualities)
    robot_supervision_grade = _robot_supervision_grade(manifest)

    metrics: dict[str, Any] = {
        "schema_version": SPATIAL_QA_SCHEMA_VERSION,
        "dataset": root.as_posix(),
        "source_manifest_schema_version": manifest.get("schema_version"),
        "example_count": example_count,
        "manifest_example_count": len(frame_records),
        "missing_count": missing_count,
        "shape_error_count": shape_error_count,
        "nan_count": nan_count,
        "weak_label_false_count": weak_label_false_count,
        "control_safe_true_count": control_safe_true_count,
        "not_robot_frame_truth_false_count": not_robot_frame_truth_false_count,
        "pose_label_count": pose_label_count,
        "pose_label_missing_mask_count": pose_label_missing_mask_count,
        "pose_label_frame": manifest.get("pose_label_frame", "none"),
        "pose_label_convention": manifest.get("pose_label_convention"),
        "control_safe": False if manifest.get("control_safe") is False and control_safe_true_count == 0 else True,
        "robot_supervision_grade": robot_supervision_grade,
        "split_counts": split_counts,
        "free_ratio_mean": mean(free_ratios),
        "free_ratio_min": min(free_ratios) if free_ratios else 0.0,
        "free_ratio_max": max(free_ratios) if free_ratios else 0.0,
        "obstacle_ratio_mean": mean(obstacle_ratios),
        "obstacle_ratio_min": min(obstacle_ratios) if obstacle_ratios else 0.0,
        "obstacle_ratio_max": max(obstacle_ratios) if obstacle_ratios else 0.0,
        "unknown_ratio_mean": mean(unknown_ratios),
        "unknown_ratio_min": min(unknown_ratios) if unknown_ratios else 0.0,
        "unknown_ratio_max": max(unknown_ratios) if unknown_ratios else 0.0,
        "confidence_mean": mean(confidence_means),
        "confidence_min": min(confidence_means) if confidence_means else 0.0,
        "confidence_median": percentile(confidence_means, 50),
        "confidence_max": max(confidence_means) if confidence_means else 0.0,
        "confidence_nonzero_ratio_mean": mean(confidence_nonzero_ratios),
        "label_density": mean(label_densities),
        "label_density_mean": mean(label_densities),
        "label_density_min": min(label_densities) if label_densities else 0.0,
        "label_density_median": percentile(label_densities, 50),
        "observed_ratio_mean": mean(observed_ratios),
        "visible_confidence_positive_ratio_mean": mean(confidence_nonzero_ratios),
        "source_observed_ratio_mean": mean(source_observed_ratios),
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "low_confidence_frame_count": low_confidence_frame_count,
        "empty_label_frame_count": empty_label_frame_count,
        "label_overlap_cell_count": label_overlap_cell_count,
        "label_sum_error_cell_count": label_sum_error_cell_count,
        "errors": errors[:50],
        **temporal,
    }
    structural_blockers = _structural_blockers(metrics)
    metrics["structurally_trainable"] = not structural_blockers
    metrics["structural_blockers"] = structural_blockers
    quarantine_reasons = _quarantine_reasons(metrics)
    metrics["trainable_candidate"] = not quarantine_reasons
    metrics["trainable_candidate_control_safe"] = False
    metrics["quality_status"] = "candidate_requires_human_review" if not quarantine_reasons else "quarantined_low_quality"
    metrics["quarantine_reasons"] = quarantine_reasons
    return metrics


def write_qa_metrics(metrics: dict[str, Any], out_path: str | Path) -> None:
    write_json(out_path, metrics, pretty=True)


def _failed_manifest_metrics(error: str) -> dict[str, Any]:
    return {
        "schema_version": SPATIAL_QA_SCHEMA_VERSION,
        "example_count": 0,
        "manifest_example_count": 0,
        "missing_count": 1,
        "shape_error_count": 0,
        "nan_count": 0,
        "control_safe": False,
        "robot_supervision_grade": "unknown",
        "structurally_trainable": False,
        "structural_blockers": ["missing_or_malformed_dataset_manifest"],
        "trainable_candidate": False,
        "trainable_candidate_control_safe": False,
        "quality_status": "quarantined_low_quality",
        "quarantine_reasons": ["missing_or_malformed_dataset_manifest"],
        "errors": [error],
    }


def _expected_shape(manifest: dict[str, Any]) -> tuple[int, int]:
    raw = manifest.get("grid_shape")
    if isinstance(raw, list) and len(raw) == 2:
        try:
            shape = (int(raw[0]), int(raw[1]))
        except (TypeError, ValueError):
            return (0, 0)
        if shape[0] > 0 and shape[1] > 0:
            return shape
    return (0, 0)


def _count_shape_errors(
    arrays: dict[str, np.ndarray],
    expected_shape: tuple[int, int],
    example_relpath: str,
    errors: list[str],
) -> int:
    count = 0
    for name, array in arrays.items():
        if array.ndim < 2:
            count += 1
            errors.append(f"{example_relpath}:{name} must be at least 2D")
            continue
        if expected_shape != (0, 0) and tuple(array.shape) != expected_shape:
            count += 1
            errors.append(f"{example_relpath}:{name} shape {array.shape} != {expected_shape}")
    return count


def _count_nan_values(arrays: dict[str, np.ndarray]) -> int:
    count = 0
    for array in arrays.values():
        if np.issubdtype(array.dtype, np.number):
            count += int(np.count_nonzero(~np.isfinite(array)))
    return count


def _safe_scalar_str(example: dict[str, np.ndarray], field: str, default: str) -> str:
    try:
        return np_scalar_to_str(example[field])
    except Exception:  # noqa: BLE001
        return default


def _safe_scalar_int(example: dict[str, np.ndarray], field: str, default: int) -> int:
    try:
        return int(np.asarray(example[field]).item())
    except Exception:  # noqa: BLE001
        return default


def _safe_scalar_float(example: dict[str, np.ndarray], field: str, default: float) -> float:
    try:
        return float(np.asarray(example[field]).item())
    except Exception:  # noqa: BLE001
        return default


def _safe_scalar_bool(example: dict[str, np.ndarray], field: str, default: bool) -> bool:
    try:
        return np_scalar_to_bool(example[field])
    except Exception:  # noqa: BLE001
        return default


def _source_observed_ratio(record: JsonDict, total: int) -> float:
    stats = record.get("source_bev_stats")
    if not isinstance(stats, dict):
        return 0.0
    observed_count = as_float(stats.get("observed_cell_count"), 0.0)
    if observed_count > 0.0 and total > 0:
        return float(observed_count / total)
    return 0.0


def _temporal_metrics(frames: list[FrameQuality]) -> dict[str, float | int]:
    full_jitter: list[float] = []
    visible_jitter: list[float] = []
    label_flicker: list[float] = []
    high_visible_jitter_count = 0
    for previous, current in zip(frames, frames[1:]):
        if previous.class_grid.shape != current.class_grid.shape or previous.class_grid.size == 0:
            continue
        changed = previous.class_grid != current.class_grid
        full_jitter.append(float(np.count_nonzero(changed) / previous.class_grid.size))
        visible = previous.observed_mask | current.observed_mask
        visible_count = int(np.count_nonzero(visible))
        if visible_count:
            value = float(np.count_nonzero(changed & visible) / visible_count)
            visible_jitter.append(value)
            if value > MAX_TEMPORAL_VISIBLE_JITTER:
                high_visible_jitter_count += 1
        labeled = previous.labeled_mask | current.labeled_mask
        labeled_count = int(np.count_nonzero(labeled))
        if labeled_count:
            label_flicker.append(float(np.count_nonzero(changed & labeled) / labeled_count))
    return {
        "temporal_jitter_mean": mean(full_jitter),
        "temporal_jitter_p95": percentile(full_jitter, 95),
        "temporal_visible_jitter_mean": mean(visible_jitter),
        "temporal_visible_jitter_p95": percentile(visible_jitter, 95),
        "temporal_label_flicker_mean": mean(label_flicker),
        "temporal_label_flicker_p95": percentile(label_flicker, 95),
        "high_temporal_visible_jitter_pair_count": high_visible_jitter_count,
    }


def _quarantine_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if int(metrics.get("missing_count", 0)) > 0:
        reasons.append("missing_examples_or_fields")
    if int(metrics.get("shape_error_count", 0)) > 0:
        reasons.append("shape_errors_present")
    if int(metrics.get("nan_count", 0)) > 0:
        reasons.append("nonfinite_values_present")
    if int(metrics.get("weak_label_false_count", 0)) > 0:
        reasons.append("weak_label_flag_missing")
    if int(metrics.get("control_safe_true_count", 0)) > 0:
        reasons.append("control_safe_flag_must_remain_false")
    if int(metrics.get("label_overlap_cell_count", 0)) > 0:
        reasons.append("overlapping_free_obstacle_unknown_labels")
    if int(metrics.get("label_sum_error_cell_count", 0)) > 0:
        reasons.append("free_obstacle_unknown_do_not_partition_grid")
    example_count = int(metrics.get("example_count", 0))
    if example_count < MIN_EXAMPLES_FOR_TRAINING_GATE:
        reasons.append(f"example_count_below_{MIN_EXAMPLES_FOR_TRAINING_GATE}")
    confidence_mean = float(metrics.get("confidence_mean", 0.0))
    if confidence_mean < LOW_CONFIDENCE_THRESHOLD:
        reasons.append(f"mean_confidence_below_{LOW_CONFIDENCE_THRESHOLD:.2f}")
    low_confidence_count = int(metrics.get("low_confidence_frame_count", 0))
    if example_count and low_confidence_count / example_count > MAX_LOW_CONFIDENCE_FRAME_FRACTION:
        reasons.append(f"low_confidence_frame_fraction_above_{MAX_LOW_CONFIDENCE_FRAME_FRACTION:.2f}")
    if float(metrics.get("label_density_mean", 0.0)) < MIN_LABEL_DENSITY:
        reasons.append(f"label_density_below_{MIN_LABEL_DENSITY:.2f}")
    observed_ratio = max(
        float(metrics.get("observed_ratio_mean", 0.0)),
        float(metrics.get("source_observed_ratio_mean", 0.0)),
        float(metrics.get("visible_confidence_positive_ratio_mean", 0.0)),
    )
    if observed_ratio < MIN_OBSERVED_RATIO:
        reasons.append(f"observed_visible_ratio_below_{MIN_OBSERVED_RATIO:.2f}")
    if int(metrics.get("empty_label_frame_count", 0)) > 0:
        reasons.append("empty_label_frames_present")
    if float(metrics.get("temporal_visible_jitter_mean", 0.0)) > MAX_TEMPORAL_VISIBLE_JITTER:
        reasons.append(f"temporal_visible_jitter_above_{MAX_TEMPORAL_VISIBLE_JITTER:.2f}")
    return reasons


def _structural_blockers(metrics: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if int(metrics.get("missing_count", 0)) > 0:
        blockers.append("missing_examples_or_fields")
    if int(metrics.get("shape_error_count", 0)) > 0:
        blockers.append("shape_errors_present")
    if int(metrics.get("nan_count", 0)) > 0:
        blockers.append("nonfinite_values_present")
    if int(metrics.get("weak_label_false_count", 0)) > 0:
        blockers.append("weak_label_flag_missing")
    if int(metrics.get("control_safe_true_count", 0)) > 0:
        blockers.append("control_safe_flag_must_remain_false")
    if int(metrics.get("label_overlap_cell_count", 0)) > 0:
        blockers.append("overlapping_free_obstacle_unknown_labels")
    if int(metrics.get("label_sum_error_cell_count", 0)) > 0:
        blockers.append("free_obstacle_unknown_do_not_partition_grid")
    if int(metrics.get("example_count", 0)) < MIN_EXAMPLES_FOR_TRAINING_GATE:
        blockers.append(f"example_count_below_{MIN_EXAMPLES_FOR_TRAINING_GATE}")
    return blockers


def _robot_supervision_grade(manifest: dict[str, Any]) -> str:
    existing = manifest.get("robot_supervision_grade")
    if isinstance(existing, str) and existing in ROBOT_SUPERVISION_GRADES:
        return existing

    source_name = str(manifest.get("source_depth_teacher_name", "")).lower()
    source_backend = str(manifest.get("source_depth_backend", "")).lower()
    source_bev = str(manifest.get("source_bev", "")).lower()
    if "robot_frame_metric" in {source_name, source_backend}:
        return "robot_frame_metric"
    if (
        "tum" in source_name
        or "rgbd_truth" in source_name
        or "public_rgbd" in source_backend
        or "rgbd_truth" in source_bev
    ):
        return "public_rgbd_anchor"
    if source_name in {"da3", "depth_pro"} or "weak_bev" in source_bev or source_backend in {"real", "fake"}:
        return "weak_visual_geometry"
    return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QA a HomeBrain SpatialTrainPack and emit quarantine metrics.")
    parser.add_argument("--dataset", required=True, help="Input SpatialTrainPack directory.")
    parser.add_argument("--out", required=True, help="Output QA JSON path.")
    args = parser.parse_args(argv)
    metrics = qa_spatial_dataset(args.dataset)
    write_qa_metrics(metrics, args.out)
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
