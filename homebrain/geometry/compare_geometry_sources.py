from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from homebrain.geometry.bev_projector import BEV_ARTIFACT_KINDS
from homebrain.geometry.validate_bev import load_bev_manifest
from homebrain.messages.schema import JsonDict, deterministic_json
from homebrain.teachers.artifacts import frame_key, load_array, load_teacher_manifest

COMPARE_SCHEMA_VERSION = "homebrain.geometry.compare_sources.v0"


def compare_geometry_sources(
    *,
    log_dir: str | Path,
    source_a: str | Path,
    source_b: str | Path,
    out_path: str | Path,
) -> Path:
    a = load_geometry_source(source_a)
    b = load_geometry_source(source_b)
    common = sorted(set(a["frames"]).intersection(set(b["frames"])))
    disagreements = [_compare_frame(a["frames"][key], b["frames"][key]) for key in common]
    report: JsonDict = {
        "schema_version": COMPARE_SCHEMA_VERSION,
        "source_log": Path(log_dir).as_posix(),
        "a": a["summary"],
        "b": b["summary"],
        "overlap_frame_count": len(common),
        "valid_mask_disagreement_mean": _mean(
            [item["valid_mask_disagreement"] for item in disagreements if item["valid_mask_disagreement"] is not None]
        ),
        "valid_mask_iou_mean": _mean(
            [item["valid_mask_iou"] for item in disagreements if item["valid_mask_iou"] is not None]
        ),
        "occupancy_disagreement_mean": _mean(
            [item["occupancy_disagreement"] for item in disagreements if item["occupancy_disagreement"] is not None]
        ),
        "confidence_absdiff_mean": _mean(
            [item["confidence_absdiff"] for item in disagreements if item["confidence_absdiff"] is not None]
        ),
        "frame_comparisons": disagreements,
        "control_safe": False,
        "comparison_note": "Disagreement metrics compare geometry artifacts only; they are not control-safety evidence.",
    }
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(deterministic_json(report))
        handle.write("\n")
    return output


def load_geometry_source(path: str | Path) -> JsonDict:
    root = Path(path)
    if (root / "bev_manifest.json").exists():
        return _load_bev_source(root)
    if (root / "teacher_manifest.json").exists():
        return _load_teacher_source(root)
    raise FileNotFoundError(f"source must contain bev_manifest.json or teacher_manifest.json: {root}")


def _load_bev_source(root: Path) -> JsonDict:
    manifest = load_bev_manifest(root)
    frames: dict[str, JsonDict] = {}
    free_ratios: list[float] = []
    obstacle_ratios: list[float] = []
    unknown_ratios: list[float] = []
    valid_ratios: list[float] = []
    confidence_means: list[float] = []
    class_grids: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    for record in manifest["frames"]:
        if not isinstance(record, dict):
            continue
        arrays = _load_bev_arrays(root, record)
        free = arrays["bev_free"] > 0
        obstacle = arrays["bev_obstacle"] > 0
        unknown = arrays["bev_unknown"] > 0
        confidence = arrays["bev_confidence"].astype(np.float32)
        valid = (confidence > np.float32(0.0)) | free | obstacle
        class_grid = np.where(obstacle, 2, np.where(free, 1, 0)).astype(np.uint8)
        total = max(free.size, 1)
        free_ratios.append(float(np.count_nonzero(free) / total))
        obstacle_ratios.append(float(np.count_nonzero(obstacle) / total))
        unknown_ratios.append(float(np.count_nonzero(unknown) / total))
        valid_ratios.append(float(np.count_nonzero(valid) / total))
        confidence_means.append(float(np.mean(confidence)))
        class_grids.append(class_grid)
        valid_masks.append(valid)
        frames[frame_key(record)] = {
            "frame_id": int(record["frame_id"]),
            "valid_mask": valid,
            "class_grid": class_grid,
            "confidence": confidence,
        }
    return {
        "summary": {
            "path": root.as_posix(),
            "source_type": "bev",
            "frame_count": len(frames),
            "weak_label": manifest.get("weak_label") is True,
            "control_safe": manifest.get("control_safe") is True,
            "free_ratio_mean": _mean(free_ratios),
            "obstacle_ratio_mean": _mean(obstacle_ratios),
            "unknown_ratio_mean": _mean(unknown_ratios),
            "valid_mask_ratio_mean": _mean(valid_ratios),
            "confidence_mean": _mean(confidence_means),
            "temporal_valid_stability": _temporal_stability(valid_masks),
            "temporal_occupancy_stability": _temporal_stability(class_grids),
        },
        "frames": frames,
    }


def _load_teacher_source(root: Path) -> JsonDict:
    manifest = load_teacher_manifest(root)
    frames: dict[str, JsonDict] = {}
    valid_ratios: list[float] = []
    confidence_means: list[float] = []
    valid_masks: list[np.ndarray] = []
    for record in manifest["frames"]:
        if not isinstance(record, dict):
            continue
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        depth = _load_first(root, artifacts, ("depth", "depth_m", "depth_relative"))
        if depth is None:
            continue
        depth_values = np.asarray(depth, dtype=np.float32)
        valid = np.isfinite(depth_values) & (depth_values > np.float32(0.0))
        confidence = _load_first(root, artifacts, ("confidence", "depth_confidence"))
        if confidence is not None:
            confidence_values = np.asarray(confidence, dtype=np.float32)
            finite = confidence_values[np.isfinite(confidence_values)]
            confidence_mean = float(np.mean(finite)) if finite.size else 0.0
        else:
            confidence_values = valid.astype(np.float32)
            confidence_mean = 0.0
        valid_ratios.append(float(np.count_nonzero(valid) / max(valid.size, 1)))
        confidence_means.append(confidence_mean)
        valid_masks.append(valid)
        frames[frame_key(record)] = {
            "frame_id": int(record["frame_id"]),
            "valid_mask": valid,
            "class_grid": None,
            "confidence": confidence_values,
        }
    return {
        "summary": {
            "path": root.as_posix(),
            "source_type": "teacher_depth",
            "teacher_name": manifest.get("teacher_name"),
            "backend": manifest.get("backend"),
            "frame_count": len(frames),
            "weak_label": False,
            "control_safe": False,
            "valid_mask_ratio_mean": _mean(valid_ratios),
            "confidence_mean": _mean(confidence_means),
            "temporal_valid_stability": _temporal_stability(valid_masks),
        },
        "frames": frames,
    }


def _load_bev_arrays(root: Path, record: JsonDict) -> dict[str, np.ndarray]:
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"BEV frame {record.get('frame_id')} missing artifacts")
    arrays: dict[str, np.ndarray] = {}
    for kind in BEV_ARTIFACT_KINDS:
        artifact = artifacts.get(kind)
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValueError(f"BEV frame {record.get('frame_id')} missing {kind}")
        arrays[kind] = load_array(root / str(artifact["path"]))
    return arrays


def _load_first(root: Path, artifacts: JsonDict, kinds: tuple[str, ...]) -> np.ndarray | None:
    for kind in kinds:
        record = artifacts.get(kind)
        if isinstance(record, dict) and isinstance(record.get("path"), str):
            path = root / str(record["path"])
            if path.exists():
                return load_array(path)
    return None


def _compare_frame(frame_a: JsonDict, frame_b: JsonDict) -> JsonDict:
    valid_a = np.asarray(frame_a["valid_mask"], dtype=bool)
    valid_b = _resize_to(np.asarray(frame_b["valid_mask"], dtype=bool), valid_a.shape).astype(bool)
    union = valid_a | valid_b
    intersection = valid_a & valid_b
    valid_disagreement = float(np.count_nonzero(valid_a != valid_b) / max(valid_a.size, 1))
    valid_iou = float(np.count_nonzero(intersection) / max(np.count_nonzero(union), 1))

    class_a = frame_a.get("class_grid")
    class_b = frame_b.get("class_grid")
    occupancy_disagreement: float | None = None
    if class_a is not None and class_b is not None:
        a_grid = np.asarray(class_a, dtype=np.uint8)
        b_grid = _resize_to(np.asarray(class_b, dtype=np.uint8), a_grid.shape).astype(np.uint8)
        mask = valid_a | _resize_to(valid_b, valid_a.shape).astype(bool)
        if np.any(mask):
            occupancy_disagreement = float(np.count_nonzero((a_grid != b_grid) & mask) / np.count_nonzero(mask))

    confidence_a = np.asarray(frame_a.get("confidence"), dtype=np.float32)
    confidence_b = _resize_to(np.asarray(frame_b.get("confidence"), dtype=np.float32), confidence_a.shape).astype(np.float32)
    confidence_mask = np.isfinite(confidence_a) & np.isfinite(confidence_b)
    confidence_absdiff = (
        float(np.mean(np.abs(confidence_a[confidence_mask] - confidence_b[confidence_mask])))
        if np.any(confidence_mask)
        else None
    )
    return {
        "frame_id": int(frame_a["frame_id"]),
        "valid_mask_disagreement": valid_disagreement,
        "valid_mask_iou": valid_iou,
        "occupancy_disagreement": occupancy_disagreement,
        "confidence_absdiff": confidence_absdiff,
    }


def _resize_to(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(array)
    values = np.squeeze(values)
    if values.ndim != 2:
        values = values.reshape(1, -1)
    if values.shape == shape:
        return values
    out_h, out_w = shape
    row_index = np.linspace(0, values.shape[0] - 1, out_h).round().astype(np.int64)
    col_index = np.linspace(0, values.shape[1] - 1, out_w).round().astype(np.int64)
    return values[row_index[:, None], col_index[None, :]]


def _temporal_stability(arrays: list[np.ndarray]) -> float:
    if len(arrays) < 2:
        return 0.0
    values: list[float] = []
    for previous, current in zip(arrays, arrays[1:]):
        if previous.shape != current.shape or previous.size == 0:
            continue
        values.append(1.0 - float(np.count_nonzero(previous != current) / previous.size))
    return _mean(values)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two HomeBrain geometry sources.")
    parser.add_argument("--log", required=True, help="Input route log used by both sources.")
    parser.add_argument("--a", required=True, help="First geometry source directory.")
    parser.add_argument("--b", required=True, help="Second geometry source directory.")
    parser.add_argument("--out", required=True, help="Output comparison JSON.")
    args = parser.parse_args(argv)
    path = compare_geometry_sources(log_dir=args.log, source_a=args.a, source_b=args.b, out_path=args.out)
    print(f"wrote geometry source comparison to {path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
