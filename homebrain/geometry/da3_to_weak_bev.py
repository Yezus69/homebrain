from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from homebrain.geometry.bev_projector import BEV_ARTIFACT_KINDS, BEV_MANIFEST_FILE, BEV_SCHEMA_VERSION, bev_array_stats
from homebrain.messages.schema import JsonDict
from homebrain.replay.segment_log import load_manifest
from homebrain.teachers.artifacts import (
    file_sha256,
    load_array,
    load_teacher_manifest,
    relative_to_root,
    save_array,
    write_json,
)

DEFAULT_GRID_SHAPE = (32, 32)


def convert_da3_to_weak_bev(
    *,
    log_dir: str | Path,
    teacher_artifacts_dir: str | Path,
    qa_path: str | Path,
    out_dir: str | Path,
    force_review: bool = False,
) -> Path:
    log_root = Path(log_dir)
    teacher_root = Path(teacher_artifacts_dir)
    output = Path(out_dir)
    qa = _read_json(qa_path)
    if qa.get("promotable_to_weak_bev") is not True and not force_review:
        reasons = qa.get("quarantine_reasons")
        raise ValueError(f"QA did not approve weak BEV promotion; reasons={reasons}")

    route_manifest = load_manifest(log_root)
    teacher_manifest = load_teacher_manifest(teacher_root)
    output.mkdir(parents=True, exist_ok=True)

    frame_records: list[JsonDict] = []
    warnings: list[str] = []
    for source_record in teacher_manifest["frames"]:
        if not isinstance(source_record, dict):
            continue
        try:
            record = _write_frame(output=output, teacher_root=teacher_root, source_record=source_record)
        except Exception as exc:  # noqa: BLE001 - preserve frame failure in manifest.
            warnings.append(f"frame_{source_record.get('frame_id')}_failed:{exc}")
            continue
        frame_records.append(record)
        warnings.extend(str(item) for item in record.get("warnings", []))

    manifest: JsonDict = {
        "schema_version": BEV_SCHEMA_VERSION,
        "source_log": log_root.as_posix(),
        "source_segment_id": route_manifest.segment_id,
        "source_schema_version": route_manifest.schema_version,
        "source_depth_artifacts": teacher_root.as_posix(),
        "source_depth_manifest_sha256": file_sha256(teacher_root / "teacher_manifest.json"),
        "source_depth_teacher_name": teacher_manifest.get("teacher_name"),
        "source_depth_backend": teacher_manifest.get("backend"),
        "source_depth_mock": bool(teacher_manifest.get("mock", False)),
        "source_depth_real_perception": bool(teacher_manifest.get("real_perception", False)),
        "source_qa": Path(qa_path).as_posix(),
        "source_qa_sha256": file_sha256(qa_path),
        "grid_shape": list(DEFAULT_GRID_SHAPE),
        "grid_orientation": {
            "note": "image_plane_relative_depth_proxy_not_robot_frame_truth",
        },
        "coordinate_frame": "teacher_estimated_visual_frame_not_robot_frame_truth",
        "projection_model": "da3_relative_depth_image_plane_pseudo_bev",
        "artifact_kinds": list(BEV_ARTIFACT_KINDS),
        "weak_label": True,
        "control_safe": False,
        "calibration_class": "teacher_estimated",
        "intrinsics_source": "teacher_estimated",
        "extrinsics_source": "teacher_estimated",
        "pose_source": "teacher_estimated_camera_pose",
        "scale_source": qa.get("scale_source", "teacher_relative_not_metric"),
        "gravity_floor_source": qa.get("gravity_floor_source", "qa_floor_proxy"),
        "not_robot_frame_truth": True,
        "trainable_for": "geometry_pretrain_only",
        "control_safety": "not_control_safe_weak_offline_geometry_label",
        "warnings": sorted(set(warnings)),
        "assumptions": [
            "DA3 relative visual geometry is not metric robot-frame truth",
            "weak labels are for geometry pretraining/review only",
        ],
        "frame_count": len(frame_records),
        "frames": frame_records,
    }
    manifest_path = output / BEV_MANIFEST_FILE
    write_json(manifest_path, manifest)
    return manifest_path


def _write_frame(*, output: Path, teacher_root: Path, source_record: JsonDict) -> JsonDict:
    artifacts = source_record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("missing artifacts object")
    depth = _load_required(teacher_root, artifacts, ("depth", "depth_relative", "depth_m"))
    confidence = _load_optional(teacher_root, artifacts, ("confidence", "depth_confidence"), depth.shape)
    arrays = _pseudo_bev_arrays(depth, confidence)
    frame_id = int(source_record["frame_id"])
    camera_id = str(source_record["camera_id"])
    frame_dir = output / "frames" / f"{camera_id}_{frame_id:06d}"

    artifact_records: dict[str, JsonDict] = {}
    for kind in BEV_ARTIFACT_KINDS:
        target = frame_dir / f"{kind}.npy"
        record = save_array(target, arrays[kind])
        artifact_records[kind] = {
            "kind": kind,
            "path": relative_to_root(target, output),
            **record,
        }

    metadata: JsonDict = {
        "schema_version": BEV_SCHEMA_VERSION,
        "sequence_id": source_record["sequence_id"],
        "camera_id": camera_id,
        "frame_id": frame_id,
        "timestamp_ns": source_record["timestamp_ns"],
        "source_da3_artifacts": {
            "depth": _artifact_path(artifacts, ("depth", "depth_relative", "depth_m")),
            "confidence": _artifact_path(artifacts, ("confidence", "depth_confidence")),
            "intrinsics": _artifact_path(artifacts, ("intrinsics",)),
            "extrinsics": _artifact_path(artifacts, ("extrinsics", "camera_pose")),
        },
        "source_data_ref": source_record.get("source_data_ref"),
        "width": int(source_record["width"]),
        "height": int(source_record["height"]),
        "weak_label": True,
        "control_safe": False,
        "calibration_class": "teacher_estimated",
        "intrinsics_source": "teacher_estimated",
        "extrinsics_source": "teacher_estimated",
        "pose_source": "teacher_estimated_camera_pose",
        "scale_source": "teacher_relative_not_metric",
        "gravity_floor_source": "qa_floor_proxy",
        "not_robot_frame_truth": True,
        "trainable_for": "geometry_pretrain_only",
        "control_safety": "not_control_safe_weak_offline_geometry_label",
        "projection_model": "da3_relative_depth_image_plane_pseudo_bev",
        "grid_shape": list(DEFAULT_GRID_SHAPE),
        "stats": bev_array_stats(arrays),
        "warnings": ["not_robot_frame_truth", "relative_depth_pseudo_bev"],
        "artifact_shapes": {kind: artifact_records[kind]["shape"] for kind in BEV_ARTIFACT_KINDS},
        "artifact_dtypes": {kind: artifact_records[kind]["dtype"] for kind in BEV_ARTIFACT_KINDS},
    }
    metadata_path = frame_dir / "metadata.json"
    write_json(metadata_path, metadata)

    return {
        "sequence_id": source_record["sequence_id"],
        "camera_id": camera_id,
        "frame_id": frame_id,
        "timestamp_ns": source_record["timestamp_ns"],
        "width": int(source_record["width"]),
        "height": int(source_record["height"]),
        "weak_label": True,
        "control_safe": False,
        "calibration_class": "teacher_estimated",
        "not_robot_frame_truth": True,
        "trainable_for": "geometry_pretrain_only",
        "metadata_path": relative_to_root(metadata_path, output),
        "metadata_sha256": file_sha256(metadata_path),
        "artifacts": artifact_records,
        "stats": metadata["stats"],
        "warnings": metadata["warnings"],
    }


def _pseudo_bev_arrays(depth: np.ndarray, confidence: np.ndarray | None) -> dict[str, np.ndarray]:
    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > np.float32(0.0))
    if not np.any(valid):
        norm = np.zeros(values.shape, dtype=np.float32)
    else:
        finite = values[valid]
        lo = float(np.percentile(finite, 5))
        hi = float(np.percentile(finite, 95))
        scale = max(hi - lo, 1.0e-6)
        norm = np.where(valid, (values - np.float32(lo)) / np.float32(scale), np.float32(0.0))
        norm = np.clip(norm, np.float32(0.0), np.float32(1.0)).astype(np.float32)

    depth_grid = _resize_nearest(norm, DEFAULT_GRID_SHAPE)
    if confidence is None:
        conf_source = np.where(valid, np.float32(0.5), np.float32(0.0)).astype(np.float32)
    else:
        conf_source = np.clip(np.asarray(confidence, dtype=np.float32), np.float32(0.0), np.float32(1.0))
    conf_grid = _resize_nearest(conf_source, DEFAULT_GRID_SHAPE).astype(np.float32)

    observed = conf_grid > np.float32(0.05)
    obstacle = observed & (depth_grid < np.float32(0.35))
    floor_candidate = observed & (depth_grid >= np.float32(0.45))
    free = floor_candidate & ~obstacle
    unknown = ~(free | obstacle)
    height = (np.float32(0.5) - depth_grid).astype(np.float32)
    return {
        "bev_free": free.astype(np.uint8),
        "bev_obstacle": obstacle.astype(np.uint8),
        "bev_unknown": unknown.astype(np.uint8),
        "bev_floor_candidate": floor_candidate.astype(np.uint8),
        "bev_height": height,
        "bev_confidence": np.where(observed, conf_grid, np.float32(0.0)).astype(np.float32),
    }


def _load_required(root: Path, artifacts: JsonDict, kinds: tuple[str, ...]) -> np.ndarray:
    array = _load_optional(root, artifacts, kinds, None)
    if array is None:
        raise ValueError(f"missing required DA3 artifact from {kinds}")
    return array


def _load_optional(
    root: Path,
    artifacts: JsonDict,
    kinds: tuple[str, ...],
    expected_shape: tuple[int, ...] | None,
) -> np.ndarray | None:
    for kind in kinds:
        record = artifacts.get(kind)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            continue
        array = load_array(root / str(record["path"]))
        if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
            return _resize_nearest(np.asarray(array, dtype=np.float32), (expected_shape[0], expected_shape[1]))
        return array
    return None


def _artifact_path(artifacts: JsonDict, kinds: tuple[str, ...]) -> str | None:
    for kind in kinds:
        record = artifacts.get(kind)
        if isinstance(record, dict) and isinstance(record.get("path"), str):
            return str(record["path"])
    return None


def _resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(array)
    values = np.squeeze(values)
    if values.ndim != 2:
        values = values.reshape(1, -1)
    out_h, out_w = shape
    row_index = np.linspace(0, values.shape[0] - 1, out_h).round().astype(np.int64)
    col_index = np.linspace(0, values.shape[1] - 1, out_w).round().astype(np.int64)
    return values[row_index[:, None], col_index[None, :]]


def _read_json(path: str | Path) -> JsonDict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote DA3 QA-passed artifacts into weak visual BEV labels.")
    parser.add_argument("--log", required=True, help="Input HomeBrain route log directory.")
    parser.add_argument("--teacher-artifacts", required=True, help="Input DA3 teacher artifact directory.")
    parser.add_argument("--qa", required=True, help="Self-calibration QA JSON.")
    parser.add_argument("--out", required=True, help="Output weak BEV directory.")
    parser.add_argument("--force-review", action="store_true", help="Write weak BEV despite QA quarantine for manual review only.")
    args = parser.parse_args(argv)
    manifest_path = convert_da3_to_weak_bev(
        log_dir=args.log,
        teacher_artifacts_dir=args.teacher_artifacts,
        qa_path=args.qa,
        out_dir=args.out,
        force_review=args.force_review,
    )
    print(f"wrote DA3 weak BEV manifest to {manifest_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
