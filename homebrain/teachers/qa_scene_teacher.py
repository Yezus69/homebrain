from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.messages.schema import JsonDict
from homebrain.teachers.artifacts import load_array
from homebrain.teachers.scene_teacher import (
    SCENE_TEACHER_FRAME_ARTIFACT_KINDS,
    SCENE_TEACHER_MANIFEST_FILE,
    SCENE_TEACHER_REQUIRED_GEOMETRY_KINDS,
    load_scene_teacher_manifest,
    write_scene_teacher_json,
)


@dataclass(frozen=True)
class SceneTeacherQA:
    metrics: JsonDict

    @property
    def promotable_to_spatial_pack(self) -> bool:
        return self.metrics.get("promotable_to_spatial_pack") is True


def run_scene_teacher_qa(*, artifacts_dir: str | Path, out_path: str | Path) -> SceneTeacherQA:
    metrics = qa_scene_teacher_artifacts(artifacts_dir)
    write_scene_teacher_json(out_path, metrics, pretty=True)
    return SceneTeacherQA(metrics=metrics)


def qa_scene_teacher_artifacts(artifacts_dir: str | Path) -> JsonDict:
    root = Path(artifacts_dir)
    errors: list[str] = []
    quarantine_reasons: list[str] = []
    try:
        manifest = load_scene_teacher_manifest(root)
    except Exception as exc:  # noqa: BLE001 - QA should report malformed packs.
        return {
            "schema_version": "homebrain.scene_teacher_qa.v0",
            "scene_teacher_manifest": (root / SCENE_TEACHER_MANIFEST_FILE).as_posix(),
            "frame_count": 0,
            "missing_artifact_count": 1,
            "artifact_shape_error_count": 0,
            "depth_valid_ratio": 0.0,
            "pose_valid_ratio": 0.0,
            "track_valid_ratio": 0.0,
            "temporal_geometry_consistency": 0.0,
            "scale_status": "missing_manifest",
            "control_safe": False,
            "promotable_to_spatial_pack": False,
            "quarantine_reasons": ["manifest_load_failed"],
            "errors": [str(exc)],
        }

    frames = [frame for frame in manifest.get("frames", []) if isinstance(frame, dict)]
    frame_count = len(frames)
    missing_count = 0
    shape_error_count = 0
    depth_valid_samples = 0
    depth_total_samples = 0
    pose_valid_count = 0
    confidence_valid_count = 0
    depth_maps: list[np.ndarray] = []

    for frame in frames:
        metadata_path = frame.get("metadata_path")
        if not isinstance(metadata_path, str) or not (root / metadata_path).exists():
            missing_count += 1
            errors.append(f"missing metadata for frame {frame.get('frame_id')}")

        artifacts = frame.get("artifacts")
        if not isinstance(artifacts, dict):
            missing_count += len(SCENE_TEACHER_FRAME_ARTIFACT_KINDS)
            errors.append(f"missing artifacts object for frame {frame.get('frame_id')}")
            continue

        geometry_present = False
        frame_geometry_valid = False
        for kind in SCENE_TEACHER_REQUIRED_GEOMETRY_KINDS:
            if kind not in artifacts:
                continue
            geometry_present = True
            array = _load_artifact_array(root, artifacts, kind, errors)
            if array is None:
                missing_count += 1
                continue
            ok, valid_count, total_count = _geometry_validity(kind, array, frame)
            if not ok:
                shape_error_count += 1
                errors.append(f"invalid {kind} for frame {frame.get('frame_id')}: shape={array.shape}")
            depth_valid_samples += valid_count
            depth_total_samples += total_count
            frame_geometry_valid = frame_geometry_valid or valid_count > 0
            if kind == "depth" and array.ndim == 2:
                depth_maps.append(np.asarray(array, dtype=np.float32))
            elif kind == "point_map" and array.ndim == 3 and array.shape[-1] == 3:
                depth_maps.append(np.asarray(array[:, :, 2], dtype=np.float32))
        if not geometry_present:
            missing_count += 1
            errors.append(f"frame {frame.get('frame_id')} missing depth or point_map")

        confidence = _optional_artifact_array(root, artifacts, "confidence", errors)
        if confidence is None:
            missing_count += 1
        elif _mask_like_shape_ok(confidence, frame):
            finite = np.isfinite(confidence)
            if bool(np.any(finite)) and frame_geometry_valid:
                confidence_valid_count += 1
        else:
            shape_error_count += 1
            errors.append(f"invalid confidence for frame {frame.get('frame_id')}: shape={confidence.shape}")

        for kind in ("validity_mask", "floor_traversable_mask", "obstacle_risk_mask", "dynamic_motion_mask"):
            array = _optional_artifact_array(root, artifacts, kind, errors)
            if array is None:
                missing_count += 1
                continue
            if not _mask_like_shape_ok(array, frame):
                shape_error_count += 1
                errors.append(f"invalid {kind} for frame {frame.get('frame_id')}: shape={array.shape}")

        intrinsics = _optional_artifact_array(root, artifacts, "intrinsics", errors)
        extrinsics = _optional_artifact_array(root, artifacts, "extrinsics", errors)
        intrinsics_ok = intrinsics is not None and _matrix_ok(intrinsics, (3, 3))
        extrinsics_ok = extrinsics is not None and _extrinsics_ok(extrinsics)
        if intrinsics is not None and not intrinsics_ok:
            shape_error_count += 1
            errors.append(f"invalid intrinsics for frame {frame.get('frame_id')}: shape={intrinsics.shape}")
        if extrinsics is not None and not extrinsics_ok:
            shape_error_count += 1
            errors.append(f"invalid extrinsics for frame {frame.get('frame_id')}: shape={extrinsics.shape}")
        if intrinsics_ok and extrinsics_ok:
            pose_valid_count += 1

        for kind, record in artifacts.items():
            if kind not in SCENE_TEACHER_FRAME_ARTIFACT_KINDS:
                continue
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                missing_count += 1
                errors.append(f"missing {kind} path for frame {frame.get('frame_id')}")
                continue
            if not (root / str(record["path"])).exists():
                missing_count += 1
                errors.append(f"missing {kind} file for frame {frame.get('frame_id')}")

    track_valid_count, track_total_count, track_errors = _track_validity(root, manifest)
    errors.extend(track_errors)
    shape_error_count += len(track_errors)

    control_safe = manifest.get("control_safe") is True
    replay_only = manifest.get("replay_only") is True
    not_executed = manifest.get("not_executed") is True
    product_training_approved = manifest.get("product_training_approved") is True
    scale_status = str(manifest.get("scale_status", "unknown"))

    depth_valid_ratio = _ratio(depth_valid_samples, depth_total_samples)
    pose_valid_ratio = _ratio(pose_valid_count, frame_count)
    confidence_valid_ratio = _ratio(confidence_valid_count, frame_count)
    track_valid_ratio = _ratio(track_valid_count, track_total_count)
    temporal_consistency = _temporal_geometry_consistency(depth_maps)

    if missing_count:
        quarantine_reasons.append("missing_artifacts")
    if shape_error_count:
        quarantine_reasons.append("invalid_artifact_shapes")
    if bool(manifest.get("mock", False)) or bool(manifest.get("synthetic", False)):
        quarantine_reasons.append("mock_or_synthetic_teacher")
    if depth_valid_ratio < 0.95:
        quarantine_reasons.append("low_depth_valid_ratio")
    if confidence_valid_ratio < 0.95:
        quarantine_reasons.append("low_confidence_valid_ratio")
    if pose_valid_ratio < 0.95:
        quarantine_reasons.append("low_pose_valid_ratio")
    if scale_status not in {"metric", "metric_or_route_measured", "measured_metric"}:
        quarantine_reasons.append("relative_or_unknown_scale")
    if control_safe:
        quarantine_reasons.append("control_safe_claim_present")
    if not replay_only or not not_executed:
        quarantine_reasons.append("replay_safety_flags_missing")
    if product_training_approved:
        quarantine_reasons.append("product_training_claim_present")

    promotable = (
        not quarantine_reasons
        and missing_count == 0
        and shape_error_count == 0
        and depth_valid_ratio >= 0.95
        and confidence_valid_ratio >= 0.95
        and pose_valid_ratio >= 0.95
        and not control_safe
    )

    return {
        "schema_version": "homebrain.scene_teacher_qa.v0",
        "scene_teacher_manifest": (root / SCENE_TEACHER_MANIFEST_FILE).as_posix(),
        "teacher_name": manifest.get("teacher_name"),
        "backend": manifest.get("backend"),
        "mock": bool(manifest.get("mock", False)),
        "synthetic": bool(manifest.get("synthetic", False)),
        "real_perception": bool(manifest.get("real_perception", False)),
        "frame_count": frame_count,
        "missing_artifact_count": missing_count,
        "artifact_shape_error_count": shape_error_count,
        "depth_valid_ratio": depth_valid_ratio,
        "pose_valid_ratio": pose_valid_ratio,
        "track_valid_ratio": track_valid_ratio,
        "temporal_geometry_consistency": temporal_consistency,
        "confidence_valid_ratio": confidence_valid_ratio,
        "scale_status": scale_status,
        "control_safe": control_safe,
        "replay_only": replay_only,
        "not_executed": not_executed,
        "product_training_approved": product_training_approved,
        "raw_pwm_emitted": manifest.get("raw_pwm_emitted") is True,
        "promotable_to_spatial_pack": promotable,
        "quarantine_reasons": sorted(set(quarantine_reasons)),
        "errors": errors[:50],
    }


def _load_artifact_array(root: Path, artifacts: JsonDict, kind: str, errors: list[str]) -> np.ndarray | None:
    record = artifacts.get(kind)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        errors.append(f"missing {kind} artifact record")
        return None
    path = root / str(record["path"])
    if not path.exists():
        return None
    try:
        return load_array(path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"failed to load {kind}: {exc}")
        return None


def _optional_artifact_array(root: Path, artifacts: JsonDict, kind: str, errors: list[str]) -> np.ndarray | None:
    if kind not in artifacts:
        return None
    return _load_artifact_array(root, artifacts, kind, errors)


def _geometry_validity(kind: str, array: np.ndarray, frame: JsonDict) -> tuple[bool, int, int]:
    height = int(frame.get("height", 0))
    width = int(frame.get("width", 0))
    if kind == "depth":
        ok = array.ndim == 2 and tuple(array.shape) == (height, width)
        finite = np.isfinite(array) & (np.asarray(array) > np.float32(0.0)) if ok else np.asarray([], dtype=bool)
        return ok, int(np.count_nonzero(finite)), int(height * width if ok else 0)
    if kind == "point_map":
        ok = array.ndim == 3 and tuple(array.shape[:2]) == (height, width) and array.shape[-1] == 3
        finite = np.isfinite(array).all(axis=2) if ok else np.asarray([], dtype=bool)
        return ok, int(np.count_nonzero(finite)), int(height * width if ok else 0)
    return False, 0, 0


def _mask_like_shape_ok(array: np.ndarray, frame: JsonDict) -> bool:
    return array.ndim == 2 and tuple(array.shape) == (int(frame.get("height", 0)), int(frame.get("width", 0)))


def _matrix_ok(array: np.ndarray, shape: tuple[int, int]) -> bool:
    return array.shape == shape and bool(np.isfinite(array).all())


def _extrinsics_ok(array: np.ndarray) -> bool:
    return array.shape in {(3, 4), (4, 4)} and bool(np.isfinite(array).all())


def _track_validity(root: Path, manifest: JsonDict) -> tuple[int, int, list[str]]:
    errors: list[str] = []
    valid = 0
    total = 0
    windows = manifest.get("windows", [])
    if not isinstance(windows, list):
        return 0, 0, ["scene teacher windows must be a list"]
    for window in windows:
        if not isinstance(window, dict):
            errors.append("window record is not an object")
            continue
        artifacts = window.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        tracks = _load_window_array(root, artifacts, "point_tracks", errors)
        validity = _load_window_array(root, artifacts, "track_validity", errors)
        if tracks is None or validity is None:
            continue
        if tracks.ndim != 3 or tracks.shape[-1] != 2 or validity.shape != tracks.shape[:2]:
            errors.append(f"invalid track shapes: point_tracks={tracks.shape}, track_validity={validity.shape}")
            continue
        finite = np.isfinite(tracks).all(axis=2) & (validity > 0)
        valid += int(np.count_nonzero(finite))
        total += int(finite.size)
    return valid, total, errors


def _load_window_array(root: Path, artifacts: JsonDict, kind: str, errors: list[str]) -> np.ndarray | None:
    record = artifacts.get(kind)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return None
    path = root / str(record["path"])
    if not path.exists():
        errors.append(f"missing {kind} file for scene window")
        return None
    try:
        return load_array(path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"failed to load {kind} for scene window: {exc}")
        return None


def _temporal_geometry_consistency(depth_maps: list[np.ndarray]) -> float:
    if len(depth_maps) < 2:
        return 0.0
    values: list[float] = []
    for previous, current in zip(depth_maps, depth_maps[1:]):
        if previous.shape != current.shape:
            continue
        valid = np.isfinite(previous) & np.isfinite(current) & (previous > 0.0) & (current > 0.0)
        if not np.any(valid):
            continue
        denom = np.maximum(np.maximum(np.abs(previous[valid]), np.abs(current[valid])), np.float32(1.0e-3))
        delta = np.abs(current[valid] - previous[valid]) / denom
        values.append(float(1.0 - np.clip(np.mean(delta), 0.0, 1.0)))
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QA a HomeBrain SceneTeacherPack.")
    parser.add_argument("--artifacts", required=True, help="Input SceneTeacherPack directory.")
    parser.add_argument("--out", required=True, help="Output QA JSON path.")
    args = parser.parse_args(argv)
    qa = run_scene_teacher_qa(artifacts_dir=args.artifacts, out_path=args.out)
    print(f"wrote scene teacher QA to {Path(args.out).as_posix()}")
    print(
        "scene teacher QA: "
        f"frames={qa.metrics['frame_count']} "
        f"missing={qa.metrics['missing_artifact_count']} "
        f"promotable={qa.metrics['promotable_to_spatial_pack']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
