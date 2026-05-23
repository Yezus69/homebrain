from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.messages.schema import JsonDict, deterministic_json
from homebrain.replay.replayd import order_events_for_replay
from homebrain.replay.segment_log import read_events
from homebrain.teachers.artifacts import frame_key, load_array, load_teacher_manifest, read_json

QA_SCHEMA_VERSION = "homebrain.geometry.self_calibration_qa.v0"


def run_self_calibration_qa(
    *,
    log_dir: str | Path,
    teacher_artifacts_dir: str | Path,
    out_path: str | Path,
) -> Path:
    log_root = Path(log_dir)
    teacher_root = Path(teacher_artifacts_dir)
    output = Path(out_path)
    manifest = load_teacher_manifest(teacher_root)
    route_frame_keys = _route_frame_keys(log_root)
    frame_records = [record for record in manifest.get("frames", []) if isinstance(record, dict)]

    depth_valid_ratios: list[float] = []
    confidence_means: list[float] = []
    intrinsics_valid = 0
    pose_valid = 0
    pose_translations: list[np.ndarray] = []
    normalized_depths: list[np.ndarray] = []
    floor_found_flags: list[bool] = []
    floor_medians: list[float] = []
    route_missing_count = 0
    loaded_depth_previews: list[np.ndarray] = []
    loaded_confidence_previews: list[np.ndarray] = []

    for frame_record in frame_records:
        if frame_key(frame_record) not in route_frame_keys:
            route_missing_count += 1
        artifacts = frame_record.get("artifacts") if isinstance(frame_record.get("artifacts"), dict) else {}
        if not isinstance(artifacts, dict):
            continue

        depth = _load_first_array(teacher_root, artifacts, ("depth", "depth_relative", "depth_m"))
        if depth is not None:
            depth = np.asarray(depth, dtype=np.float32)
            valid = np.isfinite(depth) & (depth > np.float32(0.0))
            depth_valid_ratios.append(float(np.count_nonzero(valid) / max(depth.size, 1)))
            normalized = _normalized_depth(depth)
            normalized_depths.append(normalized)
            floor_found, floor_median = _floor_proxy(normalized, valid)
            floor_found_flags.append(floor_found)
            if floor_median is not None:
                floor_medians.append(floor_median)
            if len(loaded_depth_previews) < 8:
                loaded_depth_previews.append(depth)

        confidence = _load_first_array(teacher_root, artifacts, ("confidence", "depth_confidence"))
        if confidence is not None:
            conf = np.asarray(confidence, dtype=np.float32)
            finite = conf[np.isfinite(conf)]
            confidence_means.append(float(np.mean(finite)) if finite.size else 0.0)
            if len(loaded_confidence_previews) < 8:
                loaded_confidence_previews.append(conf)

        intrinsics = _load_first_array(teacher_root, artifacts, ("intrinsics",))
        if intrinsics is not None and _valid_intrinsics(intrinsics):
            intrinsics_valid += 1

        pose = _load_first_array(teacher_root, artifacts, ("extrinsics", "camera_pose"))
        if pose is not None and _valid_pose(pose):
            pose_valid += 1
            pose_translations.append(_pose_translation(pose))

    frame_count = len(frame_records)
    pose_jump_outliers = _pose_jump_outlier_count(pose_translations)
    temporal_depth_consistency = _temporal_depth_consistency(normalized_depths)
    floor_plane_found_ratio = _ratio(floor_found_flags)
    floor_plane_stability = _floor_stability(floor_medians)
    scale_source = _scale_source(manifest, frame_records, teacher_root)

    quarantine_reasons = _quarantine_reasons(
        manifest=manifest,
        frame_count=frame_count,
        route_missing_count=route_missing_count,
        depth_valid_ratio=_mean(depth_valid_ratios),
        confidence_mean=_mean(confidence_means),
        confidence_available=bool(confidence_means),
        intrinsics_valid_ratio=intrinsics_valid / frame_count if frame_count else 0.0,
        pose_valid_ratio=pose_valid / frame_count if frame_count else 0.0,
        pose_jump_outlier_count=pose_jump_outliers,
        temporal_depth_consistency=temporal_depth_consistency,
        floor_plane_found_ratio=floor_plane_found_ratio,
        floor_plane_stability=floor_plane_stability,
        scale_source=scale_source,
    )
    contact_dir = _contact_sheet_dir(output)
    contact_sheets = _write_contact_sheets(
        contact_dir,
        depth_previews=loaded_depth_previews,
        confidence_previews=loaded_confidence_previews,
    )

    report: JsonDict = {
        "schema_version": QA_SCHEMA_VERSION,
        "source_log": log_root.as_posix(),
        "teacher_artifacts": teacher_root.as_posix(),
        "teacher_name": manifest.get("teacher_name"),
        "backend": manifest.get("backend"),
        "mock": bool(manifest.get("mock", False)),
        "synthetic": bool(manifest.get("synthetic", False)),
        "real_perception": bool(manifest.get("real_perception", False)),
        "model_id": manifest.get("model_id"),
        "calibration_class": manifest.get("calibration_class", "unknown"),
        "intrinsics_source": manifest.get("intrinsics_source", "unknown"),
        "extrinsics_source": manifest.get("extrinsics_source", "unknown"),
        "pose_source": manifest.get("pose_source", "unknown"),
        "scale_source": scale_source,
        "gravity_floor_source": manifest.get("gravity_floor_source", "unknown"),
        "not_robot_frame_truth": bool(manifest.get("not_robot_frame_truth", True)),
        "frame_count": frame_count,
        "route_missing_frame_count": route_missing_count,
        "depth_valid_ratio": _mean(depth_valid_ratios),
        "confidence_mean": _mean(confidence_means),
        "intrinsics_valid_ratio": intrinsics_valid / frame_count if frame_count else 0.0,
        "pose_valid_ratio": pose_valid / frame_count if frame_count else 0.0,
        "pose_jump_outlier_count": pose_jump_outliers,
        "temporal_depth_consistency": temporal_depth_consistency,
        "floor_plane_found_ratio": floor_plane_found_ratio,
        "floor_plane_stability": floor_plane_stability,
        "promotable_to_weak_bev": not quarantine_reasons,
        "quarantine_reasons": quarantine_reasons,
        "contact_sheet_dir": contact_dir.as_posix(),
        "contact_sheets": contact_sheets,
        "control_safe": False,
        "qa_note": "Self-calibration QA is a weak offline review gate; it is not robot-frame metric truth.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return output


def _route_frame_keys(log_dir: Path) -> set[str]:
    events = order_events_for_replay(read_events(log_dir))
    return {frame_key(event) for event in events if getattr(event, "event_type", None) == "frame"}


def _load_first_array(root: Path, artifacts: JsonDict, kinds: tuple[str, ...]) -> np.ndarray | None:
    for kind in kinds:
        record = artifacts.get(kind)
        if isinstance(record, dict) and isinstance(record.get("path"), str):
            path = root / str(record["path"])
            if path.exists():
                return load_array(path)
    return None


def _normalized_depth(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > np.float32(0.0))
    if not np.any(valid):
        return np.full(values.shape, np.nan, dtype=np.float32)
    median = float(np.nanmedian(values[valid]))
    scale = max(median, 1.0e-6)
    return np.where(valid, values / np.float32(scale), np.nan).astype(np.float32)


def _floor_proxy(normalized_depth: np.ndarray, valid: np.ndarray) -> tuple[bool, float | None]:
    if normalized_depth.ndim != 2 or normalized_depth.size == 0:
        return False, None
    start = int(normalized_depth.shape[0] * 0.55)
    bottom = normalized_depth[start:, :]
    bottom_valid = np.isfinite(bottom) & valid[start:, :]
    if not np.any(bottom_valid):
        return False, None
    valid_ratio = float(np.count_nonzero(bottom_valid) / max(bottom_valid.size, 1))
    median = float(np.nanmedian(bottom[bottom_valid]))
    if bottom.shape[0] >= 2:
        row_profile = np.nanmedian(np.where(bottom_valid, bottom, np.nan), axis=1)
        gradient = np.nanmedian(np.diff(row_profile)) if np.count_nonzero(np.isfinite(row_profile)) >= 2 else 0.0
    else:
        gradient = 0.0
    found = valid_ratio >= 0.5 and np.isfinite(median) and gradient >= -0.05
    return bool(found), median


def _valid_intrinsics(array: np.ndarray) -> bool:
    values = np.asarray(array, dtype=np.float32)
    return values.shape == (3, 3) and bool(np.all(np.isfinite(values))) and float(values[2, 2]) != 0.0


def _valid_pose(array: np.ndarray) -> bool:
    values = np.asarray(array, dtype=np.float32)
    return values.shape in {(3, 4), (4, 4)} and bool(np.all(np.isfinite(values)))


def _pose_translation(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.shape == (4, 4):
        return values[:3, 3].astype(np.float32)
    return values[:3, 3].astype(np.float32)


def _pose_jump_outlier_count(translations: list[np.ndarray]) -> int:
    if len(translations) < 3:
        return 0
    jumps = np.asarray(
        [float(np.linalg.norm(current - previous)) for previous, current in zip(translations, translations[1:])],
        dtype=np.float32,
    )
    if jumps.size < 2:
        return 0
    median = float(np.median(jumps))
    mad = float(np.median(np.abs(jumps - np.float32(median))))
    threshold = max(0.25, median + 6.0 * max(mad, 1.0e-6))
    return int(np.count_nonzero(jumps > np.float32(threshold)))


def _temporal_depth_consistency(depths: list[np.ndarray]) -> float:
    if len(depths) < 2:
        return 0.0
    scores: list[float] = []
    for previous, current in zip(depths, depths[1:]):
        if previous.shape != current.shape:
            continue
        valid = np.isfinite(previous) & np.isfinite(current)
        if not np.any(valid):
            continue
        mae = float(np.mean(np.abs(previous[valid] - current[valid])))
        scores.append(max(0.0, 1.0 - min(mae, 1.0)))
    return _mean(scores)


def _floor_stability(medians: list[float]) -> float:
    if len(medians) < 2:
        return 0.0
    values = np.asarray(medians, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    return float(max(0.0, 1.0 - min(float(np.std(values)), 1.0)))


def _scale_source(manifest: JsonDict, frame_records: list[JsonDict], root: Path) -> str:
    value = manifest.get("scale_source")
    if isinstance(value, str):
        return value
    for frame_record in frame_records:
        metadata_path = frame_record.get("metadata_path")
        if isinstance(metadata_path, str):
            try:
                metadata = read_json(root / metadata_path)
            except Exception:  # noqa: BLE001
                continue
            value = metadata.get("scale_source")
            if isinstance(value, str):
                return value
    return "unknown"


def _quarantine_reasons(
    *,
    manifest: JsonDict,
    frame_count: int,
    route_missing_count: int,
    depth_valid_ratio: float,
    confidence_mean: float,
    confidence_available: bool,
    intrinsics_valid_ratio: float,
    pose_valid_ratio: float,
    pose_jump_outlier_count: int,
    temporal_depth_consistency: float,
    floor_plane_found_ratio: float,
    floor_plane_stability: float,
    scale_source: str,
) -> list[str]:
    reasons: list[str] = []
    if frame_count == 0:
        reasons.append("no_teacher_frames")
    if route_missing_count:
        reasons.append("teacher_frames_missing_from_route_log")
    if manifest.get("mock") is True or manifest.get("synthetic") is True:
        reasons.append("mock_or_synthetic_teacher")
    if manifest.get("real_perception") is not True:
        reasons.append("real_perception_false")
    if depth_valid_ratio < 0.95:
        reasons.append("depth_valid_ratio_below_0.95")
    if confidence_available and confidence_mean < 0.20:
        reasons.append("confidence_mean_below_0.20")
    if intrinsics_valid_ratio < 0.80:
        reasons.append("intrinsics_valid_ratio_below_0.80")
    if pose_valid_ratio < 0.80:
        reasons.append("pose_valid_ratio_below_0.80")
    if pose_jump_outlier_count > 0:
        reasons.append("pose_jump_outliers_present")
    if temporal_depth_consistency < 0.70:
        reasons.append("temporal_depth_consistency_below_0.70")
    if floor_plane_found_ratio < 0.50:
        reasons.append("floor_plane_found_ratio_below_0.50")
    if floor_plane_stability < 0.50:
        reasons.append("floor_plane_stability_below_0.50")
    if scale_source in {"unknown", "missing_not_supplied"}:
        reasons.append("scale_source_unknown")
    return reasons


def _contact_sheet_dir(out_path: Path) -> Path:
    stem_path = out_path.with_suffix("")
    return Path(f"{stem_path.as_posix()}_contact_sheets")


def _write_contact_sheets(
    out_dir: Path,
    *,
    depth_previews: list[np.ndarray],
    confidence_previews: list[np.ndarray],
) -> list[JsonDict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[JsonDict] = []
    if depth_previews:
        target = out_dir / "depth_first_frames.pgm"
        _write_contact_sheet(target, depth_previews)
        outputs.append({"name": "depth_first_frames", "path": target.as_posix()})
    if confidence_previews:
        target = out_dir / "confidence_first_frames.pgm"
        _write_contact_sheet(target, confidence_previews)
        outputs.append({"name": "confidence_first_frames", "path": target.as_posix()})
    manifest = {
        "schema_version": "homebrain.geometry.self_calibration_contact_sheets.v0",
        "contact_sheets": outputs,
    }
    with (out_dir / "contact_sheet_manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(deterministic_json(manifest))
        handle.write("\n")
    return outputs


def _write_contact_sheet(path: Path, arrays: list[np.ndarray]) -> None:
    thumbs = [_preview_uint8(array) for array in arrays]
    if not thumbs:
        thumbs = [np.zeros((1, 1), dtype=np.uint8)]
    padding = 4
    height = max(image.shape[0] for image in thumbs)
    width = sum(image.shape[1] for image in thumbs) + padding * (len(thumbs) - 1)
    sheet = np.full((height, width), 12, dtype=np.uint8)
    x = 0
    for image in thumbs:
        h, w = image.shape
        sheet[:h, x : x + w] = image
        x += w + padding
    _write_pgm(path, sheet)


def _preview_uint8(array: np.ndarray, max_side: int = 96) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim != 2:
        values = values.reshape(1, -1)
    stride = max(1, int(np.ceil(max(values.shape) / max_side)))
    values = values[::stride, ::stride]
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.uint8)
    finite_values = values[finite]
    lo = float(np.percentile(finite_values, 2))
    hi = float(np.percentile(finite_values, 98))
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = (values - np.float32(lo)) / np.float32(hi - lo)
    return np.where(finite, np.clip(normalized * np.float32(255.0), 0, 255), 0).astype(np.uint8)


def _write_pgm(path: Path, image: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    h, w = image.shape
    with target.open("wb") as handle:
        handle.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        handle.write(image.astype(np.uint8).tobytes(order="C"))


def _ratio(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(1 for flag in flags if flag) / len(flags))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QA DA3/self-calibration teacher geometry before weak BEV promotion.")
    parser.add_argument("--log", required=True, help="Input HomeBrain route log directory.")
    parser.add_argument("--teacher-artifacts", required=True, help="Input DA3 teacher artifact directory.")
    parser.add_argument("--out", required=True, help="Output QA JSON path.")
    args = parser.parse_args(argv)
    report = run_self_calibration_qa(
        log_dir=args.log,
        teacher_artifacts_dir=args.teacher_artifacts,
        out_path=args.out,
    )
    print(f"wrote self-calibration QA to {report.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
