from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from homebrain.geometry.pose_sequence import (
    POSE_SEQUENCE_SCHEMA_VERSION,
    build_outlier_windows,
    compute_pose_deltas,
    frame_metric_to_json,
    load_pose_frame_metrics,
    pose_delta_to_json,
)
from homebrain.messages.schema import JsonDict, deterministic_json


def inspect_pose_sequence(
    *,
    log_dir: str | Path,
    teacher_artifacts_dir: str | Path,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest, frames = load_pose_frame_metrics(log_dir=log_dir, teacher_artifacts_dir=teacher_artifacts_dir)
    deltas, thresholds = compute_pose_deltas(frames)
    outlier_deltas = [delta for delta in deltas if delta.is_outlier]

    pose_deltas: JsonDict = {
        "schema_version": POSE_SEQUENCE_SCHEMA_VERSION,
        "source_log": Path(log_dir).as_posix(),
        "teacher_artifacts": Path(teacher_artifacts_dir).as_posix(),
        "teacher_name": manifest.get("teacher_name"),
        "backend": manifest.get("backend"),
        "mock": bool(manifest.get("mock", False)),
        "synthetic": bool(manifest.get("synthetic", False)),
        "real_perception": bool(manifest.get("real_perception", False)),
        "calibration_class": manifest.get("calibration_class", "unknown"),
        "pose_source": manifest.get("pose_source", "unknown"),
        "scale_source": manifest.get("scale_source", "unknown"),
        "not_robot_frame_truth": bool(manifest.get("not_robot_frame_truth", True)),
        "control_safe": False,
        "frame_count": len(frames),
        "pose_valid_frame_count": sum(1 for frame in frames if frame.pose_valid),
        "route_missing_frame_count": sum(1 for frame in frames if not frame.route_present),
        "pose_jump_outlier_count": len(outlier_deltas),
        **thresholds,
        "outlier_deltas": [pose_delta_to_json(delta) for delta in outlier_deltas],
        "frames": [frame_metric_to_json(frame) for frame in frames],
        "deltas": [pose_delta_to_json(delta) for delta in deltas],
        "diagnostic_note": "Pose diagnostics identify suspect teacher pose jumps; they do not repair or certify robot-frame truth.",
    }
    pose_deltas_path = output / "pose_deltas.json"
    _write_json(pose_deltas_path, pose_deltas)

    windows = build_outlier_windows(frames, deltas, radius=3)
    contact_sheets = _write_outlier_contact_sheets(output / "contact_sheets", frames, windows)
    for window, sheets in zip(windows, contact_sheets):
        window["contact_sheets"] = sheets
    outlier_windows: JsonDict = {
        "schema_version": "homebrain.geometry.pose_outlier_windows.v0",
        "source_log": Path(log_dir).as_posix(),
        "teacher_artifacts": Path(teacher_artifacts_dir).as_posix(),
        "frame_count": len(frames),
        "pose_jump_outlier_count": len(outlier_deltas),
        "outlier_windows": windows,
        "contact_sheet_dir": (output / "contact_sheets").as_posix(),
        "control_safe": False,
    }
    outlier_windows_path = output / "outlier_windows.json"
    _write_json(outlier_windows_path, outlier_windows)
    return pose_deltas_path, outlier_windows_path


def _write_outlier_contact_sheets(
    out_dir: Path,
    frames: list,
    windows: list[JsonDict],
) -> list[list[JsonDict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_outputs: list[list[JsonDict]] = []
    for window in windows:
        outlier_index = int(window["outlier_index"])
        start = int(window["start_index"])
        end = int(window["end_index"])
        selected = frames[start : end + 1]
        outputs: list[JsonDict] = []
        depth_arrays = [
            frame.normalized_depth_preview for frame in selected if frame.normalized_depth_preview is not None
        ]
        if depth_arrays:
            path = out_dir / f"outlier_{outlier_index:02d}_depth.pgm"
            write_contact_sheet(path, depth_arrays)
            outputs.append({"name": "depth", "path": path.as_posix()})
        confidence_arrays = [
            frame.confidence_preview for frame in selected if frame.confidence_preview is not None
        ]
        if confidence_arrays:
            path = out_dir / f"outlier_{outlier_index:02d}_confidence.pgm"
            write_contact_sheet(path, confidence_arrays)
            outputs.append({"name": "confidence", "path": path.as_posix()})
        refs_path = out_dir / f"outlier_{outlier_index:02d}_frame_refs.json"
        _write_json(
            refs_path,
            {
                "schema_version": "homebrain.geometry.pose_contact_refs.v0",
                "frame_refs": [
                    {
                        "frame_id": frame.frame_id,
                        "timestamp_ns": frame.timestamp_ns,
                        "source_data_ref": frame.source_record.get("source_data_ref"),
                    }
                    for frame in selected
                ],
            },
        )
        outputs.append({"name": "frame_refs", "path": refs_path.as_posix()})
        all_outputs.append(outputs)

    manifest = {
        "schema_version": "homebrain.geometry.pose_contact_sheets.v0",
        "contact_sheets": [item for group in all_outputs for item in group],
    }
    _write_json(out_dir / "contact_sheet_manifest.json", manifest)
    return all_outputs


def write_contact_sheet(path: Path, arrays: list[np.ndarray]) -> None:
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
    write_pgm(path, sheet)


def _preview_uint8(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim != 2:
        values = values.reshape(1, -1)
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


def write_pgm(path: Path, image: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    h, w = image.shape
    with target.open("wb") as handle:
        handle.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        handle.write(image.astype(np.uint8).tobytes(order="C"))


def _write_json(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(deterministic_json(data))
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect DA3 teacher pose deltas and pose jump windows.")
    parser.add_argument("--log", required=True, help="Input HomeBrain route log directory.")
    parser.add_argument("--teacher-artifacts", required=True, help="Input DA3 teacher artifact directory.")
    parser.add_argument("--out", required=True, help="Output diagnostics directory.")
    args = parser.parse_args(argv)
    pose_deltas, outlier_windows = inspect_pose_sequence(
        log_dir=args.log,
        teacher_artifacts_dir=args.teacher_artifacts,
        out_dir=args.out,
    )
    print(f"wrote pose deltas to {pose_deltas.as_posix()}")
    print(f"wrote outlier windows to {outlier_windows.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

