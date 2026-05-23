from __future__ import annotations

import argparse
from pathlib import Path

from homebrain.geometry.pose_sequence import (
    STABLE_WINDOWS_SCHEMA_VERSION,
    compute_pose_deltas,
    load_pose_frame_metrics,
    pose_delta_to_json,
    select_stable_windows,
)
from homebrain.messages.schema import JsonDict, deterministic_json


def select_route_stable_windows(
    *,
    log_dir: str | Path,
    teacher_artifacts_dir: str | Path,
    out_path: str | Path,
    min_window: int = 20,
) -> Path:
    if min_window < 1:
        raise ValueError("min-window must be at least 1")
    manifest, frames = load_pose_frame_metrics(log_dir=log_dir, teacher_artifacts_dir=teacher_artifacts_dir)
    deltas, thresholds = compute_pose_deltas(frames)
    ranked, accepted, excluded_indices = select_stable_windows(frames, deltas, min_window=min_window)
    outlier_deltas = [delta for delta in deltas if delta.is_outlier]
    excluded_frame_ids = [frames[index].frame_id for index in excluded_indices if 0 <= index < len(frames)]

    report: JsonDict = {
        "schema_version": STABLE_WINDOWS_SCHEMA_VERSION,
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
        "min_window": min_window,
        "frame_count": len(frames),
        "pose_jump_outlier_count": len(outlier_deltas),
        "pose_outlier_deltas": [pose_delta_to_json(delta) for delta in outlier_deltas],
        "excluded_frame_indices": excluded_indices,
        "excluded_frame_ids": excluded_frame_ids,
        **thresholds,
        "window_count": len(ranked),
        "accepted_window_count": len(accepted),
        "accepted_windows": accepted,
        "ranked_windows": ranked,
        "selection_note": "Accepted windows exclude pose-jump target frames and remain offline geometry-pretrain candidates only.",
    }
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(deterministic_json(report))
        handle.write("\n")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select stable DA3 pose/depth windows for weak geometry labels.")
    parser.add_argument("--log", required=True, help="Input HomeBrain route log directory.")
    parser.add_argument("--teacher-artifacts", required=True, help="Input DA3 teacher artifact directory.")
    parser.add_argument("--out", required=True, help="Output stable window JSON path.")
    parser.add_argument("--min-window", type=int, default=20, help="Minimum contiguous accepted window length.")
    args = parser.parse_args(argv)
    path = select_route_stable_windows(
        log_dir=args.log,
        teacher_artifacts_dir=args.teacher_artifacts,
        out_path=args.out,
        min_window=args.min_window,
    )
    print(f"wrote stable windows to {path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

