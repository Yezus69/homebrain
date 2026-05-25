from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.data.spatial_dataset import read_json, write_json
from homebrain.geometry.validate_bev import load_bev_manifest
from homebrain.messages.schema import JsonDict

ROBOT_FRAME_BEV_QA_SCHEMA_VERSION = "homebrain.geometry.robot_frame_bev_qa.v0"


def qa_robot_frame_bev(
    *,
    bev_dir: str | Path,
    spatial_pack: str | Path | None = None,
) -> JsonDict:
    root = Path(bev_dir)
    manifest = load_bev_manifest(root)
    frames = [frame for frame in manifest.get("frames", []) if isinstance(frame, dict)]
    frame_metadata = [_frame_metadata(root, frame) for frame in frames]
    calibration = _calibration_metrics(manifest, frame_metadata)
    pose = _pose_metrics(frame_metadata)
    depth = _depth_metrics(frame_metadata)
    density = _density_metrics(frame_metadata)
    temporal = _temporal_metrics(frame_metadata)
    leakage = _route_split_leakage_metrics(spatial_pack) if spatial_pack is not None else {
        "checked": False,
        "reason": "spatial_pack_not_supplied",
        "route_split_leakage_count": 0,
    }
    blockers = [
        *calibration["blockers"],
        *pose["blockers"],
        *depth["blockers"],
        *density["blockers"],
        *temporal["blockers"],
        *leakage.get("blockers", []),
    ]
    return {
        "schema_version": ROBOT_FRAME_BEV_QA_SCHEMA_VERSION,
        "bev_dir": root.as_posix(),
        "spatial_pack": Path(spatial_pack).as_posix() if spatial_pack is not None else None,
        "frame_count": len(frame_metadata),
        "calibration": calibration,
        "pose": pose,
        "depth": depth,
        "density": density,
        "temporal": temporal,
        "route_split_leakage": leakage,
        "qa_pass": not blockers,
        "blockers": blockers,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "product_training_approved": False,
    }


def _frame_metadata(root: Path, frame: JsonDict) -> JsonDict:
    metadata_path = frame.get("metadata_path")
    if not isinstance(metadata_path, str):
        return {"frame_id": frame.get("frame_id"), "metadata_missing": True}
    path = root / metadata_path
    if not path.exists():
        return {"frame_id": frame.get("frame_id"), "metadata_missing": True}
    data = read_json(path)
    return data if isinstance(data, dict) else {"frame_id": frame.get("frame_id"), "metadata_malformed": True}


def _calibration_metrics(manifest: JsonDict, frames: list[JsonDict]) -> JsonDict:
    assumed_count = sum(1 for frame in frames if bool(frame.get("assumed_extrinsics", False)))
    robot_truth_count = sum(1 for frame in frames if bool(frame.get("robot_frame_truth", False)))
    matrices = [frame.get("camera_to_base") for frame in frames if _valid_matrix(frame.get("camera_to_base"))]
    determinants = [_rotation_det(matrix) for matrix in matrices]
    determinant_bad_count = sum(1 for value in determinants if not 0.8 <= value <= 1.2)
    blockers: list[str] = []
    if manifest.get("robot_frame_truth") is not True:
        blockers.append("bev_manifest_not_robot_frame_truth")
    if assumed_count:
        blockers.append("assumed_extrinsics_present")
    if robot_truth_count != len(frames):
        blockers.append("not_all_frames_robot_frame_truth")
    if determinant_bad_count:
        blockers.append("camera_to_base_rotation_determinant_out_of_range")
    return {
        "robot_frame_truth_frame_count": robot_truth_count,
        "assumed_extrinsics_count": assumed_count,
        "camera_to_base_count": len(matrices),
        "rotation_determinant_min": min(determinants) if determinants else 0.0,
        "rotation_determinant_max": max(determinants) if determinants else 0.0,
        "blockers": blockers,
    }


def _pose_metrics(frames: list[JsonDict]) -> JsonDict:
    poses = [
        frame.get("source_base_pose")
        for frame in sorted(frames, key=lambda item: int(item.get("frame_id", 0)))
        if isinstance(frame.get("source_base_pose"), dict)
    ]
    translations: list[float] = []
    rotations_deg: list[float] = []
    for previous, current in zip(poses, poses[1:]):
        translations.append(_pose_translation_delta(previous, current))
        rotations_deg.append(abs(math.degrees(_wrap_angle(_pose_yaw(current) - _pose_yaw(previous)))))
    translation_threshold = max(1.0, _median(translations) + 6.0 * max(_mad(translations), 1.0e-6))
    rotation_threshold = max(45.0, _median(rotations_deg) + 6.0 * max(_mad(rotations_deg), 1.0e-6))
    translation_jump_count = sum(1 for value in translations if value > translation_threshold)
    rotation_jump_count = sum(1 for value in rotations_deg if value > rotation_threshold)
    blockers = []
    if not poses:
        blockers.append("pose_missing")
    if translation_jump_count:
        blockers.append("pose_translation_discontinuities_present")
    if rotation_jump_count:
        blockers.append("pose_rotation_discontinuities_present")
    return {
        "pose_frame_count": len(poses),
        "translation_delta_median_m": _median(translations),
        "translation_delta_max_m": max(translations) if translations else 0.0,
        "translation_jump_threshold_m": translation_threshold,
        "translation_jump_count": translation_jump_count,
        "rotation_delta_median_deg": _median(rotations_deg),
        "rotation_delta_max_deg": max(rotations_deg) if rotations_deg else 0.0,
        "rotation_jump_threshold_deg": rotation_threshold,
        "rotation_jump_count": rotation_jump_count,
        "blockers": blockers,
    }


def _depth_metrics(frames: list[JsonDict]) -> JsonDict:
    valid_ratios: list[float] = []
    observed_ratios: list[float] = []
    for frame in frames:
        stats = frame.get("stats") if isinstance(frame.get("stats"), dict) else {}
        sampled = float(stats.get("sampled_pixel_count", 0.0) or 0.0)
        valid = float(stats.get("valid_depth_point_count", 0.0) or 0.0)
        observed = float(stats.get("observed_cell_count", 0.0) or 0.0)
        grid_shape = frame.get("grid_shape")
        cells = float(grid_shape[0] * grid_shape[1]) if isinstance(grid_shape, list) and len(grid_shape) == 2 else 0.0
        if sampled > 0.0:
            valid_ratios.append(valid / sampled)
        if cells > 0.0:
            observed_ratios.append(observed / cells)
    blockers = []
    if _mean(valid_ratios) < 0.20:
        blockers.append("depth_valid_ratio_mean_below_0.20")
    if _mean(observed_ratios) < 0.05:
        blockers.append("observed_cell_ratio_mean_below_0.05")
    return {
        "depth_valid_ratio_mean": _mean(valid_ratios),
        "depth_valid_ratio_min": min(valid_ratios) if valid_ratios else 0.0,
        "observed_cell_ratio_mean": _mean(observed_ratios),
        "observed_cell_ratio_min": min(observed_ratios) if observed_ratios else 0.0,
        "blockers": blockers,
    }


def _density_metrics(frames: list[JsonDict]) -> JsonDict:
    obstacle = [_stat(frame, "obstacle_ratio") for frame in frames]
    free = [_stat(frame, "free_ratio") for frame in frames]
    unknown = [_stat(frame, "unknown_ratio") for frame in frames]
    blockers = []
    if _mean(obstacle) > 0.75:
        blockers.append("obstacle_ratio_mean_above_0.75")
    if max(obstacle, default=0.0) > 0.95:
        blockers.append("obstacle_ratio_max_above_0.95")
    if _mean(free) <= 0.0 and _mean(obstacle) <= 0.0:
        blockers.append("free_and_obstacle_labels_empty")
    return {
        "free_ratio_mean": _mean(free),
        "obstacle_ratio_mean": _mean(obstacle),
        "obstacle_ratio_max": max(obstacle) if obstacle else 0.0,
        "unknown_ratio_mean": _mean(unknown),
        "blockers": blockers,
    }


def _temporal_metrics(frames: list[JsonDict]) -> JsonDict:
    frame_stats = sorted(frames, key=lambda item: int(item.get("frame_id", 0)))
    visible_jitter: list[float] = []
    for previous, current in zip(frame_stats, frame_stats[1:]):
        prev_free = _stat(previous, "free_ratio")
        prev_obstacle = _stat(previous, "obstacle_ratio")
        curr_free = _stat(current, "free_ratio")
        curr_obstacle = _stat(current, "obstacle_ratio")
        visible_jitter.append(abs(curr_free - prev_free) + abs(curr_obstacle - prev_obstacle))
    blockers = []
    if _mean(visible_jitter) > 0.30:
        blockers.append("temporal_density_jitter_mean_above_0.30")
    if max(visible_jitter, default=0.0) > 0.75:
        blockers.append("temporal_density_jitter_max_above_0.75")
    return {
        "temporal_density_jitter_mean": _mean(visible_jitter),
        "temporal_density_jitter_p95": _percentile(visible_jitter, 95),
        "temporal_density_jitter_max": max(visible_jitter) if visible_jitter else 0.0,
        "blockers": blockers,
    }


def _route_split_leakage_metrics(spatial_pack: str | Path) -> JsonDict:
    root = Path(spatial_pack)
    manifest = read_json(root / "manifest.json")
    records = [record for record in manifest.get("examples", []) if isinstance(record, dict)]
    records.sort(key=lambda item: (str(item.get("sequence_id")), str(item.get("camera_id")), int(item.get("frame_id", 0))))
    leaks: list[JsonDict] = []
    for previous, current in zip(records, records[1:]):
        if str(previous.get("sequence_id")) != str(current.get("sequence_id")):
            continue
        if str(previous.get("camera_id")) != str(current.get("camera_id")):
            continue
        if int(current.get("frame_id", 0)) != int(previous.get("frame_id", 0)) + 1:
            continue
        pair = {str(previous.get("split")), str(current.get("split"))}
        if pair == {"train", "val"}:
            leaks.append(
                {
                    "previous_frame_id": previous.get("frame_id"),
                    "current_frame_id": current.get("frame_id"),
                    "sequence_id": current.get("sequence_id"),
                    "camera_id": current.get("camera_id"),
                }
            )
    blockers = ["train_val_adjacent_route_split_leakage"] if leaks else []
    return {
        "checked": True,
        "spatial_pack": root.as_posix(),
        "route_split_leakage_count": len(leaks),
        "examples": leaks[:20],
        "blockers": blockers,
    }


def _stat(frame: JsonDict, key: str) -> float:
    stats = frame.get("stats") if isinstance(frame.get("stats"), dict) else {}
    value = stats.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _valid_matrix(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in value)
    )


def _rotation_det(matrix: object) -> float:
    if not _valid_matrix(matrix):
        return 0.0
    values = np.asarray(matrix, dtype=np.float64)
    return float(np.linalg.det(values[:3, :3]))


def _pose_translation_delta(previous: JsonDict, current: JsonDict) -> float:
    return float(
        math.sqrt(
            (float(current["tx"]) - float(previous["tx"])) ** 2
            + (float(current["ty"]) - float(previous["ty"])) ** 2
            + (float(current["tz"]) - float(previous["tz"])) ** 2
        )
    )


def _pose_yaw(pose: JsonDict) -> float:
    qx = float(pose["qx"])
    qy = float(pose["qy"])
    qz = float(pose["qz"])
    qw = float(pose["qw"])
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _wrap_angle(value: float) -> float:
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def _percentile(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QA robot-frame RGB-D BEV artifacts from real route data.")
    parser.add_argument("--bev", required=True, help="Input robot RGB-D BEV directory.")
    parser.add_argument("--spatial-pack", default=None, help="Optional SpatialTrainPack for split leakage checks.")
    parser.add_argument("--out", required=True, help="Output QA JSON path.")
    args = parser.parse_args(argv)
    metrics = qa_robot_frame_bev(bev_dir=args.bev, spatial_pack=args.spatial_pack)
    write_json(args.out, metrics, pretty=True)
    print(json.dumps(metrics, sort_keys=True))
    return 0 if metrics["qa_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
