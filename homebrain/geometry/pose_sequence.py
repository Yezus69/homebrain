from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from homebrain.messages.schema import FrameEvent, JsonDict
from homebrain.replay.replayd import order_events_for_replay
from homebrain.replay.segment_log import read_events
from homebrain.teachers.artifacts import frame_key, load_array, load_teacher_manifest

POSE_SEQUENCE_SCHEMA_VERSION = "homebrain.geometry.pose_sequence.v0"
STABLE_WINDOWS_SCHEMA_VERSION = "homebrain.geometry.stable_windows.v0"


@dataclass(frozen=True)
class PoseFrameMetric:
    index: int
    sequence_id: str
    camera_id: str
    frame_id: int
    timestamp_ns: int
    key: str
    route_present: bool
    source_record: JsonDict
    depth_valid_ratio: float
    confidence_mean: float
    intrinsics_valid: bool
    pose_valid: bool
    translation: np.ndarray | None
    rotation: np.ndarray | None
    normalized_depth_preview: np.ndarray | None
    confidence_preview: np.ndarray | None
    floor_plane_found: bool
    floor_median: float | None


@dataclass(frozen=True)
class PoseDeltaMetric:
    index: int
    previous_index: int
    current_index: int
    previous_frame_id: int
    current_frame_id: int
    previous_timestamp_ns: int
    current_timestamp_ns: int
    time_delta_ns: int
    translation_delta: float | None
    rotation_delta_deg: float | None
    translation_outlier: bool
    rotation_outlier: bool

    @property
    def is_outlier(self) -> bool:
        return self.translation_outlier or self.rotation_outlier


def load_pose_frame_metrics(
    *,
    log_dir: str | Path,
    teacher_artifacts_dir: str | Path,
    preview_max_side: int = 96,
) -> tuple[JsonDict, list[PoseFrameMetric]]:
    log_root = Path(log_dir)
    teacher_root = Path(teacher_artifacts_dir)
    manifest = load_teacher_manifest(teacher_root)
    route_keys = _route_frame_keys(log_root)
    frames: list[PoseFrameMetric] = []

    for index, frame_record in enumerate(manifest.get("frames", [])):
        if not isinstance(frame_record, dict):
            continue
        artifacts = frame_record.get("artifacts") if isinstance(frame_record.get("artifacts"), dict) else {}
        if not isinstance(artifacts, dict):
            artifacts = {}

        depth = _load_first_array(teacher_root, artifacts, ("depth", "depth_relative", "depth_m"))
        depth_valid_ratio = 0.0
        normalized_depth_preview: np.ndarray | None = None
        floor_found = False
        floor_median: float | None = None
        if depth is not None:
            depth_values = np.asarray(depth, dtype=np.float32)
            valid = np.isfinite(depth_values) & (depth_values > np.float32(0.0))
            depth_valid_ratio = float(np.count_nonzero(valid) / max(depth_values.size, 1))
            normalized = normalized_depth(depth_values)
            normalized_depth_preview = downsample_nearest(normalized, max_side=preview_max_side)
            valid_preview = np.isfinite(normalized_depth_preview)
            floor_found, floor_median = floor_proxy(normalized_depth_preview, valid_preview)

        confidence = _load_first_array(teacher_root, artifacts, ("confidence", "depth_confidence"))
        confidence_mean = 0.0
        confidence_preview: np.ndarray | None = None
        if confidence is not None:
            conf = np.asarray(confidence, dtype=np.float32)
            finite = conf[np.isfinite(conf)]
            confidence_mean = float(np.mean(finite)) if finite.size else 0.0
            confidence_preview = downsample_nearest(conf, max_side=preview_max_side)

        intrinsics = _load_first_array(teacher_root, artifacts, ("intrinsics",))
        intrinsics_valid = intrinsics is not None and valid_intrinsics(intrinsics)

        pose = _load_first_array(teacher_root, artifacts, ("extrinsics", "camera_pose"))
        pose_valid = pose is not None and valid_pose(pose)
        translation = pose_translation(pose) if pose_valid and pose is not None else None
        rotation = pose_rotation(pose) if pose_valid and pose is not None else None

        frames.append(
            PoseFrameMetric(
                index=index,
                sequence_id=str(frame_record.get("sequence_id")),
                camera_id=str(frame_record.get("camera_id")),
                frame_id=int(frame_record.get("frame_id", index)),
                timestamp_ns=int(frame_record.get("timestamp_ns", 0)),
                key=frame_key(frame_record),
                route_present=frame_key(frame_record) in route_keys,
                source_record=frame_record,
                depth_valid_ratio=depth_valid_ratio,
                confidence_mean=confidence_mean,
                intrinsics_valid=intrinsics_valid,
                pose_valid=pose_valid,
                translation=translation,
                rotation=rotation,
                normalized_depth_preview=normalized_depth_preview,
                confidence_preview=confidence_preview,
                floor_plane_found=floor_found,
                floor_median=floor_median,
            )
        )
    return manifest, frames


def compute_pose_deltas(frames: list[PoseFrameMetric]) -> tuple[list[PoseDeltaMetric], JsonDict]:
    raw_translation: list[float] = []
    raw_rotation: list[float] = []
    pairs: list[tuple[int, int, float | None, float | None]] = []

    for current_index, (previous, current) in enumerate(zip(frames, frames[1:]), start=1):
        translation_delta: float | None = None
        rotation_delta: float | None = None
        if previous.translation is not None and current.translation is not None:
            translation_delta = float(np.linalg.norm(current.translation - previous.translation))
            raw_translation.append(translation_delta)
        if previous.rotation is not None and current.rotation is not None:
            rotation_delta = rotation_angle_deg(previous.rotation, current.rotation)
            raw_rotation.append(rotation_delta)
        pairs.append((current_index - 1, current_index, translation_delta, rotation_delta))

    translation_threshold = robust_outlier_threshold(raw_translation, floor=0.25)
    rotation_threshold = robust_outlier_threshold(raw_rotation, floor=20.0)
    deltas: list[PoseDeltaMetric] = []
    for index, (previous_index, current_index, translation_delta, rotation_delta) in enumerate(pairs):
        previous = frames[previous_index]
        current = frames[current_index]
        deltas.append(
            PoseDeltaMetric(
                index=index,
                previous_index=previous_index,
                current_index=current_index,
                previous_frame_id=previous.frame_id,
                current_frame_id=current.frame_id,
                previous_timestamp_ns=previous.timestamp_ns,
                current_timestamp_ns=current.timestamp_ns,
                time_delta_ns=current.timestamp_ns - previous.timestamp_ns,
                translation_delta=translation_delta,
                rotation_delta_deg=rotation_delta,
                translation_outlier=(
                    translation_delta is not None and translation_delta > translation_threshold
                ),
                rotation_outlier=rotation_delta is not None and rotation_delta > rotation_threshold,
            )
        )

    thresholds: JsonDict = {
        "translation_jump_threshold": translation_threshold,
        "rotation_jump_threshold_deg": rotation_threshold,
        "translation_jump_median": median(raw_translation),
        "translation_jump_mad": mad(raw_translation),
        "rotation_jump_median_deg": median(raw_rotation),
        "rotation_jump_mad_deg": mad(raw_rotation),
    }
    return deltas, thresholds


def build_outlier_windows(
    frames: list[PoseFrameMetric],
    deltas: list[PoseDeltaMetric],
    *,
    radius: int = 3,
) -> list[JsonDict]:
    windows: list[JsonDict] = []
    for outlier_index, delta in enumerate(delta for delta in deltas if delta.is_outlier):
        start = max(0, delta.previous_index - radius)
        end = min(len(frames) - 1, delta.current_index + radius)
        window_frames = frames[start : end + 1]
        windows.append(
            {
                "outlier_index": outlier_index,
                "start_index": start,
                "end_index": end,
                "start_frame_id": frames[start].frame_id,
                "end_frame_id": frames[end].frame_id,
                "frame_ids": [frame.frame_id for frame in window_frames],
                "timestamps_ns": [frame.timestamp_ns for frame in window_frames],
                "outlier_delta": pose_delta_to_json(delta),
            }
        )
    return windows


def select_stable_windows(
    frames: list[PoseFrameMetric],
    deltas: list[PoseDeltaMetric],
    *,
    min_window: int,
) -> tuple[list[JsonDict], list[JsonDict], list[int]]:
    excluded = excluded_frame_indices(frames, deltas)
    candidates = contiguous_index_spans(len(frames), excluded)
    ranked: list[JsonDict] = []
    for candidate_index, (start, end) in enumerate(candidates):
        length = end - start + 1
        if length < min_window:
            metrics = window_metrics(frames, start, end)
            reasons = [f"frame_count_below_{min_window}"]
            accepted = False
        else:
            metrics = window_metrics(frames, start, end)
            reasons = stable_window_quarantine_reasons(metrics)
            accepted = not reasons
        score = stable_window_score(metrics)
        record: JsonDict = {
            "window_id": f"window_{start:06d}_{end:06d}",
            "rank": None,
            "candidate_index": candidate_index,
            "accepted": accepted,
            "start_index": start,
            "end_index": end,
            "start_frame_id": frames[start].frame_id,
            "end_frame_id": frames[end].frame_id,
            "start_timestamp_ns": frames[start].timestamp_ns,
            "end_timestamp_ns": frames[end].timestamp_ns,
            "frame_count": length,
            "frame_ids": [frame.frame_id for frame in frames[start : end + 1]],
            "frame_keys": [frame.key for frame in frames[start : end + 1]],
            "score": score,
            "metrics": metrics,
            "quarantine_reasons": reasons,
            "control_safe": False,
        }
        ranked.append(record)

    ranked.sort(key=lambda item: (-float(item["score"]), -int(item["frame_count"]), int(item["start_index"])))
    for rank, record in enumerate(ranked, start=1):
        record["rank"] = rank
    accepted_windows = [record for record in ranked if record.get("accepted") is True]
    return ranked, accepted_windows, sorted(excluded)


def excluded_frame_indices(frames: list[PoseFrameMetric], deltas: list[PoseDeltaMetric]) -> set[int]:
    excluded = {frame.index for frame in frames if not frame.pose_valid or not frame.route_present}
    for delta in deltas:
        if delta.is_outlier:
            excluded.add(delta.current_index)
    return excluded


def contiguous_index_spans(frame_count: int, excluded: set[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(frame_count):
        if index in excluded:
            if start is not None:
                spans.append((start, index - 1))
                start = None
            continue
        if start is None:
            start = index
    if start is not None:
        spans.append((start, frame_count - 1))
    return spans


def window_metrics(frames: list[PoseFrameMetric], start: int, end: int) -> JsonDict:
    selected = frames[start : end + 1]
    depths = [frame.normalized_depth_preview for frame in selected if frame.normalized_depth_preview is not None]
    floor_medians = [frame.floor_median for frame in selected if frame.floor_median is not None]
    return {
        "depth_valid_ratio": mean([frame.depth_valid_ratio for frame in selected]),
        "confidence_mean": mean([frame.confidence_mean for frame in selected]),
        "pose_valid_ratio": mean([1.0 if frame.pose_valid else 0.0 for frame in selected]),
        "intrinsics_valid_ratio": mean([1.0 if frame.intrinsics_valid else 0.0 for frame in selected]),
        "route_present_ratio": mean([1.0 if frame.route_present else 0.0 for frame in selected]),
        "temporal_depth_consistency": temporal_depth_consistency(depths),
        "floor_plane_found_ratio": mean([1.0 if frame.floor_plane_found else 0.0 for frame in selected]),
        "floor_plane_stability": floor_stability(floor_medians),
    }


def stable_window_quarantine_reasons(metrics: JsonDict) -> list[str]:
    reasons: list[str] = []
    if float(metrics.get("depth_valid_ratio", 0.0)) < 0.95:
        reasons.append("depth_valid_ratio_below_0.95")
    if float(metrics.get("pose_valid_ratio", 0.0)) < 0.95:
        reasons.append("pose_valid_ratio_below_0.95")
    if float(metrics.get("route_present_ratio", 0.0)) < 1.0:
        reasons.append("route_present_ratio_below_1.0")
    if float(metrics.get("temporal_depth_consistency", 0.0)) < 0.70:
        reasons.append("temporal_depth_consistency_below_0.70")
    if float(metrics.get("floor_plane_found_ratio", 0.0)) < 0.50:
        reasons.append("floor_plane_found_ratio_below_0.50")
    if float(metrics.get("floor_plane_stability", 0.0)) < 0.50:
        reasons.append("floor_plane_stability_below_0.50")
    return reasons


def stable_window_score(metrics: JsonDict) -> float:
    keys = (
        "depth_valid_ratio",
        "pose_valid_ratio",
        "temporal_depth_consistency",
        "floor_plane_found_ratio",
        "floor_plane_stability",
    )
    return mean([float(metrics.get(key, 0.0)) for key in keys])


def pose_delta_to_json(delta: PoseDeltaMetric) -> JsonDict:
    return {
        "index": delta.index,
        "previous_index": delta.previous_index,
        "current_index": delta.current_index,
        "previous_frame_id": delta.previous_frame_id,
        "current_frame_id": delta.current_frame_id,
        "previous_timestamp_ns": delta.previous_timestamp_ns,
        "current_timestamp_ns": delta.current_timestamp_ns,
        "time_delta_ns": delta.time_delta_ns,
        "translation_delta": delta.translation_delta,
        "rotation_delta_deg": delta.rotation_delta_deg,
        "translation_outlier": delta.translation_outlier,
        "rotation_outlier": delta.rotation_outlier,
        "is_outlier": delta.is_outlier,
    }


def frame_metric_to_json(frame: PoseFrameMetric) -> JsonDict:
    return {
        "index": frame.index,
        "sequence_id": frame.sequence_id,
        "camera_id": frame.camera_id,
        "frame_id": frame.frame_id,
        "timestamp_ns": frame.timestamp_ns,
        "frame_key": frame.key,
        "route_present": frame.route_present,
        "depth_valid_ratio": frame.depth_valid_ratio,
        "confidence_mean": frame.confidence_mean,
        "intrinsics_valid": frame.intrinsics_valid,
        "pose_valid": frame.pose_valid,
        "translation_xyz": frame.translation.tolist() if frame.translation is not None else None,
        "floor_plane_found": frame.floor_plane_found,
        "floor_median": frame.floor_median,
    }


def _route_frame_keys(log_dir: Path) -> set[str]:
    events = order_events_for_replay(read_events(log_dir))
    return {frame_key(event) for event in events if isinstance(event, FrameEvent)}


def _load_first_array(root: Path, artifacts: JsonDict, kinds: tuple[str, ...]) -> np.ndarray | None:
    for kind in kinds:
        record = artifacts.get(kind)
        if isinstance(record, dict) and isinstance(record.get("path"), str):
            path = root / str(record["path"])
            if path.exists():
                return load_array(path)
    return None


def valid_intrinsics(array: np.ndarray) -> bool:
    values = np.asarray(array, dtype=np.float32)
    return values.shape == (3, 3) and bool(np.all(np.isfinite(values))) and float(values[2, 2]) != 0.0


def valid_pose(array: np.ndarray) -> bool:
    values = np.asarray(array, dtype=np.float32)
    return values.shape in {(3, 4), (4, 4)} and bool(np.all(np.isfinite(values)))


def pose_translation(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    return values[:3, 3].astype(np.float32)


def pose_rotation(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    return values[:3, :3].astype(np.float32)


def rotation_angle_deg(previous: np.ndarray, current: np.ndarray) -> float:
    relative = np.asarray(current, dtype=np.float64) @ np.asarray(previous, dtype=np.float64).T
    trace_value = float(np.trace(relative))
    cosine = max(-1.0, min(1.0, (trace_value - 1.0) / 2.0))
    return float(np.degrees(np.arccos(cosine)))


def robust_outlier_threshold(values: list[float], *, floor: float) -> float:
    if not values:
        return float(floor)
    value_median = median(values)
    value_mad = mad(values)
    return float(max(floor, value_median + 6.0 * max(value_mad, 1.0e-6)))


def normalized_depth(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > np.float32(0.0))
    if not np.any(valid):
        return np.full(values.shape, np.nan, dtype=np.float32)
    scale = max(float(np.nanmedian(values[valid])), 1.0e-6)
    return np.where(valid, values / np.float32(scale), np.nan).astype(np.float32)


def downsample_nearest(array: np.ndarray, *, max_side: int) -> np.ndarray:
    values = np.asarray(array)
    values = np.squeeze(values)
    if values.ndim != 2:
        values = values.reshape(1, -1)
    stride = max(1, int(np.ceil(max(values.shape) / max_side)))
    return values[::stride, ::stride]


def floor_proxy(normalized_depth: np.ndarray, valid: np.ndarray) -> tuple[bool, float | None]:
    if normalized_depth.ndim != 2 or normalized_depth.size == 0:
        return False, None
    start = int(normalized_depth.shape[0] * 0.55)
    bottom = normalized_depth[start:, :]
    bottom_valid = np.isfinite(bottom) & valid[start:, :]
    if not np.any(bottom_valid):
        return False, None
    valid_ratio = float(np.count_nonzero(bottom_valid) / max(bottom_valid.size, 1))
    median_value = float(np.nanmedian(bottom[bottom_valid]))
    if bottom.shape[0] >= 2:
        row_profile = np.nanmedian(np.where(bottom_valid, bottom, np.nan), axis=1)
        gradient = (
            np.nanmedian(np.diff(row_profile))
            if np.count_nonzero(np.isfinite(row_profile)) >= 2
            else 0.0
        )
    else:
        gradient = 0.0
    found = valid_ratio >= 0.5 and np.isfinite(median_value) and gradient >= -0.05
    return bool(found), median_value


def temporal_depth_consistency(depths: list[np.ndarray | None]) -> float:
    usable = [depth for depth in depths if depth is not None]
    if len(usable) < 2:
        return 0.0
    scores: list[float] = []
    for previous, current in zip(usable, usable[1:]):
        if previous.shape != current.shape:
            continue
        valid = np.isfinite(previous) & np.isfinite(current)
        if not np.any(valid):
            continue
        mae = float(np.mean(np.abs(previous[valid] - current[valid])))
        scores.append(max(0.0, 1.0 - min(mae, 1.0)))
    return mean(scores)


def floor_stability(medians: list[float]) -> float:
    if len(medians) < 2:
        return 0.0
    values = np.asarray(medians, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    return float(max(0.0, 1.0 - min(float(np.std(values)), 1.0)))


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def mad(values: list[float]) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    value_median = float(np.median(array))
    return float(np.median(np.abs(array - value_median)))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
