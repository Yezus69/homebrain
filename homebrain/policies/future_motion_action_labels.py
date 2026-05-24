from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from homebrain.messages.schema import JsonDict, OdomEvent, PoseEvent
from homebrain.policies.candidate_trajectories import CandidateTrajectory, Pose2D


DEFAULT_YAW_WEIGHT_M_PER_RAD = 0.5
DEFAULT_STATIONARY_TRANSLATION_M = 0.03
DEFAULT_STATIONARY_YAW_RAD = 0.05


@dataclass(frozen=True)
class TimedPose2D:
    timestamp_ns: int
    frame_id: int
    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class CandidateMatchDistance:
    candidate_id: str
    total_distance: float
    position_l2_m: float
    yaw_l1_rad: float

    def to_dict(self) -> JsonDict:
        return {
            "candidate_id": self.candidate_id,
            "total_distance": round(float(self.total_distance), 6),
            "position_l2_m": round(float(self.position_l2_m), 6),
            "yaw_l1_rad": round(float(self.yaw_l1_rad), 6),
        }


@dataclass(frozen=True)
class FutureMotionActionLabel:
    candidate_id: str
    bc_label_valid: bool
    invalid_reason: str | None
    bc_label_confidence: float
    best_distance: float
    second_best_distance: float
    margin: float
    horizon_s: float
    target_frame_id: int
    dx_m: float
    dy_m: float
    dyaw_rad: float
    top3: tuple[CandidateMatchDistance, ...]

    def to_dict(self) -> JsonDict:
        return {
            "candidate_id": self.candidate_id,
            "bc_label_valid": bool(self.bc_label_valid),
            "invalid_reason": self.invalid_reason,
            "bc_label_confidence": round(float(self.bc_label_confidence), 6),
            "best_distance": round(float(self.best_distance), 6) if math.isfinite(self.best_distance) else "inf",
            "second_best_distance": round(float(self.second_best_distance), 6)
            if math.isfinite(self.second_best_distance)
            else "inf",
            "margin": round(float(self.margin), 6) if math.isfinite(self.margin) else 0.0,
            "horizon_s": round(float(self.horizon_s), 6),
            "target_frame_id": int(self.target_frame_id),
            "dx_m": round(float(self.dx_m), 6),
            "dy_m": round(float(self.dy_m), 6),
            "dyaw_rad": round(float(self.dyaw_rad), 6),
            "top3": [item.to_dict() for item in self.top3],
        }


def pose_sample_from_event(event: PoseEvent | OdomEvent, *, frame_id: int = -1) -> TimedPose2D:
    return TimedPose2D(
        timestamp_ns=int(event.timestamp_ns),
        frame_id=int(frame_id),
        x_m=float(event.position_m[0]),
        y_m=float(event.position_m[1]),
        yaw_rad=_yaw_from_xyzw(event.orientation_xyzw),
    )


def pose_sample_from_base_pose(
    pose: JsonDict | None,
    *,
    timestamp_ns: int,
    frame_id: int,
) -> TimedPose2D | None:
    if not isinstance(pose, dict):
        return None
    try:
        xyzw = (
            float(pose["qx"]),
            float(pose["qy"]),
            float(pose["qz"]),
            float(pose["qw"]),
        )
        return TimedPose2D(
            timestamp_ns=int(timestamp_ns),
            frame_id=int(frame_id),
            x_m=float(pose["tx"]),
            y_m=float(pose["ty"]),
            yaw_rad=_yaw_from_xyzw(xyzw),
        )
    except (KeyError, TypeError, ValueError):
        return None


def match_future_motion_to_candidate(
    *,
    sequence: Iterable[TimedPose2D | None],
    current_frame_id: int,
    current_timestamp_ns: int,
    candidates: list[CandidateTrajectory],
    horizon_s: float | None = None,
    yaw_weight_m_per_rad: float = DEFAULT_YAW_WEIGHT_M_PER_RAD,
    stationary_translation_m: float = DEFAULT_STATIONARY_TRANSLATION_M,
    stationary_yaw_rad: float = DEFAULT_STATIONARY_YAW_RAD,
) -> FutureMotionActionLabel:
    if not candidates:
        raise ValueError("at least one candidate trajectory is required")
    if yaw_weight_m_per_rad < 0.0:
        raise ValueError("yaw_weight_m_per_rad must be non-negative")
    ordered = sorted((item for item in sequence if item is not None), key=lambda item: item.timestamp_ns)
    if horizon_s is None:
        horizon_s = max(float(candidate.duration_s) for candidate in candidates)
    if horizon_s <= 0.0:
        raise ValueError("horizon_s must be positive")
    current = _find_pose_at_or_before(ordered, current_timestamp_ns, current_frame_id)
    if current is None:
        return _invalid_label("pose_missing", horizon_s=horizon_s)
    target_ns = int(current_timestamp_ns + round(horizon_s * 1_000_000_000))
    if not ordered or ordered[-1].timestamp_ns < target_ns:
        return _invalid_label("future_horizon_truncated", horizon_s=horizon_s)

    sample_count = max(2, max(len(candidate.poses) for candidate in candidates))
    time_grid = np.linspace(0.0, float(horizon_s), sample_count, dtype=np.float64)
    actual = _actual_relative_trajectory(
        ordered,
        current=current,
        time_grid=time_grid,
        current_timestamp_ns=current_timestamp_ns,
    )
    if actual is None:
        return _invalid_label("pose_missing", horizon_s=horizon_s)
    terminal = actual[-1]
    terminal_motion = math.hypot(terminal.x_m, terminal.y_m)
    terminal_yaw = abs(_wrap_angle(terminal.yaw_rad))
    if terminal_motion < stationary_translation_m and terminal_yaw < stationary_yaw_rad:
        return _invalid_label(
            "stationary_below_threshold",
            horizon_s=horizon_s,
            target_frame_id=_target_frame_id(ordered, target_ns),
            dx_m=terminal.x_m,
            dy_m=terminal.y_m,
            dyaw_rad=terminal.yaw_rad,
        )

    distances = tuple(
        _candidate_distance(
            candidate,
            actual=actual,
            time_grid=time_grid,
            yaw_weight_m_per_rad=yaw_weight_m_per_rad,
        )
        for candidate in candidates
    )
    ranked = tuple(sorted(distances, key=lambda item: (item.total_distance, _candidate_index(candidates, item.candidate_id))))
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else best
    margin = max(0.0, float(second.total_distance - best.total_distance))
    confidence = margin / (margin + float(best.total_distance) + 1.0e-6)
    return FutureMotionActionLabel(
        candidate_id=best.candidate_id,
        bc_label_valid=True,
        invalid_reason=None,
        bc_label_confidence=float(max(0.0, min(1.0, confidence))),
        best_distance=float(best.total_distance),
        second_best_distance=float(second.total_distance),
        margin=float(margin),
        horizon_s=float(horizon_s),
        target_frame_id=_target_frame_id(ordered, target_ns),
        dx_m=float(terminal.x_m),
        dy_m=float(terminal.y_m),
        dyaw_rad=float(terminal.yaw_rad),
        top3=ranked[:3],
    )


def _actual_relative_trajectory(
    poses: list[TimedPose2D],
    *,
    current: TimedPose2D,
    time_grid: np.ndarray,
    current_timestamp_ns: int,
) -> tuple[Pose2D, ...] | None:
    result: list[Pose2D] = []
    for offset_s in time_grid.tolist():
        sample = _interpolate_pose(poses, current_timestamp_ns + int(round(float(offset_s) * 1_000_000_000)))
        if sample is None:
            return None
        result.append(_relative_pose(current, sample))
    return tuple(result)


def _candidate_distance(
    candidate: CandidateTrajectory,
    *,
    actual: tuple[Pose2D, ...],
    time_grid: np.ndarray,
    yaw_weight_m_per_rad: float,
) -> CandidateMatchDistance:
    candidate_poses = [_candidate_pose_at(candidate, float(offset_s)) for offset_s in time_grid.tolist()]
    position_errors = [
        math.hypot(float(actual_pose.x_m - candidate_pose.x_m), float(actual_pose.y_m - candidate_pose.y_m))
        for actual_pose, candidate_pose in zip(actual, candidate_poses)
    ]
    yaw_errors = [
        abs(_wrap_angle(float(actual_pose.yaw_rad - candidate_pose.yaw_rad)))
        for actual_pose, candidate_pose in zip(actual, candidate_poses)
    ]
    position_l2 = float(sum(position_errors) / max(len(position_errors), 1))
    yaw_l1 = float(sum(yaw_errors) / max(len(yaw_errors), 1))
    return CandidateMatchDistance(
        candidate_id=candidate.id,
        total_distance=float(position_l2 + yaw_weight_m_per_rad * yaw_l1),
        position_l2_m=position_l2,
        yaw_l1_rad=yaw_l1,
    )


def _candidate_pose_at(candidate: CandidateTrajectory, t_s: float) -> Pose2D:
    if not candidate.poses:
        raise ValueError(f"candidate {candidate.id} has no integrated poses")
    if t_s <= 0.0 or candidate.duration_s <= 0.0:
        return candidate.poses[0]
    if t_s >= candidate.duration_s:
        return candidate.poses[-1]
    scaled = (t_s / candidate.duration_s) * (len(candidate.poses) - 1)
    left_index = int(math.floor(scaled))
    right_index = min(left_index + 1, len(candidate.poses) - 1)
    alpha = float(scaled - left_index)
    left = candidate.poses[left_index]
    right = candidate.poses[right_index]
    return Pose2D(
        x_m=float((1.0 - alpha) * left.x_m + alpha * right.x_m),
        y_m=float((1.0 - alpha) * left.y_m + alpha * right.y_m),
        yaw_rad=float(_wrap_angle(left.yaw_rad + alpha * _wrap_angle(right.yaw_rad - left.yaw_rad))),
    )


def _interpolate_pose(poses: list[TimedPose2D], timestamp_ns: int) -> TimedPose2D | None:
    if not poses:
        return None
    if timestamp_ns < poses[0].timestamp_ns or timestamp_ns > poses[-1].timestamp_ns:
        return None
    for index, pose in enumerate(poses):
        if pose.timestamp_ns == timestamp_ns:
            return pose
        if pose.timestamp_ns > timestamp_ns and index > 0:
            left = poses[index - 1]
            right = pose
            span = max(1, int(right.timestamp_ns - left.timestamp_ns))
            alpha = float(timestamp_ns - left.timestamp_ns) / float(span)
            return TimedPose2D(
                timestamp_ns=int(timestamp_ns),
                frame_id=right.frame_id,
                x_m=float((1.0 - alpha) * left.x_m + alpha * right.x_m),
                y_m=float((1.0 - alpha) * left.y_m + alpha * right.y_m),
                yaw_rad=float(_wrap_angle(left.yaw_rad + alpha * _wrap_angle(right.yaw_rad - left.yaw_rad))),
            )
    return poses[-1] if timestamp_ns == poses[-1].timestamp_ns else None


def _relative_pose(origin: TimedPose2D, target: TimedPose2D) -> Pose2D:
    dx = float(target.x_m - origin.x_m)
    dy = float(target.y_m - origin.y_m)
    cos_yaw = math.cos(origin.yaw_rad)
    sin_yaw = math.sin(origin.yaw_rad)
    return Pose2D(
        x_m=float(cos_yaw * dx + sin_yaw * dy),
        y_m=float(-sin_yaw * dx + cos_yaw * dy),
        yaw_rad=float(_wrap_angle(target.yaw_rad - origin.yaw_rad)),
    )


def _find_pose_at_or_before(
    poses: list[TimedPose2D],
    timestamp_ns: int,
    frame_id: int,
) -> TimedPose2D | None:
    for pose in poses:
        if pose.frame_id == frame_id:
            return pose
    return _interpolate_pose(poses, timestamp_ns)


def _target_frame_id(poses: list[TimedPose2D], target_ns: int) -> int:
    target = _interpolate_pose(poses, target_ns)
    return int(target.frame_id) if target is not None else -1


def _invalid_label(
    reason: str,
    *,
    horizon_s: float,
    target_frame_id: int = -1,
    dx_m: float = 0.0,
    dy_m: float = 0.0,
    dyaw_rad: float = 0.0,
) -> FutureMotionActionLabel:
    return FutureMotionActionLabel(
        candidate_id="missing",
        bc_label_valid=False,
        invalid_reason=reason,
        bc_label_confidence=0.0,
        best_distance=float("inf"),
        second_best_distance=float("inf"),
        margin=0.0,
        horizon_s=float(horizon_s),
        target_frame_id=int(target_frame_id),
        dx_m=float(dx_m),
        dy_m=float(dy_m),
        dyaw_rad=float(dyaw_rad),
        top3=tuple(),
    )


def _yaw_from_xyzw(orientation_xyzw: tuple[float, float, float, float]) -> float:
    x, y, z, w = (float(value) for value in orientation_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("pose quaternion has zero norm")
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _candidate_index(candidates: list[CandidateTrajectory], candidate_id: str) -> int:
    for index, candidate in enumerate(candidates):
        if candidate.id == candidate_id:
            return index
    return 9999
