from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from homebrain.messages.schema import JsonDict
from homebrain.policies.candidate_trajectories import CandidateTrajectory

TRAJECTORY_SCORER_SCHEMA_VERSION = "homebrain.trajectory_scorer.v0"
RISKY_CANDIDATE_THRESHOLD = 0.35


@dataclass(frozen=True)
class LocalBev:
    free: np.ndarray
    occupied: np.ndarray
    unknown: np.ndarray
    traversable: np.ndarray | None = None
    risky: np.ndarray | None = None
    hazard: np.ndarray | None = None
    confidence: np.ndarray | None = None
    uncertainty: np.ndarray | None = None
    source: str = "unknown"

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.free.shape[0]), int(self.free.shape[1]))

    def validate(self) -> None:
        expected = self.free.shape
        if len(expected) != 2:
            raise ValueError("BEV arrays must be 2D")
        for name, array in (
            ("occupied", self.occupied),
            ("unknown", self.unknown),
            ("traversable", self.traversable),
            ("risky", self.risky),
            ("hazard", self.hazard),
            ("confidence", self.confidence),
            ("uncertainty", self.uncertainty),
        ):
            if array is not None and array.shape != expected:
                raise ValueError(f"{name} shape {array.shape} does not match {expected}")


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    risk_score: float
    unknown_penalty: float
    uncertainty_penalty: float
    hazard_exposure: float
    coverage_gain_proxy: float
    smoothness_penalty: float
    total_score: float
    risky: bool

    def to_dict(self) -> JsonDict:
        return {
            "candidate_id": self.candidate_id,
            "risk_score": round(float(self.risk_score), 6),
            "unknown_penalty": round(float(self.unknown_penalty), 6),
            "uncertainty_penalty": round(float(self.uncertainty_penalty), 6),
            "hazard_exposure": round(float(self.hazard_exposure), 6),
            "coverage_gain_proxy": round(float(self.coverage_gain_proxy), 6),
            "smoothness_penalty": round(float(self.smoothness_penalty), 6),
            "total_score": round(float(self.total_score), 6),
            "risky": bool(self.risky),
        }


@dataclass(frozen=True)
class TrajectoryDecision:
    selected_candidate_id: str
    reason: str
    scores: tuple[CandidateScore, ...]
    all_motion_candidates_risky: bool

    def selected_score(self) -> CandidateScore:
        for score in self.scores:
            if score.candidate_id == self.selected_candidate_id:
                return score
        raise ValueError(f"selected candidate missing score: {self.selected_candidate_id}")

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": TRAJECTORY_SCORER_SCHEMA_VERSION,
            "selected_candidate_id": self.selected_candidate_id,
            "reason": self.reason,
            "all_motion_candidates_risky": bool(self.all_motion_candidates_risky),
            "scores": [score.to_dict() for score in self.scores],
        }


class CoverageMemory:
    def __init__(self, grid_shape: tuple[int, int], *, meters_per_cell: float) -> None:
        if grid_shape[0] <= 0 or grid_shape[1] <= 0:
            raise ValueError("grid_shape must be positive")
        if meters_per_cell <= 0.0:
            raise ValueError("meters_per_cell must be positive")
        self.grid_shape = grid_shape
        self.meters_per_cell = float(meters_per_cell)
        self.seen = np.zeros(grid_shape, dtype=np.bool_)
        self.covered = np.zeros(grid_shape, dtype=np.bool_)
        self.pose_aligned = False
        self.pose_alignment_attempt_count = 0
        self.missing_pose_delta_count = 0

    @property
    def cells_seen(self) -> int:
        return int(np.count_nonzero(self.seen))

    @property
    def cells_covered(self) -> int:
        return int(np.count_nonzero(self.covered))

    def align_with_pose_delta(self, pose_delta: tuple[float, float, float] | None) -> None:
        if pose_delta is None:
            self.pose_aligned = False
            self.missing_pose_delta_count += 1
            return
        dx_m, dy_m, _dyaw_rad = pose_delta
        if not np.isfinite([dx_m, dy_m]).all():
            self.pose_aligned = False
            self.missing_pose_delta_count += 1
            return
        self.pose_alignment_attempt_count += 1
        row_shift = int(round(float(dx_m) / self.meters_per_cell))
        col_shift = int(round(-float(dy_m) / self.meters_per_cell))
        self.seen = _shift_mask(self.seen, row_shift=row_shift, col_shift=col_shift)
        self.covered = _shift_mask(self.covered, row_shift=row_shift, col_shift=col_shift)
        self.pose_aligned = True

    def update_current_frame(self, bev: LocalBev) -> None:
        bev.validate()
        traversable = _as_probability(bev.traversable) if bev.traversable is not None else _as_probability(bev.free)
        risky = _as_probability(bev.risky) if bev.risky is not None else _as_probability(bev.occupied)
        hazard = _as_probability(bev.hazard) if bev.hazard is not None else 0.0
        free = _as_probability(bev.free)
        occupied = _as_probability(bev.occupied)
        unknown = _as_probability(bev.unknown)
        confidence = _as_probability(bev.confidence) if bev.confidence is not None else None
        observed = (free >= 0.5) | (occupied >= 0.5) | (traversable >= 0.5) | (risky >= 0.5) | (hazard >= 0.5)
        if confidence is not None:
            observed |= (confidence >= 0.2) & (unknown < 0.5)
        covered = (free >= 0.5) | (traversable >= 0.5)
        self.seen |= observed
        self.covered |= covered

    def to_dict(self) -> JsonDict:
        return {
            "grid_shape": [int(self.grid_shape[0]), int(self.grid_shape[1])],
            "meters_per_cell": round(float(self.meters_per_cell), 6),
            "pose_aligned": bool(self.pose_aligned),
            "pose_alignment_attempt_count": int(self.pose_alignment_attempt_count),
            "missing_pose_delta_count": int(self.missing_pose_delta_count),
            "coverage_memory_cells_seen": self.cells_seen,
            "coverage_memory_cells_covered": self.cells_covered,
            "alignment_model": "translation_cell_shift_yaw_ignored_v0",
        }


def score_trajectories(
    *,
    bev: LocalBev,
    candidates: list[CandidateTrajectory],
    coverage_memory: CoverageMemory | None = None,
) -> TrajectoryDecision:
    bev.validate()
    if not candidates:
        raise ValueError("at least one candidate is required")

    raw_scores = [
        _raw_score_candidate(candidate, bev=bev, coverage_memory=coverage_memory)
        for candidate in candidates
    ]
    motion_scores = [score for score in raw_scores if score.candidate_id != "stop"]
    all_motion_risky = bool(motion_scores) and all(score.risky for score in motion_scores)
    adjusted_scores = tuple(
        _adjust_stop_score(score, all_motion_candidates_risky=all_motion_risky)
        for score in raw_scores
    )
    selected = min(adjusted_scores, key=lambda score: (score.total_score, _candidate_order(candidates, score.candidate_id)))
    return TrajectoryDecision(
        selected_candidate_id=selected.candidate_id,
        reason=_selection_reason(selected, all_motion_candidates_risky=all_motion_risky),
        scores=adjusted_scores,
        all_motion_candidates_risky=all_motion_risky,
    )


def risky_candidate_fraction(decision: TrajectoryDecision) -> float:
    if not decision.scores:
        return 0.0
    return float(sum(1 for score in decision.scores if score.risky) / len(decision.scores))


def _raw_score_candidate(
    candidate: CandidateTrajectory,
    *,
    bev: LocalBev,
    coverage_memory: CoverageMemory | None,
) -> CandidateScore:
    cells = candidate.footprint_cells
    occupied = np.maximum(_as_probability(bev.occupied), _as_probability(bev.risky) if bev.risky is not None else 0.0)
    hazard = _as_probability(bev.hazard) if bev.hazard is not None else np.zeros_like(occupied, dtype=np.float32)
    traversable = _as_probability(bev.traversable) if bev.traversable is not None else _as_probability(bev.free)
    free = np.maximum(_as_probability(bev.free), traversable)
    unknown = _as_probability(bev.unknown)
    if bev.uncertainty is not None:
        uncertainty = _as_probability(bev.uncertainty)
    elif bev.confidence is not None:
        uncertainty = 1.0 - _as_probability(bev.confidence)
    else:
        uncertainty = unknown

    occupied_values = _cell_values(occupied, cells, default=1.0)
    hazard_values = _cell_values(hazard, cells, default=1.0)
    unknown_values = _cell_values(unknown, cells, default=1.0)
    uncertainty_values = _cell_values(uncertainty, cells, default=1.0)
    risk_score = max(float(np.max(occupied_values)), float(np.mean(occupied_values)))
    hazard_exposure = max(float(np.max(hazard_values)), float(np.mean(hazard_values)))
    unknown_penalty = float(np.mean(unknown_values))
    uncertainty_penalty = float(np.mean(uncertainty_values))
    coverage_gain = _coverage_gain_proxy(candidate, free=free, coverage_memory=coverage_memory)
    smoothness = _smoothness_penalty(candidate)
    risky = max(risk_score, hazard_exposure) >= RISKY_CANDIDATE_THRESHOLD
    total = (
        12.0 * risk_score
        + 8.0 * hazard_exposure
        + 1.6 * unknown_penalty
        + 0.8 * uncertainty_penalty
        + 0.25 * smoothness
        - 0.18 * coverage_gain
    )
    if not cells:
        total += 5.0
        risky = True
    return CandidateScore(
        candidate_id=candidate.id,
        risk_score=risk_score,
        unknown_penalty=unknown_penalty,
        uncertainty_penalty=uncertainty_penalty,
        hazard_exposure=hazard_exposure,
        coverage_gain_proxy=coverage_gain,
        smoothness_penalty=smoothness,
        total_score=total,
        risky=risky,
    )


def _adjust_stop_score(score: CandidateScore, *, all_motion_candidates_risky: bool) -> CandidateScore:
    if score.candidate_id != "stop":
        return score
    stop_delta = -8.0 if all_motion_candidates_risky else 8.0
    return CandidateScore(
        candidate_id=score.candidate_id,
        risk_score=score.risk_score,
        unknown_penalty=score.unknown_penalty,
        uncertainty_penalty=score.uncertainty_penalty,
        hazard_exposure=score.hazard_exposure,
        coverage_gain_proxy=score.coverage_gain_proxy,
        smoothness_penalty=score.smoothness_penalty,
        total_score=score.total_score + stop_delta,
        risky=score.risky,
    )


def _selection_reason(score: CandidateScore, *, all_motion_candidates_risky: bool) -> str:
    if score.candidate_id == "stop" and all_motion_candidates_risky:
        return "all_motion_candidates_risky_select_stop"
    if score.coverage_gain_proxy > 0.0:
        return "lowest_total_score_prefers_low_risk_coverage_gain"
    return "lowest_total_score_prefers_low_risk_low_uncertainty"


def _coverage_gain_proxy(
    candidate: CandidateTrajectory,
    *,
    free: np.ndarray,
    coverage_memory: CoverageMemory | None,
) -> float:
    if not candidate.footprint_cells:
        return 0.0
    count = 0
    covered = coverage_memory.covered if coverage_memory is not None else np.zeros_like(free, dtype=np.bool_)
    for row, col in candidate.footprint_cells:
        if 0 <= row < free.shape[0] and 0 <= col < free.shape[1]:
            if free[row, col] >= 0.5 and not bool(covered[row, col]):
                count += 1
    return float(count)


def _smoothness_penalty(candidate: CandidateTrajectory) -> float:
    angular = candidate.cmd_vel_proxy.get("angular_velocity_radps", 0.0)
    linear = candidate.cmd_vel_proxy.get("linear_velocity_mps", 0.0)
    try:
        angular_abs = abs(float(angular))
        linear_abs = abs(float(linear))
    except (TypeError, ValueError):
        return 1.0
    return float(angular_abs * candidate.duration_s + (0.05 if linear_abs == 0.0 and angular_abs > 0.0 else 0.0))


def _candidate_order(candidates: Iterable[CandidateTrajectory], candidate_id: str) -> int:
    for index, candidate in enumerate(candidates):
        if candidate.id == candidate_id:
            return index
    return 9999


def _cell_values(array: np.ndarray, cells: tuple[tuple[int, int], ...], *, default: float) -> np.ndarray:
    if not cells:
        return np.asarray([default], dtype=np.float32)
    values: list[float] = []
    for row, col in cells:
        if 0 <= row < array.shape[0] and 0 <= col < array.shape[1]:
            values.append(float(array[row, col]))
        else:
            values.append(default)
    if not values:
        values.append(default)
    return np.asarray(values, dtype=np.float32)


def _as_probability(array: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)


def _shift_mask(mask: np.ndarray, *, row_shift: int, col_shift: int) -> np.ndarray:
    shifted = np.zeros_like(mask, dtype=np.bool_)
    height, width = mask.shape
    src_row_start = max(0, -row_shift)
    src_row_end = min(height, height - row_shift)
    dst_row_start = max(0, row_shift)
    dst_row_end = min(height, height + row_shift)
    src_col_start = max(0, -col_shift)
    src_col_end = min(width, width - col_shift)
    dst_col_start = max(0, col_shift)
    dst_col_end = min(width, width + col_shift)
    if src_row_start >= src_row_end or src_col_start >= src_col_end:
        return shifted
    shifted[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = mask[
        src_row_start:src_row_end,
        src_col_start:src_col_end,
    ]
    return shifted
