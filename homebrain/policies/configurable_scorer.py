from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from homebrain.messages.schema import JsonDict, deterministic_json
from homebrain.policies.bev_action_sanity import LoadedBevFrame
from homebrain.policies.candidate_trajectories import CandidateTrajectory, generate_default_candidates
from homebrain.policies.trajectory_scorer import LocalBev


@dataclass(frozen=True)
class TransparentScorerConfig:
    unknown_weight: float
    risk_weight: float
    obstacle_threshold: float
    footprint_radius_cells: int
    obstacle_inflation_cells: int
    stop_bias: float

    def to_dict(self) -> JsonDict:
        return {
            "unknown_weight": float(self.unknown_weight),
            "risk_weight": float(self.risk_weight),
            "obstacle_threshold": float(self.obstacle_threshold),
            "footprint_radius_cells": int(self.footprint_radius_cells),
            "obstacle_inflation_cells": int(self.obstacle_inflation_cells),
            "stop_bias": float(self.stop_bias),
        }

    @staticmethod
    def from_dict(data: JsonDict) -> "TransparentScorerConfig":
        return TransparentScorerConfig(
            unknown_weight=float(data["unknown_weight"]),
            risk_weight=float(data["risk_weight"]),
            obstacle_threshold=float(data["obstacle_threshold"]),
            footprint_radius_cells=int(data["footprint_radius_cells"]),
            obstacle_inflation_cells=int(data["obstacle_inflation_cells"]),
            stop_bias=float(data["stop_bias"]),
        )


@dataclass(frozen=True)
class ConfiguredDecision:
    selected_candidate_id: str
    scores: tuple[JsonDict, ...]
    all_motion_blocked: bool

    def selected_score(self) -> JsonDict:
        for score in self.scores:
            if score["candidate_id"] == self.selected_candidate_id:
                return score
        raise ValueError(f"missing selected score: {self.selected_candidate_id}")


def default_sweep_configs() -> list[TransparentScorerConfig]:
    configs: list[TransparentScorerConfig] = []
    for unknown_weight in (0.8, 1.6):
        for risk_weight in (12.0, 18.0):
            for obstacle_threshold in (0.35, 0.50):
                for footprint_radius_cells in (2, 3):
                    for obstacle_inflation_cells in (0, 1):
                        for stop_bias in (8.0, 10.0):
                            configs.append(
                                TransparentScorerConfig(
                                    unknown_weight=unknown_weight,
                                    risk_weight=risk_weight,
                                    obstacle_threshold=obstacle_threshold,
                                    footprint_radius_cells=footprint_radius_cells,
                                    obstacle_inflation_cells=obstacle_inflation_cells,
                                    stop_bias=stop_bias,
                                )
                            )
    return configs


def score_frame_with_config(
    frame: LoadedBevFrame,
    *,
    config: TransparentScorerConfig,
    candidates: list[CandidateTrajectory] | None = None,
) -> ConfiguredDecision:
    if candidates is None:
        candidates = generate_default_candidates(
            grid_shape=frame.record.bev.shape,
            meters_per_cell=frame.meters_per_cell,
            robot_radius_m=max(frame.meters_per_cell, config.footprint_radius_cells * frame.meters_per_cell),
        )
    free = _free(frame.record.bev)
    risk = _risk(frame.record.bev)
    if config.obstacle_inflation_cells > 0:
        risk = np.maximum(
            risk,
            _inflate_binary(risk >= np.float32(config.obstacle_threshold), radius_cells=config.obstacle_inflation_cells),
        )
    unknown = _prob(frame.record.bev.unknown)
    raw_scores = tuple(
        _score_candidate(candidate, free=free, risk=risk, unknown=unknown, config=config)
        for candidate in candidates
    )
    translational_scores = [
        score
        for candidate, score in zip(candidates, raw_scores)
        if score["candidate_id"] != "stop" and abs(float(candidate.cmd_vel_proxy.get("linear_velocity_mps", 0.0))) > 0.0
    ]
    all_motion_blocked = bool(translational_scores) and all(
        bool(score["blocked"]) or float(score.get("coverage_gain_proxy", 0.0)) <= 0.0
        for score in translational_scores
    )
    adjusted: list[JsonDict] = []
    for score in raw_scores:
        adjusted_score = dict(score)
        if score["candidate_id"] == "stop":
            delta = -abs(config.stop_bias) if all_motion_blocked else abs(config.stop_bias)
            adjusted_score["total_score"] = round(float(adjusted_score["total_score"]) + delta, 6)
        adjusted.append(adjusted_score)
    selected = min(adjusted, key=lambda score: (float(score["total_score"]), _candidate_order(candidates, str(score["candidate_id"]))))
    return ConfiguredDecision(
        selected_candidate_id=str(selected["candidate_id"]),
        scores=tuple(adjusted),
        all_motion_blocked=all_motion_blocked,
    )


def evaluate_config(
    frames: list[LoadedBevFrame],
    *,
    config: TransparentScorerConfig,
) -> JsonDict:
    decisions: list[tuple[LoadedBevFrame, ConfiguredDecision]] = []
    candidates_by_key: dict[tuple[tuple[int, int], float, int], list[CandidateTrajectory]] = {}
    for frame in frames:
        key = (frame.record.bev.shape, frame.meters_per_cell, config.footprint_radius_cells)
        if key not in candidates_by_key:
            candidates_by_key[key] = generate_default_candidates(
                grid_shape=frame.record.bev.shape,
                meters_per_cell=frame.meters_per_cell,
                robot_radius_m=max(frame.meters_per_cell, config.footprint_radius_cells * frame.meters_per_cell),
            )
        decisions.append((frame, score_frame_with_config(frame, config=config, candidates=candidates_by_key[key])))

    selected_ids = [decision.selected_candidate_id for _frame, decision in decisions]
    selected_scores = [decision.selected_score() for _frame, decision in decisions]
    controlled_open = [
        decision.selected_candidate_id != "stop"
        for frame, decision in decisions
        if frame.source_family == "controlled_bev" and frame.scenario_name == "open_room"
    ]
    controlled_blocked = [
        decision.selected_candidate_id == "stop"
        for frame, decision in decisions
        if frame.source_family == "controlled_bev" and frame.scenario_name == "blocked_path"
    ]
    reviewed = [
        decision.selected_candidate_id != "stop"
        for frame, decision in decisions
        if frame.source_family != "controlled_bev"
    ]
    collision_positive = [
        float(score.get("collision_proxy", 0.0)) >= float(config.obstacle_threshold)
        for score in selected_scores
        if score.get("candidate_id") != "stop"
    ]
    stop_count = sum(1 for value in selected_ids if value == "stop")
    return {
        "config": config.to_dict(),
        "frame_count": len(frames),
        "controlled_open_frame_count": len(controlled_open),
        "controlled_blocked_frame_count": len(controlled_blocked),
        "reviewed_frame_count": len(reviewed),
        "controlled_open_motion_rate": _fraction(controlled_open),
        "blocked_map_stop_rate": _fraction(controlled_blocked),
        "reviewed_motion_rate": _fraction(reviewed),
        "collision_proxy_rate": _fraction(collision_positive),
        "stop_fraction": float(stop_count / max(len(selected_ids), 1)),
        "selected_distribution": _distribution(selected_ids),
        "passes_controlled_open": bool(controlled_open) and _fraction(controlled_open) >= 0.95,
        "passes_controlled_blocked": bool(controlled_blocked) and _fraction(controlled_blocked) >= 0.95,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
    }


def decisions_jsonl_records(
    frames: list[LoadedBevFrame],
    *,
    config: TransparentScorerConfig,
) -> list[JsonDict]:
    records: list[JsonDict] = []
    candidates_by_key: dict[tuple[tuple[int, int], float, int], list[CandidateTrajectory]] = {}
    for frame_index, frame in enumerate(frames):
        key = (frame.record.bev.shape, frame.meters_per_cell, config.footprint_radius_cells)
        if key not in candidates_by_key:
            candidates_by_key[key] = generate_default_candidates(
                grid_shape=frame.record.bev.shape,
                meters_per_cell=frame.meters_per_cell,
                robot_radius_m=max(frame.meters_per_cell, config.footprint_radius_cells * frame.meters_per_cell),
            )
        candidates = candidates_by_key[key]
        decision = score_frame_with_config(frame, config=config, candidates=candidates)
        scores_by_id = {str(score["candidate_id"]): score for score in decision.scores}
        selected_score = decision.selected_score()
        records.append(
            {
                "schema_version": "homebrain.configured_trajectory_decision.v0",
                "event_type": "trajectory_decision",
                "frame_index": frame_index,
                "sequence_id": frame.record.sequence_id,
                "camera_id": frame.record.camera_id,
                "frame_id": int(frame.record.frame_id),
                "timestamp_ns": int(frame.record.timestamp_ns),
                "source_ref": frame.record.source_ref,
                "source_name": frame.source_name,
                "scenario_name": frame.scenario_name,
                "scorer_config": config.to_dict(),
                "candidate_count": len(candidates),
                "candidates": [
                    {**candidate.to_dict(), "score": scores_by_id[candidate.id]}
                    for candidate in candidates
                ],
                "selected_candidate_id": decision.selected_candidate_id,
                "selected_score": selected_score,
                "all_motion_candidates_blocked": bool(decision.all_motion_blocked),
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
                "raw_pwm_emitted": False,
            }
        )
    return records


def write_jsonl(path: str | Path, records: Iterable[JsonDict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(deterministic_json(record))
            handle.write("\n")


def _score_candidate(
    candidate: CandidateTrajectory,
    *,
    free: np.ndarray,
    risk: np.ndarray,
    unknown: np.ndarray,
    config: TransparentScorerConfig,
) -> JsonDict:
    cells = tuple(candidate.footprint_cells)
    risk_values = _cell_values(risk, cells, default=1.0)
    unknown_values = _cell_values(unknown, cells, default=1.0)
    free_values = _cell_values(free, cells, default=0.0)
    collision = float(np.max(risk_values))
    unknown_penalty = float(np.mean(unknown_values))
    coverage_gain = float(np.count_nonzero((free_values >= np.float32(0.5)) & (risk_values < np.float32(config.obstacle_threshold))))
    smoothness = _smoothness_cost(candidate)
    blocked = collision >= config.obstacle_threshold or unknown_penalty >= 0.95 or not cells
    total = (
        float(config.risk_weight) * collision
        + float(config.unknown_weight) * unknown_penalty
        + 0.25 * smoothness
        - 0.18 * coverage_gain
    )
    if not cells:
        total += 10.0
    return {
        "candidate_id": candidate.id,
        "risk_score": round(collision, 6),
        "collision_proxy": round(collision, 6),
        "unknown_penalty": round(unknown_penalty, 6),
        "coverage_gain_proxy": round(coverage_gain, 6),
        "smoothness_penalty": round(smoothness, 6),
        "blocked": bool(blocked),
        "risky": bool(collision >= config.obstacle_threshold),
        "total_score": round(float(total), 6),
    }


def _free(bev: LocalBev) -> np.ndarray:
    traversable = _prob(bev.traversable) if bev.traversable is not None else 0.0
    return np.maximum(_prob(bev.free), traversable)


def _risk(bev: LocalBev) -> np.ndarray:
    risky = _prob(bev.risky) if bev.risky is not None else 0.0
    return np.maximum(_prob(bev.occupied), risky)


def _prob(array: np.ndarray | float | None) -> np.ndarray:
    if array is None:
        return np.asarray(0.0, dtype=np.float32)
    return np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)


def _inflate_binary(mask: np.ndarray, *, radius_cells: int) -> np.ndarray:
    radius = max(0, int(radius_cells))
    if radius == 0:
        return mask.astype(np.float32)
    inflated = np.zeros(mask.shape, dtype=np.bool_)
    rows, cols = mask.shape
    for row in range(rows):
        for col in range(cols):
            if not bool(mask[row, col]):
                continue
            row0 = max(0, row - radius)
            row1 = min(rows, row + radius + 1)
            col0 = max(0, col - radius)
            col1 = min(cols, col + radius + 1)
            inflated[row0:row1, col0:col1] = True
    return inflated.astype(np.float32)


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


def _smoothness_cost(candidate: CandidateTrajectory) -> float:
    angular = abs(float(candidate.cmd_vel_proxy.get("angular_velocity_radps", 0.0)))
    linear = abs(float(candidate.cmd_vel_proxy.get("linear_velocity_mps", 0.0)))
    return float(angular * candidate.duration_s + (0.05 if linear == 0.0 and angular > 0.0 else 0.0))


def _candidate_order(candidates: Iterable[CandidateTrajectory], candidate_id: str) -> int:
    for index, candidate in enumerate(candidates):
        if candidate.id == candidate_id:
            return index
    return 9999


def _fraction(values: list[bool]) -> float:
    if not values:
        return 0.0
    return float(sum(1 for value in values if value) / len(values))


def _distribution(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
