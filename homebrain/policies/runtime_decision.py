from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from homebrain.brain.future_bev_rollout_v1 import (
    FUTURE_BEV_ROLLOUT_V1_SOURCE,
    load_checkpoint as load_future_rollout_checkpoint,
    score_local_bev_with_future_rollout,
)
from homebrain.messages.schema import JsonDict
from homebrain.policies.candidate_trajectories import CandidateTrajectory
from homebrain.policies.trajectory_scorer import (
    CoverageMemory,
    LocalBev,
    TrajectoryDecision,
    risky_candidate_fraction,
    score_trajectories,
)
from homebrain.policies.trajectory_scorer_net_v0 import (
    TRAJECTORY_SCORER_V0_SOURCE,
    candidate_score_records,
    load_trajectory_scorer_checkpoint,
    score_local_bev_with_model,
    scorer_checkpoint_hash,
)

RUNTIME_DECISION_SCHEMA_VERSION = "homebrain.runtime_decision.v0"
TRANSPARENT_TRAJECTORY_SCORER_SOURCE = "transparent_coverage_risk_v0"
DEFAULT_CMD_VEL_LIMITS = {
    "max_linear_velocity_mps": 0.25,
    "max_angular_velocity_radps": 1.0,
}


@dataclass(frozen=True)
class RuntimeTrajectoryScorer:
    model: torch.nn.Module
    checkpoint: Path
    metadata: JsonDict
    device: torch.device


@dataclass(frozen=True)
class RuntimeFutureRolloutScorer:
    model: torch.nn.Module
    checkpoint: Path
    metadata: JsonDict
    device: torch.device


@dataclass(frozen=True)
class RuntimeDecisionResult:
    candidate_trajectories: list[JsonDict]
    selected_candidate_id: str
    selected_candidate_index: int
    debug: JsonDict
    artifact_arrays: dict[str, np.ndarray]
    coverage_memory: CoverageMemory


def load_runtime_trajectory_scorer(
    checkpoint: str | Path,
    *,
    device: torch.device,
) -> RuntimeTrajectoryScorer:
    model, payload = load_trajectory_scorer_checkpoint(checkpoint, map_location=device)
    model.to(device)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return RuntimeTrajectoryScorer(
        model=model,
        checkpoint=Path(checkpoint),
        metadata=dict(metadata),
        device=device,
    )


def load_runtime_future_rollout_scorer(
    checkpoint: str | Path,
    *,
    device: torch.device,
) -> RuntimeFutureRolloutScorer:
    model, payload = load_future_rollout_checkpoint(checkpoint, map_location=device)
    model.to(device)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return RuntimeFutureRolloutScorer(
        model=model,
        checkpoint=Path(checkpoint),
        metadata=dict(metadata),
        device=device,
    )


def decide_trajectory(
    *,
    bev: LocalBev,
    candidates: list[CandidateTrajectory],
    coverage_memory: CoverageMemory,
    pose_delta: tuple[float, float, float] | None,
    learned_scorer: RuntimeTrajectoryScorer | None = None,
    future_rollout_scorer: RuntimeFutureRolloutScorer | None = None,
    patch_features: np.ndarray | None = None,
    sensor_mask: np.ndarray | None = None,
    policy_bev_source: str,
    coverage_memory_reset: bool = False,
    future_rollout_selection_mode: str = "argmin",
    selection_history: list[str] | None = None,
) -> RuntimeDecisionResult:
    bev.validate()
    if not candidates:
        raise ValueError("at least one candidate is required")

    coverage_before = coverage_memory.to_dict()
    coverage_memory.align_with_pose_delta(pose_delta)
    coverage_after_align = coverage_memory.to_dict()
    transparent_decision = score_trajectories(
        bev=bev,
        candidates=candidates,
        coverage_memory=coverage_memory,
    )

    learned_logits: np.ndarray | None = None
    learned_features: np.ndarray | None = None
    future_rollout_arrays: dict[str, np.ndarray] = {}
    future_scores: dict[str, np.ndarray | list[str] | str] | None = None
    future_selected_candidate_id: str | None = None
    if future_rollout_scorer is not None:
        future_scores = score_local_bev_with_future_rollout(
            model=future_rollout_scorer.model,  # type: ignore[arg-type]
            bev=bev,
            patch_features=patch_features,
            sensor_mask=sensor_mask,
            device=future_rollout_scorer.device,
        )
        _validate_future_candidate_ids(candidates, future_scores)
        lower_score = np.asarray(future_scores["candidate_lower_is_better_score"], dtype=np.float32)
        future_selected_index, guided_scores = _select_future_rollout_candidate(
            candidates=candidates,
            future_scores=future_scores,
            transparent_decision=transparent_decision,
            selection_mode=future_rollout_selection_mode,
            selection_history=selection_history,
        )
        future_selected_candidate_id = candidates[future_selected_index].id
        future_rollout_arrays = {
            "future_rollout_candidate_collision": np.asarray(future_scores["candidate_collision"], dtype=np.float32),
            "future_rollout_candidate_future_collision": np.asarray(
                future_scores.get("candidate_future_collision", future_scores["candidate_collision"]),
                dtype=np.float32,
            ),
            "future_rollout_candidate_unsafe_now": np.asarray(
                future_scores.get("candidate_unsafe_now", np.zeros_like(future_scores["candidate_collision"])),
                dtype=np.float32,
            ),
            "future_rollout_candidate_unknown_exposure": np.asarray(
                future_scores["candidate_unknown_exposure"],
                dtype=np.float32,
            ),
            "future_rollout_candidate_new_area_gain": np.asarray(future_scores["candidate_new_area_gain"], dtype=np.float32),
            "future_rollout_candidate_progress": np.asarray(future_scores["candidate_progress"], dtype=np.float32),
            "future_rollout_candidate_lower_is_better_score": lower_score,
            "future_rollout_guided_total_scores": guided_scores.astype(np.float32),
            "future_rollout_future_bev_prob": np.asarray(future_scores["future_bev_prob"], dtype=np.float32),
            "future_rollout_future_uncertainty_grid": np.asarray(
                future_scores["future_uncertainty_grid"],
                dtype=np.float32,
            ),
            "future_rollout_horizons_s": np.asarray(future_scores["future_horizons_s"], dtype=np.float32),
        }
    if learned_scorer is not None:
        learned_logits, selected_candidate_id, learned_features = score_local_bev_with_model(
            model=learned_scorer.model,  # type: ignore[arg-type]
            bev=bev,
            candidates=candidates,
            coverage_memory=coverage_memory,
            device=learned_scorer.device,
            candidate_feature_mode=str(learned_scorer.metadata.get("candidate_feature_mode", "all")),
            logit_bias_by_candidate_id=learned_scorer.metadata,
        )
        candidate_records = candidate_score_records(
            candidates=candidates,
            logits=learned_logits,
            candidate_features=learned_features,
            selected_candidate_id=selected_candidate_id,
        )
        _attach_transparent_scores(candidate_records, transparent_decision)
        if future_scores is not None and future_selected_candidate_id is not None:
            _attach_future_rollout_scores(
                candidate_records,
                candidates=candidates,
                future_scores=future_scores,
                future_selected_candidate_id=future_selected_candidate_id,
            )
        scorer_name = TRAJECTORY_SCORER_V0_SOURCE
        scorer_mode = "learned"
    elif future_scores is not None and future_selected_candidate_id is not None:
        selected_candidate_id = future_selected_candidate_id
        candidate_records = _future_rollout_candidate_score_records(
            candidates=candidates,
            future_scores=future_scores,
            selected_candidate_id=selected_candidate_id,
        )
        _attach_transparent_scores(candidate_records, transparent_decision)
        scorer_name = FUTURE_BEV_ROLLOUT_V1_SOURCE
        scorer_mode = "future_rollout"
    else:
        selected_candidate_id = transparent_decision.selected_candidate_id
        candidate_records = _transparent_candidate_score_records(
            candidates=candidates,
            decision=transparent_decision,
        )
        scorer_name = TRANSPARENT_TRAJECTORY_SCORER_SOURCE
        scorer_mode = "transparent"

    selected_index = _selected_candidate_index(candidates, selected_candidate_id)
    cmd_vel_proposal = _bounded_cmd_vel_proposal(candidates[selected_index])
    coverage_memory.update_current_frame(bev)
    coverage_after_update = coverage_memory.to_dict()
    selected_metrics = _score_dict_for_candidate(transparent_decision, selected_candidate_id)
    selected_record_score = _score_dict_from_records(candidate_records, selected_candidate_id)
    selected_metrics.update(selected_record_score)
    selected_future_score = _nested_score_dict_from_records(
        candidate_records,
        selected_candidate_id,
        "future_rollout_trajectory_score",
    )
    selected_metrics.update(selected_future_score)
    artifact_arrays: dict[str, np.ndarray] = {
        "trajectory_selected_index": np.asarray([selected_index], dtype=np.int64),
        "trajectory_transparent_total_scores": np.asarray(
            [score.total_score for score in transparent_decision.scores],
            dtype=np.float32,
        ),
        "trajectory_transparent_risk_scores": np.asarray(
            [score.risk_score for score in transparent_decision.scores],
            dtype=np.float32,
        ),
        "trajectory_transparent_unknown_penalties": np.asarray(
            [score.unknown_penalty for score in transparent_decision.scores],
            dtype=np.float32,
        ),
        "trajectory_transparent_uncertainty_penalties": np.asarray(
            [score.uncertainty_penalty for score in transparent_decision.scores],
            dtype=np.float32,
        ),
        "trajectory_transparent_coverage_gains": np.asarray(
            [score.coverage_gain_proxy for score in transparent_decision.scores],
            dtype=np.float32,
        ),
    }
    if learned_logits is not None:
        artifact_arrays["trajectory_logits"] = learned_logits.astype(np.float32)
    if learned_features is not None:
        artifact_arrays["trajectory_candidate_features"] = learned_features.astype(np.float32)
    artifact_arrays.update(future_rollout_arrays)

    debug: JsonDict = {
        "schema_version": RUNTIME_DECISION_SCHEMA_VERSION,
        "trajectory_scoring": True,
        "trajectory_scorer": scorer_name,
        "trajectory_scorer_mode": scorer_mode,
        "policy_bev_source": policy_bev_source,
        "selected_candidate_id": selected_candidate_id,
        "selected_candidate_index": selected_index,
        "selected_candidate_metrics": selected_metrics,
        "transparent_decision": transparent_decision.to_dict(),
        "risky_candidate_fraction": risky_candidate_fraction(transparent_decision),
        "coverage_memory": coverage_after_update,
        "coverage_memory_before": coverage_before,
        "coverage_memory_after_align": coverage_after_align,
        "coverage_memory_update": _coverage_update(coverage_before, coverage_after_update),
        "coverage_memory_reset": bool(coverage_memory_reset),
        "cmd_vel_proposal": cmd_vel_proposal,
        "cmd_vel_proposal_available": True,
        "cmd_vel_proposal_is_bounded": True,
        "cmd_vel_bounds": dict(DEFAULT_CMD_VEL_LIMITS),
        "cmd_vel_proposal_note": "review_only_not_executed",
        "selection_history_recent": list(selection_history or []),
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "product_training_approved": False,
        "cmd_vel_emitted": False,
        "raw_pwm_emitted": False,
    }
    if learned_scorer is not None:
        debug.update(
            {
                "trajectory_scorer_checkpoint": learned_scorer.checkpoint.as_posix(),
                "trajectory_scorer_checkpoint_sha256": scorer_checkpoint_hash(learned_scorer.checkpoint),
                "trajectory_scorer_metadata": learned_scorer.metadata,
            }
        )
    if future_rollout_scorer is not None:
        debug.update(
            {
                "future_rollout_checkpoint": future_rollout_scorer.checkpoint.as_posix(),
                "future_rollout_checkpoint_sha256": _file_sha256(future_rollout_scorer.checkpoint),
                "future_rollout_metadata": future_rollout_scorer.metadata,
                "future_rollout_selection_mode": future_rollout_selection_mode,
                "future_rollout_guided_total_scores": [
                    round(float(value), 6) for value in future_rollout_arrays["future_rollout_guided_total_scores"]
                ],
                "future_rollout_raw_argmin_candidate_id": candidates[
                    int(np.argmin(future_rollout_arrays["future_rollout_candidate_lower_is_better_score"]))
                ].id,
                "future_rollout_candidate_selected_for_prediction": future_selected_candidate_id,
                "future_rollout_policy_selection_role": "prediction_context"
                if learned_scorer is not None
                else "runtime_selector",
                "temporal_diversity_prior_enabled": future_rollout_selection_mode != "argmin"
                and learned_scorer is None,
                "future_rollout_replay_only": True,
                "future_rollout_control_safe": False,
            }
        )

    return RuntimeDecisionResult(
        candidate_trajectories=candidate_records,
        selected_candidate_id=selected_candidate_id,
        selected_candidate_index=selected_index,
        debug=debug,
        artifact_arrays=artifact_arrays,
        coverage_memory=coverage_memory,
    )


def _transparent_candidate_score_records(
    *,
    candidates: list[CandidateTrajectory],
    decision: TrajectoryDecision,
) -> list[JsonDict]:
    scores_by_id = {score.candidate_id: score.to_dict() for score in decision.scores}
    records: list[JsonDict] = []
    for candidate in candidates:
        score = scores_by_id[candidate.id]
        records.append(
            {
                **candidate.to_dict(),
                "trajectory_score": {
                    "schema_version": "homebrain.transparent_trajectory_score.v0",
                    "scorer": TRANSPARENT_TRAJECTORY_SCORER_SOURCE,
                    "candidate_id": candidate.id,
                    **score,
                    "lower_is_better": True,
                    "selected_by_runtime_policy": candidate.id == decision.selected_candidate_id,
                    "replay_only": True,
                    "not_executed": True,
                    "control_safe": False,
                    "product_training_approved": False,
                },
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
                "product_training_approved": False,
            }
        )
    return records


def _attach_transparent_scores(records: list[JsonDict], decision: TrajectoryDecision) -> None:
    scores_by_id = {score.candidate_id: score.to_dict() for score in decision.scores}
    for record in records:
        candidate_id = str(record.get("id", record.get("trajectory_id", "")))
        score = scores_by_id.get(candidate_id)
        if score is None:
            continue
        record["transparent_trajectory_score"] = {
            "schema_version": "homebrain.transparent_trajectory_score.v0",
            "scorer": TRANSPARENT_TRAJECTORY_SCORER_SOURCE,
            **score,
            "lower_is_better": True,
            "selected_by_transparent_scorer": candidate_id == decision.selected_candidate_id,
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "product_training_approved": False,
        }


def _attach_future_rollout_scores(
    records: list[JsonDict],
    *,
    candidates: list[CandidateTrajectory],
    future_scores: dict[str, np.ndarray | list[str] | str],
    future_selected_candidate_id: str,
) -> None:
    future_records = _future_rollout_candidate_score_records(
        candidates=candidates,
        future_scores=future_scores,
        selected_candidate_id=future_selected_candidate_id,
    )
    by_id = {
        str(record.get("id", record.get("trajectory_id", ""))): record.get("trajectory_score")
        for record in future_records
    }
    for record in records:
        candidate_id = str(record.get("id", record.get("trajectory_id", "")))
        score = by_id.get(candidate_id)
        if isinstance(score, dict):
            record["future_rollout_trajectory_score"] = dict(score)


def _future_rollout_candidate_score_records(
    *,
    candidates: list[CandidateTrajectory],
    future_scores: dict[str, np.ndarray | list[str] | str],
    selected_candidate_id: str,
) -> list[JsonDict]:
    collision = np.asarray(future_scores["candidate_collision"], dtype=np.float32)
    future_collision = np.asarray(future_scores.get("candidate_future_collision", collision), dtype=np.float32)
    unsafe_now = np.asarray(future_scores.get("candidate_unsafe_now", np.zeros_like(collision)), dtype=np.float32)
    unknown = np.asarray(future_scores["candidate_unknown_exposure"], dtype=np.float32)
    gain = np.asarray(future_scores["candidate_new_area_gain"], dtype=np.float32)
    progress = np.asarray(future_scores["candidate_progress"], dtype=np.float32)
    lower_score = np.asarray(future_scores["candidate_lower_is_better_score"], dtype=np.float32)
    formula = str(future_scores.get("scoring_formula", "unknown"))
    records: list[JsonDict] = []
    for index, candidate in enumerate(candidates):
        records.append(
            {
                **candidate.to_dict(),
                "trajectory_score": {
                    "schema_version": "homebrain.future_bev_rollout_trajectory_score.v1",
                    "scorer": FUTURE_BEV_ROLLOUT_V1_SOURCE,
                    "candidate_id": candidate.id,
                    "future_collision_probability": round(float(collision[index]), 6),
                    "future_collision_only_probability": round(float(future_collision[index]), 6),
                    "unsafe_now_probability": round(float(unsafe_now[index]), 6),
                    "future_unknown_exposure": round(float(unknown[index]), 6),
                    "future_new_area_gain": round(float(gain[index]), 6),
                    "future_progress": round(float(progress[index]), 6),
                    "total_score": round(float(lower_score[index]), 6),
                    "scoring_formula": formula,
                    "lower_is_better": True,
                    "selected_by_runtime_policy": candidate.id == selected_candidate_id,
                    "replay_only": True,
                    "not_executed": True,
                    "control_safe": False,
                    "product_training_approved": False,
                },
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
                "product_training_approved": False,
            }
        )
    return records


def _validate_future_candidate_ids(
    candidates: list[CandidateTrajectory],
    future_scores: dict[str, np.ndarray | list[str] | str],
) -> None:
    scored_ids = [str(value) for value in future_scores.get("candidate_ids", [])]
    expected = [candidate.id for candidate in candidates]
    if scored_ids != expected:
        raise ValueError(f"Future BEV rollout candidate ids do not match runtime candidates: {scored_ids} != {expected}")


def _select_future_rollout_candidate(
    *,
    candidates: list[CandidateTrajectory],
    future_scores: dict[str, np.ndarray | list[str] | str],
    transparent_decision: TrajectoryDecision,
    selection_mode: str,
    selection_history: list[str] | None,
) -> tuple[int, np.ndarray]:
    lower_score = np.asarray(future_scores["candidate_lower_is_better_score"], dtype=np.float32)
    if selection_mode == "argmin":
        return int(np.argmin(lower_score)), lower_score.copy()
    if selection_mode != "guided_transparent":
        raise ValueError("future_rollout_selection_mode must be 'argmin' or 'guided_transparent'")

    transparent_scores = np.asarray([score.total_score for score in transparent_decision.scores], dtype=np.float32)
    collision = np.asarray(future_scores["candidate_collision"], dtype=np.float32)
    unsafe_now = np.asarray(future_scores.get("candidate_unsafe_now", np.zeros_like(collision)), dtype=np.float32)
    unknown = np.asarray(future_scores["candidate_unknown_exposure"], dtype=np.float32)
    gain = np.asarray(future_scores["candidate_new_area_gain"], dtype=np.float32)
    progress = np.asarray(future_scores["candidate_progress"], dtype=np.float32)
    repeated = _recent_selection_penalty(candidates, selection_history or [])
    nonprogress = np.asarray(
        [
            0.18
            if abs(float(candidate.cmd_vel_proxy.get("linear_velocity_mps", 0.0))) < 1.0e-6
            and candidate.id != "stop"
            else 0.0
            for candidate in candidates
        ],
        dtype=np.float32,
    )
    future_norm = _normalise_scores(lower_score)
    guided = (
        transparent_scores
        + 0.20 * future_norm
        + 0.75 * collision
        + 0.50 * unsafe_now
        + 0.08 * unknown
        - 0.05 * gain
        - 0.03 * progress
        + repeated
        + nonprogress
    ).astype(np.float32)
    hard_unsafe = (collision >= 0.5) | (unsafe_now >= 0.5)
    guided = guided + hard_unsafe.astype(np.float32) * np.float32(100.0)
    return int(np.argmin(guided)), guided


def _normalise_scores(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if upper - lower <= 1.0e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - np.float32(lower)) / np.float32(upper - lower)).astype(np.float32)


def _recent_selection_penalty(candidates: list[CandidateTrajectory], history: list[str]) -> np.ndarray:
    if not history:
        return np.zeros((len(candidates),), dtype=np.float32)
    recent = history[-8:]
    values: list[float] = []
    for candidate in candidates:
        count = sum(1 for item in recent if item == candidate.id)
        last_penalty = 0.22 if recent and recent[-1] == candidate.id else 0.0
        values.append(last_penalty + 0.035 * count)
    return np.asarray(values, dtype=np.float32)


def _selected_candidate_index(candidates: list[CandidateTrajectory], selected_candidate_id: str) -> int:
    for index, candidate in enumerate(candidates):
        if candidate.id == selected_candidate_id:
            return index
    raise ValueError(f"selected candidate is not in candidate set: {selected_candidate_id}")


def _score_dict_for_candidate(decision: TrajectoryDecision, candidate_id: str) -> JsonDict:
    for score in decision.scores:
        if score.candidate_id == candidate_id:
            return score.to_dict()
    raise ValueError(f"candidate score missing for selected candidate: {candidate_id}")


def _score_dict_from_records(records: list[JsonDict], candidate_id: str) -> JsonDict:
    for record in records:
        record_id = str(record.get("id", record.get("trajectory_id", "")))
        if record_id != candidate_id:
            continue
        score = record.get("trajectory_score")
        return dict(score) if isinstance(score, dict) else {}
    return {}


def _nested_score_dict_from_records(records: list[JsonDict], candidate_id: str, field: str) -> JsonDict:
    for record in records:
        record_id = str(record.get("id", record.get("trajectory_id", "")))
        if record_id != candidate_id:
            continue
        score = record.get(field)
        return dict(score) if isinstance(score, dict) else {}
    return {}


def _coverage_update(before: JsonDict, after: JsonDict) -> JsonDict:
    return {
        "cells_seen_delta": int(after.get("coverage_memory_cells_seen", 0))
        - int(before.get("coverage_memory_cells_seen", 0)),
        "cells_covered_delta": int(after.get("coverage_memory_cells_covered", 0))
        - int(before.get("coverage_memory_cells_covered", 0)),
        "pose_aligned": bool(after.get("pose_aligned", False)),
        "pose_alignment_attempt_count": int(after.get("pose_alignment_attempt_count", 0)),
        "missing_pose_delta_count": int(after.get("missing_pose_delta_count", 0)),
    }


def _bounded_cmd_vel_proposal(candidate: CandidateTrajectory) -> JsonDict:
    proxy = candidate.cmd_vel_proxy
    linear = _clip_float(
        proxy.get("linear_velocity_mps", 0.0),
        -float(DEFAULT_CMD_VEL_LIMITS["max_linear_velocity_mps"]),
        float(DEFAULT_CMD_VEL_LIMITS["max_linear_velocity_mps"]),
    )
    angular = _clip_float(
        proxy.get("angular_velocity_radps", 0.0),
        -float(DEFAULT_CMD_VEL_LIMITS["max_angular_velocity_radps"]),
        float(DEFAULT_CMD_VEL_LIMITS["max_angular_velocity_radps"]),
    )
    return {
        "linear_velocity_mps": linear,
        "angular_velocity_radps": angular,
        "selected_candidate_id": candidate.id,
        "source": "selected_candidate_cmd_vel_proxy",
        "bounded": True,
        "not_hardware_control": True,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
    }


def _clip_float(value: object, lower: float, upper: float) -> float:
    number = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
    if not np.isfinite(number):
        number = 0.0
    return float(np.clip(number, lower, upper))


def metadata_float(metadata: dict[str, Any], key: str, default: float) -> float:
    value = metadata.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if np.isfinite(number) and number > 0.0:
            return number
    camera_config = metadata.get("camera_config")
    if isinstance(camera_config, dict):
        nested = camera_config.get(key)
        if isinstance(nested, (int, float)) and not isinstance(nested, bool):
            number = float(nested)
            if np.isfinite(number) and number > 0.0:
                return number
    return default


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
