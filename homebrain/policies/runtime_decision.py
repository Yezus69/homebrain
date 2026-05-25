from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

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


@dataclass(frozen=True)
class RuntimeTrajectoryScorer:
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


def decide_trajectory(
    *,
    bev: LocalBev,
    candidates: list[CandidateTrajectory],
    coverage_memory: CoverageMemory,
    pose_delta: tuple[float, float, float] | None,
    learned_scorer: RuntimeTrajectoryScorer | None = None,
    policy_bev_source: str,
    coverage_memory_reset: bool = False,
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
    if learned_scorer is None:
        selected_candidate_id = transparent_decision.selected_candidate_id
        candidate_records = _transparent_candidate_score_records(
            candidates=candidates,
            decision=transparent_decision,
        )
        scorer_name = TRANSPARENT_TRAJECTORY_SCORER_SOURCE
        scorer_mode = "transparent"
    else:
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
        scorer_name = TRAJECTORY_SCORER_V0_SOURCE
        scorer_mode = "learned"

    selected_index = _selected_candidate_index(candidates, selected_candidate_id)
    coverage_memory.update_current_frame(bev)
    coverage_after_update = coverage_memory.to_dict()
    selected_metrics = _score_dict_for_candidate(transparent_decision, selected_candidate_id)
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
