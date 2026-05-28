from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib

import numpy as np
import torch

from homebrain.brain.counterfactual_dynamic_bev_world_model_v0 import (
    CounterfactualDynamicBEVWorldModelV0,
    load_checkpoint,
)
from homebrain.brain.modeld import SceneState
from homebrain.messages.schema import JsonDict
from homebrain.policies.candidate_trajectories import CandidateTrajectory
from homebrain.policies.trajectory_scorer import LocalBev

RUNTIME_SAFETY_DEBUG: JsonDict = {
    "no_future_labels_used_at_runtime": True,
    "no_teacher_fields_at_runtime": True,
    "replay_only": True,
    "not_executed": True,
    "control_safe": False,
    "raw_pwm_emitted": False,
    "hardware_validated": False,
}


@dataclass(frozen=True)
class RuntimeCounterfactualWorldModel:
    model: CounterfactualDynamicBEVWorldModelV0
    checkpoint: Path
    checkpoint_sha256: str
    metadata: JsonDict
    device: torch.device


def load_counterfactual_world_model(
    checkpoint: str | Path,
    device: str | torch.device | None = None,
) -> RuntimeCounterfactualWorldModel:
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_checkpoint(checkpoint, map_location=resolved_device)
    model.to(resolved_device)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return RuntimeCounterfactualWorldModel(
        model=model,
        checkpoint=Path(checkpoint),
        checkpoint_sha256=_file_sha256(Path(checkpoint)),
        metadata=dict(metadata),
        device=resolved_device,
    )


def predict_future_bev_from_scene_state(
    runtime: RuntimeCounterfactualWorldModel,
    bev_history: np.ndarray | torch.Tensor,
    pose_delta_history: np.ndarray | torch.Tensor,
    candidate_trajectories: np.ndarray | torch.Tensor,
    previous_action_history: np.ndarray | torch.Tensor | None = None,
    sensor_mask: np.ndarray | torch.Tensor | None = None,
) -> dict[str, Any]:
    model = runtime.model
    model.eval()
    history = _tensor(bev_history, device=runtime.device)
    if history.ndim == 4:
        history = history.unsqueeze(0)
    pose = _tensor(pose_delta_history, device=runtime.device)
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
    action = _optional_tensor(previous_action_history, device=runtime.device)
    if action is not None and action.ndim == 2:
        action = action.unsqueeze(0)
    sensor = _optional_tensor(sensor_mask, device=runtime.device)
    if sensor is not None and sensor.ndim == 2:
        sensor = sensor.unsqueeze(0)
    candidates = _tensor(candidate_trajectories, device=runtime.device)
    if candidates.ndim == 3:
        candidates = candidates.unsqueeze(0)
    with torch.no_grad():
        outputs = model(
            history,
            pose,
            candidates,
            previous_action_history=action,
            sensor_mask=sensor,
        )
    arrays = _outputs_to_numpy(outputs)
    return {
        "future": arrays,
        "debug": {
            "counterfactual_world_model_checkpoint": runtime.checkpoint.as_posix(),
            "counterfactual_world_model_checkpoint_sha256": runtime.checkpoint_sha256,
            "runtime_inputs": [
                "SceneState_or_LocalBev_history",
                "pose_delta_history",
                "previous_action_history",
                "candidate_trajectories",
                "sensor_mask",
            ],
            **RUNTIME_SAFETY_DEBUG,
        },
    }


def score_candidates_with_world_model(
    runtime: RuntimeCounterfactualWorldModel,
    local_bev_history: list[LocalBev] | np.ndarray | None = None,
    pose_delta_history: np.ndarray | torch.Tensor | None = None,
    candidates: list[CandidateTrajectory] | None = None,
    candidate_trajectories: np.ndarray | torch.Tensor | None = None,
    previous_action_history: np.ndarray | torch.Tensor | None = None,
    sensor_mask: np.ndarray | torch.Tensor | None = None,
    scene_state: SceneState | None = None,
) -> dict[str, Any]:
    history = _history_from_inputs(runtime, local_bev_history=local_bev_history, scene_state=scene_state)
    pose = _pose_history(runtime, pose_delta_history, history_steps=history.shape[0])
    trajectory_array = _trajectory_array(candidates=candidates, candidate_trajectories=candidate_trajectories)
    result = predict_future_bev_from_scene_state(
        runtime,
        history,
        pose,
        trajectory_array,
        previous_action_history=previous_action_history,
        sensor_mask=sensor_mask,
    )
    future = result["future"]
    candidate_ids = [candidate.id for candidate in candidates] if candidates is not None else [f"candidate_{i}" for i in range(trajectory_array.shape[0])]
    scores = np.asarray(future["candidate_score"], dtype=np.float32).reshape(-1)
    risk = np.asarray(future["candidate_risk"], dtype=np.float32).reshape(-1)
    dynamic = np.asarray(future["candidate_dynamic_risk"], dtype=np.float32).reshape(-1)
    unknown = np.asarray(future["candidate_unknown_exposure"], dtype=np.float32).reshape(-1)
    records = [
        {
            "candidate_id": candidate_id,
            "world_model_score": round(float(scores[index]), 6),
            "candidate_risk_probability": round(float(risk[index]), 6),
            "candidate_dynamic_risk_probability": round(float(dynamic[index]), 6),
            "candidate_unknown_exposure_probability": round(float(unknown[index]), 6),
            "lower_is_better": True,
            **RUNTIME_SAFETY_DEBUG,
        }
        for index, candidate_id in enumerate(candidate_ids)
    ]
    selected_index = int(np.argmin(scores)) if scores.size else 0
    return {
        "candidate_scores": scores,
        "candidate_risk": risk,
        "candidate_dynamic_risk": dynamic,
        "candidate_unknown_exposure": unknown,
        "selected_candidate_index": selected_index,
        "selected_candidate_id": candidate_ids[selected_index] if candidate_ids else None,
        "candidate_records": records,
        "future_risk_debug_maps": {
            "future_occupied": future["future_occupied"],
            "future_dynamic_risk": future["future_dynamic_risk"],
            "future_uncertainty": future["future_uncertainty"],
            "future_flow_xy": future["future_flow_xy"],
        },
        "replay_only_decision_debug_artifact": {
            "trajectory_scorer": "counterfactual_dynamic_bev_world_model_v0",
            "selected_candidate_id": candidate_ids[selected_index] if candidate_ids else None,
            **RUNTIME_SAFETY_DEBUG,
        },
        "debug": {**result["debug"], **RUNTIME_SAFETY_DEBUG},
    }


def _history_from_inputs(
    runtime: RuntimeCounterfactualWorldModel,
    *,
    local_bev_history: list[LocalBev] | np.ndarray | None,
    scene_state: SceneState | None,
) -> np.ndarray:
    steps = int(runtime.model.config.history_frames)
    if local_bev_history is None and scene_state is not None:
        history = scene_state.history_for_rollout(steps)
        if history is not None:
            if history.shape[1] == 5:
                pad = np.zeros((history.shape[0], 2, *history.shape[-2:]), dtype=np.float32)
                pad[:, 0] = history[:, 4]
                pad[:, 1] = history[:, 2]
                return np.concatenate([history, pad], axis=1).astype(np.float32)
    if isinstance(local_bev_history, np.ndarray):
        history = np.asarray(local_bev_history, dtype=np.float32)
        return _fit_history_channels(history, runtime.model.config.bev_channels)
    if isinstance(local_bev_history, list) and local_bev_history:
        return _fit_history_channels(np.stack([_local_bev_stack(item) for item in local_bev_history], axis=0), runtime.model.config.bev_channels)
    height, width = runtime.model.config.bev_shape
    history = np.zeros((steps, runtime.model.config.bev_channels, height, width), dtype=np.float32)
    history[:, 2] = 1.0
    history[:, 6] = 1.0
    return history


def _pose_history(
    runtime: RuntimeCounterfactualWorldModel,
    pose_delta_history: np.ndarray | torch.Tensor | None,
    *,
    history_steps: int,
) -> np.ndarray:
    dim = int(runtime.model.config.pose_delta_dim)
    if pose_delta_history is None:
        return np.zeros((history_steps, dim), dtype=np.float32)
    values = np.asarray(pose_delta_history, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    out = np.zeros((history_steps, dim), dtype=np.float32)
    clipped = values[-history_steps:]
    out[-clipped.shape[0] :, : min(dim, clipped.shape[1])] = clipped[:, : min(dim, clipped.shape[1])]
    return out


def _trajectory_array(
    *,
    candidates: list[CandidateTrajectory] | None,
    candidate_trajectories: np.ndarray | torch.Tensor | None,
) -> np.ndarray:
    if candidate_trajectories is not None:
        return np.asarray(candidate_trajectories, dtype=np.float32)
    if not candidates:
        raise ValueError("score_candidates_with_world_model requires candidates or candidate_trajectories")
    max_points = max(len(candidate.poses) for candidate in candidates)
    out = np.zeros((len(candidates), max_points, 3), dtype=np.float32)
    for candidate_index, candidate in enumerate(candidates):
        for pose_index in range(max_points):
            pose = candidate.poses[min(pose_index, len(candidate.poses) - 1)]
            out[candidate_index, pose_index] = [pose.x_m, pose.y_m, pose.yaw_rad]
    return out


def _fit_history_channels(history: np.ndarray, channels: int) -> np.ndarray:
    array = np.asarray(history, dtype=np.float32)
    if array.ndim == 3:
        array = array[None, ...]
    out = np.zeros((array.shape[0], channels, *array.shape[-2:]), dtype=np.float32)
    width = min(channels, array.shape[1])
    out[:, :width] = array[:, :width]
    if channels > 2 and width <= 2:
        out[:, 2] = 1.0
    if channels > 6 and width <= 6:
        out[:, 6] = out[:, 2]
    return out


def _local_bev_stack(bev: LocalBev) -> np.ndarray:
    free = np.asarray(bev.free, dtype=np.float32)
    occupied = np.asarray(bev.occupied, dtype=np.float32)
    unknown = np.asarray(bev.unknown, dtype=np.float32)
    traversable = np.asarray(bev.traversable, dtype=np.float32) if bev.traversable is not None else free
    risky = np.asarray(bev.risky, dtype=np.float32) if bev.risky is not None else occupied
    uncertainty = np.asarray(bev.uncertainty, dtype=np.float32) if bev.uncertainty is not None else unknown
    return np.stack([free, occupied, unknown, traversable, risky, risky, uncertainty], axis=0).astype(np.float32)


def _outputs_to_numpy(outputs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        "future_occupied": torch.sigmoid(outputs["future_occupied_logits"])[0].detach().cpu().numpy().astype(np.float32),
        "future_free": torch.sigmoid(outputs["future_free_logits"])[0].detach().cpu().numpy().astype(np.float32),
        "future_unknown": torch.sigmoid(outputs["future_unknown_logits"])[0].detach().cpu().numpy().astype(np.float32),
        "future_risky": torch.sigmoid(outputs["future_risky_logits"])[0].detach().cpu().numpy().astype(np.float32),
        "future_dynamic_risk": torch.sigmoid(outputs["future_dynamic_risk_logits"])[0].detach().cpu().numpy().astype(np.float32),
        "future_uncertainty": torch.sigmoid(outputs["future_uncertainty_logits"])[0].detach().cpu().numpy().astype(np.float32),
        "future_flow_xy": outputs["future_flow_xy"][0].detach().cpu().numpy().astype(np.float32),
        "candidate_risk": torch.sigmoid(outputs["candidate_risk_logits"])[0].detach().cpu().numpy().astype(np.float32),
        "candidate_dynamic_risk": torch.sigmoid(outputs["candidate_dynamic_risk_logits"])[0].detach().cpu().numpy().astype(np.float32),
        "candidate_unknown_exposure": torch.sigmoid(outputs["candidate_unknown_exposure_logits"])[0].detach().cpu().numpy().astype(np.float32),
        "candidate_score": outputs["candidate_score"][0].detach().cpu().numpy().astype(np.float32),
    }


def _tensor(value: np.ndarray | torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.from_numpy(np.asarray(value, dtype=np.float32).copy()).to(device)


def _optional_tensor(value: np.ndarray | torch.Tensor | None, *, device: torch.device) -> torch.Tensor | None:
    return None if value is None else _tensor(value, device=device)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
