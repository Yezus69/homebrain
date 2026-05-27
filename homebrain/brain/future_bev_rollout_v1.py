from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from homebrain.policies.candidate_trajectories import CandidateTrajectory, generate_default_candidates
from homebrain.policies.trajectory_scorer import LocalBev
from homebrain.train import FUTURE_BEV_ROLLOUT_V1_CHECKPOINT_VERSION
from homebrain.train.future_bev_rollout_dataset import DEFAULT_FUTURE_HORIZONS_S, FUTURE_BEV_CHANNELS
from homebrain.train.spatial_dataset import BEV_OUTPUT_CHANNELS

FUTURE_BEV_ROLLOUT_V1_SOURCE = "future_bev_rollout_v1"
FUTURE_BEV_ROLLOUT_V1_CONFIG_SCHEMA_VERSION = "homebrain.future_bev_rollout_v1.config.v1"
CANDIDATE_OUTCOME_CHANNELS: tuple[str, ...] = (
    "collision",
    "future_collision",
    "unsafe_now",
    "unknown_exposure",
    "new_area_gain",
    "progress",
)


@dataclass(frozen=True)
class FutureBEVRolloutV1Config:
    bev_shape: tuple[int, int]
    feature_dim: int = 1
    sensor_dim: int = 4
    hidden_channels: int = 48
    history_steps: int = 1
    horizons_s: tuple[float, ...] = DEFAULT_FUTURE_HORIZONS_S
    meters_per_cell: float = 0.05
    robot_radius_m: float = 0.18
    schema_version: str = FUTURE_BEV_ROLLOUT_V1_CONFIG_SCHEMA_VERSION
    model_version: str = "FutureBEVRolloutV1"

    def __post_init__(self) -> None:
        if self.bev_shape[0] <= 1 or self.bev_shape[1] <= 1:
            raise ValueError("bev_shape dimensions must be greater than one")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive; use a masked 1-channel placeholder when DINO is absent")
        if self.sensor_dim <= 0:
            raise ValueError("sensor_dim must be positive")
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if self.history_steps <= 0:
            raise ValueError("history_steps must be positive")
        if not self.horizons_s or any(float(value) <= 0.0 for value in self.horizons_s):
            raise ValueError("horizons_s must contain positive values")
        if self.meters_per_cell <= 0.0:
            raise ValueError("meters_per_cell must be positive")
        if self.robot_radius_m < 0.0:
            raise ValueError("robot_radius_m must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        candidates = generate_default_candidates(
            grid_shape=self.bev_shape,
            meters_per_cell=self.meters_per_cell,
            robot_radius_m=self.robot_radius_m,
        )
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "bev_shape": list(self.bev_shape),
            "feature_dim": int(self.feature_dim),
            "sensor_dim": int(self.sensor_dim),
            "hidden_channels": int(self.hidden_channels),
            "history_steps": int(self.history_steps),
            "horizons_s": [float(value) for value in self.horizons_s],
            "meters_per_cell": float(self.meters_per_cell),
            "robot_radius_m": float(self.robot_radius_m),
            "bev_input_channels": list(BEV_OUTPUT_CHANNELS),
            "future_bev_channels": list(FUTURE_BEV_CHANNELS),
            "future_risk_channels": ["risk"],
            "candidate_outcome_channels": list(CANDIDATE_OUTCOME_CHANNELS),
            "candidate_ids": [candidate.id for candidate in candidates],
            "uses_memory_bev_input": True,
            "uses_bev_history_input": True,
            "uses_uncertainty_map_input": True,
            "uses_action_conditioned_future_risk": True,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FutureBEVRolloutV1Config":
        shape = data.get("bev_shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError("checkpoint config must include bev_shape")
        horizons_raw = data.get("horizons_s", DEFAULT_FUTURE_HORIZONS_S)
        if not isinstance(horizons_raw, (list, tuple)):
            raise ValueError("checkpoint config horizons_s must be a list")
        return cls(
            bev_shape=(int(shape[0]), int(shape[1])),
            feature_dim=int(data.get("feature_dim", 1)),
            sensor_dim=int(data.get("sensor_dim", 4)),
            hidden_channels=int(data.get("hidden_channels", 48)),
            history_steps=int(data.get("history_steps", 1)),
            horizons_s=tuple(float(value) for value in horizons_raw),
            meters_per_cell=float(data.get("meters_per_cell", 0.05)),
            robot_radius_m=float(data.get("robot_radius_m", 0.18)),
        )


class FutureBEVRolloutV1(nn.Module):
    def __init__(self, config: FutureBEVRolloutV1Config) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_channels)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(len(BEV_OUTPUT_CHANNELS), hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(config.feature_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.memory_encoder = nn.Sequential(
            nn.Conv2d(len(BEV_OUTPUT_CHANNELS), hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.history_encoder = nn.Sequential(
            nn.Conv2d(len(BEV_OUTPUT_CHANNELS) * int(config.history_steps), hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.uncertainty_encoder = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.sensor_adapter = nn.Sequential(
            nn.Linear(config.sensor_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.trunk = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.future_head = nn.Conv2d(hidden, len(config.horizons_s) * len(FUTURE_BEV_CHANNELS), kernel_size=1)
        self.future_risk_head = nn.Conv2d(hidden, len(config.horizons_s), kernel_size=1)
        self.uncertainty_head = nn.Conv2d(hidden, len(config.horizons_s), kernel_size=1)
        self.candidate_head = nn.Sequential(
            nn.Linear(hidden + 5, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(CANDIDATE_OUTCOME_CHANNELS)),
        )
        self.candidate_future_risk_head = nn.Sequential(
            nn.Linear(hidden + 5, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(config.horizons_s)),
        )
        _zero_module(self.memory_encoder)
        _zero_module(self.history_encoder)
        _zero_module(self.uncertainty_encoder)
        candidates = generate_default_candidates(
            grid_shape=config.bev_shape,
            meters_per_cell=config.meters_per_cell,
            robot_radius_m=config.robot_radius_m,
        )
        self.candidates = candidates
        self.candidate_ids = tuple(candidate.id for candidate in candidates)
        masks, static = _candidate_tensors(candidates, config.bev_shape)
        self.register_buffer("candidate_masks", masks, persistent=False)
        self.register_buffer("candidate_static_features", static, persistent=False)

    def forward(
        self,
        current_bev: torch.Tensor,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        sensor_mask: torch.Tensor,
        memory_bev: torch.Tensor | None = None,
        bev_history: torch.Tensor | None = None,
        uncertainty_map: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if current_bev.ndim != 4:
            raise ValueError("current_bev must be BCHW")
        if current_bev.shape[1] != len(BEV_OUTPUT_CHANNELS):
            raise ValueError(f"current_bev must have {len(BEV_OUTPUT_CHANNELS)} channels")
        batch = current_bev.shape[0]
        if sensor_mask.ndim != 2 or sensor_mask.shape[1] != self.config.sensor_dim:
            raise ValueError(f"sensor_mask must be Bx{self.config.sensor_dim}")
        bev_latent = self.bev_encoder(current_bev)
        feature_latent = self.feature_encoder(features)
        feature_latent = F.interpolate(
            feature_latent,
            size=self.config.bev_shape,
            mode="bilinear",
            align_corners=False,
        )
        feature_weight = feature_mask.view(batch, 1, 1, 1).to(device=current_bev.device, dtype=current_bev.dtype)
        memory_latent = self.memory_encoder(_memory_input(current_bev, memory_bev))
        history_latent = self.history_encoder(_history_input(current_bev, bev_history, self.config.history_steps))
        uncertainty_latent = self.uncertainty_encoder(_uncertainty_input(current_bev, uncertainty_map))
        context = self.sensor_adapter(sensor_mask).view(batch, -1, 1, 1)
        latent = self.trunk(bev_latent + memory_latent + history_latent + uncertainty_latent + feature_weight * feature_latent + context)
        future = self.future_head(latent)
        future = future.view(
            batch,
            len(self.config.horizons_s),
            len(FUTURE_BEV_CHANNELS),
            self.config.bev_shape[0],
            self.config.bev_shape[1],
        )
        future_risk = self.future_risk_head(latent).view(
            batch,
            len(self.config.horizons_s),
            1,
            self.config.bev_shape[0],
            self.config.bev_shape[1],
        )
        uncertainty = self.uncertainty_head(latent).view(
            batch,
            len(self.config.horizons_s),
            1,
            self.config.bev_shape[0],
            self.config.bev_shape[1],
        )
        pooled = self._candidate_pool(latent)
        static = self.candidate_static_features.to(device=latent.device, dtype=latent.dtype)
        static = static.unsqueeze(0).expand(batch, -1, -1)
        candidate_context = torch.cat([pooled, static], dim=-1)
        candidate_raw = self.candidate_head(candidate_context)
        candidate_future_risk = self.candidate_future_risk_head(candidate_context)
        return {
            "future_bev_logits": future,
            "future_risk_logits": future_risk,
            "future_risk_prob": torch.sigmoid(future_risk),
            "uncertainty_logits": uncertainty,
            "uncertainty_grid": torch.sigmoid(uncertainty),
            "candidate_outcome_logits": candidate_raw,
            "candidate_future_risk_logits": candidate_future_risk,
            "candidate_horizon_future_risk": torch.sigmoid(candidate_future_risk),
            "candidate_collision_logit": candidate_raw[..., 0],
            "candidate_future_collision_logit": candidate_raw[..., 1],
            "candidate_unsafe_now_logit": candidate_raw[..., 2],
            "candidate_unknown_exposure_logit": candidate_raw[..., 3],
            "candidate_new_area_gain_logit": candidate_raw[..., 4],
            "candidate_progress_logit": candidate_raw[..., 5],
            "candidate_future_collision": torch.sigmoid(candidate_raw[..., 1]),
            "candidate_unsafe_now": torch.sigmoid(candidate_raw[..., 2]),
            "candidate_unknown_exposure": torch.sigmoid(candidate_raw[..., 3]),
            "candidate_new_area_gain": torch.sigmoid(candidate_raw[..., 4]),
            "candidate_progress": torch.sigmoid(candidate_raw[..., 5]),
        }

    def _candidate_pool(self, latent: torch.Tensor) -> torch.Tensor:
        batch, hidden, _height, _width = latent.shape
        masks = self.candidate_masks.to(device=latent.device, dtype=latent.dtype)
        denom = masks.sum(dim=(2, 3)).clamp_min(1.0)
        pooled = torch.einsum("bchw,nthw->bnc", latent, masks) / denom.view(1, -1, 1)
        if pooled.shape != (batch, len(self.candidate_ids), hidden):
            raise RuntimeError("candidate pooling produced an unexpected shape")
        return pooled


def _zero_module(module: nn.Module) -> None:
    layers = [layer for layer in module.modules() if isinstance(layer, (nn.Conv2d, nn.Linear))]
    if not layers:
        return
    layer = layers[-1]
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _memory_input(current_bev: torch.Tensor, memory_bev: torch.Tensor | None) -> torch.Tensor:
    if memory_bev is None:
        return current_bev
    if memory_bev.shape != current_bev.shape:
        return current_bev
    return memory_bev.to(device=current_bev.device, dtype=current_bev.dtype).clamp(0.0, 1.0)


def _history_input(current_bev: torch.Tensor, bev_history: torch.Tensor | None, history_steps: int) -> torch.Tensor:
    batch, channels, height, width = current_bev.shape
    steps = max(1, int(history_steps))
    if bev_history is None or bev_history.ndim != 5:
        history = current_bev.unsqueeze(1).expand(batch, steps, channels, height, width)
    else:
        history = bev_history.to(device=current_bev.device, dtype=current_bev.dtype)
        if history.shape[0] != batch or history.shape[2:] != (channels, height, width):
            history = current_bev.unsqueeze(1).expand(batch, steps, channels, height, width)
        elif history.shape[1] > steps:
            history = history[:, -steps:]
        elif history.shape[1] < steps:
            pad = history[:, :1].expand(batch, steps - history.shape[1], channels, height, width)
            history = torch.cat([pad, history], dim=1)
    return history.clamp(0.0, 1.0).reshape(batch, steps * channels, height, width)


def _uncertainty_input(current_bev: torch.Tensor, uncertainty_map: torch.Tensor | None) -> torch.Tensor:
    if uncertainty_map is None or uncertainty_map.ndim != 4 or uncertainty_map.shape[1] != 1:
        return current_bev[:, 2:3].clamp(0.0, 1.0)
    if uncertainty_map.shape[0] != current_bev.shape[0] or uncertainty_map.shape[-2:] != current_bev.shape[-2:]:
        return current_bev[:, 2:3].clamp(0.0, 1.0)
    return uncertainty_map.to(device=current_bev.device, dtype=current_bev.dtype).clamp(0.0, 1.0)


def current_bev_from_local_bev(bev: LocalBev) -> np.ndarray:
    bev.validate()
    free = _prob(bev.free)
    occupied = _prob(bev.occupied)
    unknown = _prob(bev.unknown)
    traversable = _prob(bev.traversable) if bev.traversable is not None else free.copy()
    risky = _prob(bev.risky) if bev.risky is not None else occupied.copy()
    return np.stack([free, occupied, unknown, traversable, risky], axis=0).astype(np.float32)


def prepare_feature_tensor(
    *,
    model: FutureBEVRolloutV1,
    patch_features: np.ndarray | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_dim = int(model.config.feature_dim)
    if patch_features is None:
        features = torch.zeros((1, feature_dim, 1, 1), dtype=torch.float32, device=device)
        mask = torch.zeros((1,), dtype=torch.float32, device=device)
        return features, mask
    array = np.asarray(patch_features, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != feature_dim:
        features = torch.zeros((1, feature_dim, 1, 1), dtype=torch.float32, device=device)
        mask = torch.zeros((1,), dtype=torch.float32, device=device)
        return features, mask
    features = torch.from_numpy(np.transpose(array, (2, 0, 1))[None, ...].copy()).to(device)
    mask = torch.ones((1,), dtype=torch.float32, device=device)
    return features, mask


def score_local_bev_with_future_rollout(
    *,
    model: FutureBEVRolloutV1,
    bev: LocalBev,
    patch_features: np.ndarray | None,
    sensor_mask: np.ndarray | None,
    device: torch.device,
    memory_bev: np.ndarray | None = None,
    bev_history: np.ndarray | None = None,
    uncertainty_map: np.ndarray | None = None,
) -> dict[str, np.ndarray | list[str] | str]:
    model.eval()
    current = torch.from_numpy(current_bev_from_local_bev(bev)[None, ...]).to(device)
    features, feature_mask = prepare_feature_tensor(model=model, patch_features=patch_features, device=device)
    if sensor_mask is None:
        sensor = torch.zeros((1, model.config.sensor_dim), dtype=torch.float32, device=device)
    else:
        sensor_array = np.asarray(sensor_mask, dtype=np.float32).reshape(1, -1)
        if sensor_array.shape[1] != model.config.sensor_dim:
            sensor_array = np.zeros((1, model.config.sensor_dim), dtype=np.float32)
        sensor = torch.from_numpy(sensor_array).to(device)
    memory_tensor = _optional_batched_bev_tensor(memory_bev, current=current, device=device)
    history_tensor = _optional_history_tensor(
        bev_history,
        current=current,
        history_steps=int(model.config.history_steps),
        device=device,
    )
    uncertainty_tensor = _optional_uncertainty_tensor(uncertainty_map, current=current, device=device)
    with torch.no_grad():
        outputs = model(
            current,
            features,
            feature_mask,
            sensor,
            memory_bev=memory_tensor,
            bev_history=history_tensor,
            uncertainty_map=uncertainty_tensor,
        )
    future_bev = torch.sigmoid(outputs["future_bev_logits"])[0].detach().cpu().numpy().astype(np.float32)
    future_risk_grid = outputs["future_risk_prob"][0, :, 0].detach().cpu().numpy().astype(np.float32)
    future_uncertainty = outputs["uncertainty_grid"][0, :, 0].detach().cpu().numpy().astype(np.float32)
    collision = torch.sigmoid(outputs["candidate_collision_logit"])[0].detach().cpu().numpy().astype(np.float32)
    future_collision = outputs["candidate_future_collision"][0].detach().cpu().numpy().astype(np.float32)
    candidate_horizon_risk = outputs["candidate_horizon_future_risk"][0].detach().cpu().numpy().astype(np.float32)
    candidate_future_risk = np.max(candidate_horizon_risk, axis=1).astype(np.float32)
    unsafe_now = outputs["candidate_unsafe_now"][0].detach().cpu().numpy().astype(np.float32)
    unknown = outputs["candidate_unknown_exposure"][0].detach().cpu().numpy().astype(np.float32)
    gain = outputs["candidate_new_area_gain"][0].detach().cpu().numpy().astype(np.float32)
    progress = outputs["candidate_progress"][0].detach().cpu().numpy().astype(np.float32)
    stop_penalty = np.asarray([0.8 if candidate_id == "stop" else 0.0 for candidate_id in model.candidate_ids], dtype=np.float32)
    combined_collision = np.maximum(collision, np.maximum(future_collision, candidate_future_risk))
    lower_score = (
        8.0 * combined_collision + 2.0 * unsafe_now + 1.25 * unknown - 2.4 * gain - 2.0 * progress + stop_penalty
    ).astype(np.float32)
    return {
        "candidate_ids": list(model.candidate_ids),
        "candidate_collision": combined_collision,
        "candidate_future_collision": future_collision,
        "candidate_future_risk": candidate_future_risk,
        "candidate_horizon_future_risk": candidate_horizon_risk,
        "candidate_unsafe_now": unsafe_now,
        "candidate_unknown_exposure": unknown,
        "candidate_new_area_gain": gain,
        "candidate_progress": progress,
        "candidate_lower_is_better_score": lower_score,
        "future_bev_prob": future_bev,
        "future_risk_prob": future_risk_grid,
        "future_uncertainty_grid": future_uncertainty,
        "future_horizons_s": np.asarray(model.config.horizons_s, dtype=np.float32),
        "scoring_formula": (
            "8*max(collision,future_collision,candidate_future_risk) + 2*unsafe_now + 1.25*unknown "
            "- 2.4*new_area_gain - 2*progress + 0.8*stop"
        ),
    }


def _optional_batched_bev_tensor(
    value: np.ndarray | None,
    *,
    current: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    expected = tuple(int(size) for size in current.shape[1:])
    if array.shape == expected:
        array = array[None, ...]
    elif array.shape != tuple(int(size) for size in current.shape):
        return None
    return torch.from_numpy(np.clip(array, 0.0, 1.0).astype(np.float32).copy()).to(device)


def _optional_history_tensor(
    value: np.ndarray | None,
    *,
    current: torch.Tensor,
    history_steps: int,
    device: torch.device,
) -> torch.Tensor | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    expected_step_shape = tuple(int(size) for size in current.shape[1:])
    if array.ndim == 4 and array.shape[1:] == expected_step_shape:
        array = array[None, ...]
    elif array.ndim != 5 or array.shape[0] != current.shape[0] or array.shape[2:] != expected_step_shape:
        return None
    steps = max(1, int(history_steps))
    if array.shape[1] > steps:
        array = array[:, -steps:]
    elif array.shape[1] < steps:
        pad = np.repeat(array[:, :1], steps - array.shape[1], axis=1)
        array = np.concatenate([pad, array], axis=1)
    return torch.from_numpy(np.clip(array, 0.0, 1.0).astype(np.float32).copy()).to(device)


def _optional_uncertainty_tensor(
    value: np.ndarray | None,
    *,
    current: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    expected_hw = tuple(int(size) for size in current.shape[-2:])
    if array.shape == expected_hw:
        array = array[None, None, ...]
    elif array.shape == (1, *expected_hw):
        array = array[None, ...]
    elif array.shape != (int(current.shape[0]), 1, *expected_hw):
        return None
    return torch.from_numpy(np.clip(array, 0.0, 1.0).astype(np.float32).copy()).to(device)


def checkpoint_payload(
    model: FutureBEVRolloutV1,
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_version": FUTURE_BEV_ROLLOUT_V1_CHECKPOINT_VERSION,
        "model_name": "FutureBEVRolloutV1",
        "model_config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "metadata": metadata,
        "metrics": metrics,
    }


def save_checkpoint(
    path: str | Path,
    model: FutureBEVRolloutV1,
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, metadata=metadata, metrics=metrics), target)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[FutureBEVRolloutV1, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("checkpoint_version") != FUTURE_BEV_ROLLOUT_V1_CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {payload.get('checkpoint_version')!r}; "
            f"expected {FUTURE_BEV_ROLLOUT_V1_CHECKPOINT_VERSION!r}"
        )
    config = FutureBEVRolloutV1Config.from_dict(payload["model_config"])
    model = FutureBEVRolloutV1(config)
    state_dict = payload["state_dict"]
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
    if unexpected_keys:
        raise ValueError(f"unexpected FutureBEV rollout checkpoint keys: {unexpected_keys}")
    if len(compatible_state) != len(state_dict):
        payload["load_warnings"] = {
            "skipped_incompatible_keys": sorted(set(state_dict) - set(compatible_state)),
            "missing_new_model_keys": sorted(missing_keys),
        }
    model.eval()
    return model, payload


def _candidate_tensors(candidates: list[CandidateTrajectory], shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    masks = torch.zeros((len(candidates), 1, shape[0], shape[1]), dtype=torch.float32)
    max_forward = max((max((pose.x_m for pose in candidate.poses), default=0.0) for candidate in candidates), default=1.0)
    max_forward = max(float(max_forward), 1.0e-6)
    static: list[list[float]] = []
    for index, candidate in enumerate(candidates):
        for row, col in candidate.footprint_cells:
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                masks[index, 0, row, col] = 1.0
        linear = float(candidate.cmd_vel_proxy.get("linear_velocity_mps", 0.0))
        angular = float(candidate.cmd_vel_proxy.get("angular_velocity_radps", 0.0))
        terminal = candidate.poses[-1] if candidate.poses else None
        terminal_forward = 0.0 if terminal is None else max(0.0, float(terminal.x_m)) / max_forward
        static.append(
            [
                linear / 0.30,
                angular / 1.25,
                candidate.duration_s / 2.5,
                1.0 if candidate.id == "stop" else 0.0,
                terminal_forward,
            ]
        )
    return masks, torch.tensor(static, dtype=torch.float32)


def _prob(array: np.ndarray | None) -> np.ndarray:
    if array is None:
        raise ValueError("BEV probability array is missing")
    return np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
