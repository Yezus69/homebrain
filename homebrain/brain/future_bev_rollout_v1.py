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
            "horizons_s": [float(value) for value in self.horizons_s],
            "meters_per_cell": float(self.meters_per_cell),
            "robot_radius_m": float(self.robot_radius_m),
            "bev_input_channels": list(BEV_OUTPUT_CHANNELS),
            "future_bev_channels": list(FUTURE_BEV_CHANNELS),
            "candidate_outcome_channels": list(CANDIDATE_OUTCOME_CHANNELS),
            "candidate_ids": [candidate.id for candidate in candidates],
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
        self.uncertainty_head = nn.Conv2d(hidden, len(config.horizons_s), kernel_size=1)
        self.candidate_head = nn.Sequential(
            nn.Linear(hidden + 5, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(CANDIDATE_OUTCOME_CHANNELS)),
        )
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
        context = self.sensor_adapter(sensor_mask).view(batch, -1, 1, 1)
        latent = self.trunk(bev_latent + feature_weight * feature_latent + context)
        future = self.future_head(latent)
        future = future.view(
            batch,
            len(self.config.horizons_s),
            len(FUTURE_BEV_CHANNELS),
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
        candidate_raw = self.candidate_head(torch.cat([pooled, static], dim=-1))
        return {
            "future_bev_logits": future,
            "uncertainty_logits": uncertainty,
            "uncertainty_grid": torch.sigmoid(uncertainty),
            "candidate_outcome_logits": candidate_raw,
            "candidate_collision_logit": candidate_raw[..., 0],
            "candidate_unknown_exposure_logit": candidate_raw[..., 1],
            "candidate_new_area_gain_logit": candidate_raw[..., 2],
            "candidate_progress_logit": candidate_raw[..., 3],
            "candidate_unknown_exposure": torch.sigmoid(candidate_raw[..., 1]),
            "candidate_new_area_gain": torch.sigmoid(candidate_raw[..., 2]),
            "candidate_progress": torch.sigmoid(candidate_raw[..., 3]),
        }

    def _candidate_pool(self, latent: torch.Tensor) -> torch.Tensor:
        batch, hidden, _height, _width = latent.shape
        masks = self.candidate_masks.to(device=latent.device, dtype=latent.dtype)
        denom = masks.sum(dim=(2, 3)).clamp_min(1.0)
        pooled = torch.einsum("bchw,nthw->bnc", latent, masks) / denom.view(1, -1, 1)
        if pooled.shape != (batch, len(self.candidate_ids), hidden):
            raise RuntimeError("candidate pooling produced an unexpected shape")
        return pooled


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
    with torch.no_grad():
        outputs = model(current, features, feature_mask, sensor)
    collision = torch.sigmoid(outputs["candidate_collision_logit"])[0].detach().cpu().numpy().astype(np.float32)
    unknown = outputs["candidate_unknown_exposure"][0].detach().cpu().numpy().astype(np.float32)
    gain = outputs["candidate_new_area_gain"][0].detach().cpu().numpy().astype(np.float32)
    progress = outputs["candidate_progress"][0].detach().cpu().numpy().astype(np.float32)
    lower_score = (8.0 * collision + 1.25 * unknown - 1.6 * gain - 0.8 * progress).astype(np.float32)
    return {
        "candidate_ids": list(model.candidate_ids),
        "candidate_collision": collision,
        "candidate_unknown_exposure": unknown,
        "candidate_new_area_gain": gain,
        "candidate_progress": progress,
        "candidate_lower_is_better_score": lower_score,
        "scoring_formula": "8*collision + 1.25*unknown - 1.6*new_area_gain - 0.8*progress",
    }


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
    model.load_state_dict(payload["state_dict"])
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
