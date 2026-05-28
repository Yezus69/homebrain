from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from homebrain.train import COUNTERFACTUAL_DYNAMIC_BEV_WORLD_MODEL_V0_CHECKPOINT_VERSION
from homebrain.train.dynamic_bev_world_pack_v0 import DYNAMIC_BEV_HISTORY_CHANNELS

COUNTERFACTUAL_DYNAMIC_BEV_WORLD_MODEL_V0_SOURCE = "counterfactual_dynamic_bev_world_model_v0"
COUNTERFACTUAL_DYNAMIC_BEV_WORLD_MODEL_V0_CONFIG_SCHEMA_VERSION = (
    "homebrain.counterfactual_dynamic_bev_world_model_v0.config.v0"
)


@dataclass(frozen=True)
class CounterfactualDynamicBEVWorldModelV0Config:
    bev_shape: tuple[int, int]
    history_frames: int = 6
    bev_channels: int = len(DYNAMIC_BEV_HISTORY_CHANNELS)
    pose_delta_dim: int = 3
    previous_action_dim: int = 2
    sensor_mask_dim: int = 4
    hidden_channels: int = 32
    future_horizons_sec: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    meters_per_cell: float = 0.05
    robot_radius_m: float = 0.18
    schema_version: str = COUNTERFACTUAL_DYNAMIC_BEV_WORLD_MODEL_V0_CONFIG_SCHEMA_VERSION
    model_version: str = "CounterfactualDynamicBEVWorldModelV0"

    def __post_init__(self) -> None:
        if self.bev_shape[0] <= 1 or self.bev_shape[1] <= 1:
            raise ValueError("bev_shape dimensions must be greater than one")
        if self.history_frames <= 0:
            raise ValueError("history_frames must be positive")
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if not self.future_horizons_sec:
            raise ValueError("at least one future horizon is required")
        if self.meters_per_cell <= 0.0:
            raise ValueError("meters_per_cell must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "bev_shape": [int(self.bev_shape[0]), int(self.bev_shape[1])],
            "history_frames": int(self.history_frames),
            "bev_channels": int(self.bev_channels),
            "bev_channel_names": list(DYNAMIC_BEV_HISTORY_CHANNELS),
            "pose_delta_dim": int(self.pose_delta_dim),
            "previous_action_dim": int(self.previous_action_dim),
            "sensor_mask_dim": int(self.sensor_mask_dim),
            "hidden_channels": int(self.hidden_channels),
            "future_horizons_sec": [float(value) for value in self.future_horizons_sec],
            "meters_per_cell": float(self.meters_per_cell),
            "robot_radius_m": float(self.robot_radius_m),
            "runtime_inputs": [
                "SceneState_or_LocalBev_history",
                "pose_delta_history",
                "previous_action_history",
                "candidate_trajectories",
                "sensor_mask",
                "checkpoint",
            ],
            "forbidden_runtime_fields": [
                "target_bev",
                "future_bev",
                "future_labels",
                "teacher_masks",
                "route_ground_truth",
                "candidate_oracle_cost",
                "ground_truth_global_trajectory",
                "future_frames",
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CounterfactualDynamicBEVWorldModelV0Config":
        shape = data.get("bev_shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("checkpoint config must include bev_shape")
        horizons = data.get("future_horizons_sec", data.get("horizons_s", (0.5, 1.0, 2.0, 3.0)))
        if not isinstance(horizons, (list, tuple)):
            raise ValueError("checkpoint horizons must be a list")
        return cls(
            bev_shape=(int(shape[0]), int(shape[1])),
            history_frames=int(data.get("history_frames", 6)),
            bev_channels=int(data.get("bev_channels", len(DYNAMIC_BEV_HISTORY_CHANNELS))),
            pose_delta_dim=int(data.get("pose_delta_dim", 3)),
            previous_action_dim=int(data.get("previous_action_dim", 2)),
            sensor_mask_dim=int(data.get("sensor_mask_dim", 4)),
            hidden_channels=int(data.get("hidden_channels", 32)),
            future_horizons_sec=tuple(float(value) for value in horizons),
            meters_per_cell=float(data.get("meters_per_cell", 0.05)),
            robot_radius_m=float(data.get("robot_radius_m", 0.18)),
        )


class CounterfactualDynamicBEVWorldModelV0(nn.Module):
    def __init__(self, config: CounterfactualDynamicBEVWorldModelV0Config) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_channels)
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(config.bev_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, min(8, hidden // 4)), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.step_context = nn.Sequential(
            nn.Linear(config.pose_delta_dim + config.previous_action_dim + config.sensor_mask_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.temporal = _ConvGRUCell(hidden, hidden)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        horizon_count = len(config.future_horizons_sec)
        self.future_logits_head = nn.Conv2d(hidden, horizon_count * 6, kernel_size=1)
        self.flow_head = nn.Conv2d(hidden, horizon_count * 2, kernel_size=1)
        self.candidate_mlp = nn.Sequential(
            nn.Linear(hidden + 8, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def forward(
        self,
        bev_history: torch.Tensor,
        pose_delta_history: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        previous_action_history: torch.Tensor | None = None,
        sensor_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if bev_history.ndim != 5:
            raise ValueError("bev_history must be BxTxCxHxW")
        batch, steps, channels, height, width = bev_history.shape
        if channels != self.config.bev_channels:
            raise ValueError(f"bev_history must have {self.config.bev_channels} channels")
        if (height, width) != self.config.bev_shape:
            raise ValueError(f"bev_history shape {(height, width)} does not match config {self.config.bev_shape}")
        history = _fit_history(bev_history.float(), self.config.history_frames)
        pose = _fit_sequence(
            pose_delta_history,
            batch=batch,
            steps=self.config.history_frames,
            dim=self.config.pose_delta_dim,
            device=history.device,
            dtype=history.dtype,
        )
        action = _fit_sequence(
            previous_action_history,
            batch=batch,
            steps=self.config.history_frames,
            dim=self.config.previous_action_dim,
            device=history.device,
            dtype=history.dtype,
        )
        sensor = _fit_sensor(
            sensor_mask,
            batch=batch,
            steps=self.config.history_frames,
            dim=self.config.sensor_mask_dim,
            device=history.device,
            dtype=history.dtype,
        )
        state = torch.zeros((batch, self.config.hidden_channels, height, width), dtype=history.dtype, device=history.device)
        for step in range(self.config.history_frames):
            frame = self.frame_encoder(history[:, step].clamp(0.0, 1.0))
            context = torch.cat([pose[:, step], action[:, step], sensor[:, step]], dim=1)
            frame = frame + self.step_context(context).view(batch, -1, 1, 1)
            state = self.temporal(frame, state)
        latent = self.decoder(state)
        horizon_count = len(self.config.future_horizons_sec)
        future_residual = self.future_logits_head(latent).view(batch, horizon_count, 6, height, width)
        current = history[:, -1]
        base = torch.stack(
            [
                current[:, 1],
                current[:, 0],
                current[:, 2],
                current[:, 4],
                current[:, 5],
                current[:, 6],
            ],
            dim=1,
        )
        future = future_residual + _prob_to_logit(base).unsqueeze(1)
        flow = self.flow_head(latent).view(batch, horizon_count, 2, height, width)
        masks = candidate_masks_from_trajectories(
            candidate_trajectories,
            grid_shape=self.config.bev_shape,
            meters_per_cell=self.config.meters_per_cell,
            robot_radius_m=self.config.robot_radius_m,
        ).to(device=history.device, dtype=history.dtype)
        pooled = _pool_candidate_features(
            future=future,
            latent=latent,
            masks=masks,
            trajectories=candidate_trajectories.to(device=history.device, dtype=history.dtype),
        )
        mlp = self.candidate_mlp(pooled)
        risk_logit = _prob_to_logit(pooled[..., 0]).add(mlp[..., 0])
        dynamic_logit = _prob_to_logit(pooled[..., 2]).add(mlp[..., 1])
        unknown_logit = _prob_to_logit(pooled[..., 3]).add(mlp[..., 2])
        candidate_score = (
            torch.sigmoid(risk_logit)
            + 0.50 * torch.sigmoid(dynamic_logit)
        )
        return {
            "future_occupied_logits": future[:, :, 0],
            "future_free_logits": future[:, :, 1],
            "future_unknown_logits": future[:, :, 2],
            "future_risky_logits": future[:, :, 3],
            "future_dynamic_risk_logits": future[:, :, 4],
            "future_uncertainty_logits": future[:, :, 5],
            "future_flow_xy": flow,
            "candidate_risk_logits": risk_logit,
            "candidate_dynamic_risk_logits": dynamic_logit,
            "candidate_unknown_exposure_logits": unknown_logit,
            "candidate_score": candidate_score,
            "latent_memory_debug": state.detach(),
        }


class _ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.gates = nn.Conv2d(input_channels + hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1)

    def forward(self, value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([value, state], dim=1)
        reset, update = torch.chunk(torch.sigmoid(self.gates(combined)), 2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([value, reset * state], dim=1)))
        return (1.0 - update) * state + update * candidate


def candidate_masks_from_trajectories(
    candidate_trajectories: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
    meters_per_cell: float,
    robot_radius_m: float,
) -> torch.Tensor:
    if candidate_trajectories.ndim != 4 or candidate_trajectories.shape[-1] < 2:
        raise ValueError("candidate_trajectories must be BxKxPx3")
    device = candidate_trajectories.device
    batch, candidate_count, point_count, _dims = candidate_trajectories.shape
    height, width = grid_shape
    x = candidate_trajectories[..., 0]
    y = candidate_trajectories[..., 1]
    rows = height - 1 - torch.floor(x / meters_per_cell).to(torch.long)
    cols = torch.floor((y + (width * meters_per_cell) / 2.0) / meters_per_cell).to(torch.long)
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    flat_index = (rows.clamp(0, height - 1) * width + cols.clamp(0, width - 1)).view(batch, candidate_count, point_count)
    masks = torch.zeros((batch, candidate_count, height * width), dtype=torch.float32, device=device)
    masks.scatter_add_(2, flat_index, valid.to(dtype=masks.dtype))
    masks = masks.view(batch * candidate_count, 1, height, width).clamp(0.0, 1.0)
    radius_cells = max(0, int(round(float(robot_radius_m) / float(meters_per_cell))))
    if radius_cells > 0:
        kernel = 2 * radius_cells + 1
        masks = F.max_pool2d(masks, kernel_size=kernel, stride=1, padding=radius_cells)
    return masks.view(batch, candidate_count, height, width).clamp(0.0, 1.0)


def checkpoint_payload(
    model: CounterfactualDynamicBEVWorldModelV0,
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_version": COUNTERFACTUAL_DYNAMIC_BEV_WORLD_MODEL_V0_CHECKPOINT_VERSION,
        "model_name": "CounterfactualDynamicBEVWorldModelV0",
        "model_config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "metadata": metadata,
        "metrics": metrics,
    }


def save_checkpoint(
    path: str | Path,
    model: CounterfactualDynamicBEVWorldModelV0,
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
) -> tuple[CounterfactualDynamicBEVWorldModelV0, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("checkpoint_version") != COUNTERFACTUAL_DYNAMIC_BEV_WORLD_MODEL_V0_CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint {payload.get('checkpoint_version')!r}; "
            f"expected {COUNTERFACTUAL_DYNAMIC_BEV_WORLD_MODEL_V0_CHECKPOINT_VERSION!r}"
        )
    model = CounterfactualDynamicBEVWorldModelV0(CounterfactualDynamicBEVWorldModelV0Config.from_dict(payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _fit_history(history: torch.Tensor, steps: int) -> torch.Tensor:
    if history.shape[1] == steps:
        return history
    if history.shape[1] > steps:
        return history[:, -steps:]
    pad = history[:, :1].expand(history.shape[0], steps - history.shape[1], *history.shape[2:])
    return torch.cat([pad, history], dim=1)


def _fit_sequence(
    value: torch.Tensor | None,
    *,
    batch: int,
    steps: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return torch.zeros((batch, steps, dim), dtype=dtype, device=device)
    tensor = value.to(device=device, dtype=dtype)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 3 or tensor.shape[0] != batch:
        return torch.zeros((batch, steps, dim), dtype=dtype, device=device)
    if tensor.shape[1] > steps:
        tensor = tensor[:, -steps:]
    elif tensor.shape[1] < steps:
        tensor = torch.cat([tensor[:, :1].expand(batch, steps - tensor.shape[1], tensor.shape[2]), tensor], dim=1)
    out = torch.zeros((batch, steps, dim), dtype=dtype, device=device)
    width = min(dim, int(tensor.shape[2]))
    out[:, :, :width] = tensor[:, :, :width]
    return out


def _fit_sensor(
    value: torch.Tensor | None,
    *,
    batch: int,
    steps: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return torch.zeros((batch, steps, dim), dtype=dtype, device=device)
    return _fit_sequence(value, batch=batch, steps=steps, dim=dim, device=device, dtype=dtype)


def _pool_candidate_features(
    *,
    future: torch.Tensor,
    latent: torch.Tensor,
    masks: torch.Tensor,
    trajectories: torch.Tensor,
) -> torch.Tensor:
    occupied = torch.sigmoid(future[:, :, 0])
    risky = torch.sigmoid(future[:, :, 3])
    dynamic = torch.sigmoid(future[:, :, 4])
    unknown = torch.sigmoid(future[:, :, 2])
    combined = torch.maximum(occupied, torch.maximum(risky, dynamic))
    max_risk = _masked_max_over_horizons(combined, masks)
    mean_risk = _masked_mean_over_horizons(combined, masks)
    max_dynamic = _masked_max_over_horizons(dynamic, masks)
    mean_unknown = _masked_mean_over_horizons(unknown, masks)
    distance = torch.linalg.norm(trajectories[:, :, 1:, :2] - trajectories[:, :, :-1, :2], dim=-1).sum(dim=-1)
    terminal = torch.linalg.norm(trajectories[:, :, -1, :2], dim=-1)
    yaw_span = torch.abs(trajectories[:, :, -1, 2] - trajectories[:, :, 0, 2])
    stop_like = (distance < 1.0e-4).to(dtype=future.dtype)
    compact = torch.stack([max_risk, mean_risk, max_dynamic, mean_unknown, distance, terminal, yaw_span, stop_like], dim=-1)
    latent_pooled = _masked_mean_latent(latent, masks)
    return torch.cat([latent_pooled, compact], dim=-1)


def _masked_mean_over_horizons(values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    masked = values.unsqueeze(2) * masks.unsqueeze(1)
    denom = masks.sum(dim=(2, 3)).clamp_min(1.0).unsqueeze(1)
    return (masked.sum(dim=(3, 4)) / denom).mean(dim=1)


def _masked_mean_latent(latent: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    masked = latent.unsqueeze(1) * masks.unsqueeze(2)
    denom = masks.sum(dim=(2, 3)).clamp_min(1.0).unsqueeze(-1)
    return masked.sum(dim=(3, 4)) / denom


def _masked_max_over_horizons(values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    masked = values.unsqueeze(2).masked_fill(masks.unsqueeze(1) <= 0.0, -1.0)
    return masked.amax(dim=(1, 3, 4)).clamp_min(0.0)


def _prob_to_logit(value: torch.Tensor) -> torch.Tensor:
    clipped = value.clamp(1.0e-4, 1.0 - 1.0e-4)
    return torch.log(clipped / (1.0 - clipped))
