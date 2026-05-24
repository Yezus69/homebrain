from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F

from homebrain.train import SPATIAL_V1_CHECKPOINT_VERSION
from homebrain.train.spatial_dataset import BEV_OUTPUT_CHANNELS, SENSOR_MASK_FIELDS

SPATIAL_MEMORY_V1_SOURCE = "spatial_memory_net_v1"
SPATIAL_MEMORY_V1_SCHEMA_VERSION = "homebrain.spatial_memory_v1.config.v0"
MISSING_POSE_BEHAVIORS = ("reset", "no_warp", "masked_update")
MissingPoseBehavior = Literal["reset", "no_warp", "masked_update"]


@dataclass(frozen=True)
class SpatialMemoryNetV1Config:
    feature_dim: int
    bev_shape: tuple[int, int]
    hidden_channels: int = 64
    sensor_dim: int = len(SENSOR_MASK_FIELDS) + 1
    meters_per_cell: float = 0.05
    memory_update_alpha: float = 0.65
    memory_decay: float = 1.0
    missing_pose_behavior: MissingPoseBehavior = "reset"
    unknown_prior_logit: float = 4.0
    known_prior_logit: float = -4.0
    predicted_observation_confidence_threshold: float = 0.65
    predicted_observation_unknown_threshold: float = 0.45
    schema_version: str = SPATIAL_MEMORY_V1_SCHEMA_VERSION
    model_version: str = "SpatialMemoryNetV1"

    def __post_init__(self) -> None:
        if self.bev_shape[0] <= 0 or self.bev_shape[1] <= 0:
            raise ValueError("bev_shape dimensions must be positive")
        if self.meters_per_cell <= 0.0:
            raise ValueError("meters_per_cell must be positive")
        if not 0.0 <= self.memory_update_alpha <= 1.0:
            raise ValueError("memory_update_alpha must be in [0, 1]")
        if not 0.0 <= self.memory_decay <= 1.0:
            raise ValueError("memory_decay must be in [0, 1]")
        if self.missing_pose_behavior not in MISSING_POSE_BEHAVIORS:
            raise ValueError(f"missing_pose_behavior must be one of {MISSING_POSE_BEHAVIORS}")
        if self.unknown_prior_logit <= self.known_prior_logit:
            raise ValueError("unknown_prior_logit must be greater than known_prior_logit")
        if not 0.0 <= self.predicted_observation_confidence_threshold <= 1.0:
            raise ValueError("predicted_observation_confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.predicted_observation_unknown_threshold <= 1.0:
            raise ValueError("predicted_observation_unknown_threshold must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "feature_dim": self.feature_dim,
            "bev_shape": list(self.bev_shape),
            "hidden_channels": self.hidden_channels,
            "sensor_dim": self.sensor_dim,
            "meters_per_cell": self.meters_per_cell,
            "memory_update_alpha": self.memory_update_alpha,
            "memory_decay": self.memory_decay,
            "missing_pose_behavior": self.missing_pose_behavior,
            "unknown_prior_logit": self.unknown_prior_logit,
            "known_prior_logit": self.known_prior_logit,
            "predicted_observation_confidence_threshold": self.predicted_observation_confidence_threshold,
            "predicted_observation_unknown_threshold": self.predicted_observation_unknown_threshold,
            "output_channels": list(BEV_OUTPUT_CHANNELS),
            "memory_channels": list(BEV_OUTPUT_CHANNELS),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpatialMemoryNetV1Config":
        shape = data.get("bev_shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError("checkpoint config must include bev_shape")
        behavior = str(data.get("missing_pose_behavior", "reset"))
        if behavior not in MISSING_POSE_BEHAVIORS:
            raise ValueError(f"unsupported missing_pose_behavior {behavior!r}")
        return cls(
            feature_dim=int(data["feature_dim"]),
            bev_shape=(int(shape[0]), int(shape[1])),
            hidden_channels=int(data.get("hidden_channels", 64)),
            sensor_dim=int(data.get("sensor_dim", len(SENSOR_MASK_FIELDS) + 1)),
            meters_per_cell=float(data.get("meters_per_cell", 0.05)),
            memory_update_alpha=float(data.get("memory_update_alpha", 0.65)),
            memory_decay=float(data.get("memory_decay", 1.0)),
            missing_pose_behavior=behavior,  # type: ignore[arg-type]
            unknown_prior_logit=float(data.get("unknown_prior_logit", 4.0)),
            known_prior_logit=float(data.get("known_prior_logit", -4.0)),
            predicted_observation_confidence_threshold=float(
                data.get("predicted_observation_confidence_threshold", 0.65)
            ),
            predicted_observation_unknown_threshold=float(data.get("predicted_observation_unknown_threshold", 0.45)),
        )


@dataclass
class SpatialMemoryState:
    memory_logits: torch.Tensor
    observed_mask: torch.Tensor
    step_index: torch.Tensor

    def to(self, device: torch.device | str) -> "SpatialMemoryState":
        return SpatialMemoryState(
            memory_logits=self.memory_logits.to(device),
            observed_mask=self.observed_mask.to(device),
            step_index=self.step_index.to(device),
        )

    def detach(self) -> "SpatialMemoryState":
        return SpatialMemoryState(
            memory_logits=self.memory_logits.detach(),
            observed_mask=self.observed_mask.detach(),
            step_index=self.step_index.detach(),
        )


@dataclass(frozen=True)
class SpatialWarpResult:
    tensor: torch.Tensor
    pose_warp_valid: torch.Tensor
    memory_reset: torch.Tensor
    pose_warp_used: torch.Tensor


def warp_memory_se2(
    memory: torch.Tensor,
    pose_delta: torch.Tensor | None,
    pose_valid: torch.Tensor | None,
    *,
    meters_per_cell: float,
    missing_pose_behavior: MissingPoseBehavior = "reset",
    mode: str = "bilinear",
    fill_value: float = 0.0,
) -> SpatialWarpResult:
    """Warp a batch of egocentric BEV memories into the current robot frame.

    Pose deltas use the HomeBrain BEV convention: +x is forward, +y is left,
    +yaw is left turn. Forward motion makes old world cells move toward larger
    row indices in the new egocentric grid.
    """

    if memory.ndim != 4:
        raise ValueError("memory must be a BCHW tensor")
    if meters_per_cell <= 0.0:
        raise ValueError("meters_per_cell must be positive")
    if missing_pose_behavior not in MISSING_POSE_BEHAVIORS:
        raise ValueError(f"missing_pose_behavior must be one of {MISSING_POSE_BEHAVIORS}")
    batch, _channels, height, width = memory.shape
    device = memory.device
    dtype = memory.dtype

    if pose_delta is None:
        pose_delta_t = torch.zeros((batch, 3), dtype=dtype, device=device)
        valid = torch.zeros((batch,), dtype=torch.bool, device=device)
    else:
        pose_delta_t = pose_delta.to(device=device, dtype=dtype).view(batch, 3)
        if pose_valid is None:
            valid = torch.isfinite(pose_delta_t).all(dim=1)
        else:
            valid = pose_valid.to(device=device).view(batch) > 0.0
            valid = valid & torch.isfinite(pose_delta_t).all(dim=1)

    if height == 1 or width == 1:
        raise ValueError("memory warp requires height and width greater than 1")

    rows = torch.arange(height, dtype=dtype, device=device)
    cols = torch.arange(width, dtype=dtype, device=device)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
    origin_row = torch.tensor(float(height - 1), dtype=dtype, device=device)
    origin_col = torch.tensor(float(width // 2), dtype=dtype, device=device)
    x_new = (origin_row - row_grid)[None, :, :] * meters_per_cell
    y_new = (col_grid - origin_col)[None, :, :] * meters_per_cell

    dx = pose_delta_t[:, 0].view(batch, 1, 1)
    dy = pose_delta_t[:, 1].view(batch, 1, 1)
    dyaw = pose_delta_t[:, 2].view(batch, 1, 1)
    cos_yaw = torch.cos(dyaw)
    sin_yaw = torch.sin(dyaw)
    x_old = cos_yaw * x_new - sin_yaw * y_new + dx
    y_old = sin_yaw * x_new + cos_yaw * y_new + dy
    row_src = origin_row - (x_old / meters_per_cell)
    col_src = origin_col + (y_old / meters_per_cell)
    x_norm = (2.0 * col_src / float(width - 1)) - 1.0
    y_norm = (2.0 * row_src / float(height - 1)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)

    warped = F.grid_sample(
        memory,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )
    if fill_value != 0.0:
        in_bounds = (
            (x_norm >= -1.0)
            & (x_norm <= 1.0)
            & (y_norm >= -1.0)
            & (y_norm <= 1.0)
        ).to(dtype).unsqueeze(1)
        warped = warped * in_bounds + torch.full_like(warped, fill_value) * (1.0 - in_bounds)

    invalid = ~valid
    reset = invalid & (missing_pose_behavior == "reset")
    if torch.any(invalid):
        keep_original = invalid & (missing_pose_behavior != "reset")
        if torch.any(keep_original):
            warped = torch.where(keep_original.view(batch, 1, 1, 1), memory, warped)
        if torch.any(reset):
            warped = torch.where(reset.view(batch, 1, 1, 1), torch.full_like(warped, fill_value), warped)

    return SpatialWarpResult(
        tensor=warped,
        pose_warp_valid=valid,
        memory_reset=reset,
        pose_warp_used=valid,
    )


class SpatialMemoryNetV1(nn.Module):
    def __init__(self, config: SpatialMemoryNetV1Config) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(config.feature_dim, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.sensor_adapter = nn.Sequential(
            nn.Linear(config.sensor_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.current_bev_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, len(BEV_OUTPUT_CHANNELS), kernel_size=1),
        )
        self.memory_refine = nn.Sequential(
            nn.Conv2d(len(BEV_OUTPUT_CHANNELS), hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, len(BEV_OUTPUT_CHANNELS), kernel_size=1),
        )
        nn.init.zeros_(self.memory_refine[-1].weight)
        nn.init.zeros_(self.memory_refine[-1].bias)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(hidden + len(BEV_OUTPUT_CHANNELS) + 1, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.pose_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.uncertainty_scalar_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def init_memory(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SpatialMemoryState:
        param = next(self.parameters())
        resolved_device = torch.device(device) if device is not None else param.device
        resolved_dtype = dtype or param.dtype
        height, width = self.config.bev_shape
        return SpatialMemoryState(
            memory_logits=self.unknown_prior_logits(
                batch_size,
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            observed_mask=torch.zeros((batch_size, 1, height, width), dtype=resolved_dtype, device=resolved_device),
            step_index=torch.zeros((batch_size,), dtype=torch.int64, device=resolved_device),
        )

    def unknown_prior_logits(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        height, width = self.config.bev_shape
        memory = torch.full(
            (batch_size, len(BEV_OUTPUT_CHANNELS), height, width),
            fill_value=float(self.config.known_prior_logit),
            dtype=dtype,
            device=device,
        )
        memory[:, BEV_OUTPUT_CHANNELS.index("unknown")] = float(self.config.unknown_prior_logit)
        return memory

    def derive_predicted_observation_mask(self, current_bev_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(current_bev_logits)
        unknown_index = BEV_OUTPUT_CHANNELS.index("unknown")
        evidence_indices = [
            index
            for index, channel in enumerate(BEV_OUTPUT_CHANNELS)
            if channel != "unknown"
        ]
        evidence_confidence = probs[:, evidence_indices].amax(dim=1, keepdim=True)
        unknown_prob = probs[:, unknown_index : unknown_index + 1]
        return (
            (evidence_confidence >= float(self.config.predicted_observation_confidence_threshold))
            & (unknown_prob <= float(self.config.predicted_observation_unknown_threshold))
        ).to(dtype=current_bev_logits.dtype)

    def forward(
        self,
        features: torch.Tensor,
        timestamp_s: torch.Tensor,
        sensor_mask: torch.Tensor,
        *,
        memory_state: SpatialMemoryState | None = None,
        pose_delta_to_current: torch.Tensor | None = None,
        pose_delta_to_current_mask: torch.Tensor | None = None,
        observation_mask: torch.Tensor | None = None,
        force_reset: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        return self.step(
            features,
            timestamp_s,
            sensor_mask,
            memory_state=memory_state,
            pose_delta_to_current=pose_delta_to_current,
            pose_delta_to_current_mask=pose_delta_to_current_mask,
            observation_mask=observation_mask,
            force_reset=force_reset,
        )

    def step(
        self,
        features: torch.Tensor,
        timestamp_s: torch.Tensor,
        sensor_mask: torch.Tensor,
        *,
        memory_state: SpatialMemoryState | None = None,
        pose_delta_to_current: torch.Tensor | None = None,
        pose_delta_to_current_mask: torch.Tensor | None = None,
        observation_mask: torch.Tensor | None = None,
        force_reset: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        batch = int(features.shape[0])
        if memory_state is None:
            memory_state = self.init_memory(batch, device=features.device, dtype=features.dtype)
            first_step_reset = torch.ones((batch,), dtype=torch.bool, device=features.device)
        else:
            memory_state = memory_state.to(features.device)
            first_step_reset = memory_state.step_index == 0

        if force_reset is not None:
            explicit_reset = force_reset.to(device=features.device).view(batch) > 0.0
        else:
            explicit_reset = torch.zeros((batch,), dtype=torch.bool, device=features.device)
        reset_before = first_step_reset | explicit_reset
        if torch.any(reset_before):
            zero_state = self.init_memory(batch, device=features.device, dtype=features.dtype)
            memory_state = SpatialMemoryState(
                memory_logits=torch.where(reset_before.view(batch, 1, 1, 1), zero_state.memory_logits, memory_state.memory_logits),
                observed_mask=torch.where(reset_before.view(batch, 1, 1, 1), zero_state.observed_mask, memory_state.observed_mask),
                step_index=torch.where(reset_before, zero_state.step_index, memory_state.step_index),
            )

        encoded = self.encoder(features)
        context = torch.cat([timestamp_s, sensor_mask], dim=1)
        bias = self.sensor_adapter(context).view(batch, -1, 1, 1)
        fused_features = encoded + bias
        pooled = fused_features.mean(dim=(2, 3))
        bev_latent = F.interpolate(fused_features, size=self.config.bev_shape, mode="bilinear", align_corners=False)
        current_bev_logits = self.current_bev_head(bev_latent)
        pose_delta = self.pose_head(pooled)

        pose_valid = (
            pose_delta_to_current_mask.view(batch)
            if pose_delta_to_current_mask is not None
            else torch.ones((batch,), dtype=features.dtype, device=features.device)
        )
        warp = warp_memory_se2(
            memory_state.memory_logits,
            pose_delta_to_current,
            pose_valid,
            meters_per_cell=self.config.meters_per_cell,
            missing_pose_behavior=self.config.missing_pose_behavior,
        )
        observed_warp = warp_memory_se2(
            memory_state.observed_mask,
            pose_delta_to_current,
            pose_valid,
            meters_per_cell=self.config.meters_per_cell,
            missing_pose_behavior=self.config.missing_pose_behavior,
            mode="nearest",
        )
        warped_memory = warp.tensor * self.config.memory_decay
        warped_observed = observed_warp.tensor.clamp(0.0, 1.0)
        unknown_prior = self.unknown_prior_logits(batch, device=features.device, dtype=features.dtype)
        warped_memory = torch.where(warped_observed > 0.0, warped_memory, unknown_prior)

        if observation_mask is None:
            current_observed = self.derive_predicted_observation_mask(current_bev_logits)
            observation_mask_source = "predicted_current_bev_confidence"
        else:
            current_observed = observation_mask.to(device=features.device, dtype=features.dtype).view(
                batch,
                1,
                self.config.bev_shape[0],
                self.config.bev_shape[1],
            ).clamp(0.0, 1.0)
            observation_mask_source = "trusted_observation_mask"

        old_known = warped_observed > 0.0
        current_known = current_observed > 0.0
        alpha = torch.tensor(self.config.memory_update_alpha, dtype=features.dtype, device=features.device)
        blended = (1.0 - alpha) * warped_memory + alpha * current_bev_logits
        memory_logits = torch.where(old_known & current_known, blended, torch.where(current_known, current_bev_logits, warped_memory))
        update_mask_coverage = current_known.to(features.dtype).mean(dim=(1, 2, 3))
        memory_overwrite_fraction = (old_known & current_known).to(features.dtype).mean(dim=(1, 2, 3))
        if self.config.missing_pose_behavior == "masked_update":
            invalid_pose = (~warp.pose_warp_valid).view(batch, 1, 1, 1)
            memory_logits = torch.where(
                invalid_pose & ~current_known,
                memory_state.memory_logits,
                memory_logits,
            )
            warped_observed = torch.where(invalid_pose & ~current_known, memory_state.observed_mask, warped_observed)
        observed_mask = torch.maximum(warped_observed, current_observed)
        refinement = self.memory_refine(memory_logits)
        refined_memory_logits = torch.where(current_known, memory_logits + refinement, memory_logits)
        uncertainty_input = torch.cat([bev_latent, refined_memory_logits, observed_mask], dim=1)
        uncertainty_logits = self.uncertainty_head(uncertainty_input)
        uncertainty_scalar = torch.sigmoid(self.uncertainty_scalar_head(pooled)).squeeze(1)

        next_state = SpatialMemoryState(
            memory_logits=refined_memory_logits,
            observed_mask=observed_mask,
            step_index=memory_state.step_index + 1,
        )
        memory_reset = warp.memory_reset | reset_before
        debug = {
            "memory_used": (next_state.step_index > 1),
            "pose_warp_used": warp.pose_warp_used,
            "pose_warp_valid": warp.pose_warp_valid,
            "memory_reset": memory_reset,
            "update_mask_coverage": update_mask_coverage,
            "memory_overwrite_fraction": memory_overwrite_fraction,
            "missing_pose_behavior": self.config.missing_pose_behavior,
            "observation_mask_source": observation_mask_source,
        }
        return {
            "current_bev_logits": current_bev_logits,
            "fused_memory_bev_logits": refined_memory_logits,
            "bev_logits": refined_memory_logits,
            "uncertainty_logits": uncertainty_logits,
            "uncertainty_grid": torch.sigmoid(uncertainty_logits),
            "uncertainty_scalar": uncertainty_scalar,
            "pose_delta": pose_delta,
            "memory_state": next_state,
            "update_mask": current_observed,
            "debug": debug,
        }

    def forward_sequence(
        self,
        features: torch.Tensor,
        timestamp_s: torch.Tensor,
        sensor_mask: torch.Tensor,
        *,
        pose_delta_to_current: torch.Tensor | None = None,
        pose_delta_to_current_mask: torch.Tensor | None = None,
        observation_mask: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
        initial_state: SpatialMemoryState | None = None,
    ) -> dict[str, Any]:
        if features.ndim != 5:
            raise ValueError("features must be BTCHW")
        batch, steps = int(features.shape[0]), int(features.shape[1])
        state = initial_state
        current_logits: list[torch.Tensor] = []
        memory_logits: list[torch.Tensor] = []
        update_masks: list[torch.Tensor] = []
        uncertainty_logits: list[torch.Tensor] = []
        uncertainty_scalars: list[torch.Tensor] = []
        pose_deltas: list[torch.Tensor] = []
        debug_values: dict[str, list[torch.Tensor]] = {
            "memory_used": [],
            "pose_warp_used": [],
            "pose_warp_valid": [],
            "memory_reset": [],
            "update_mask_coverage": [],
            "memory_overwrite_fraction": [],
        }
        for index in range(steps):
            force_reset = reset_mask[:, index] if reset_mask is not None else None
            result = self.step(
                features[:, index],
                timestamp_s[:, index],
                sensor_mask[:, index],
                memory_state=state,
                pose_delta_to_current=pose_delta_to_current[:, index] if pose_delta_to_current is not None else None,
                pose_delta_to_current_mask=pose_delta_to_current_mask[:, index]
                if pose_delta_to_current_mask is not None
                else None,
                observation_mask=observation_mask[:, index] if observation_mask is not None else None,
                force_reset=force_reset,
            )
            state = result["memory_state"]
            current_logits.append(result["current_bev_logits"])
            memory_logits.append(result["fused_memory_bev_logits"])
            update_masks.append(result["update_mask"])
            uncertainty_logits.append(result["uncertainty_logits"])
            uncertainty_scalars.append(result["uncertainty_scalar"])
            pose_deltas.append(result["pose_delta"])
            for key in debug_values:
                debug_values[key].append(result["debug"][key].to(features.device))
        return {
            "current_bev_logits": torch.stack(current_logits, dim=1),
            "fused_memory_bev_logits": torch.stack(memory_logits, dim=1),
            "bev_logits": torch.stack(memory_logits, dim=1),
            "update_mask": torch.stack(update_masks, dim=1),
            "uncertainty_logits": torch.stack(uncertainty_logits, dim=1),
            "uncertainty_grid": torch.sigmoid(torch.stack(uncertainty_logits, dim=1)),
            "uncertainty_scalar": torch.stack(uncertainty_scalars, dim=1),
            "pose_delta": torch.stack(pose_deltas, dim=1),
            "memory_state": state,
            "debug": {key: torch.stack(values, dim=1) for key, values in debug_values.items()},
        }


def checkpoint_payload(
    model: SpatialMemoryNetV1,
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_version": SPATIAL_V1_CHECKPOINT_VERSION,
        "model_name": "SpatialMemoryNetV1",
        "model_config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "metadata": metadata,
        "metrics": metrics,
    }


def save_checkpoint(
    path: str | Path,
    model: SpatialMemoryNetV1,
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, metadata=metadata, metrics=metrics), target)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple[SpatialMemoryNetV1, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("checkpoint_version") != SPATIAL_V1_CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {payload.get('checkpoint_version')!r}; "
            f"expected {SPATIAL_V1_CHECKPOINT_VERSION!r}"
        )
    config = SpatialMemoryNetV1Config.from_dict(payload["model_config"])
    model = SpatialMemoryNetV1(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def warm_start_current_bev_from_v0(
    model: SpatialMemoryNetV1,
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint), map_location=map_location, weights_only=False)
    if payload.get("model_name") != "SpatialMemoryNetV0":
        raise ValueError(f"expected a SpatialMemoryNetV0 checkpoint, got {payload.get('model_name')!r}")
    source_state = payload.get("state_dict")
    if not isinstance(source_state, dict):
        raise ValueError("v0 checkpoint is missing state_dict")
    target_state = model.state_dict()
    copied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for source_key, source_tensor in source_state.items():
        target_key = _v0_to_v1_key(str(source_key))
        if target_key is None:
            continue
        target_tensor = target_state.get(target_key)
        if target_tensor is None:
            skipped.append({"source": str(source_key), "target": target_key, "reason": "missing_target_key"})
            continue
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            skipped.append(
                {
                    "source": str(source_key),
                    "target": target_key,
                    "reason": "shape_mismatch",
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                }
            )
            continue
        target_state[target_key] = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
        copied.append({"source": str(source_key), "target": target_key, "shape": list(target_tensor.shape)})
    model.load_state_dict(target_state, strict=True)
    return {
        "schema_version": "homebrain.spatial_memory_v1.warm_start.v0",
        "source_checkpoint": Path(checkpoint).as_posix(),
        "source_model_name": payload.get("model_name"),
        "copied_count": len(copied),
        "skipped_count": len(skipped),
        "copied": copied,
        "skipped": skipped,
        "shape_safe": True,
    }


def _v0_to_v1_key(source_key: str) -> str | None:
    if source_key.startswith("encoder."):
        return source_key
    if source_key.startswith("sensor_adapter."):
        return source_key
    if source_key.startswith("bev_head."):
        return "current_bev_head." + source_key[len("bev_head.") :]
    if source_key.startswith("pose_head."):
        return source_key
    if source_key.startswith("uncertainty_scalar_head."):
        return source_key
    return None
