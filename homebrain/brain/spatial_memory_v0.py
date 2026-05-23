from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from homebrain.train import SPATIAL_V0_CHECKPOINT_VERSION
from homebrain.train.spatial_dataset import BEV_OUTPUT_CHANNELS, SENSOR_MASK_FIELDS

SPATIAL_MEMORY_V0_SOURCE = "spatial_memory_net_v0"


@dataclass(frozen=True)
class SpatialMemoryNetConfig:
    feature_dim: int
    bev_shape: tuple[int, int]
    hidden_channels: int = 64
    sensor_dim: int = len(SENSOR_MASK_FIELDS) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "bev_shape": list(self.bev_shape),
            "hidden_channels": self.hidden_channels,
            "sensor_dim": self.sensor_dim,
            "output_channels": list(BEV_OUTPUT_CHANNELS),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpatialMemoryNetConfig":
        shape = data.get("bev_shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError("checkpoint config must include bev_shape")
        return cls(
            feature_dim=int(data["feature_dim"]),
            bev_shape=(int(shape[0]), int(shape[1])),
            hidden_channels=int(data.get("hidden_channels", 64)),
            sensor_dim=int(data.get("sensor_dim", len(SENSOR_MASK_FIELDS) + 1)),
        )


class SpatialMemoryNetV0(nn.Module):
    def __init__(self, config: SpatialMemoryNetConfig) -> None:
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
        self.bev_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, len(BEV_OUTPUT_CHANNELS), kernel_size=1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
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

    def forward(
        self,
        features: torch.Tensor,
        timestamp_s: torch.Tensor,
        sensor_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(features)
        context = torch.cat([timestamp_s, sensor_mask], dim=1)
        bias = self.sensor_adapter(context).view(features.shape[0], -1, 1, 1)
        fused = encoded + bias
        pooled = fused.mean(dim=(2, 3))
        bev_latent = F.interpolate(fused, size=self.config.bev_shape, mode="bilinear", align_corners=False)
        bev_logits = self.bev_head(bev_latent)
        uncertainty_logits = self.uncertainty_head(bev_latent)
        return {
            "bev_logits": bev_logits,
            "uncertainty_logits": uncertainty_logits,
            "pose_delta": self.pose_head(pooled),
            "uncertainty_scalar": torch.sigmoid(self.uncertainty_scalar_head(pooled)).squeeze(1),
        }


def checkpoint_payload(
    model: SpatialMemoryNetV0,
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_version": SPATIAL_V0_CHECKPOINT_VERSION,
        "model_name": "SpatialMemoryNetV0",
        "model_config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "metadata": metadata,
        "metrics": metrics,
    }


def save_checkpoint(
    path: str | Path,
    model: SpatialMemoryNetV0,
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, metadata=metadata, metrics=metrics), target)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple[SpatialMemoryNetV0, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("checkpoint_version") != SPATIAL_V0_CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {payload.get('checkpoint_version')!r}; "
            f"expected {SPATIAL_V0_CHECKPOINT_VERSION!r}"
        )
    config = SpatialMemoryNetConfig.from_dict(payload["model_config"])
    model = SpatialMemoryNetV0(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
