from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from homebrain.train import DIRECT_BEV_STUDENT_V0_CHECKPOINT_VERSION

DIRECT_BEV_STUDENT_V0_SOURCE = "direct_bev_student_v0"
DIRECT_BEV_CHANNELS: tuple[str, ...] = ("free", "occupied", "unknown", "traversable", "risky")
DIRECT_BEV_CONFIG_SCHEMA_VERSION = "homebrain.direct_bev_student_v0.config.v0"


@dataclass(frozen=True)
class DirectBEVStudentV0Config:
    bev_shape: tuple[int, int]
    image_size: tuple[int, int] = (160, 96)
    hidden_channels: int = 32
    pose_dim: int = 3
    previous_action_dim: int = 2
    sensor_mask_dim: int = 4
    use_depth: bool = True
    compact_feature_channels: int = 16
    schema_version: str = DIRECT_BEV_CONFIG_SCHEMA_VERSION
    model_version: str = "DirectBEVStudentV0"

    def __post_init__(self) -> None:
        if self.bev_shape[0] <= 1 or self.bev_shape[1] <= 1:
            raise ValueError("bev_shape dimensions must be greater than one")
        if self.image_size[0] <= 1 or self.image_size[1] <= 1:
            raise ValueError("image_size must be width,height greater than one")
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "bev_shape": [int(self.bev_shape[0]), int(self.bev_shape[1])],
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "hidden_channels": int(self.hidden_channels),
            "pose_dim": int(self.pose_dim),
            "previous_action_dim": int(self.previous_action_dim),
            "sensor_mask_dim": int(self.sensor_mask_dim),
            "use_depth": bool(self.use_depth),
            "compact_feature_channels": int(self.compact_feature_channels),
            "output_channels": list(DIRECT_BEV_CHANNELS),
            "hazard_head": True,
            "hazard_output_channel": "bev_hazard",
            "runtime_inputs": [
                "current_rgb",
                "optional_current_depth",
                "optional_previous_frame",
                "relative_pose_delta_or_odom_like_delta",
                "optional_previous_action",
                "model_checkpoint",
            ],
            "forbidden_runtime_fields": [
                "target_bev",
                "future_labels",
                "candidate_oracle_cost",
                "teacher_masks",
                "hazard_masks",
                "hazard_boxes",
                "open_vocab_detector",
                "ground_truth_global_trajectory",
                "future_frames",
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DirectBEVStudentV0Config":
        bev_shape = data.get("bev_shape")
        image_size = data.get("image_size", (160, 96))
        if not isinstance(bev_shape, (list, tuple)) or len(bev_shape) != 2:
            raise ValueError("checkpoint config must include bev_shape")
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            raise ValueError("checkpoint config image_size must be width,height")
        return cls(
            bev_shape=(int(bev_shape[0]), int(bev_shape[1])),
            image_size=(int(image_size[0]), int(image_size[1])),
            hidden_channels=int(data.get("hidden_channels", 32)),
            pose_dim=int(data.get("pose_dim", 3)),
            previous_action_dim=int(data.get("previous_action_dim", 2)),
            sensor_mask_dim=int(data.get("sensor_mask_dim", 4)),
            use_depth=bool(data.get("use_depth", True)),
            compact_feature_channels=int(data.get("compact_feature_channels", 16)),
        )


class DirectBEVStudentV0(nn.Module):
    def __init__(self, config: DirectBEVStudentV0Config) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_channels)
        in_channels = 4 if config.use_depth else 3
        self.encoder = nn.Sequential(
            _conv_block(in_channels, hidden, stride=2),
            _conv_block(hidden, hidden, stride=1),
            _conv_block(hidden, hidden * 2, stride=2),
            _conv_block(hidden * 2, hidden * 2, stride=1),
            _conv_block(hidden * 2, hidden * 4, stride=2),
            _conv_block(hidden * 4, hidden * 4, stride=1),
        )
        context_dim = int(config.sensor_mask_dim + config.pose_dim + config.previous_action_dim)
        self.context = nn.Sequential(
            nn.Linear(context_dim, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden * 4),
        )
        self.bev_trunk = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden * 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden * 4, hidden * 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.bev_head = nn.Conv2d(hidden * 2, len(DIRECT_BEV_CHANNELS), kernel_size=1)
        self.hazard_head = nn.Conv2d(hidden * 2, 1, kernel_size=1)
        self.uncertainty_head = nn.Conv2d(hidden * 2, 1, kernel_size=1)
        self.dynamic_risk_head = nn.Conv2d(hidden * 2, 1, kernel_size=1)
        self.feature_head = nn.Conv2d(hidden * 2, int(config.compact_feature_channels), kernel_size=1)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor | None = None,
        sensor_mask: torch.Tensor | None = None,
        pose_delta_prev: torch.Tensor | None = None,
        previous_action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must be Bx3xHxW")
        batch = int(rgb.shape[0])
        rgb = _resize(rgb.float(), self.config.image_size).clamp(0.0, 1.0)
        if self.config.use_depth:
            depth_tensor = _depth_input(depth, batch=batch, image_size=self.config.image_size, device=rgb.device, dtype=rgb.dtype)
            image = torch.cat([rgb, depth_tensor], dim=1)
        else:
            image = rgb
        latent = self.encoder(image)
        context = torch.cat(
            [
                _context_input(sensor_mask, batch=batch, dim=self.config.sensor_mask_dim, device=rgb.device, dtype=rgb.dtype),
                _context_input(pose_delta_prev, batch=batch, dim=self.config.pose_dim, device=rgb.device, dtype=rgb.dtype),
                _context_input(previous_action, batch=batch, dim=self.config.previous_action_dim, device=rgb.device, dtype=rgb.dtype),
            ],
            dim=1,
        )
        latent = latent + self.context(context).view(batch, -1, 1, 1)
        bev_latent = self.bev_trunk(
            F.interpolate(latent, size=self.config.bev_shape, mode="bilinear", align_corners=False)
        )
        return {
            "bev_logits": self.bev_head(bev_latent),
            "hazard_logits": self.hazard_head(bev_latent),
            "uncertainty_logits": self.uncertainty_head(bev_latent),
            "dynamic_risk_logits": self.dynamic_risk_head(bev_latent),
            "compact_features": self.feature_head(bev_latent),
        }


def checkpoint_payload(
    model: DirectBEVStudentV0,
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_version": DIRECT_BEV_STUDENT_V0_CHECKPOINT_VERSION,
        "model_name": "DirectBEVStudentV0",
        "model_config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "metadata": metadata,
        "metrics": metrics,
    }


def save_checkpoint(
    path: str | Path,
    model: DirectBEVStudentV0,
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
) -> tuple[DirectBEVStudentV0, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("checkpoint_version") != DIRECT_BEV_STUDENT_V0_CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported DirectBEV checkpoint {payload.get('checkpoint_version')!r}; "
            f"expected {DIRECT_BEV_STUDENT_V0_CHECKPOINT_VERSION!r}"
        )
    model = DirectBEVStudentV0(DirectBEVStudentV0Config.from_dict(payload["model_config"]))
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    unexpected_keys = [key for key in unexpected if not key.startswith("hazard_head.")]
    missing_keys = [key for key in missing if not key.startswith("hazard_head.")]
    if unexpected_keys or missing_keys:
        raise ValueError(f"checkpoint state_dict mismatch: missing={missing_keys}, unexpected={unexpected_keys}")
    model.eval()
    return model, payload


def _conv_block(in_channels: int, out_channels: int, *, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(max(1, min(8, out_channels // 4)), out_channels),
        nn.GELU(),
    )


def _resize(tensor: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    width, height = image_size
    if tuple(tensor.shape[-2:]) == (height, width):
        return tensor
    return F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)


def _depth_input(
    depth: torch.Tensor | None,
    *,
    batch: int,
    image_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    width, height = image_size
    if depth is None:
        return torch.zeros((batch, 1, height, width), dtype=dtype, device=device)
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if depth.ndim != 4 or depth.shape[1] != 1 or depth.shape[0] != batch:
        return torch.zeros((batch, 1, height, width), dtype=dtype, device=device)
    return _resize(depth.to(device=device, dtype=dtype).clamp(0.0, 6.0) / 6.0, image_size)


def _context_input(
    value: torch.Tensor | None,
    *,
    batch: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return torch.zeros((batch, dim), dtype=dtype, device=device)
    tensor = value.to(device=device, dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.view(1, -1).expand(batch, -1)
    if tensor.ndim != 2 or tensor.shape[0] != batch:
        return torch.zeros((batch, dim), dtype=dtype, device=device)
    if tensor.shape[1] == dim:
        return tensor
    out = torch.zeros((batch, dim), dtype=dtype, device=device)
    width = min(dim, int(tensor.shape[1]))
    out[:, :width] = tensor[:, :width]
    return out
