from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from homebrain.brain.counterfactual_dynamic_bev_world_model_v0 import (
    CounterfactualDynamicBEVWorldModelV0,
    CounterfactualDynamicBEVWorldModelV0Config,
)
from homebrain.brain.direct_bev_student_v0 import DIRECT_BEV_CHANNELS, DirectBEVStudentV0, DirectBEVStudentV0Config
from homebrain.brain.spatial_memory_v1 import SpatialMemoryState, warp_memory_se2
from homebrain.train import HOMEBRAIN_NET_V0_CHECKPOINT_VERSION
from homebrain.train.dynamic_bev_world_pack_v0 import DYNAMIC_BEV_HISTORY_CHANNELS

HOMEBRAIN_NET_V0_SOURCE = "homebrain_net_v0"
HOMEBRAIN_NET_V0_CONFIG_SCHEMA_VERSION = "homebrain.homebrain_net_v0.config.v0"
HOMEBRAIN_NET_V0_FORBIDDEN_RUNTIME_FIELDS: tuple[str, ...] = ("target_bev", "future_bev", "future_labels", "teacher_masks", "route_ground_truth", "candidate_oracle_cost", "ground_truth_global_trajectory", "future_frames")


@dataclass(frozen=True)
class HomeBrainNetV0Config:
    bev_shape: tuple[int, int]
    image_size: tuple[int, int] = (160, 96)
    hidden_channels: int = 32
    compact_feature_channels: int = 16
    pose_dim: int = 3
    previous_action_dim: int = 2
    sensor_mask_dim: int = 4
    history_frames: int = 6
    future_horizons_sec: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    meters_per_cell: float = 0.05
    robot_radius_m: float = 0.18
    pose_correction_scale: float = 0.5
    enable_pose: bool = True
    enable_bev_occupancy: bool = True
    enable_metric_depth: bool = True
    enable_dynamic: bool = True
    enable_memory: bool = True
    enable_world_model: bool = True
    schema_version: str = HOMEBRAIN_NET_V0_CONFIG_SCHEMA_VERSION
    model_version: str = "HomeBrainNetV0"

    def __post_init__(self) -> None:
        if self.bev_shape[0] <= 1 or self.bev_shape[1] <= 1:
            raise ValueError("bev_shape dimensions must be greater than one")
        if self.image_size[0] <= 1 or self.image_size[1] <= 1:
            raise ValueError("image_size must be width,height greater than one")
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if self.history_frames <= 0:
            raise ValueError("history_frames must be positive")
        if not self.future_horizons_sec:
            raise ValueError("future_horizons_sec must not be empty")
        if not 0.0 <= self.pose_correction_scale <= 1.0:
            raise ValueError("pose_correction_scale must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bev_shape"] = list(self.bev_shape)
        data["image_size"] = list(self.image_size)
        data["future_horizons_sec"] = list(self.future_horizons_sec)
        data["enabled_heads"] = {name.removeprefix("enable_"): data.pop(name) for name in list(data) if name.startswith("enable_")}
        data["output_channels"] = list(DIRECT_BEV_CHANNELS)
        data["world_history_channels"] = list(DYNAMIC_BEV_HISTORY_CHANNELS)
        data["runtime_inputs"] = ["current_rgb", "optional_current_depth", "pose_delta_prev_or_odom_like_delta", "previous_action", "sensor_mask", "SceneState_or_LocalBev_history", "candidate_trajectories", "checkpoint"]
        data["forbidden_runtime_fields"] = list(HOMEBRAIN_NET_V0_FORBIDDEN_RUNTIME_FIELDS)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HomeBrainNetV0Config":
        shape = data.get("bev_shape")
        image = data.get("image_size", (160, 96))
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("checkpoint config must include bev_shape")
        if not isinstance(image, (list, tuple)) or len(image) != 2:
            raise ValueError("checkpoint config image_size must be width,height")
        heads = data.get("enabled_heads") if isinstance(data.get("enabled_heads"), dict) else {}
        return cls(
            bev_shape=(int(shape[0]), int(shape[1])),
            image_size=(int(image[0]), int(image[1])),
            hidden_channels=int(data.get("hidden_channels", 32)),
            compact_feature_channels=int(data.get("compact_feature_channels", 16)),
            pose_dim=int(data.get("pose_dim", 3)),
            previous_action_dim=int(data.get("previous_action_dim", 2)),
            sensor_mask_dim=int(data.get("sensor_mask_dim", 4)),
            history_frames=int(data.get("history_frames", 6)),
            future_horizons_sec=tuple(float(v) for v in data.get("future_horizons_sec", (0.5, 1.0, 2.0, 3.0))),
            meters_per_cell=float(data.get("meters_per_cell", 0.05)),
            robot_radius_m=float(data.get("robot_radius_m", 0.18)),
            pose_correction_scale=float(data.get("pose_correction_scale", 0.5)),
            enable_pose=bool(heads.get("pose", data.get("enable_pose", True))),
            enable_bev_occupancy=bool(heads.get("bev_occupancy", data.get("enable_bev_occupancy", True))),
            enable_metric_depth=bool(heads.get("metric_depth", data.get("enable_metric_depth", True))),
            enable_dynamic=bool(heads.get("dynamic", data.get("enable_dynamic", True))),
            enable_memory=bool(heads.get("memory", data.get("enable_memory", True))),
            enable_world_model=bool(heads.get("world_model", data.get("enable_world_model", True))),
        )


class HomeBrainNetV0(nn.Module):
    def __init__(self, config: HomeBrainNetV0Config) -> None:
        super().__init__()
        self.config = config
        direct_config = DirectBEVStudentV0Config(bev_shape=config.bev_shape, image_size=config.image_size, hidden_channels=config.hidden_channels, pose_dim=config.pose_dim, previous_action_dim=config.previous_action_dim, sensor_mask_dim=config.sensor_mask_dim, use_depth=True, compact_feature_channels=config.compact_feature_channels)
        self.perception = DirectBEVStudentV0(direct_config)
        latent_channels = config.hidden_channels * 4
        bev_channels = config.hidden_channels * 2
        self.pose_head = nn.Sequential(nn.Linear(latent_channels, latent_channels), nn.GELU(), nn.Linear(latent_channels, 3))
        self.metric_depth_head = nn.Sequential(
            nn.Conv2d(latent_channels, bev_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(bev_channels, 1, kernel_size=1),
        )
        self.bev_flow_head = nn.Conv2d(bev_channels, 2, kernel_size=1)
        self.memory = _HomeBrainMemoryFusion(config)
        world_config = CounterfactualDynamicBEVWorldModelV0Config(bev_shape=config.bev_shape, history_frames=config.history_frames, bev_channels=len(DYNAMIC_BEV_HISTORY_CHANNELS), pose_delta_dim=config.pose_dim, previous_action_dim=config.previous_action_dim, sensor_mask_dim=config.sensor_mask_dim, hidden_channels=config.hidden_channels, future_horizons_sec=config.future_horizons_sec, meters_per_cell=config.meters_per_cell, robot_radius_m=config.robot_radius_m)
        self.world_model = CounterfactualDynamicBEVWorldModelV0(world_config)

    def step(self, rgb: torch.Tensor, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.forward(rgb, *args, **kwargs)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor | None = None,
        sensor_mask: torch.Tensor | None = None,
        pose_delta_prev: torch.Tensor | None = None,
        previous_action: torch.Tensor | None = None,
        *,
        memory_state: SpatialMemoryState | None = None,
        pose_delta_to_current: torch.Tensor | None = None,
        pose_delta_to_current_mask: torch.Tensor | None = None,
        observation_mask: torch.Tensor | None = None,
        force_reset: torch.Tensor | None = None,
        bev_history: torch.Tensor | None = None,
        pose_delta_history: torch.Tensor | None = None,
        previous_action_history: torch.Tensor | None = None,
        sensor_mask_history: torch.Tensor | None = None,
        candidate_trajectories: torch.Tensor | None = None,
        enabled_heads: set[str] | tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, Any]:
        heads = set(enabled_heads) if enabled_heads is not None else None
        latent, bev_latent, context = self._shared_features(rgb, depth, sensor_mask, pose_delta_prev, previous_action)
        batch = int(latent.shape[0])
        outputs: dict[str, Any] = {"shared_rgbd_latent": latent, "compact_features": self.perception.feature_head(bev_latent)}
        current_logits = self.perception.bev_head(bev_latent)
        uncertainty_logits = self.perception.uncertainty_head(bev_latent)
        dynamic_logits = self.perception.dynamic_risk_head(bev_latent)
        if self._on("bev_occupancy", heads):
            outputs.update({"bev_logits": current_logits, "bev_occupancy_logits": current_logits[:, :4], "uncertainty_logits": uncertainty_logits})
        if self._on("pose", heads):
            pose_delta = self.pose_head(latent.mean(dim=(2, 3)))
            if pose_delta_prev is not None:
                odom_delta = pose_delta_prev.to(device=pose_delta.device, dtype=pose_delta.dtype).view(batch, -1)[:, : self.config.pose_dim]
                pose_delta = odom_delta + float(self.config.pose_correction_scale) * pose_delta
            outputs["pose_delta"] = pose_delta
        if self._on("metric_depth", heads):
            depth_logits = F.interpolate(self.metric_depth_head(latent), size=rgb.shape[-2:], mode="bilinear", align_corners=False)
            outputs["metric_depth_logits"] = depth_logits
            outputs["metric_depth_m"] = F.softplus(depth_logits)
        if self._on("dynamic", heads):
            outputs["dynamic_occupancy_logits"] = dynamic_logits
            outputs["bev_flow_xy"] = self.bev_flow_head(bev_latent)
        memory_logits = current_logits
        if self._on("memory", heads):
            memory_out = self.memory(
                current_logits,
                uncertainty_logits,
                memory_state=memory_state,
                pose_delta_to_current=pose_delta_to_current,
                pose_delta_to_current_mask=pose_delta_to_current_mask,
                observation_mask=observation_mask,
                force_reset=force_reset,
            )
            outputs.update(memory_out)
            memory_logits = memory_out["fused_memory_bev_logits"]
            if self._on("dynamic", heads):
                occ = DIRECT_BEV_CHANNELS.index("occupied")
                free = DIRECT_BEV_CHANNELS.index("free")
                current_occ = torch.sigmoid(current_logits[:, occ : occ + 1])
                current_free = torch.sigmoid(current_logits[:, free : free + 1])
                previous_occ = torch.sigmoid(memory_out["warped_memory_bev_logits"][:, occ : occ + 1])
                previous_free = torch.sigmoid(memory_out["warped_memory_bev_logits"][:, free : free + 1])
                observed = memory_out["warped_observed_mask"].clamp(0.0, 1.0)
                appeared = current_occ * (1.0 - previous_occ)
                disappeared = previous_occ * current_free
                free_to_occ = previous_free * current_occ
                temporal_dynamic = torch.maximum(torch.maximum(appeared, disappeared), free_to_occ) * observed
                temporal_dynamic = F.max_pool2d(temporal_dynamic, kernel_size=3, stride=1, padding=1)
                learned_dynamic = torch.sigmoid(dynamic_logits)
                outputs["dynamic_occupancy_logits"] = torch.logit(torch.maximum(learned_dynamic, temporal_dynamic).clamp(1.0e-4, 1.0 - 1.0e-4))
        if self._on("world_model", heads):
            world_history = _world_history(current_logits=memory_logits, dynamic_logits=dynamic_logits, uncertainty_logits=uncertainty_logits, configured_history=bev_history, history_frames=self.config.history_frames)
            candidates = _candidate_tensor(candidate_trajectories, batch=batch, device=rgb.device, dtype=rgb.dtype)
            outputs.update(
                self.world_model(
                    world_history,
                    _sequence_or_current(pose_delta_history, pose_delta_prev, batch, self.config.history_frames, 3, rgb),
                    candidates,
                    previous_action_history=_sequence_or_current(previous_action_history, previous_action, batch, self.config.history_frames, 2, rgb),
                    sensor_mask=sensor_mask_history if sensor_mask_history is not None else sensor_mask,
                )
            )
        outputs["debug"] = {"model": HOMEBRAIN_NET_V0_SOURCE, "shared_rgbd_encoder": True, "runtime_forbidden_fields": list(HOMEBRAIN_NET_V0_FORBIDDEN_RUNTIME_FIELDS), "context_shape": list(context.shape), "replay_only": True, "not_executed": True, "control_safe": False, "raw_pwm_emitted": False, "hardware_validated": False}
        return outputs

    def _shared_features(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor | None,
        sensor_mask: torch.Tensor | None,
        pose_delta_prev: torch.Tensor | None,
        previous_action: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must be Bx3xHxW")
        batch = int(rgb.shape[0])
        rgb_r = _resize(rgb.float(), self.config.image_size).clamp(0.0, 1.0)
        image = torch.cat([rgb_r, _depth_input(depth, batch, self.config.image_size, rgb_r)], dim=1)
        latent = self.perception.encoder(image)
        context = torch.cat([_context(sensor_mask, batch, self.config.sensor_mask_dim, rgb_r), _context(pose_delta_prev, batch, self.config.pose_dim, rgb_r), _context(previous_action, batch, self.config.previous_action_dim, rgb_r)], dim=1)
        latent = latent + self.perception.context(context).view(batch, -1, 1, 1)
        bev_latent = self.perception.bev_trunk(F.interpolate(latent, size=self.config.bev_shape, mode="bilinear", align_corners=False))
        return latent, bev_latent, context

    def _on(self, name: str, enabled_heads: set[str] | None) -> bool:
        if enabled_heads is not None:
            return name in enabled_heads
        return bool(getattr(self.config, f"enable_{name}", True))


class _HomeBrainMemoryFusion(nn.Module):
    def __init__(self, config: HomeBrainNetV0Config) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_channels
        channels = len(DIRECT_BEV_CHANNELS)
        self.memory_refine = nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, channels, 1))
        self.uncertainty_refine = nn.Sequential(nn.Conv2d(channels + 2, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.memory_refine[-1].weight)
        nn.init.zeros_(self.memory_refine[-1].bias)

    def init_memory(self, batch: int, like: torch.Tensor) -> SpatialMemoryState:
        h, w = self.config.bev_shape
        memory = torch.full((batch, len(DIRECT_BEV_CHANNELS), h, w), -4.0, dtype=like.dtype, device=like.device)
        memory[:, DIRECT_BEV_CHANNELS.index("unknown")] = 4.0
        return SpatialMemoryState(memory, torch.zeros((batch, 1, h, w), dtype=like.dtype, device=like.device), torch.zeros((batch,), dtype=torch.int64, device=like.device))

    def forward(
        self,
        current_logits: torch.Tensor,
        uncertainty_logits: torch.Tensor,
        *,
        memory_state: SpatialMemoryState | None,
        pose_delta_to_current: torch.Tensor | None,
        pose_delta_to_current_mask: torch.Tensor | None,
        observation_mask: torch.Tensor | None,
        force_reset: torch.Tensor | None,
    ) -> dict[str, Any]:
        batch = int(current_logits.shape[0])
        state = memory_state.to(current_logits.device) if memory_state is not None else self.init_memory(batch, current_logits)
        if force_reset is not None:
            reset = force_reset.to(current_logits.device).view(batch) > 0
            zero = self.init_memory(batch, current_logits)
            state = SpatialMemoryState(torch.where(reset[:, None, None, None], zero.memory_logits, state.memory_logits), torch.where(reset[:, None, None, None], zero.observed_mask, state.observed_mask), torch.where(reset, zero.step_index, state.step_index))
        pose_valid = pose_delta_to_current_mask.view(batch) if pose_delta_to_current_mask is not None else None
        warp = warp_memory_se2(state.memory_logits, pose_delta_to_current, pose_valid, meters_per_cell=self.config.meters_per_cell, missing_pose_behavior="reset")
        obs_warp = warp_memory_se2(state.observed_mask, pose_delta_to_current, pose_valid, meters_per_cell=self.config.meters_per_cell, missing_pose_behavior="reset", mode="nearest")
        current_obs = observation_mask.to(current_logits.device, dtype=current_logits.dtype) if observation_mask is not None else _observed_from_logits(current_logits)
        alpha = torch.tensor(0.65, dtype=current_logits.dtype, device=current_logits.device)
        fused = torch.where(obs_warp.tensor > 0, (1.0 - alpha) * warp.tensor + alpha * current_logits, current_logits)
        fused = fused + self.memory_refine(fused)
        observed = torch.maximum(obs_warp.tensor.clamp(0, 1), current_obs.view(batch, 1, *self.config.bev_shape).clamp(0, 1))
        next_state = SpatialMemoryState(fused, observed, state.step_index + 1)
        uncertainty_grid = self.uncertainty_refine(torch.cat([fused, observed, uncertainty_logits], dim=1))
        return {
            "fused_memory_bev_logits": fused,
            "memory_state": next_state,
            "memory_uncertainty_logits": uncertainty_grid,
            "uncertainty_grid": torch.sigmoid(uncertainty_grid),
            "update_mask": current_obs,
            "warped_memory_bev_logits": warp.tensor,
            "warped_observed_mask": obs_warp.tensor,
            "memory_debug": {
                "pose_warp_valid": warp.pose_warp_valid,
                "pose_warp_used": warp.pose_warp_used,
                "memory_reset": warp.memory_reset,
                "memory_used": next_state.step_index > 1,
            },
        }


def checkpoint_payload(model: HomeBrainNetV0, *, metadata: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {"checkpoint_version": HOMEBRAIN_NET_V0_CHECKPOINT_VERSION, "model_name": "HomeBrainNetV0", "model_config": model.config.to_dict(), "state_dict": model.state_dict(), "metadata": metadata, "metrics": metrics}


def save_checkpoint(path: str | Path, model: HomeBrainNetV0, *, metadata: dict[str, Any], metrics: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, metadata=metadata, metrics=metrics), target)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple[HomeBrainNetV0, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("checkpoint_version") != HOMEBRAIN_NET_V0_CHECKPOINT_VERSION:
        raise ValueError(f"unsupported HomeBrainNetV0 checkpoint {payload.get('checkpoint_version')!r}")
    model = HomeBrainNetV0(HomeBrainNetV0Config.from_dict(payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def warm_start_homebrain_net_v0(model: HomeBrainNetV0, checkpoints: dict[str, str | Path]) -> dict[str, Any]:
    state = model.state_dict()
    copied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for prefix, checkpoint in checkpoints.items():
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        source = payload.get("state_dict", {})
        for key, tensor in source.items():
            target_key = _warm_target(prefix, str(key))
            if target_key is None:
                continue
            if target_key not in state or tuple(state[target_key].shape) != tuple(tensor.shape):
                skipped.append({"source": str(key), "target": target_key, "reason": "missing_or_shape_mismatch"})
                continue
            state[target_key] = tensor.to(dtype=state[target_key].dtype)
            copied.append({"source": str(key), "target": target_key, "shape": list(tensor.shape)})
    model.load_state_dict(state, strict=True)
    return {"schema_version": "homebrain.homebrain_net_v0.warm_start.v0", "copied_count": len(copied), "skipped_count": len(skipped), "copied": copied, "skipped": skipped, "shape_safe": True}


def _warm_target(prefix: str, key: str) -> str | None:
    if prefix == "spatial_memory_v1":
        return key if key.startswith("pose_head.") else f"memory.{key}" if key.startswith("memory_refine.") else None
    return {"direct_bev": "perception.", "world_model": "world_model.", "homebrain": ""}.get(prefix, prefix) + key


def _resize(tensor: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    width, height = image_size
    return tensor if tuple(tensor.shape[-2:]) == (height, width) else F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)


def _depth_input(depth: torch.Tensor | None, batch: int, image_size: tuple[int, int], like: torch.Tensor) -> torch.Tensor:
    width, height = image_size
    if depth is None:
        return torch.zeros((batch, 1, height, width), dtype=like.dtype, device=like.device)
    tensor = depth.to(device=like.device, dtype=like.dtype)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 4 or tensor.shape[0] != batch or tensor.shape[1] != 1:
        return torch.zeros((batch, 1, height, width), dtype=like.dtype, device=like.device)
    return _resize(tensor.clamp(0.0, 6.0) / 6.0, image_size)


def _context(value: torch.Tensor | None, batch: int, dim: int, like: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.zeros((batch, dim), dtype=like.dtype, device=like.device)
    tensor = value.to(device=like.device, dtype=like.dtype)
    if tensor.ndim == 1:
        tensor = tensor.view(1, -1).expand(batch, -1)
    out = torch.zeros((batch, dim), dtype=like.dtype, device=like.device)
    if tensor.ndim == 2 and tensor.shape[0] == batch:
        out[:, : min(dim, tensor.shape[1])] = tensor[:, : min(dim, tensor.shape[1])]
    return out


def _observed_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    evidence = probs[:, [0, 1, 3, 4]].amax(dim=1, keepdim=True)
    unknown = probs[:, 2:3]
    return ((evidence >= 0.65) & (unknown <= 0.45)).to(dtype=logits.dtype)


def _world_history(current_logits: torch.Tensor, dynamic_logits: torch.Tensor, uncertainty_logits: torch.Tensor, configured_history: torch.Tensor | None, history_frames: int) -> torch.Tensor:
    if configured_history is not None:
        return configured_history.to(device=current_logits.device, dtype=current_logits.dtype)
    prob = torch.sigmoid(current_logits)
    frame = torch.stack([prob[:, 0], prob[:, 1], prob[:, 2], prob[:, 3], prob[:, 4], torch.sigmoid(dynamic_logits[:, 0]), torch.sigmoid(uncertainty_logits[:, 0])], dim=1)
    return frame.unsqueeze(1).expand(current_logits.shape[0], history_frames, frame.shape[1], frame.shape[2], frame.shape[3]).contiguous()


def _sequence_or_current(history: torch.Tensor | None, current: torch.Tensor | None, batch: int, steps: int, dim: int, like: torch.Tensor) -> torch.Tensor:
    if history is not None:
        return history.to(device=like.device, dtype=like.dtype)
    base = _context(current, batch, dim, like)
    return base.unsqueeze(1).expand(batch, steps, dim).contiguous()


def _candidate_tensor(value: torch.Tensor | None, *, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if value is not None:
        return value.to(device=device, dtype=dtype)
    return torch.zeros((batch, 1, 2, 3), dtype=dtype, device=device)
