from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib

import numpy as np
import torch

from homebrain.brain.direct_bev_student_v0 import DIRECT_BEV_CHANNELS, DirectBEVStudentV0, load_checkpoint
from homebrain.messages.schema import JsonDict
from homebrain.policies.trajectory_scorer import LocalBev


@dataclass(frozen=True)
class RuntimeDirectBEVStudent:
    model: DirectBEVStudentV0
    checkpoint: Path
    checkpoint_sha256: str
    metadata: JsonDict
    device: torch.device


def load_runtime_direct_bev_student(
    checkpoint: str | Path,
    device: str | torch.device | None = None,
) -> RuntimeDirectBEVStudent:
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_checkpoint(checkpoint, map_location=resolved_device)
    model.to(resolved_device)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return RuntimeDirectBEVStudent(
        model=model,
        checkpoint=Path(checkpoint),
        checkpoint_sha256=_file_sha256(Path(checkpoint)),
        metadata=dict(metadata),
        device=resolved_device,
    )


def predict_local_bev_from_rgbd(
    runtime: RuntimeDirectBEVStudent,
    rgb: np.ndarray | torch.Tensor,
    depth: np.ndarray | torch.Tensor | None = None,
    pose_delta_prev: tuple[float, float, float] | np.ndarray | torch.Tensor | None = None,
    previous_action: tuple[float, float] | np.ndarray | torch.Tensor | None = None,
    sensor_mask: np.ndarray | torch.Tensor | None = None,
) -> dict[str, Any]:
    model = runtime.model
    model.eval()
    rgb_tensor = _rgb_tensor(rgb, device=runtime.device)
    depth_tensor = _depth_tensor(depth, device=runtime.device)
    pose_tensor = _vector_tensor(pose_delta_prev, width=model.config.pose_dim, device=runtime.device)
    action_tensor = _vector_tensor(previous_action, width=model.config.previous_action_dim, device=runtime.device)
    sensor_tensor = _sensor_tensor(sensor_mask, depth_present=depth is not None, width=model.config.sensor_mask_dim, device=runtime.device)
    with torch.no_grad():
        outputs = model(
            rgb_tensor,
            depth=depth_tensor,
            sensor_mask=sensor_tensor,
            pose_delta_prev=pose_tensor,
            previous_action=action_tensor,
        )
    arrays = {
        "bev_prob": torch.sigmoid(outputs["bev_logits"])[0].detach().cpu().numpy().astype(np.float32),
        "uncertainty_prob": torch.sigmoid(outputs["uncertainty_logits"])[0, 0].detach().cpu().numpy().astype(np.float32),
        "dynamic_risk_prob": torch.sigmoid(outputs["dynamic_risk_logits"])[0, 0].detach().cpu().numpy().astype(np.float32),
        "compact_features": outputs["compact_features"][0].detach().cpu().numpy().astype(np.float32),
    }
    local_bev = direct_bev_to_local_bev(arrays, source="direct_bev_student_v0_runtime")
    return {
        "local_bev": local_bev,
        "arrays": arrays,
        "debug": {
            "direct_bev_student_checkpoint": runtime.checkpoint.as_posix(),
            "direct_bev_student_checkpoint_sha256": runtime.checkpoint_sha256,
            "direct_bev_student_source_dataset": runtime.metadata.get("source_dataset_name"),
            "direct_bev_runtime_inputs": [
                "current_rgb",
                "current_depth" if depth is not None else "depth_missing_zero_filled",
                "pose_delta_prev" if pose_delta_prev is not None else "pose_delta_prev_missing_zero_filled",
                "previous_action" if previous_action is not None else "previous_action_missing_zero_filled",
            ],
            "direct_bev_teacher_fields_used_at_runtime": False,
            "direct_bev_future_labels_used_at_runtime": False,
            "direct_bev_replay_only": True,
            "direct_bev_control_safe": False,
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "raw_pwm_emitted": False,
            "hardware_validated": False,
        },
    }


def direct_bev_to_local_bev(
    prediction: dict[str, np.ndarray] | torch.Tensor,
    *,
    source: str = "direct_bev_student_v0",
) -> LocalBev:
    if isinstance(prediction, torch.Tensor):
        bev = torch.sigmoid(prediction)[0].detach().cpu().numpy().astype(np.float32)
        uncertainty = None
        dynamic = None
    else:
        bev = np.asarray(prediction["bev_prob"], dtype=np.float32)
        uncertainty = np.asarray(prediction.get("uncertainty_prob"), dtype=np.float32) if "uncertainty_prob" in prediction else None
        dynamic = np.asarray(prediction.get("dynamic_risk_prob"), dtype=np.float32) if "dynamic_risk_prob" in prediction else None
    if bev.shape[0] != len(DIRECT_BEV_CHANNELS):
        raise ValueError("DirectBEV prediction must have five BEV channels")
    risky = np.maximum(bev[4], dynamic).astype(np.float32) if dynamic is not None else bev[4].astype(np.float32)
    return LocalBev(
        free=np.clip(bev[0], 0.0, 1.0).astype(np.float32),
        occupied=np.clip(bev[1], 0.0, 1.0).astype(np.float32),
        unknown=np.clip(bev[2], 0.0, 1.0).astype(np.float32),
        traversable=np.clip(bev[3], 0.0, 1.0).astype(np.float32),
        risky=np.clip(risky, 0.0, 1.0).astype(np.float32),
        confidence=(1.0 - np.clip(uncertainty, 0.0, 1.0)).astype(np.float32) if uncertainty is not None else None,
        uncertainty=np.clip(uncertainty, 0.0, 1.0).astype(np.float32) if uncertainty is not None else None,
        source=source,
    )


def _rgb_tensor(rgb: np.ndarray | torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if isinstance(rgb, torch.Tensor):
        tensor = rgb.detach().to(device=device, dtype=torch.float32)
    else:
        array = np.asarray(rgb)
        if array.ndim == 3 and array.shape[-1] == 3:
            array = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(array.astype(np.float32)).to(device)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.max().detach().item() > 2.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def _depth_tensor(depth: np.ndarray | torch.Tensor | None, *, device: torch.device) -> torch.Tensor | None:
    if depth is None:
        return None
    if isinstance(depth, torch.Tensor):
        tensor = depth.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.from_numpy(np.asarray(depth, dtype=np.float32)).to(device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0) if tensor.shape[0] == 1 else tensor[:, None, :, :]
    return tensor


def _vector_tensor(
    value: tuple[float, ...] | np.ndarray | torch.Tensor | None,
    *,
    width: int,
    device: torch.device,
) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device, dtype=torch.float32).view(1, -1)
    else:
        tensor = torch.tensor(np.asarray(value, dtype=np.float32).reshape(1, -1), device=device)
    if tensor.shape[1] == width:
        return tensor
    out = torch.zeros((1, width), dtype=torch.float32, device=device)
    count = min(width, int(tensor.shape[1]))
    out[:, :count] = tensor[:, :count]
    return out


def _sensor_tensor(
    value: np.ndarray | torch.Tensor | None,
    *,
    depth_present: bool,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if value is not None:
        tensor = _vector_tensor(value, width=width, device=device)
        if tensor is not None:
            return tensor
    base = torch.zeros((1, width), dtype=torch.float32, device=device)
    if width > 0:
        base[:, 0] = 1.0
    if width > 1:
        base[:, 1] = 1.0 if depth_present else 0.0
    return base


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
