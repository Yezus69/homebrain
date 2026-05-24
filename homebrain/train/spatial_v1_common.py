from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from homebrain.brain.spatial_memory_v1 import warp_memory_se2
from homebrain.train.spatial_dataset import BEV_OUTPUT_CHANNELS
from homebrain.train.spatial_v0_common import batch_to_device


def compute_spatial_v1_losses(
    outputs: dict[str, Any],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    current_raw = F.binary_cross_entropy_with_logits(
        outputs["current_bev_logits"],
        batch["bev_labels"],
        reduction="none",
    )
    current_loss = (current_raw * batch["bev_label_mask"]).sum() / batch["bev_label_mask"].sum().clamp_min(1.0)

    memory_raw = F.binary_cross_entropy_with_logits(
        outputs["fused_memory_bev_logits"],
        batch["fused_memory_targets"],
        reduction="none",
    )
    memory_mask = batch["fused_memory_target_mask"]
    memory_loss = (memory_raw * memory_mask).sum() / memory_mask.sum().clamp_min(1.0)

    pose_mask = batch["pose_mask"].view(*batch["pose_mask"].shape[:2], 1)
    if torch.count_nonzero(pose_mask) > 0:
        pose_loss = (((outputs["pose_delta"] - batch["pose_delta"]) ** 2) * pose_mask).sum() / pose_mask.sum().clamp_min(1.0)
    else:
        pose_loss = torch.zeros((), dtype=current_loss.dtype, device=current_loss.device)

    with torch.no_grad():
        per_cell_error = (
            torch.sigmoid(outputs["fused_memory_bev_logits"]) - batch["fused_memory_targets"]
        ).abs().mean(dim=2, keepdim=True)
    uncertainty_grid = torch.sigmoid(outputs["uncertainty_logits"])
    uncertainty_loss = F.mse_loss(uncertainty_grid, per_cell_error)
    scalar_target = per_cell_error.mean(dim=(2, 3, 4)).detach()
    scalar_uncertainty_loss = F.mse_loss(outputs["uncertainty_scalar"], scalar_target)

    total = current_loss + memory_loss
    total = total + torch.tensor(0.1, device=total.device) * pose_loss
    total = total + torch.tensor(0.05, device=total.device) * (uncertainty_loss + scalar_uncertainty_loss)
    return {
        "loss": total,
        "current_bev_loss": current_loss,
        "fused_memory_bev_loss": memory_loss,
        "pose_loss": pose_loss,
        "uncertainty_loss": uncertainty_loss + scalar_uncertainty_loss,
    }


@torch.no_grad()
def evaluate_spatial_v1(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    current_losses: list[float] = []
    memory_losses: list[float] = []
    current_ious: list[float] = []
    memory_ious: list[float] = []
    temporal_ious: list[float] = []
    unknown_reductions: list[float] = []
    uncertainty_errors: list[float] = []
    pose_sq_error_sum = 0.0
    pose_count = 0
    warp_valid_count = 0
    reset_count = 0
    frame_count = 0
    started = time.perf_counter()

    for batch in loader:
        batch = batch_to_device(batch, device)
        outputs = model.forward_sequence(
            batch["features"],
            batch["timestamp_s"],
            batch["sensor_mask"],
            pose_delta_to_current=batch["pose_delta_to_current"],
            pose_delta_to_current_mask=batch["pose_delta_to_current_mask"],
            observation_mask=batch["observation_mask"],
        )
        loss_parts = compute_spatial_v1_losses(outputs, batch)
        losses.append(float(loss_parts["loss"].detach().cpu()))
        current_losses.append(float(loss_parts["current_bev_loss"].detach().cpu()))
        memory_losses.append(float(loss_parts["fused_memory_bev_loss"].detach().cpu()))

        current_probs = torch.sigmoid(outputs["current_bev_logits"])
        memory_probs = torch.sigmoid(outputs["fused_memory_bev_logits"])
        current_ious.extend(_masked_iou(current_probs > 0.5, batch["bev_labels"] > 0.5, batch["bev_label_mask"] > 0.0))
        memory_ious.extend(
            _masked_iou(
                memory_probs > 0.5,
                batch["fused_memory_targets"] > 0.5,
                batch["fused_memory_target_mask"] > 0.0,
            )
        )
        unknown_index = BEV_OUTPUT_CHANNELS.index("unknown")
        unknown_reductions.append(
            float((current_probs[:, :, unknown_index] - memory_probs[:, :, unknown_index]).mean().detach().cpu())
        )
        temporal_ious.extend(
            _temporal_reprojection_iou(
                memory_probs,
                batch["pose_delta_to_current"],
                batch["pose_delta_to_current_mask"],
                meters_per_cell=float(model.config.meters_per_cell),
            )
        )

        per_cell_error = (memory_probs - batch["fused_memory_targets"]).abs().mean(dim=2, keepdim=True)
        uncertainty_grid = torch.sigmoid(outputs["uncertainty_logits"])
        uncertainty_errors.append(float((uncertainty_grid - per_cell_error).abs().mean().detach().cpu()))

        pose_mask = batch["pose_mask"].view(*batch["pose_mask"].shape[:2]) > 0.0
        if torch.count_nonzero(pose_mask) > 0:
            error = outputs["pose_delta"][pose_mask] - batch["pose_delta"][pose_mask]
            pose_sq_error_sum += float((error**2).sum().detach().cpu())
            pose_count += int(error.numel())

        debug = outputs["debug"]
        warp_valid_count += int(torch.count_nonzero(debug["pose_warp_valid"]).detach().cpu())
        reset_count += int(torch.count_nonzero(debug["memory_reset"]).detach().cpu())
        frame_count += int(batch["features"].shape[0] * batch["features"].shape[1])

    elapsed = max(time.perf_counter() - started, 1e-9)
    pose_rmse = (pose_sq_error_sum / pose_count) ** 0.5 if pose_count else None
    return {
        "loss": _mean(losses),
        "current_bev_loss": _mean(current_losses),
        "fused_memory_bev_loss": _mean(memory_losses),
        "current_bev_iou_or_proxy": _mean(current_ious),
        "fused_memory_bev_iou_or_proxy": _mean(memory_ious),
        "unknown_reduction_vs_current": _mean(unknown_reductions),
        "temporal_reprojection_consistency_iou": _mean(temporal_ious),
        "pose_delta_rmse": pose_rmse,
        "uncertainty_calibration_proxy": _mean(uncertainty_errors),
        "inference_fps": frame_count / elapsed,
        "memory_warp_valid_fraction": warp_valid_count / float(frame_count) if frame_count else 0.0,
        "memory_reset_fraction": reset_count / float(frame_count) if frame_count else 0.0,
        "frame_count": frame_count,
    }


def memory_disabled_baseline_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_name": "v1_window_length_1_memory_disabled_proxy",
        "current_bev_iou_or_proxy": metrics["current_bev_iou_or_proxy"],
        "fused_memory_bev_iou_or_proxy": metrics["current_bev_iou_or_proxy"],
        "temporal_reprojection_consistency_iou": 0.0,
        "unknown_reduction_vs_current": 0.0,
    }


def memory_benefit_result(
    *,
    v1_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    current_regression_tolerance: float = 0.02,
) -> dict[str, Any]:
    fused_delta = float(v1_metrics["fused_memory_bev_iou_or_proxy"]) - float(
        baseline_metrics["fused_memory_bev_iou_or_proxy"]
    )
    temporal_delta = float(v1_metrics["temporal_reprojection_consistency_iou"]) - float(
        baseline_metrics.get("temporal_reprojection_consistency_iou", 0.0)
    )
    current_delta = float(v1_metrics["current_bev_iou_or_proxy"]) - float(
        baseline_metrics.get("current_bev_iou_or_proxy", v1_metrics["current_bev_iou_or_proxy"])
    )
    pass_gate = (fused_delta > 0.0 or temporal_delta > 0.0) and current_delta >= -current_regression_tolerance
    reasons: list[str] = []
    if fused_delta <= 0.0 and temporal_delta <= 0.0:
        reasons.append("no_fused_iou_or_temporal_consistency_improvement")
    if current_delta < -current_regression_tolerance:
        reasons.append("current_bev_regressed_beyond_tolerance")
    return {
        "memory_benefit_pass": bool(pass_gate),
        "fused_memory_iou_delta": fused_delta,
        "temporal_consistency_iou_delta": temporal_delta,
        "current_bev_iou_delta": current_delta,
        "current_regression_tolerance": current_regression_tolerance,
        "failure_reasons": reasons,
    }


def _masked_iou(predictions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> list[float]:
    values: list[float] = []
    channels = int(predictions.shape[2])
    for channel in range(channels):
        pred = predictions[:, :, channel] & mask[:, :, channel]
        label = labels[:, :, channel] & mask[:, :, channel]
        union = torch.count_nonzero(pred | label)
        if int(union.item()) == 0:
            continue
        intersection = torch.count_nonzero(pred & label)
        values.append(float((intersection.float() / union.float()).detach().cpu()))
    return values


def _temporal_reprojection_iou(
    probs: torch.Tensor,
    pose_delta_to_current: torch.Tensor,
    pose_delta_to_current_mask: torch.Tensor,
    *,
    meters_per_cell: float,
) -> list[float]:
    values: list[float] = []
    _batch, steps, _channels, _height, _width = probs.shape
    for step_index in range(1, steps):
        valid = pose_delta_to_current_mask[:, step_index].view(-1) > 0.0
        if torch.count_nonzero(valid) == 0:
            continue
        warped = warp_memory_se2(
            probs[:, step_index - 1],
            pose_delta_to_current[:, step_index],
            pose_delta_to_current_mask[:, step_index],
            meters_per_cell=meters_per_cell,
            missing_pose_behavior="no_warp",
        ).tensor
        pred = probs[:, step_index][valid] > 0.5
        reproj = warped[valid] > 0.5
        for channel in range(int(probs.shape[2])):
            union = torch.count_nonzero(pred[:, channel] | reproj[:, channel])
            if int(union.item()) == 0:
                continue
            intersection = torch.count_nonzero(pred[:, channel] & reproj[:, channel])
            values.append(float((intersection.float() / union.float()).detach().cpu()))
    return values


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
