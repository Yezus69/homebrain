from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def compute_spatial_v0_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    labels = batch["bev_labels"]
    mask = batch["bev_label_mask"]
    raw_bce = F.binary_cross_entropy_with_logits(outputs["bev_logits"], labels, reduction="none")
    bev_loss = (raw_bce * mask).sum() / mask.sum().clamp_min(1.0)

    with torch.no_grad():
        per_cell_error = (torch.sigmoid(outputs["bev_logits"]) - labels).abs().mean(dim=1, keepdim=True)
    uncertainty_grid = torch.sigmoid(outputs["uncertainty_logits"])
    uncertainty_loss = F.mse_loss(uncertainty_grid, per_cell_error)

    pose_mask = batch["pose_mask"].view(-1, 1)
    if torch.count_nonzero(pose_mask) > 0:
        pose_loss = (((outputs["pose_delta"] - batch["pose_delta"]) ** 2) * pose_mask).sum() / pose_mask.sum().clamp_min(1.0)
    else:
        pose_loss = torch.zeros((), dtype=bev_loss.dtype, device=bev_loss.device)

    scalar_target = per_cell_error.mean(dim=(1, 2, 3)).detach()
    scalar_uncertainty_loss = F.mse_loss(outputs["uncertainty_scalar"], scalar_target)
    total = bev_loss + torch.tensor(0.1, device=bev_loss.device) * pose_loss
    total = total + torch.tensor(0.05, device=bev_loss.device) * (uncertainty_loss + scalar_uncertainty_loss)
    return {
        "loss": total,
        "bev_loss": bev_loss,
        "pose_loss": pose_loss,
        "uncertainty_loss": uncertainty_loss + scalar_uncertainty_loss,
    }


@torch.no_grad()
def evaluate_spatial_v0(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    bev_losses: list[float] = []
    uncertainty_errors: list[float] = []
    ious: list[float] = []
    pose_sq_error_sum = 0.0
    pose_count = 0
    inference_frames = 0
    started = time.perf_counter()

    for batch in loader:
        batch = batch_to_device(batch, device)
        outputs = model(batch["features"], batch["timestamp_s"], batch["sensor_mask"])
        loss_parts = compute_spatial_v0_losses(outputs, batch)
        losses.append(float(loss_parts["loss"].detach().cpu()))
        bev_losses.append(float(loss_parts["bev_loss"].detach().cpu()))
        probs = torch.sigmoid(outputs["bev_logits"])
        predictions = probs > 0.5
        labels = batch["bev_labels"] > 0.5
        mask = batch["bev_label_mask"] > 0.0
        ious.extend(_masked_iou(predictions, labels, mask))
        per_cell_error = (probs - batch["bev_labels"]).abs().mean(dim=1, keepdim=True)
        uncertainty_grid = torch.sigmoid(outputs["uncertainty_logits"])
        uncertainty_errors.append(float((uncertainty_grid - per_cell_error).abs().mean().detach().cpu()))
        pose_mask = batch["pose_mask"].view(-1) > 0.0
        if torch.count_nonzero(pose_mask) > 0:
            error = outputs["pose_delta"][pose_mask] - batch["pose_delta"][pose_mask]
            pose_sq_error_sum += float((error**2).sum().detach().cpu())
            pose_count += int(error.numel())
        inference_frames += int(batch["features"].shape[0])

    elapsed = max(time.perf_counter() - started, 1e-9)
    pose_rmse = (pose_sq_error_sum / pose_count) ** 0.5 if pose_count else None
    return {
        "loss": _mean(losses),
        "bev_loss": _mean(bev_losses),
        "bev_iou_or_proxy": _mean(ious),
        "pose_delta_rmse": pose_rmse,
        "uncertainty_calibration_proxy": _mean(uncertainty_errors),
        "inference_fps": inference_frames / elapsed,
    }


def _masked_iou(predictions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> list[float]:
    values: list[float] = []
    channels = int(predictions.shape[1])
    for channel in range(channels):
        pred = predictions[:, channel] & mask[:, channel]
        label = labels[:, channel] & mask[:, channel]
        union = torch.count_nonzero(pred | label)
        if int(union.item()) == 0:
            continue
        intersection = torch.count_nonzero(pred & label)
        values.append(float((intersection.float() / union.float()).detach().cpu()))
    return values


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
