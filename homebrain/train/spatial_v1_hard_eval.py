from __future__ import annotations

import time
from typing import Any, Iterable, Literal

import torch
from torch.utils.data import DataLoader

from homebrain.brain.spatial_memory_v1 import warp_memory_se2
from homebrain.train.spatial_dataset import BEV_OUTPUT_CHANNELS
from homebrain.train.spatial_v0_common import batch_to_device

PoseAblationMode = Literal["route_pose", "no_warp", "corrupted_route_pose", "predicted_pose"]
OcclusionMode = Literal["random_block_dropout", "forward_corridor_dropout", "side_region_dropout"]

OCCLUSION_MODES: tuple[OcclusionMode, ...] = (
    "random_block_dropout",
    "forward_corridor_dropout",
    "side_region_dropout",
)
POSE_ABLATION_MODES: tuple[PoseAblationMode, ...] = (
    "route_pose",
    "no_warp",
    "corrupted_route_pose",
    "predicted_pose",
)


@torch.no_grad()
def evaluate_spatial_v1_hard(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    future_horizon: int = 3,
) -> dict[str, Any]:
    deployment = evaluate_deployment_memory(model, loader, device=device, pose_mode="route_pose")
    future = evaluate_future_observation_memory(
        model,
        loader,
        device=device,
        future_horizon=future_horizon,
    )
    occlusion = evaluate_occlusion_stress(model, loader, device=device)
    pose_ablation = evaluate_pose_ablations(model, loader, device=device)
    return {
        "schema_version": "homebrain.spatial_v1_hard_eval.v0",
        "deployment_style": deployment,
        "future_observation": future,
        "occlusion_stress": occlusion,
        "pose_ablation": pose_ablation,
        "pass_gates": {
            "deployment_memory_benefit_pass": deployment["deployment_memory_benefit_pass"],
            "hidden_memory_benefit_pass": future["hidden_memory_benefit_pass"],
            "occlusion_recovery_pass": occlusion["occlusion_recovery_pass"],
            "pose_ablation_pass": pose_ablation["pose_ablation_pass"],
            "hard_validation_pass": bool(
                deployment["deployment_memory_benefit_pass"]
                and future["hidden_memory_benefit_pass"]
                and occlusion["occlusion_recovery_pass"]
                and pose_ablation["pose_ablation_pass"]
            ),
        },
        "safety_flags": {
            "uses_label_update_masks": False,
            "labels_used_only_after_prediction": True,
            "control_safe": False,
            "replay_only": True,
            "not_executed": True,
            "product_training_approved": False,
        },
    }


@torch.no_grad()
def evaluate_deployment_memory(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    pose_mode: PoseAblationMode = "route_pose",
) -> dict[str, Any]:
    model.eval()
    current_ious: list[float] = []
    memory_ious: list[float] = []
    update_coverages: list[float] = []
    overwrite_fractions: list[float] = []
    frame_count = 0
    started = time.perf_counter()
    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device)
        outputs = _forward_deployment(model, batch, pose_mode=pose_mode)
        labels = batch["bev_labels"] > 0.5
        label_mask = batch["bev_label_mask"] > 0.0
        current_probs = torch.sigmoid(outputs["current_bev_logits"])
        memory_probs = torch.sigmoid(outputs["fused_memory_bev_logits"])
        current_ious.extend(_masked_iou_values(current_probs, labels, label_mask))
        memory_ious.extend(_masked_iou_values(memory_probs, labels, label_mask))
        update_coverages.append(float(outputs["debug"]["update_mask_coverage"].float().mean().detach().cpu()))
        overwrite_fractions.append(float(outputs["debug"]["memory_overwrite_fraction"].float().mean().detach().cpu()))
        frame_count += int(batch["features"].shape[0] * batch["features"].shape[1])
    current = _mean(current_ious)
    memory = _mean(memory_ious)
    delta = memory - current
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "deployment_memory_iou": memory,
        "deployment_current_iou": current,
        "deployment_memory_delta": delta,
        "deployment_update_mask_coverage": _mean(update_coverages),
        "deployment_overwrite_fraction": _mean(overwrite_fractions),
        "deployment_memory_benefit_pass": bool(delta > 0.0),
        "frame_count": frame_count,
        "inference_fps": frame_count / elapsed,
        "pose_mode": pose_mode,
        "observation_mask_source": "model_predicted_current_bev_confidence",
        "uses_label_update_masks": False,
        "labels_used_only_after_prediction": True,
    }


@torch.no_grad()
def evaluate_future_observation_memory(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    future_horizon: int = 3,
) -> dict[str, Any]:
    if future_horizon < 1:
        raise ValueError("future_horizon must be at least 1")
    model.eval()
    hidden_memory_ious: list[float] = []
    hidden_current_ious: list[float] = []
    hidden_free_memory_ious: list[float] = []
    hidden_free_current_ious: list[float] = []
    hidden_obstacle_memory_ious: list[float] = []
    hidden_obstacle_current_ious: list[float] = []
    reveal_count = 0
    pair_count = 0
    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device)
        outputs = _forward_deployment(model, batch, pose_mode="route_pose")
        current_probs = torch.sigmoid(outputs["current_bev_logits"])
        memory_probs = torch.sigmoid(outputs["fused_memory_bev_logits"])
        current_trusted = outputs["update_mask"] > 0.5
        batch_reveals = _future_reveal_masks(
            batch,
            future_horizon=future_horizon,
            meters_per_cell=float(model.config.meters_per_cell),
        )
        for reveal in batch_reveals:
            step = int(reveal["step"])
            future_labels = reveal["labels"] > 0.5
            future_observed = reveal["observed"] > 0.5
            chain_valid = reveal["valid"].view(-1, 1, 1, 1)
            hidden_cells = future_observed & (~current_trusted[:, step]) & chain_valid
            cells = int(torch.count_nonzero(hidden_cells).detach().cpu())
            if cells == 0:
                continue
            mask = hidden_cells.repeat(1, len(BEV_OUTPUT_CHANNELS), 1, 1)
            reveal_count += cells
            pair_count += 1
            hidden_memory_ious.extend(
                _masked_iou_values(memory_probs[:, step], future_labels, mask, channel_indices=(0, 1, 3, 4))
            )
            hidden_current_ious.extend(
                _masked_iou_values(current_probs[:, step], future_labels, mask, channel_indices=(0, 1, 3, 4))
            )
            free_mask = hidden_cells.repeat(1, len(BEV_OUTPUT_CHANNELS), 1, 1)
            hidden_free_memory_ious.extend(
                _masked_iou_values(memory_probs[:, step], future_labels, free_mask, channel_indices=(0, 3))
            )
            hidden_free_current_ious.extend(
                _masked_iou_values(current_probs[:, step], future_labels, free_mask, channel_indices=(0, 3))
            )
            hidden_obstacle_memory_ious.extend(
                _masked_iou_values(memory_probs[:, step], future_labels, free_mask, channel_indices=(1, 4))
            )
            hidden_obstacle_current_ious.extend(
                _masked_iou_values(current_probs[:, step], future_labels, free_mask, channel_indices=(1, 4))
            )
    hidden_memory = _mean(hidden_memory_ious)
    hidden_current = _mean(hidden_current_ious)
    hidden_delta = hidden_memory - hidden_current
    return {
        "hidden_cell_iou": hidden_memory,
        "hidden_current_iou": hidden_current,
        "hidden_memory_delta": hidden_delta,
        "hidden_free_iou": _mean(hidden_free_memory_ious),
        "hidden_free_current_iou": _mean(hidden_free_current_ious),
        "hidden_free_delta": _mean(hidden_free_memory_ious) - _mean(hidden_free_current_ious),
        "hidden_obstacle_iou": _mean(hidden_obstacle_memory_ious),
        "hidden_obstacle_current_iou": _mean(hidden_obstacle_current_ious),
        "hidden_obstacle_delta": _mean(hidden_obstacle_memory_ious) - _mean(hidden_obstacle_current_ious),
        "future_reveal_count": reveal_count,
        "future_pair_count": pair_count,
        "future_horizon": int(future_horizon),
        "hidden_memory_benefit_pass": bool(reveal_count > 0 and hidden_delta > 0.0),
        "alignment_pose_source": "route_pose",
        "current_trust_source": "model_update_mask",
        "uses_label_update_masks": False,
    }


@torch.no_grad()
def evaluate_occlusion_stress(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    modes: Iterable[OcclusionMode] = OCCLUSION_MODES,
) -> dict[str, Any]:
    model.eval()
    records: list[dict[str, Any]] = []
    for mode in modes:
        current_ious: list[float] = []
        memory_ious: list[float] = []
        dropped_counts: list[int] = []
        for raw_batch in loader:
            batch = batch_to_device(raw_batch, device)
            base = _forward_deployment(model, batch, pose_mode="route_pose")
            base_update = (base["update_mask"] > 0.5).to(dtype=batch["features"].dtype)
            keep = deterministic_occlusion_keep_mask(base_update, mode=mode, frame_ids=batch["frame_id"])
            occluded_update = base_update * keep
            outputs = _forward_with_observation_mask(model, batch, observation_mask=occluded_update, pose_mode="route_pose")
            dropped = (base_update > 0.5) & (occluded_update <= 0.0)
            dropped_count = int(torch.count_nonzero(dropped).detach().cpu())
            if dropped_count == 0:
                continue
            dropped_counts.append(dropped_count)
            labels = batch["bev_labels"] > 0.5
            mask = dropped.repeat(1, 1, len(BEV_OUTPUT_CHANNELS), 1, 1)
            current_probs = _unknown_out_current(torch.sigmoid(outputs["current_bev_logits"]), dropped)
            memory_probs = torch.sigmoid(outputs["fused_memory_bev_logits"])
            current_ious.extend(_masked_iou_values(current_probs, labels, mask))
            memory_ious.extend(_masked_iou_values(memory_probs, labels, mask))
        current = _mean(current_ious)
        memory = _mean(memory_ious)
        records.append(
            {
                "mode": mode,
                "occluded_current_iou": current,
                "occluded_memory_iou": memory,
                "occlusion_recovery_delta": memory - current,
                "dropped_cell_count": int(sum(dropped_counts)),
                "pass": bool(sum(dropped_counts) > 0 and memory > current),
            }
        )
    deltas = [float(record["occlusion_recovery_delta"]) for record in records]
    return {
        "modes": records,
        "occluded_current_iou": _mean([float(record["occluded_current_iou"]) for record in records]),
        "occluded_memory_iou": _mean([float(record["occluded_memory_iou"]) for record in records]),
        "occlusion_recovery_delta": _mean(deltas),
        "occlusion_recovery_pass": bool(records and all(bool(record["pass"]) for record in records)),
        "mask_source": "deterministic_eval_only_model_update_mask_dropout",
        "uses_label_update_masks": False,
    }


@torch.no_grad()
def evaluate_pose_ablations(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    modes: Iterable[PoseAblationMode] = POSE_ABLATION_MODES,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for mode in modes:
        metrics = evaluate_deployment_memory(model, loader, device=device, pose_mode=mode)
        rows.append(
            {
                "pose_mode": mode,
                "deployment_memory_iou": metrics["deployment_memory_iou"],
                "deployment_current_iou": metrics["deployment_current_iou"],
                "deployment_memory_delta": metrics["deployment_memory_delta"],
                "deployment_update_mask_coverage": metrics["deployment_update_mask_coverage"],
                "deployment_overwrite_fraction": metrics["deployment_overwrite_fraction"],
                "predicted_pose_implemented": mode == "predicted_pose",
            }
        )
    by_mode = {str(row["pose_mode"]): row for row in rows}
    route = float(by_mode.get("route_pose", {}).get("deployment_memory_iou", 0.0))
    no_warp = float(by_mode.get("no_warp", {}).get("deployment_memory_iou", 0.0))
    corrupted = float(by_mode.get("corrupted_route_pose", {}).get("deployment_memory_iou", 0.0))
    predicted = float(by_mode.get("predicted_pose", {}).get("deployment_memory_iou", 0.0))
    best = max(route, no_warp, corrupted, predicted)
    tolerance = 0.005
    return {
        "table": rows,
        "pose_ablation_pass": bool(route >= best - tolerance and route >= corrupted and route >= predicted),
        "expected_best_pose_mode": "route_pose",
        "route_pose_best_tolerance": tolerance,
        "deterministic_corruption": "sinusoidal_pose_delta_noise",
    }


def deterministic_occlusion_keep_mask(
    update_mask: torch.Tensor,
    *,
    mode: OcclusionMode,
    frame_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if update_mask.ndim != 5:
        raise ValueError("update_mask must be BT1HW")
    batch, steps, _channels, height, width = update_mask.shape
    keep = torch.ones_like(update_mask)
    if mode == "forward_corridor_dropout":
        col0 = max(0, width // 2 - max(1, width // 8))
        col1 = min(width, width // 2 + max(1, width // 8) + 1)
        keep[:, :, :, : max(1, (height * 3) // 4), col0:col1] = 0.0
        return keep
    if mode == "side_region_dropout":
        if frame_ids is None:
            parity = torch.arange(steps, device=update_mask.device).view(1, steps).repeat(batch, 1)
        else:
            parity = frame_ids.to(device=update_mask.device).view(batch, steps)
        left = parity.remainder(2) == 0
        for batch_index in range(batch):
            for step_index in range(steps):
                if bool(left[batch_index, step_index]):
                    keep[batch_index, step_index, :, :, : width // 2] = 0.0
                else:
                    keep[batch_index, step_index, :, :, width // 2 :] = 0.0
        return keep
    if mode == "random_block_dropout":
        block_h = max(1, height // 3)
        block_w = max(1, width // 3)
        ids = frame_ids.to(device=update_mask.device).view(batch, steps) if frame_ids is not None else None
        for batch_index in range(batch):
            for step_index in range(steps):
                frame_value = int(ids[batch_index, step_index].detach().cpu()) if ids is not None else step_index
                top_span = max(1, height - block_h + 1)
                left_span = max(1, width - block_w + 1)
                top = (frame_value * 3 + batch_index * 5 + step_index * 7) % top_span
                left_col = (frame_value * 5 + batch_index * 11 + step_index * 13) % left_span
                keep[batch_index, step_index, :, top : top + block_h, left_col : left_col + block_w] = 0.0
        return keep
    raise ValueError(f"unsupported occlusion mode: {mode}")


def _forward_deployment(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    pose_mode: PoseAblationMode,
) -> dict[str, Any]:
    if pose_mode == "predicted_pose":
        first_pose, first_mask = _pose_inputs(batch, mode="route_pose")
        first = model.forward_sequence(
            batch["features"],
            batch["timestamp_s"],
            batch["sensor_mask"],
            pose_delta_to_current=first_pose,
            pose_delta_to_current_mask=first_mask,
            observation_mask=None,
        )
        pose, pose_mask = _pose_inputs(batch, mode="predicted_pose", first_pass_outputs=first)
    else:
        pose, pose_mask = _pose_inputs(batch, mode=pose_mode)
    return model.forward_sequence(
        batch["features"],
        batch["timestamp_s"],
        batch["sensor_mask"],
        pose_delta_to_current=pose,
        pose_delta_to_current_mask=pose_mask,
        observation_mask=None,
    )


def _forward_with_observation_mask(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    observation_mask: torch.Tensor,
    pose_mode: PoseAblationMode,
) -> dict[str, Any]:
    pose, pose_mask = _pose_inputs(batch, mode=pose_mode)
    return model.forward_sequence(
        batch["features"],
        batch["timestamp_s"],
        batch["sensor_mask"],
        pose_delta_to_current=pose,
        pose_delta_to_current_mask=pose_mask,
        observation_mask=observation_mask,
    )


def _pose_inputs(
    batch: dict[str, Any],
    *,
    mode: PoseAblationMode,
    first_pass_outputs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pose = batch["pose_delta_to_current"]
    mask = batch["pose_delta_to_current_mask"]
    if mode == "route_pose":
        return pose, mask
    if mode == "no_warp":
        return torch.zeros_like(pose), torch.ones_like(mask)
    if mode == "corrupted_route_pose":
        frame_ids = batch["frame_id"].to(dtype=pose.dtype, device=pose.device).unsqueeze(-1)
        noise = torch.cat(
            [
                0.03 * torch.sin(frame_ids * 0.37),
                0.02 * torch.cos(frame_ids * 0.19),
                0.08 * torch.sin(frame_ids * 0.11),
            ],
            dim=-1,
        )
        return pose + noise * mask, mask
    if mode == "predicted_pose":
        if first_pass_outputs is None:
            raise ValueError("predicted_pose mode requires first_pass_outputs")
        predicted = first_pass_outputs["pose_delta"].detach()
        predicted_mask = torch.ones_like(mask)
        predicted_mask[:, 0] = 0.0
        return predicted, predicted_mask
    raise ValueError(f"unsupported pose ablation mode: {mode}")


def _future_reveal_masks(
    batch: dict[str, torch.Tensor],
    *,
    future_horizon: int,
    meters_per_cell: float,
) -> list[dict[str, torch.Tensor | int]]:
    labels = batch["bev_labels"].to(dtype=torch.float32)
    observed = _label_observed_mask(labels)
    pose = batch["pose_delta_to_current"]
    pose_mask = batch["pose_delta_to_current_mask"]
    _batch, steps, _channels, _height, _width = labels.shape
    reveals: list[dict[str, torch.Tensor | int]] = []
    for step in range(steps - 1):
        for future in range(step + 1, min(steps, step + future_horizon + 1)):
            future_labels = labels[:, future]
            future_observed = observed[:, future]
            valid = torch.ones((labels.shape[0],), dtype=torch.bool, device=labels.device)
            aligned_labels = future_labels
            aligned_observed = future_observed
            for cursor in range(future, step, -1):
                inv_pose = invert_pose_delta(pose[:, cursor])
                cursor_valid = pose_mask[:, cursor].view(-1) > 0.0
                valid = valid & cursor_valid
                aligned_labels = warp_memory_se2(
                    aligned_labels,
                    inv_pose,
                    cursor_valid.to(dtype=pose.dtype),
                    meters_per_cell=meters_per_cell,
                    missing_pose_behavior="no_warp",
                ).tensor
                aligned_observed = warp_memory_se2(
                    aligned_observed,
                    inv_pose,
                    cursor_valid.to(dtype=pose.dtype),
                    meters_per_cell=meters_per_cell,
                    missing_pose_behavior="no_warp",
                    mode="nearest",
                ).tensor.clamp(0.0, 1.0)
            reveals.append(
                {
                    "step": step,
                    "future_step": future,
                    "labels": aligned_labels,
                    "observed": aligned_observed,
                    "valid": valid,
                }
            )
    return reveals


def invert_pose_delta(pose_delta: torch.Tensor) -> torch.Tensor:
    if pose_delta.shape[-1] != 3:
        raise ValueError("pose_delta must end in 3 values")
    dx = pose_delta[..., 0]
    dy = pose_delta[..., 1]
    yaw = pose_delta[..., 2]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    inv_dx = -(cos_yaw * dx + sin_yaw * dy)
    inv_dy = -(-sin_yaw * dx + cos_yaw * dy)
    inv_yaw = -yaw
    return torch.stack([inv_dx, inv_dy, inv_yaw], dim=-1)


def _unknown_out_current(probs: torch.Tensor, hidden_mask: torch.Tensor) -> torch.Tensor:
    result = probs.clone()
    hidden = hidden_mask.to(dtype=torch.bool)
    for channel, name in enumerate(BEV_OUTPUT_CHANNELS):
        fill = 1.0 if name == "unknown" else 0.0
        result[:, :, channel] = torch.where(hidden[:, :, 0], torch.full_like(result[:, :, channel], fill), result[:, :, channel])
    return result


def _label_observed_mask(labels: torch.Tensor) -> torch.Tensor:
    free = labels[:, :, BEV_OUTPUT_CHANNELS.index("free") : BEV_OUTPUT_CHANNELS.index("free") + 1]
    occupied = labels[:, :, BEV_OUTPUT_CHANNELS.index("occupied") : BEV_OUTPUT_CHANNELS.index("occupied") + 1]
    return torch.clamp(free + occupied, 0.0, 1.0)


def _masked_iou_values(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    channel_indices: Iterable[int] | None = None,
) -> list[float]:
    if probabilities.ndim == 4:
        probabilities = probabilities.unsqueeze(1)
        labels = labels.unsqueeze(1)
        mask = mask.unsqueeze(1)
    if mask.shape[2] == 1:
        mask = mask.repeat(1, 1, probabilities.shape[2], 1, 1)
    values: list[float] = []
    channels = tuple(channel_indices) if channel_indices is not None else tuple(range(int(probabilities.shape[2])))
    for channel in channels:
        pred = (probabilities[:, :, channel] > 0.5) & mask[:, :, channel]
        label = labels[:, :, channel] & mask[:, :, channel]
        union = torch.count_nonzero(pred | label)
        if int(union.detach().cpu()) == 0:
            continue
        intersection = torch.count_nonzero(pred & label)
        values.append(float((intersection.float() / union.float()).detach().cpu()))
    return values


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
