from __future__ import annotations

import argparse
import json
import math
from itertools import cycle
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from homebrain.brain.future_bev_rollout_v1 import (
    CANDIDATE_OUTCOME_CHANNELS,
    FUTURE_BEV_ROLLOUT_V1_SOURCE,
    FutureBEVRolloutV1,
    FutureBEVRolloutV1Config,
    save_checkpoint,
)
from homebrain.data.spatial_dataset import write_json
from homebrain.train.future_bev_rollout_dataset import FutureBEVRolloutDataset
from homebrain.train.spatial_v0_common import batch_to_device


def train_future_bev_rollout_v1(
    *,
    rollout_pack: str | Path,
    out_dir: str | Path,
    max_steps: int = 300,
    batch_size: int = 4,
    learning_rate: float = 1.0e-3,
    hidden_channels: int = 48,
    tiny_overfit: bool = False,
    device_name: str | None = None,
    seed: int = 24,
    command: str | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_dataset = FutureBEVRolloutDataset(rollout_pack, split="train", tiny_overfit=tiny_overfit)
    try:
        val_dataset = train_dataset if tiny_overfit else FutureBEVRolloutDataset(rollout_pack, split="val")
    except ValueError:
        val_dataset = train_dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        generator=_generator(seed),
    )
    train_eval_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)
    config = FutureBEVRolloutV1Config(
        bev_shape=train_dataset.grid_shape,
        feature_dim=int(train_dataset.feature_shape[-1]),
        sensor_dim=int(train_dataset.sensor_dim),
        hidden_channels=int(hidden_channels),
        history_steps=int(train_dataset.history_steps),
        horizons_s=train_dataset.horizons_s,
        meters_per_cell=float(train_dataset.manifest.get("meters_per_cell", 0.05)),
        robot_radius_m=float(train_dataset.manifest.get("robot_radius_m", 0.18)),
    )
    model = FutureBEVRolloutV1(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    initial_train = evaluate_future_bev_rollout_v1(model, train_eval_loader, device=device, dataset=train_dataset)
    initial_val = evaluate_future_bev_rollout_v1(model, val_loader, device=device, dataset=val_dataset)
    iterator = cycle(train_loader)
    model.train()
    for _step in range(max_steps):
        batch = batch_to_device(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        outputs = _model_forward(model, batch)
        losses = compute_future_bev_rollout_losses(outputs, batch)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    final_train = evaluate_future_bev_rollout_v1(model, train_eval_loader, device=device, dataset=train_dataset)
    final_val = evaluate_future_bev_rollout_v1(model, val_loader, device=device, dataset=val_dataset)
    start_loss = float(initial_train["loss"])
    end_loss = float(final_train["loss"])
    val_start_loss = float(initial_val["loss"])
    val_end_loss = float(final_val["loss"])
    loss_reduction_ratio = (start_loss - end_loss) / start_loss if start_loss > 0.0 else 0.0
    val_loss_reduction_ratio = (val_start_loss - val_end_loss) / val_start_loss if val_start_loss > 0.0 else 0.0
    failure_flags: list[str] = []
    if not np.isfinite(end_loss) or not np.isfinite(val_end_loss):
        failure_flags.append("non_finite_loss")
    if max_steps > 0 and np.isfinite(loss_reduction_ratio) and loss_reduction_ratio <= 0.0:
        failure_flags.append("train_loss_not_reduced")
    if float(final_val["action_entropy"]) <= 1.0e-6:
        failure_flags.append("action_entropy_zero")
    if float(final_val["dominant_action_fraction"]) >= 0.95:
        failure_flags.append("action_distribution_collapsed")
    metrics: dict[str, Any] = {
        "schema_version": "homebrain.future_bev_rollout_v1_train_metrics.v1",
        "model": FUTURE_BEV_ROLLOUT_V1_SOURCE,
        "rollout_pack": Path(rollout_pack).as_posix(),
        "out_dir": Path(out_dir).as_posix(),
        "train_command": command,
        "max_steps": int(max_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "device": str(device),
        "tiny_overfit": bool(tiny_overfit),
        "train_example_count": len(train_dataset),
        "val_example_count": len(val_dataset),
        "grid_shape": list(train_dataset.grid_shape),
        "feature_shape": list(train_dataset.feature_shape),
        "history_steps": int(train_dataset.history_steps),
        "horizons_s": [float(value) for value in train_dataset.horizons_s],
        "candidate_ids": list(train_dataset.candidate_ids),
        "candidate_outcome_channels": list(CANDIDATE_OUTCOME_CHANNELS),
        "train_loss_start": start_loss,
        "train_loss_end": end_loss,
        "loss_reduction_ratio": float(loss_reduction_ratio),
        "val_loss_start": val_start_loss,
        "val_loss": val_end_loss,
        "val_loss_reduction_ratio": float(val_loss_reduction_ratio),
        "future_free_iou_or_proxy": float(final_val["future_free_iou_or_proxy"]),
        "future_occupied_iou_or_proxy": float(final_val["future_occupied_iou_or_proxy"]),
        "future_unknown_iou_or_proxy": float(final_val["future_unknown_iou_or_proxy"]),
        "future_risk_iou_or_proxy": float(final_val["future_risk_iou_or_proxy"]),
        "future_risk_auc_or_proxy": final_val["future_risk_auc_or_proxy"],
        "copy_forward_free_iou_or_proxy": float(final_val["copy_forward_free_iou_or_proxy"]),
        "copy_forward_occupied_iou_or_proxy": float(final_val["copy_forward_occupied_iou_or_proxy"]),
        "copy_forward_unknown_iou_or_proxy": float(final_val["copy_forward_unknown_iou_or_proxy"]),
        "copy_forward_risk_iou_or_proxy": float(final_val["copy_forward_risk_iou_or_proxy"]),
        "improvement_vs_copy_forward": float(final_val["improvement_vs_copy_forward"]),
        "future_free_all_zero_iou_baseline": float(final_val["future_free_all_zero_iou_baseline"]),
        "future_occupied_all_zero_iou_baseline": float(final_val["future_occupied_all_zero_iou_baseline"]),
        "future_unknown_all_one_iou_baseline": float(final_val["future_unknown_all_one_iou_baseline"]),
        "newly_observed_precision": float(final_val["newly_observed_precision"]),
        "newly_observed_recall": float(final_val["newly_observed_recall"]),
        "unknown_reduction_prediction_quality": float(final_val["unknown_reduction_prediction_quality"]),
        "candidate_collision_auroc": final_val["candidate_collision_auroc"],
        "candidate_collision_accuracy": float(final_val["candidate_collision_accuracy"]),
        "candidate_collision_positive_rate": float(final_val["candidate_collision_positive_rate"]),
        "candidate_future_collision_positive_rate": float(final_val["candidate_future_collision_positive_rate"]),
        "candidate_future_risk_mse": float(final_val["candidate_future_risk_mse"]),
        "candidate_unsafe_now_positive_rate": float(final_val["candidate_unsafe_now_positive_rate"]),
        "candidate_collision_mse": float(final_val["candidate_collision_mse"]),
        "candidate_collision_always_zero_mse_baseline": float(final_val["candidate_collision_always_zero_mse_baseline"]),
        "candidate_collision_always_negative_accuracy_baseline": float(
            final_val["candidate_collision_always_negative_accuracy_baseline"]
        ),
        "candidate_unknown_exposure_mse": float(final_val["candidate_unknown_exposure_mse"]),
        "candidate_unknown_exposure_always_one_mse_baseline": float(
            final_val["candidate_unknown_exposure_always_one_mse_baseline"]
        ),
        "candidate_new_area_gain_ranking_quality": float(final_val["candidate_new_area_gain_ranking_quality"]),
        "candidate_risk_ranking_accuracy": float(final_val["candidate_risk_ranking_accuracy"]),
        "candidate_oracle_match_fraction": float(final_val["candidate_oracle_match_fraction"]),
        "unsafe_candidate_rejection_rate": float(final_val["unsafe_candidate_rejection_rate"]),
        "unsafe_selected_rate": float(final_val["unsafe_selected_rate"]),
        "stop_selected_fraction": float(final_val["stop_selected_fraction"]),
        "candidate_oracle_selected_distribution": final_val["candidate_oracle_selected_distribution"],
        "beats_collision_always_negative_baseline": bool(final_val["beats_collision_always_negative_baseline"]),
        "beats_unknown_always_one_baseline": bool(final_val["beats_unknown_always_one_baseline"]),
        "beats_future_all_unknown_baseline": bool(final_val["beats_future_all_unknown_baseline"]),
        "action_entropy": float(final_val["action_entropy"]),
        "dominant_action_fraction": float(final_val["dominant_action_fraction"]),
        "route_held_out_generalization": final_val["route_held_out_generalization"],
        "failure_flags": failure_flags,
        "source_pack_manifest_sha256": _file_hash(Path(rollout_pack) / "manifest.json"),
        "rollout_pack_metadata": {
            "valid_horizon_fraction": train_dataset.manifest.get("valid_horizon_fraction"),
            "candidate_valid_fraction": train_dataset.manifest.get("candidate_valid_fraction"),
            "target_generation": train_dataset.manifest.get("target_generation"),
        },
        "representation_pretraining_only": True,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "product_training_approved": False,
        "raw_pwm_emitted": False,
    }
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "train_metrics.json", metrics, pretty=True)
    write_json(
        output / "config.json",
        {
            "schema_version": "homebrain.future_bev_rollout_v1_train_config.v1",
            "rollout_pack": Path(rollout_pack).as_posix(),
            "max_steps": int(max_steps),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "hidden_channels": int(hidden_channels),
            "history_steps": int(train_dataset.history_steps),
            "seed": int(seed),
            "device": str(device),
            "model_config": config.to_dict(),
            "representation_pretraining_only": True,
            "control_safe": False,
        },
        pretty=True,
    )
    save_checkpoint(
        output / "checkpoint.pt",
        model.cpu(),
        metadata={
            "rollout_pack": Path(rollout_pack).as_posix(),
            "horizons_s": [float(value) for value in train_dataset.horizons_s],
            "history_steps": int(train_dataset.history_steps),
            "meters_per_cell": float(train_dataset.manifest.get("meters_per_cell", 0.05)),
            "robot_radius_m": float(train_dataset.manifest.get("robot_radius_m", 0.18)),
            "candidate_ids": list(train_dataset.candidate_ids),
            "candidate_outcome_channels": list(CANDIDATE_OUTCOME_CHANNELS),
            "train_command": command,
            "representation_pretraining_only": True,
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "product_training_approved": False,
            "raw_pwm_emitted": False,
            "hardware_validated": False,
        },
        metrics=metrics,
    )
    return metrics


def compute_future_bev_rollout_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    future_mask = batch["future_valid_mask"]
    future_labels = batch["future_bev"]
    future_bev_loss = _masked_balanced_bce_with_logits(
        outputs["future_bev_logits"],
        future_labels,
        future_mask,
        focal_gamma=0.75,
    )
    future_risk_loss = _masked_balanced_bce_with_logits(
        outputs["future_risk_logits"],
        batch["future_risk"],
        future_mask,
        focal_gamma=1.25,
    )
    unknown_target = future_labels[:, :, 2:3]
    uncertainty_bce = F.binary_cross_entropy_with_logits(outputs["uncertainty_logits"], unknown_target, reduction="none")
    uncertainty_loss = (uncertainty_bce * future_mask).sum() / future_mask.sum().clamp_min(1.0)
    candidate_mask = batch["candidate_valid_mask"]
    collision_loss = _masked_balanced_bce_with_logits(
        outputs["candidate_collision_logit"],
        batch["candidate_collision"],
        candidate_mask,
        focal_gamma=1.5,
    )
    future_collision_loss = _masked_balanced_bce_with_logits(
        outputs["candidate_future_collision_logit"],
        batch["candidate_future_collision"],
        candidate_mask,
        focal_gamma=1.5,
    )
    unsafe_now_loss = _masked_balanced_bce_with_logits(
        outputs["candidate_unsafe_now_logit"],
        batch["candidate_unsafe_now"],
        candidate_mask,
        focal_gamma=1.0,
    )
    unknown_loss = _masked_mse(
        torch.sigmoid(outputs["candidate_unknown_exposure_logit"]),
        batch["candidate_unknown_exposure"],
        candidate_mask,
    )
    gain_loss = _masked_mse(
        torch.sigmoid(outputs["candidate_new_area_gain_logit"]),
        batch["candidate_new_area_gain"],
        candidate_mask,
    )
    progress_loss = _masked_mse(
        torch.sigmoid(outputs["candidate_progress_logit"]),
        batch["candidate_progress"],
        candidate_mask,
    )
    oracle_score_loss = _masked_mse(
        _normalize_candidate_costs(_candidate_lower_score(outputs)),
        _normalize_candidate_costs(batch["candidate_oracle_cost"]),
        candidate_mask,
    )
    oracle_choice_loss = _oracle_choice_loss(_candidate_lower_score(outputs), batch["candidate_oracle_cost"], candidate_mask)
    candidate_future_risk_loss = _masked_balanced_bce_with_logits(
        outputs["candidate_future_risk_logits"],
        batch["candidate_horizon_future_risk"],
        candidate_mask.unsqueeze(-1).expand_as(batch["candidate_horizon_future_risk"]),
        focal_gamma=1.25,
    )
    candidate_loss = (
        collision_loss
        + 0.5 * future_collision_loss
        + 0.5 * unsafe_now_loss
        + 0.5 * candidate_future_risk_loss
        + 0.35 * oracle_score_loss
        + 3.0 * oracle_choice_loss
        + 0.5 * (unknown_loss + gain_loss + progress_loss)
    )
    total = future_bev_loss + 0.4 * future_risk_loss + 0.15 * uncertainty_loss + 0.75 * candidate_loss
    return {
        "loss": total,
        "future_bev_loss": future_bev_loss,
        "future_risk_loss": future_risk_loss,
        "uncertainty_loss": uncertainty_loss,
        "candidate_collision_loss": collision_loss,
        "candidate_future_collision_loss": future_collision_loss,
        "candidate_future_risk_loss": candidate_future_risk_loss,
        "candidate_oracle_score_loss": oracle_score_loss,
        "candidate_oracle_choice_loss": oracle_choice_loss,
        "candidate_unsafe_now_loss": unsafe_now_loss,
        "candidate_outcome_loss": candidate_loss,
    }


def _model_forward(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(
        batch["current_bev"],
        batch["features"],
        batch["feature_mask"],
        batch["sensor_mask"],
        memory_bev=batch.get("memory_bev"),
        bev_history=batch.get("bev_history"),
        uncertainty_map=batch.get("uncertainty_map"),
    )


@torch.no_grad()
def evaluate_future_bev_rollout_v1(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    dataset: FutureBEVRolloutDataset | None = None,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    bev_losses: list[float] = []
    free_iou_parts = _IouParts()
    occupied_iou_parts = _IouParts()
    unknown_iou_parts = _IouParts()
    risk_iou_parts = _IouParts()
    free_soft_iou_parts = _SoftIouParts()
    occupied_soft_iou_parts = _SoftIouParts()
    unknown_soft_iou_parts = _SoftIouParts()
    risk_soft_iou_parts = _SoftIouParts()
    copy_free_iou_parts = _IouParts()
    copy_occupied_iou_parts = _IouParts()
    copy_unknown_iou_parts = _IouParts()
    copy_risk_iou_parts = _IouParts()
    free_all_zero_parts = _IouParts()
    occupied_all_zero_parts = _IouParts()
    unknown_all_one_parts = _IouParts()
    future_channel_positive_sum = torch.zeros((3,), dtype=torch.float64)
    future_channel_total = 0.0
    risk_scores: list[float] = []
    risk_labels: list[float] = []
    newly_tp = newly_fp = newly_fn = 0.0
    unknown_reduction_mse: list[float] = []
    collision_scores: list[float] = []
    collision_labels: list[float] = []
    collision_correct = 0.0
    collision_total = 0.0
    collision_mse_sum = 0.0
    collision_zero_mse_sum = 0.0
    future_collision_mse_sum = 0.0
    future_risk_mse_sum = 0.0
    unsafe_now_mse_sum = 0.0
    unknown_exposure_mse_sum = 0.0
    unknown_exposure_one_mse_sum = 0.0
    valid_candidate_total = 0.0
    valid_candidate_horizon_total = 0.0
    future_collision_positive = 0.0
    unsafe_now_positive = 0.0
    ranking_values: list[float] = []
    risk_ranking_values: list[float] = []
    oracle_matches = 0.0
    oracle_total = 0.0
    unsafe_selected = 0.0
    stop_selected = 0.0
    selected_total = 0.0
    selected_ids: list[str] = []
    oracle_selected_ids: list[str] = []
    inference_examples = 0
    started = time.perf_counter()

    candidate_ids = list(getattr(model, "candidate_ids", dataset.candidate_ids if dataset is not None else []))
    for batch in loader:
        batch = batch_to_device(batch, device)
        outputs = _model_forward(model, batch)
        loss_parts = compute_future_bev_rollout_losses(outputs, batch)
        losses.append(float(loss_parts["loss"].detach().cpu()))
        bev_losses.append(float(loss_parts["future_bev_loss"].detach().cpu()))
        probs = torch.sigmoid(outputs["future_bev_logits"])
        risk_prob = torch.sigmoid(outputs["future_risk_logits"])
        mask = batch["future_valid_mask"] > 0.0
        _update_iou(free_iou_parts, probs[:, :, 0:1] > 0.5, batch["future_bev"][:, :, 0:1] > 0.5, mask)
        _update_iou(occupied_iou_parts, probs[:, :, 1:2] > 0.5, batch["future_bev"][:, :, 1:2] > 0.5, mask)
        _update_iou(unknown_iou_parts, probs[:, :, 2:3] > 0.5, batch["future_bev"][:, :, 2:3] > 0.5, mask)
        _update_iou(risk_iou_parts, risk_prob > 0.5, batch["future_risk"] > 0.5, mask)
        _update_soft_iou(free_soft_iou_parts, probs[:, :, 0:1], batch["future_bev"][:, :, 0:1], mask)
        _update_soft_iou(occupied_soft_iou_parts, probs[:, :, 1:2], batch["future_bev"][:, :, 1:2], mask)
        _update_soft_iou(unknown_soft_iou_parts, probs[:, :, 2:3], batch["future_bev"][:, :, 2:3], mask)
        _update_soft_iou(risk_soft_iou_parts, risk_prob, batch["future_risk"], mask)
        current_copy = batch["current_bev"].unsqueeze(1).expand(-1, probs.shape[1], -1, -1, -1)
        _update_iou(copy_free_iou_parts, current_copy[:, :, 0:1] > 0.5, batch["future_bev"][:, :, 0:1] > 0.5, mask)
        _update_iou(
            copy_occupied_iou_parts,
            current_copy[:, :, 1:2] > 0.5,
            batch["future_bev"][:, :, 1:2] > 0.5,
            mask,
        )
        _update_iou(
            copy_unknown_iou_parts,
            current_copy[:, :, 2:3] > 0.5,
            batch["future_bev"][:, :, 2:3] > 0.5,
            mask,
        )
        _update_iou(copy_risk_iou_parts, current_copy[:, :, 4:5] > 0.5, batch["future_risk"] > 0.5, mask)
        risk_scores.extend(risk_prob[mask].detach().cpu().numpy().astype(float).tolist())
        risk_labels.extend(batch["future_risk"][mask].detach().cpu().numpy().astype(float).tolist())
        _update_iou(free_all_zero_parts, torch.zeros_like(probs[:, :, 0:1], dtype=torch.bool), batch["future_bev"][:, :, 0:1] > 0.5, mask)
        _update_iou(
            occupied_all_zero_parts,
            torch.zeros_like(probs[:, :, 1:2], dtype=torch.bool),
            batch["future_bev"][:, :, 1:2] > 0.5,
            mask,
        )
        _update_iou(
            unknown_all_one_parts,
            torch.ones_like(probs[:, :, 2:3], dtype=torch.bool),
            batch["future_bev"][:, :, 2:3] > 0.5,
            mask,
        )
        channel_mask = mask.to(dtype=torch.float32).expand_as(batch["future_bev"])
        future_channel_positive_sum += (batch["future_bev"].detach().cpu().double() * channel_mask.detach().cpu().double()).sum(
            dim=(0, 1, 3, 4)
        )
        future_channel_total += float(channel_mask[:, :, 0:1].sum().detach().cpu())

        current_unknown = (batch["current_bev"][:, 2:3] > 0.5).unsqueeze(1)
        predicted_observed = ((probs[:, :, 0:1] > 0.5) | (probs[:, :, 1:2] > 0.5)) & mask & current_unknown
        target_new = (batch["future_derived"][:, :, 0:1] > 0.5) & mask
        newly_tp += float(torch.count_nonzero(predicted_observed & target_new).detach().cpu())
        newly_fp += float(torch.count_nonzero(predicted_observed & ~target_new).detach().cpu())
        newly_fn += float(torch.count_nonzero(~predicted_observed & target_new).detach().cpu())

        pred_reduction = current_unknown.to(probs.dtype) - probs[:, :, 2:3]
        target_reduction = current_unknown.to(probs.dtype) - batch["future_bev"][:, :, 2:3]
        reduction_error = ((pred_reduction - target_reduction) ** 2) * mask.to(probs.dtype)
        unknown_reduction_mse.append(float((reduction_error.sum() / mask.sum().clamp_min(1)).detach().cpu()))

        candidate_mask = batch["candidate_valid_mask"] > 0.0
        collision_prob = torch.sigmoid(outputs["candidate_collision_logit"])
        future_collision_prob = torch.sigmoid(outputs["candidate_future_collision_logit"])
        candidate_future_risk_prob = torch.sigmoid(outputs["candidate_future_risk_logits"])
        unsafe_now_prob = torch.sigmoid(outputs["candidate_unsafe_now_logit"])
        collision_pred = collision_prob >= 0.5
        collision_target = batch["candidate_collision"] >= 0.5
        if torch.count_nonzero(candidate_mask) > 0:
            candidate_horizon_mask = candidate_mask.unsqueeze(-1).expand_as(batch["candidate_horizon_future_risk"])
            collision_correct += float(torch.count_nonzero((collision_pred == collision_target) & candidate_mask).detach().cpu())
            collision_total += float(torch.count_nonzero(candidate_mask).detach().cpu())
            collision_scores.extend(collision_prob[candidate_mask].detach().cpu().numpy().astype(float).tolist())
            collision_labels.extend(batch["candidate_collision"][candidate_mask].detach().cpu().numpy().astype(float).tolist())
            valid_candidate_total += float(torch.count_nonzero(candidate_mask).detach().cpu())
            valid_candidate_horizon_total += float(torch.count_nonzero(candidate_horizon_mask).detach().cpu())
            collision_mse_sum += float((((collision_prob - batch["candidate_collision"]) ** 2)[candidate_mask]).sum().detach().cpu())
            collision_zero_mse_sum += float(((batch["candidate_collision"] ** 2)[candidate_mask]).sum().detach().cpu())
            future_collision_mse_sum += float(
                (((future_collision_prob - batch["candidate_future_collision"]) ** 2)[candidate_mask]).sum().detach().cpu()
            )
            future_risk_mse_sum += float(
                (((candidate_future_risk_prob - batch["candidate_horizon_future_risk"]) ** 2)[candidate_horizon_mask])
                .sum()
                .detach()
                .cpu()
            )
            unsafe_now_mse_sum += float(
                (((unsafe_now_prob - batch["candidate_unsafe_now"]) ** 2)[candidate_mask]).sum().detach().cpu()
            )
            future_collision_positive += float(
                torch.count_nonzero((batch["candidate_future_collision"] >= 0.05) & candidate_mask).detach().cpu()
            )
            unsafe_now_positive += float(
                torch.count_nonzero((batch["candidate_unsafe_now"] >= 0.05) & candidate_mask).detach().cpu()
            )
            unknown_pred = torch.sigmoid(outputs["candidate_unknown_exposure_logit"])
            unknown_exposure_mse_sum += float(
                (((unknown_pred - batch["candidate_unknown_exposure"]) ** 2)[candidate_mask]).sum().detach().cpu()
            )
            unknown_exposure_one_mse_sum += float(
                (((1.0 - batch["candidate_unknown_exposure"]) ** 2)[candidate_mask]).sum().detach().cpu()
            )
        ranking_values.extend(
            _pairwise_ranking_quality(
                torch.sigmoid(outputs["candidate_new_area_gain_logit"]),
                batch["candidate_new_area_gain"],
                batch["candidate_valid_mask"],
            )
        )
        lower_score = _candidate_lower_score(outputs, candidate_ids=candidate_ids)
        risk_ranking_values.extend(_pairwise_lower_ranking_quality(lower_score, batch["candidate_oracle_cost"], batch["candidate_valid_mask"]))
        selected = torch.argmin(lower_score, dim=1).detach().cpu().numpy().astype(int).tolist()
        oracle_cost = batch.get("candidate_oracle_cost")
        if oracle_cost is not None:
            masked_oracle = oracle_cost.clone()
            masked_oracle = masked_oracle.masked_fill(~candidate_mask, float("inf"))
            oracle_selected = torch.argmin(masked_oracle, dim=1).detach().cpu().numpy().astype(int).tolist()
        else:
            oracle_selected = []
        for index in selected:
            if 0 <= index < len(candidate_ids):
                selected_ids.append(candidate_ids[index])
                selected_total += 1.0
                if candidate_ids[index] == "stop":
                    stop_selected += 1.0
        for batch_index, index in enumerate(oracle_selected):
            if 0 <= index < len(candidate_ids) and bool(torch.any(candidate_mask[batch_index]).detach().cpu()):
                oracle_selected_ids.append(candidate_ids[index])
                oracle_total += 1.0
                if batch_index < len(selected) and int(selected[batch_index]) == int(index):
                    oracle_matches += 1.0
        for batch_index, index in enumerate(selected):
            if 0 <= index < len(candidate_ids) and candidate_ids[index] != "stop":
                if bool((batch["candidate_collision"][batch_index, index] >= 0.5).detach().cpu()):
                    unsafe_selected += 1.0
        inference_examples += int(batch["current_bev"].shape[0])

    elapsed = max(time.perf_counter() - started, 1.0e-9)
    precision = newly_tp / max(newly_tp + newly_fp, 1.0)
    recall = newly_tp / max(newly_tp + newly_fn, 1.0)
    collision_auroc = _binary_auroc(collision_scores, collision_labels)
    selected_distribution = _distribution(selected_ids)
    oracle_distribution = _distribution(oracle_selected_ids)
    collision_positive_rate = sum(1 for value in collision_labels if float(value) >= 0.05) / max(len(collision_labels), 1)
    collision_always_negative_accuracy = sum(1 for value in collision_labels if float(value) < 0.5) / max(
        len(collision_labels),
        1,
    )
    future_ratios = (future_channel_positive_sum / max(future_channel_total, 1.0)).numpy().astype(float).tolist()
    collision_mse = collision_mse_sum / max(valid_candidate_total, 1.0)
    collision_zero_mse = collision_zero_mse_sum / max(valid_candidate_total, 1.0)
    unknown_exposure_mse = unknown_exposure_mse_sum / max(valid_candidate_total, 1.0)
    unknown_exposure_one_mse = unknown_exposure_one_mse_sum / max(valid_candidate_total, 1.0)
    free_hard_iou = free_iou_parts.value()
    occupied_hard_iou = occupied_iou_parts.value()
    unknown_hard_iou = unknown_iou_parts.value()
    risk_hard_iou = risk_iou_parts.value()
    free_soft_iou = free_soft_iou_parts.value()
    occupied_soft_iou = occupied_soft_iou_parts.value()
    unknown_soft_iou = unknown_soft_iou_parts.value()
    risk_soft_iou = risk_soft_iou_parts.value()
    future_score = _mean(
        [
            free_hard_iou if free_hard_iou > 0.0 else free_soft_iou,
            occupied_hard_iou if occupied_hard_iou > 0.0 else occupied_soft_iou,
            unknown_hard_iou if unknown_hard_iou > 0.0 else unknown_soft_iou,
            risk_hard_iou if risk_hard_iou > 0.0 else risk_soft_iou,
        ]
    )
    copy_score = _mean(
        [
            copy_free_iou_parts.value(),
            copy_occupied_iou_parts.value(),
            copy_unknown_iou_parts.value(),
            copy_risk_iou_parts.value(),
        ]
    )
    return {
        "loss": _mean(losses),
        "future_bev_loss": _mean(bev_losses),
        "future_free_iou_or_proxy": free_hard_iou if free_hard_iou > 0.0 else free_soft_iou,
        "future_occupied_iou_or_proxy": occupied_hard_iou if occupied_hard_iou > 0.0 else occupied_soft_iou,
        "future_unknown_iou_or_proxy": unknown_hard_iou if unknown_hard_iou > 0.0 else unknown_soft_iou,
        "future_risk_iou_or_proxy": risk_hard_iou if risk_hard_iou > 0.0 else risk_soft_iou,
        "future_risk_auc_or_proxy": _binary_auroc(risk_scores, risk_labels) or (risk_hard_iou if risk_hard_iou > 0.0 else risk_soft_iou),
        "future_free_hard_iou": free_hard_iou,
        "future_occupied_hard_iou": occupied_hard_iou,
        "future_unknown_hard_iou": unknown_hard_iou,
        "future_risk_hard_iou": risk_hard_iou,
        "future_free_soft_iou_proxy": free_soft_iou,
        "future_occupied_soft_iou_proxy": occupied_soft_iou,
        "future_unknown_soft_iou_proxy": unknown_soft_iou,
        "future_risk_soft_iou_proxy": risk_soft_iou,
        "copy_forward_free_iou_or_proxy": copy_free_iou_parts.value(),
        "copy_forward_occupied_iou_or_proxy": copy_occupied_iou_parts.value(),
        "copy_forward_unknown_iou_or_proxy": copy_unknown_iou_parts.value(),
        "copy_forward_risk_iou_or_proxy": copy_risk_iou_parts.value(),
        "improvement_vs_copy_forward": float(future_score - copy_score),
        "future_free_all_zero_iou_baseline": free_all_zero_parts.value(),
        "future_occupied_all_zero_iou_baseline": occupied_all_zero_parts.value(),
        "future_unknown_all_one_iou_baseline": unknown_all_one_parts.value(),
        "future_free_target_ratio": float(future_ratios[0]) if len(future_ratios) > 0 else 0.0,
        "future_occupied_target_ratio": float(future_ratios[1]) if len(future_ratios) > 1 else 0.0,
        "future_unknown_target_ratio": float(future_ratios[2]) if len(future_ratios) > 2 else 0.0,
        "newly_observed_precision": float(precision),
        "newly_observed_recall": float(recall),
        "unknown_reduction_prediction_quality": float(1.0 / (1.0 + _mean(unknown_reduction_mse))),
        "candidate_collision_auroc": collision_auroc,
        "candidate_collision_accuracy": float(collision_correct / max(collision_total, 1.0)),
        "candidate_collision_positive_rate": float(collision_positive_rate),
        "candidate_future_collision_positive_rate": float(future_collision_positive / max(valid_candidate_total, 1.0)),
        "candidate_unsafe_now_positive_rate": float(unsafe_now_positive / max(valid_candidate_total, 1.0)),
        "candidate_collision_mse": float(collision_mse),
        "candidate_collision_always_zero_mse_baseline": float(collision_zero_mse),
        "candidate_collision_always_negative_accuracy_baseline": float(collision_always_negative_accuracy),
        "candidate_future_collision_mse": float(future_collision_mse_sum / max(valid_candidate_total, 1.0)),
        "candidate_future_risk_mse": float(future_risk_mse_sum / max(valid_candidate_horizon_total, 1.0)),
        "candidate_unsafe_now_mse": float(unsafe_now_mse_sum / max(valid_candidate_total, 1.0)),
        "candidate_unknown_exposure_mse": float(unknown_exposure_mse),
        "candidate_unknown_exposure_always_one_mse_baseline": float(unknown_exposure_one_mse),
        "candidate_new_area_gain_ranking_quality": _mean(ranking_values),
        "candidate_risk_ranking_accuracy": _mean(risk_ranking_values),
        "candidate_oracle_match_fraction": float(oracle_matches / max(oracle_total, 1.0)),
        "candidate_oracle_selected_distribution": oracle_distribution,
        "unsafe_selected_rate": float(unsafe_selected / max(selected_total, 1.0)),
        "unsafe_candidate_rejection_rate": float(1.0 - unsafe_selected / max(selected_total - stop_selected, 1.0)),
        "stop_selected_fraction": float(stop_selected / max(selected_total, 1.0)),
        "beats_collision_always_negative_baseline": bool(
            collision_mse < collision_zero_mse or (collision_correct / max(collision_total, 1.0)) > collision_always_negative_accuracy
        ),
        "beats_unknown_always_one_baseline": bool(unknown_exposure_mse < unknown_exposure_one_mse),
        "beats_future_all_unknown_baseline": bool(
            free_iou_parts.value() > free_all_zero_parts.value()
            or occupied_iou_parts.value() > occupied_all_zero_parts.value()
            or unknown_iou_parts.value() > unknown_all_one_parts.value()
        ),
        "action_entropy": _entropy(selected_distribution),
        "dominant_action_fraction": _dominant_fraction(selected_distribution),
        "selected_distribution": selected_distribution,
        "inference_fps": float(inference_examples / elapsed),
        "route_held_out_generalization": _route_held_out_record(dataset),
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
    }


class _IouParts:
    def __init__(self) -> None:
        self.intersection = 0.0
        self.union = 0.0

    def value(self) -> float:
        if self.union <= 0.0:
            return 0.0
        return float(self.intersection / self.union)


class _SoftIouParts:
    def __init__(self) -> None:
        self.intersection = 0.0
        self.union = 0.0

    def value(self) -> float:
        if self.union <= 0.0:
            return 0.0
        return float(self.intersection / self.union)


def _update_iou(parts: _IouParts, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
    pred = pred & mask
    target = target & mask
    parts.intersection += float(torch.count_nonzero(pred & target).detach().cpu())
    parts.union += float(torch.count_nonzero(pred | target).detach().cpu())


def _update_soft_iou(parts: _SoftIouParts, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
    mask_f = mask.to(dtype=pred.dtype)
    target_f = target.to(dtype=pred.dtype).clamp(0.0, 1.0)
    pred_f = pred.to(dtype=target_f.dtype).clamp(0.0, 1.0)
    intersection = torch.minimum(pred_f, target_f) * mask_f
    union = torch.maximum(pred_f, target_f) * mask_f
    parts.intersection += float(intersection.sum().detach().cpu())
    parts.union += float(union.sum().detach().cpu())


def _masked_balanced_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    target = target.to(dtype=logits.dtype)
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    while mask.ndim < logits.ndim:
        mask = mask.unsqueeze(1)
    mask = torch.broadcast_to(mask, logits.shape)
    positive = ((target >= 0.05).to(logits.dtype) * mask).sum()
    negative = ((target < 0.05).to(logits.dtype) * mask).sum()
    total = (positive + negative).clamp_min(1.0)
    positive_weight = (0.5 * total / positive.clamp_min(1.0)).clamp(max=20.0)
    negative_weight = (0.5 * total / negative.clamp_min(1.0)).clamp(max=20.0)
    weights = torch.where(target >= 0.05, positive_weight, negative_weight)
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if focal_gamma > 0.0:
        prob = torch.sigmoid(logits)
        pt = prob * target + (1.0 - prob) * (1.0 - target)
        raw = raw * torch.pow(1.0 - pt.clamp(0.0, 1.0), focal_gamma)
    return (raw * weights * mask).sum() / mask.sum().clamp_min(1.0)


def _candidate_lower_score(
    outputs: dict[str, torch.Tensor],
    *,
    candidate_ids: list[str] | None = None,
) -> torch.Tensor:
    candidate_future_risk = torch.max(torch.sigmoid(outputs["candidate_future_risk_logits"]), dim=2).values
    combined_collision = torch.maximum(
        torch.sigmoid(outputs["candidate_collision_logit"]),
        torch.maximum(torch.sigmoid(outputs["candidate_future_collision_logit"]), candidate_future_risk),
    )
    score = (
        8.0 * combined_collision
        + 2.0 * torch.sigmoid(outputs["candidate_unsafe_now_logit"])
        + 1.25 * torch.sigmoid(outputs["candidate_unknown_exposure_logit"])
        - 2.4 * torch.sigmoid(outputs["candidate_new_area_gain_logit"])
        - 2.0 * torch.sigmoid(outputs["candidate_progress_logit"])
    )
    if candidate_ids:
        stop_penalty_values = [0.8 if candidate_id == "stop" else 0.0 for candidate_id in candidate_ids]
    else:
        stop_penalty_values = [0.8] + [0.0 for _index in range(max(int(score.shape[1]) - 1, 0))]
    stop_penalty = torch.tensor(stop_penalty_values, dtype=score.dtype, device=score.device)
    return score + stop_penalty.view(1, -1)


def _normalize_candidate_costs(values: torch.Tensor) -> torch.Tensor:
    lower = torch.min(values, dim=1, keepdim=True).values
    upper = torch.max(values, dim=1, keepdim=True).values
    span = (upper - lower).clamp_min(1.0e-6)
    return (values - lower) / span


def _oracle_choice_loss(predicted_cost: torch.Tensor, oracle_cost: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_oracle = oracle_cost.masked_fill(mask <= 0.0, float("inf"))
    target = torch.argmin(masked_oracle, dim=1)
    masked_pred = predicted_cost.masked_fill(mask <= 0.0, 1.0e6)
    counts = torch.bincount(target, minlength=predicted_cost.shape[1]).to(device=predicted_cost.device, dtype=predicted_cost.dtype)
    weights = torch.where(counts > 0.0, 1.0 / counts.clamp_min(1.0), torch.zeros_like(counts))
    if torch.count_nonzero(weights) > 0:
        weights = weights * (torch.count_nonzero(weights).to(dtype=weights.dtype) / weights.sum().clamp_min(1.0e-6))
    return F.cross_entropy(-masked_pred, target, weight=weights)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def _pairwise_ranking_quality(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> list[float]:
    values: list[float] = []
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy() > 0.0
    for batch_index in range(pred_np.shape[0]):
        correct = 0
        total = 0
        valid = np.where(mask_np[batch_index])[0]
        for left_pos, left in enumerate(valid):
            for right in valid[left_pos + 1 :]:
                target_diff = float(target_np[batch_index, left] - target_np[batch_index, right])
                if abs(target_diff) < 1.0e-6:
                    continue
                pred_diff = float(pred_np[batch_index, left] - pred_np[batch_index, right])
                correct += int((target_diff > 0.0 and pred_diff > 0.0) or (target_diff < 0.0 and pred_diff < 0.0))
                total += 1
        if total > 0:
            values.append(float(correct / total))
    return values


def _pairwise_lower_ranking_quality(pred_cost: torch.Tensor, target_cost: torch.Tensor, mask: torch.Tensor) -> list[float]:
    values: list[float] = []
    pred_np = pred_cost.detach().cpu().numpy()
    target_np = target_cost.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy() > 0.0
    for batch_index in range(pred_np.shape[0]):
        correct = 0
        total = 0
        valid = np.where(mask_np[batch_index])[0]
        for left_pos, left in enumerate(valid):
            for right in valid[left_pos + 1 :]:
                target_diff = float(target_np[batch_index, left] - target_np[batch_index, right])
                if abs(target_diff) < 1.0e-6:
                    continue
                pred_diff = float(pred_np[batch_index, left] - pred_np[batch_index, right])
                correct += int((target_diff < 0.0 and pred_diff < 0.0) or (target_diff > 0.0 and pred_diff > 0.0))
                total += 1
        if total > 0:
            values.append(float(correct / total))
    return values


def _binary_auroc(scores: list[float], labels: list[float]) -> float | None:
    pairs = [(float(score), 1 if float(label) >= 0.5 else 0) for score, label in zip(scores, labels)]
    pos = [score for score, label in pairs if label == 1]
    neg = [score for score, label in pairs if label == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0.0
    for pos_score in pos:
        for neg_score in neg:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
            total += 1.0
    return float(wins / max(total, 1.0))


def _route_held_out_record(dataset: FutureBEVRolloutDataset | None) -> dict[str, Any]:
    if dataset is None:
        return {"available": False}
    split_units = sorted({str(record.get("split_unit_id", "")) for record in dataset.records})
    splits = sorted({str(record.get("split", "")) for record in dataset.records})
    return {
        "available": True,
        "basis": "split_unit_id",
        "split_units": split_units,
        "split_unit_count": len(split_units),
        "splits": splits,
        "held_out_split": any(split != "train" for split in splits),
    }


def _distribution(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _entropy(distribution: dict[str, int]) -> float:
    total = sum(distribution.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in distribution.values():
        p = count / total
        entropy -= p * math.log(p)
    return float(entropy)


def _dominant_fraction(distribution: dict[str, int]) -> float:
    total = sum(distribution.values())
    if total <= 0:
        return 0.0
    return float(max(distribution.values()) / total)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _file_hash(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train replay-only Future BEV Rollout v1.")
    parser.add_argument("--rollout-pack", required=True, help="Input FutureBEVRolloutPack directory.")
    parser.add_argument("--out", required=True, help="Output training directory.")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args(argv)
    metrics = train_future_bev_rollout_v1(
        rollout_pack=args.rollout_pack,
        out_dir=args.out,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_channels=args.hidden_channels,
        tiny_overfit=args.tiny_overfit,
        device_name=args.device,
        seed=args.seed,
        command=" ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
    )
    print(json.dumps({"metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
