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
        outputs = model(batch["current_bev"], batch["features"], batch["feature_mask"], batch["sensor_mask"])
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
    if max_steps > 0 and loss_reduction_ratio <= 0.0:
        failure_flags.append("train_loss_not_reduced")
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
        "newly_observed_precision": float(final_val["newly_observed_precision"]),
        "newly_observed_recall": float(final_val["newly_observed_recall"]),
        "unknown_reduction_prediction_quality": float(final_val["unknown_reduction_prediction_quality"]),
        "candidate_collision_auroc": final_val["candidate_collision_auroc"],
        "candidate_collision_accuracy": float(final_val["candidate_collision_accuracy"]),
        "candidate_new_area_gain_ranking_quality": float(final_val["candidate_new_area_gain_ranking_quality"]),
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
    raw_bce = F.binary_cross_entropy_with_logits(outputs["future_bev_logits"], future_labels, reduction="none")
    future_bev_loss = (raw_bce * future_mask).sum() / future_mask.sum().clamp_min(1.0)
    unknown_target = future_labels[:, :, 2:3]
    uncertainty_bce = F.binary_cross_entropy_with_logits(outputs["uncertainty_logits"], unknown_target, reduction="none")
    uncertainty_loss = (uncertainty_bce * future_mask).sum() / future_mask.sum().clamp_min(1.0)
    candidate_mask = batch["candidate_valid_mask"]
    collision_loss_raw = F.binary_cross_entropy_with_logits(
        outputs["candidate_collision_logit"],
        batch["candidate_collision"],
        reduction="none",
    )
    collision_loss = (collision_loss_raw * candidate_mask).sum() / candidate_mask.sum().clamp_min(1.0)
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
    candidate_loss = collision_loss + 0.5 * (unknown_loss + gain_loss + progress_loss)
    total = future_bev_loss + 0.15 * uncertainty_loss + 0.75 * candidate_loss
    return {
        "loss": total,
        "future_bev_loss": future_bev_loss,
        "uncertainty_loss": uncertainty_loss,
        "candidate_collision_loss": collision_loss,
        "candidate_outcome_loss": candidate_loss,
    }


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
    newly_tp = newly_fp = newly_fn = 0.0
    unknown_reduction_mse: list[float] = []
    collision_scores: list[float] = []
    collision_labels: list[float] = []
    collision_correct = 0.0
    collision_total = 0.0
    ranking_values: list[float] = []
    selected_ids: list[str] = []
    inference_examples = 0
    started = time.perf_counter()

    candidate_ids = list(getattr(model, "candidate_ids", dataset.candidate_ids if dataset is not None else []))
    for batch in loader:
        batch = batch_to_device(batch, device)
        outputs = model(batch["current_bev"], batch["features"], batch["feature_mask"], batch["sensor_mask"])
        loss_parts = compute_future_bev_rollout_losses(outputs, batch)
        losses.append(float(loss_parts["loss"].detach().cpu()))
        bev_losses.append(float(loss_parts["future_bev_loss"].detach().cpu()))
        probs = torch.sigmoid(outputs["future_bev_logits"])
        mask = batch["future_valid_mask"] > 0.0
        _update_iou(free_iou_parts, probs[:, :, 0:1] > 0.5, batch["future_bev"][:, :, 0:1] > 0.5, mask)
        _update_iou(occupied_iou_parts, probs[:, :, 1:2] > 0.5, batch["future_bev"][:, :, 1:2] > 0.5, mask)

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
        collision_pred = collision_prob >= 0.5
        collision_target = batch["candidate_collision"] >= 0.5
        if torch.count_nonzero(candidate_mask) > 0:
            collision_correct += float(torch.count_nonzero((collision_pred == collision_target) & candidate_mask).detach().cpu())
            collision_total += float(torch.count_nonzero(candidate_mask).detach().cpu())
            collision_scores.extend(collision_prob[candidate_mask].detach().cpu().numpy().astype(float).tolist())
            collision_labels.extend(batch["candidate_collision"][candidate_mask].detach().cpu().numpy().astype(float).tolist())
        ranking_values.extend(
            _pairwise_ranking_quality(
                torch.sigmoid(outputs["candidate_new_area_gain_logit"]),
                batch["candidate_new_area_gain"],
                batch["candidate_valid_mask"],
            )
        )
        lower_score = (
            8.0 * collision_prob
            + 1.25 * torch.sigmoid(outputs["candidate_unknown_exposure_logit"])
            - 1.6 * torch.sigmoid(outputs["candidate_new_area_gain_logit"])
            - 0.8 * torch.sigmoid(outputs["candidate_progress_logit"])
        )
        selected = torch.argmin(lower_score, dim=1).detach().cpu().numpy().astype(int).tolist()
        for index in selected:
            if 0 <= index < len(candidate_ids):
                selected_ids.append(candidate_ids[index])
        inference_examples += int(batch["current_bev"].shape[0])

    elapsed = max(time.perf_counter() - started, 1.0e-9)
    precision = newly_tp / max(newly_tp + newly_fp, 1.0)
    recall = newly_tp / max(newly_tp + newly_fn, 1.0)
    collision_auroc = _binary_auroc(collision_scores, collision_labels)
    selected_distribution = _distribution(selected_ids)
    return {
        "loss": _mean(losses),
        "future_bev_loss": _mean(bev_losses),
        "future_free_iou_or_proxy": free_iou_parts.value(),
        "future_occupied_iou_or_proxy": occupied_iou_parts.value(),
        "newly_observed_precision": float(precision),
        "newly_observed_recall": float(recall),
        "unknown_reduction_prediction_quality": float(1.0 / (1.0 + _mean(unknown_reduction_mse))),
        "candidate_collision_auroc": collision_auroc,
        "candidate_collision_accuracy": float(collision_correct / max(collision_total, 1.0)),
        "candidate_new_area_gain_ranking_quality": _mean(ranking_values),
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


def _update_iou(parts: _IouParts, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
    pred = pred & mask
    target = target & mask
    parts.intersection += float(torch.count_nonzero(pred & target).detach().cpu())
    parts.union += float(torch.count_nonzero(pred | target).detach().cpu())


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
