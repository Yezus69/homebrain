from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from homebrain.brain.counterfactual_dynamic_bev_world_model_v0 import load_checkpoint
from homebrain.data.spatial_dataset import read_json, write_json
from homebrain.train.train_counterfactual_dynamic_bev_world_model_v0 import DynamicBEVWorldDataset, model_forward

FORBIDDEN_RUNTIME_FIELDS: tuple[str, ...] = (
    "target_bev",
    "future_bev",
    "future_labels",
    "teacher_masks",
    "route_ground_truth",
    "candidate_oracle_cost",
    "ground_truth_global_trajectory",
    "future_frames",
)


def eval_counterfactual_dynamic_bev_world_model_v0(
    *,
    checkpoint: str | Path,
    pack: str | Path,
    split: str = "val",
    out_path: str | Path,
    device_name: str | None = None,
) -> Path:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model.to(device)
    dataset = DynamicBEVWorldDataset(pack, split=split)
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=False)
    metrics = _evaluate_model(model, loader, device=device)
    manifest = read_json(Path(pack) / "manifest.json")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    train_routes = set(str(value) for value in manifest.get("train_route_ids", []))
    val_routes = set(str(value) for value in manifest.get("val_route_ids", []))
    runtime_leakage = _runtime_field_leakage_passed(metadata)
    failed = _failed_checks(
        manifest=manifest,
        metadata=metadata,
        metrics=metrics,
        train_routes=train_routes,
        val_routes=val_routes,
        runtime_leakage=runtime_leakage,
    )
    report = {
        "schema_version": "homebrain.eval_counterfactual_dynamic_bev_world_model_v0.v0",
        "checkpoint": Path(checkpoint).as_posix(),
        "pack": Path(pack).as_posix(),
        "split": split,
        "baselines": [
            "static_copy_baseline",
            "previous_frame_copy_baseline",
            "constant_flow_baseline",
            "center_prior_baseline",
            "transparent_teacher_current_bev_scorer",
        ],
        **metrics,
        "route_heldout_count": len(val_routes),
        "train_route_ids": sorted(train_routes),
        "val_route_ids": sorted(val_routes),
        "runtime_field_leakage_passed": runtime_leakage,
        "no_future_labels_used_at_runtime": bool(metadata.get("no_future_labels_used_at_runtime", False)),
        "no_teacher_fields_at_runtime": bool(metadata.get("no_teacher_fields_at_runtime", False)),
        "accepted_counterfactual_dynamic_bev_world_model_v0": not failed,
        "acceptance": {"passed": not failed, "failed_checks": failed},
        "replay_only": bool(metadata.get("replay_only", False)),
        "not_executed": bool(metadata.get("not_executed", False)),
        "control_safe": bool(metadata.get("control_safe", True)),
        "raw_pwm_emitted": bool(metadata.get("raw_pwm_emitted", True)),
        "hardware_validated": bool(metadata.get("hardware_validated", True)),
    }
    write_json(out_path, report, pretty=True)
    return Path(out_path)


@torch.no_grad()
def _evaluate_model(model: torch.nn.Module, loader: DataLoader, *, device: torch.device) -> dict[str, Any]:
    future_occ_scores: list[list[float]] = []
    future_occ_labels: list[list[float]] = []
    future_dyn_scores: list[list[float]] = []
    future_dyn_labels: list[list[float]] = []
    static_occ_scores: list[list[float]] = []
    prev_occ_scores: list[list[float]] = []
    prev_dyn_scores: list[list[float]] = []
    iou_parts: list[_IouParts] | None = None
    dyn_f1_parts: list[_F1Parts] | None = None
    flow_epe: list[list[float]] = []
    brier_values: list[float] = []
    candidate_scores: list[float] = []
    candidate_labels: list[float] = []
    rank_values: list[float] = []
    top1_high_risk = 0.0
    top1_total = 0.0
    top3_contains_safe = 0.0
    static_top1_high_risk = 0.0
    center_scores_all: list[float] = []
    center_labels_all: list[float] = []
    for raw_batch in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw_batch.items()}
        outputs = model_forward(model, batch)
        occ_prob = torch.sigmoid(outputs["future_occupied_logits"]).detach().cpu()
        dyn_prob = torch.sigmoid(outputs["future_dynamic_risk_logits"]).detach().cpu()
        uncertainty = torch.sigmoid(outputs["future_uncertainty_logits"]).detach().cpu()
        flow = outputs["future_flow_xy"].detach().cpu()
        target_occ = batch["future_occupied"].detach().cpu()
        target_dyn = batch["future_dynamic_risk"].detach().cpu()
        target_flow = batch["future_flow_xy"].detach().cpu()
        valid = batch["future_valid_mask"].detach().cpu()
        current = batch["bev_history"][:, -1].detach().cpu()
        previous = batch["bev_history"][:, -2].detach().cpu() if batch["bev_history"].shape[1] > 1 else current
        if iou_parts is None:
            horizon_count = int(occ_prob.shape[1])
            iou_parts = [_IouParts() for _ in range(horizon_count)]
            dyn_f1_parts = [_F1Parts() for _ in range(horizon_count)]
            future_occ_scores = [[] for _ in range(horizon_count)]
            future_occ_labels = [[] for _ in range(horizon_count)]
            future_dyn_scores = [[] for _ in range(horizon_count)]
            future_dyn_labels = [[] for _ in range(horizon_count)]
            static_occ_scores = [[] for _ in range(horizon_count)]
            prev_occ_scores = [[] for _ in range(horizon_count)]
            prev_dyn_scores = [[] for _ in range(horizon_count)]
            flow_epe = [[] for _ in range(horizon_count)]
        for horizon in range(occ_prob.shape[1]):
            mask = valid[:, horizon] > 0.0
            _extend_binary(future_occ_scores[horizon], future_occ_labels[horizon], occ_prob[:, horizon], target_occ[:, horizon], mask)
            _extend_binary(future_dyn_scores[horizon], future_dyn_labels[horizon], dyn_prob[:, horizon], target_dyn[:, horizon], mask)
            _extend_binary(static_occ_scores[horizon], [], current[:, 1], target_occ[:, horizon], mask, include_labels=False)
            _extend_binary(prev_occ_scores[horizon], [], previous[:, 1], target_occ[:, horizon], mask, include_labels=False)
            _extend_binary(prev_dyn_scores[horizon], [], previous[:, 5], target_dyn[:, horizon], mask, include_labels=False)
            iou_parts[horizon].update(occ_prob[:, horizon] > 0.5, target_occ[:, horizon] > 0.5, mask)
            dyn_f1_parts[horizon].update(dyn_prob[:, horizon] > 0.5, target_dyn[:, horizon] > 0.5, mask)
            dyn_mask = mask.unsqueeze(1) & (target_dyn[:, horizon : horizon + 1] > 0.05)
            if bool(torch.any(dyn_mask)):
                epe = torch.linalg.norm(flow[:, horizon] - target_flow[:, horizon], dim=1)
                flow_epe[horizon].extend(epe[dyn_mask[:, 0]].numpy().astype(float).tolist())
            err = ((occ_prob[:, horizon] > 0.5).to(torch.float32) - target_occ[:, horizon]).abs()
            brier_values.extend(((uncertainty[:, horizon] - err) ** 2)[mask].numpy().astype(float).tolist())
        risk_prob = torch.sigmoid(outputs["candidate_risk_logits"]).detach().cpu()
        candidate_score = outputs["candidate_score"].detach().cpu()
        target_risk = batch["candidate_total_teacher_risk"].detach().cpu()
        target_safe = batch["candidate_safe_label"].detach().cpu()
        candidate_scores.extend(risk_prob.reshape(-1).numpy().astype(float).tolist())
        candidate_labels.extend((target_risk.reshape(-1).numpy() >= 0.5).astype(float).tolist())
        rank_values.extend(_spearman_by_row(candidate_score.numpy(), target_risk.numpy()))
        selected = torch.argmin(candidate_score, dim=1)
        static_score = _transparent_current_score(batch["bev_history"].detach().cpu(), batch["candidate_trajectories"].detach().cpu())
        static_selected = torch.argmin(static_score, dim=1)
        center_score = _center_prior_score(batch["candidate_trajectories"].detach().cpu())
        center_scores_all.extend(center_score.reshape(-1).numpy().astype(float).tolist())
        center_labels_all.extend((target_risk.reshape(-1).numpy() >= 0.5).astype(float).tolist())
        for row in range(target_risk.shape[0]):
            top1_total += 1.0
            if float(target_risk[row, selected[row]]) >= 0.5:
                top1_high_risk += 1.0
            if float(target_risk[row, static_selected[row]]) >= 0.5:
                static_top1_high_risk += 1.0
            top3 = torch.argsort(candidate_score[row])[: min(3, candidate_score.shape[1])]
            if bool(torch.any(target_safe[row, top3] >= 0.5)):
                top3_contains_safe += 1.0
    assert iou_parts is not None and dyn_f1_parts is not None
    occ_ap = [_average_precision(scores, labels) for scores, labels in zip(future_occ_scores, future_occ_labels)]
    dyn_ap = [_average_precision(scores, labels) for scores, labels in zip(future_dyn_scores, future_dyn_labels)]
    static_ap = [_average_precision(scores, labels) for scores, labels in zip(static_occ_scores, future_occ_labels)]
    prev_occ_ap = [_average_precision(scores, labels) for scores, labels in zip(prev_occ_scores, future_occ_labels)]
    prev_dyn_ap = [_average_precision(scores, labels) for scores, labels in zip(prev_dyn_scores, future_dyn_labels)]
    center_auroc = _binary_auroc(center_scores_all, center_labels_all)
    return {
        "future_occupied_auprc_by_horizon": occ_ap,
        "future_occupied_iou_by_horizon": [part.value() for part in iou_parts],
        "future_dynamic_risk_auprc_by_horizon": dyn_ap,
        "future_dynamic_risk_f1_by_horizon": [part.value() for part in dyn_f1_parts],
        "future_flow_epe_by_horizon": [float(np.mean(values)) if values else 0.0 for values in flow_epe],
        "uncertainty_brier_or_ece_proxy": float(np.mean(brier_values)) if brier_values else 0.0,
        "candidate_risk_auroc": _binary_auroc(candidate_scores, candidate_labels),
        "candidate_risk_auprc": _average_precision(candidate_scores, candidate_labels),
        "candidate_rank_spearman": float(np.mean(rank_values)) if rank_values else 0.0,
        "candidate_top1_high_risk_rate": float(top1_high_risk / max(top1_total, 1.0)),
        "candidate_top3_contains_safe_rate": float(top3_contains_safe / max(top1_total, 1.0)),
        "static_copy_baseline_future_occupied_auprc_by_horizon": static_ap,
        "previous_frame_copy_baseline_future_occupied_auprc_by_horizon": prev_occ_ap,
        "previous_frame_copy_baseline_future_dynamic_risk_auprc_by_horizon": prev_dyn_ap,
        "constant_flow_baseline_future_flow_epe_by_horizon": [0.0 for _ in flow_epe],
        "center_prior_baseline_candidate_risk_auroc": center_auroc,
        "transparent_teacher_current_bev_scorer_top1_high_risk_rate": float(static_top1_high_risk / max(top1_total, 1.0)),
        "improvement_vs_static_copy": float(np.mean(occ_ap) - np.mean(static_ap)),
        "improvement_vs_previous_frame_copy": float(np.mean(dyn_ap) - np.mean(prev_dyn_ap)),
        "improvement_vs_center_prior": float((_binary_auroc(candidate_scores, candidate_labels) or 0.0) - (center_auroc or 0.0)),
    }


def _failed_checks(
    *,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    train_routes: set[str],
    val_routes: set[str],
    runtime_leakage: bool,
) -> list[str]:
    checks = {
        "synthetic_or_fixture_false": manifest.get("synthetic_or_fixture") is False,
        "real_source_route_count_gte_3": int(manifest.get("real_source_route_count", 0)) >= 3,
        "real_source_frame_count_gte_1000": int(manifest.get("real_source_frame_count", 0)) >= 1000,
        "route_heldout_split_true": bool(train_routes) and bool(val_routes),
        "no_train_route_in_val": not (train_routes & val_routes),
        "val_dynamic_positive_frame_count_gt_0": int(manifest.get("dynamic_positive_frame_count", 0)) > 0,
        "future_dynamic_risk_auprc_beats_previous_frame_copy_by_0_05": metrics["improvement_vs_previous_frame_copy"] >= 0.05,
        "future_occupied_auprc_beats_static_copy_by_0_05": metrics["improvement_vs_static_copy"] >= 0.05,
        "candidate_risk_auroc_gte_0_70": (metrics["candidate_risk_auroc"] or 0.0) >= 0.70,
        "candidate_top1_high_risk_rate_lower_than_static_copy_scorer": _top1_high_risk_rate_passes(
            model_rate=float(metrics["candidate_top1_high_risk_rate"]),
            baseline_rate=float(metrics["transparent_teacher_current_bev_scorer_top1_high_risk_rate"]),
        ),
        "runtime_field_leakage_passed": runtime_leakage,
        "no_future_labels_used_at_runtime": metadata.get("no_future_labels_used_at_runtime") is True,
        "no_teacher_fields_at_runtime": metadata.get("no_teacher_fields_at_runtime") is True,
        "replay_only": metadata.get("replay_only") is True,
        "not_executed": metadata.get("not_executed") is True,
        "control_safe_false": metadata.get("control_safe") is False,
        "raw_pwm_emitted_false": metadata.get("raw_pwm_emitted") is False,
        "hardware_validated_false": metadata.get("hardware_validated") is False,
    }
    return [name for name, passed in checks.items() if not passed]


def _top1_high_risk_rate_passes(*, model_rate: float, baseline_rate: float) -> bool:
    if baseline_rate <= 0.0:
        return model_rate <= 0.0
    return model_rate < baseline_rate


def _runtime_field_leakage_passed(metadata: dict[str, Any]) -> bool:
    forbidden = set(FORBIDDEN_RUNTIME_FIELDS)
    runtime_inputs = set(str(value) for value in metadata.get("runtime_inputs", []))
    return forbidden.isdisjoint(runtime_inputs) and metadata.get("no_future_labels_used_at_runtime") is True


class _IouParts:
    def __init__(self) -> None:
        self.intersection = 0.0
        self.union = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        pred = pred & mask
        target = target & mask
        self.intersection += float(torch.count_nonzero(pred & target))
        self.union += float(torch.count_nonzero(pred | target))

    def value(self) -> float:
        return float(self.intersection / self.union) if self.union > 0.0 else 0.0


class _F1Parts:
    def __init__(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        pred = pred & mask
        target = target & mask
        self.tp += float(torch.count_nonzero(pred & target))
        self.fp += float(torch.count_nonzero(pred & ~target & mask))
        self.fn += float(torch.count_nonzero(~pred & target & mask))

    def value(self) -> float:
        return float(2.0 * self.tp / max(2.0 * self.tp + self.fp + self.fn, 1.0))


def _extend_binary(
    scores_out: list[float],
    labels_out: list[float],
    scores: torch.Tensor,
    label_tensor: torch.Tensor,
    mask: torch.Tensor,
    *,
    include_labels: bool = True,
) -> None:
    scores_out.extend(scores[mask].numpy().astype(float).tolist())
    if include_labels:
        labels_out.extend((label_tensor[mask].numpy() >= 0.5).astype(float).tolist())


def _average_precision(scores: list[float], labels: list[float]) -> float:
    if not scores or not labels or sum(labels) <= 0.0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    y = np.asarray(labels, dtype=np.float64)[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1, dtype=np.float64)
    return float(np.sum(precision * y) / max(np.sum(y), 1.0))


def _binary_auroc(scores: list[float], labels: list[float]) -> float | None:
    pairs = [(float(score), int(float(label) >= 0.5)) for score, label in zip(scores, labels)]
    pos = [score for score, label in pairs if label == 1]
    neg = [score for score, label in pairs if label == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0.0
    for pos_score in pos:
        for neg_score in neg:
            wins += 1.0 if pos_score > neg_score else 0.5 if pos_score == neg_score else 0.0
            total += 1.0
    return float(wins / max(total, 1.0))


def _spearman_by_row(pred: np.ndarray, target: np.ndarray) -> list[float]:
    values: list[float] = []
    for row in range(pred.shape[0]):
        pr = _rank(pred[row])
        tr = _rank(target[row])
        if np.std(pr) <= 1.0e-6 or np.std(tr) <= 1.0e-6:
            continue
        values.append(float(np.corrcoef(pr, tr)[0, 1]))
    return values


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    rank = np.zeros_like(values, dtype=np.float64)
    rank[order] = np.arange(values.size, dtype=np.float64)
    return rank


def _transparent_current_score(bev_history: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
    current = bev_history[:, -1]
    risk = torch.maximum(current[:, 1], torch.maximum(current[:, 4], current[:, 5]))
    unknown = current[:, 2]
    return _trajectory_pool(risk, trajectories) * 8.0 + _trajectory_pool(unknown, trajectories) * 1.25


def _center_prior_score(trajectories: torch.Tensor) -> torch.Tensor:
    y_abs = torch.abs(trajectories[..., 1]).mean(dim=2)
    forward = trajectories[..., 0].amax(dim=2)
    return y_abs - 0.05 * forward


def _trajectory_pool(grid: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
    batch, candidate_count, point_count, _dims = trajectories.shape
    height, width = grid.shape[-2:]
    rows = height - 1 - torch.floor(trajectories[..., 0] / 0.05).to(torch.long)
    cols = torch.floor((trajectories[..., 1] + (width * 0.05) / 2.0) / 0.05).to(torch.long)
    rows = rows.clamp(0, height - 1)
    cols = cols.clamp(0, width - 1)
    gathered = grid[torch.arange(batch).view(batch, 1, 1), rows, cols]
    return gathered.amax(dim=2).view(batch, candidate_count)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate CounterfactualDynamicBEVWorldModelV0.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    eval_counterfactual_dynamic_bev_world_model_v0(
        checkpoint=args.checkpoint,
        pack=args.pack,
        split=args.split,
        out_path=args.out,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
