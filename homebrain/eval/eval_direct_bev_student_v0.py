from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch

from homebrain.brain.direct_bev_student_v0 import load_checkpoint
from homebrain.data.spatial_dataset import read_json, write_json
from homebrain.runtime.direct_bev_runtime import predict_local_bev_from_rgbd
from homebrain.train.train_direct_bev_student_v0 import RealRGBDRouteBEVDataset, TARGET_NAMES


def eval_direct_bev_student_v0(
    *,
    checkpoint: str | Path,
    pack_dir: str | Path,
    split: str,
    out_path: str | Path,
    device_name: str | None = None,
) -> Path:
    pack_root = Path(pack_dir)
    manifest = read_json(pack_root / "manifest.json")
    records = [record for record in manifest.get("examples", []) if isinstance(record, dict) and str(record.get("split")) == split]
    if not records:
        report = _empty_report(manifest, split=split, reason=f"no_examples_for_split_{split}")
        write_json(out_path, report, pretty=True)
        return Path(out_path)

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    dataset = RealRGBDRouteBEVDataset(pack_root, split=split, image_size=tuple(model.config.image_size))
    sums = _MetricSums()
    dynamic_targets: list[np.ndarray] = []
    dynamic_scores: list[np.ndarray] = []
    previous_dynamic_scores: list[np.ndarray] = []
    static_memory_scores: list[np.ndarray] = []
    previous_by_route: dict[str, dict[str, np.ndarray]] = {}
    dynamic_positive_frames = 0
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            route_id = str(item["route_id"])
            outputs = model(
                item["rgb"][None, ...].to(device),
                depth=item["depth"][None, ...].to(device),
                sensor_mask=item["sensor_mask"][None, ...].to(device),
                pose_delta_prev=item["pose_delta_prev"][None, ...].to(device),
                previous_action=item["previous_action"][None, ...].to(device),
            )
            pred = torch.sigmoid(outputs["bev_logits"])[0].detach().cpu().numpy().astype(np.float32)
            uncertainty = torch.sigmoid(outputs["uncertainty_logits"])[0, 0].detach().cpu().numpy().astype(np.float32)
            dynamic = torch.sigmoid(outputs["dynamic_risk_logits"])[0, 0].detach().cpu().numpy().astype(np.float32)
            target = item["targets"].numpy().astype(np.float32)
            target_uncertainty = item["uncertainty"].numpy()[0].astype(np.float32)
            target_dynamic = item["dynamic"].numpy()[0].astype(np.float32)
            sums.add(pred, target, uncertainty, target_uncertainty)
            if np.count_nonzero(target_dynamic > 0.5) > 0:
                dynamic_positive_frames += 1
            previous = previous_by_route.get(route_id)
            previous_dynamic = np.zeros_like(target_dynamic) if previous is None else previous["dynamic"]
            previous_risky = np.zeros_like(target_dynamic) if previous is None else previous["risky"]
            dynamic_targets.append(target_dynamic.reshape(-1))
            dynamic_scores.append(dynamic.reshape(-1))
            previous_dynamic_scores.append(previous_dynamic.reshape(-1))
            static_memory_scores.append(previous_risky.reshape(-1))
            previous_by_route[route_id] = {"dynamic": target_dynamic, "risky": target[4]}

    dynamic_target_flat = _concat(dynamic_targets)
    dynamic_score_flat = _concat(dynamic_scores)
    previous_dynamic_flat = _concat(previous_dynamic_scores)
    static_memory_flat = _concat(static_memory_scores)
    dynamic_auprc = _average_precision(dynamic_target_flat, dynamic_score_flat)
    previous_auprc = _average_precision(dynamic_target_flat, previous_dynamic_flat)
    static_auprc = _average_precision(dynamic_target_flat, static_memory_flat)
    metrics = sums.to_metrics()
    metrics.update(
        {
            "dynamic_risk_auprc": dynamic_auprc,
            "dynamic_risk_f1_at_0_5": _f1(dynamic_target_flat, dynamic_score_flat >= 0.5),
            "route_heldout_count": int(len(set(str(record.get("route_id")) for record in records))),
            "dynamic_positive_frame_count": int(dynamic_positive_frames),
            "improvement_vs_center_prior": float(metrics["occupied_iou"] - sums.center_prior_occupied_iou()),
            "improvement_vs_previous_frame_copy": float(dynamic_auprc - previous_auprc),
            "improvement_vs_static_memory_copy": float(dynamic_auprc - static_auprc),
            "previous_frame_copy_dynamic_risk_auprc": previous_auprc,
            "static_memory_copy_dynamic_risk_auprc": static_auprc,
            "runtime_field_leakage_passed": runtime_field_leakage_passed(metadata),
            "no_future_labels_used": bool(metadata.get("no_future_labels_used") is True),
            "no_teacher_fields_at_runtime": bool(metadata.get("no_teacher_fields_at_runtime") is True),
            "replay_only": metadata.get("replay_only"),
            "not_executed": metadata.get("not_executed"),
            "control_safe": metadata.get("control_safe"),
            "raw_pwm_emitted": metadata.get("raw_pwm_emitted"),
            "hardware_validated": metadata.get("hardware_validated"),
        }
    )
    report = {
        "schema_version": "homebrain.eval_direct_bev_student_v0.v0",
        "checkpoint": Path(checkpoint).as_posix(),
        "pack": pack_root.as_posix(),
        "split": split,
        "metrics": metrics,
        "acceptance": _acceptance(manifest, metrics),
        "accepted_real_rgbd_route_bev_student": False,
    }
    report["accepted_real_rgbd_route_bev_student"] = bool(report["acceptance"]["accepted"])
    write_json(out_path, report, pretty=True)
    return Path(out_path)


def runtime_field_leakage_passed(metadata: dict[str, Any] | None = None) -> bool:
    forbidden = {
        "target_bev",
        "future_labels",
        "future_frames",
        "candidate_oracle_cost",
        "teacher_masks",
        "ground_truth_global_trajectory",
        "route_level_oracle_data",
    }
    runtime_params = set(inspect.signature(predict_local_bev_from_rgbd).parameters)
    if runtime_params & forbidden:
        return False
    if metadata is not None:
        forbidden_runtime = metadata.get("forbidden_runtime_fields")
        if isinstance(forbidden_runtime, list) and any(str(item) in runtime_params for item in forbidden_runtime):
            return False
    return True


class _MetricSums:
    def __init__(self) -> None:
        self.intersections = np.zeros((len(TARGET_NAMES),), dtype=np.float64)
        self.unions = np.zeros((len(TARGET_NAMES),), dtype=np.float64)
        self.center_intersection = 0.0
        self.center_union = 0.0
        self.uncertainty_abs = 0.0
        self.uncertainty_count = 0

    def add(self, pred: np.ndarray, target: np.ndarray, uncertainty: np.ndarray, target_uncertainty: np.ndarray) -> None:
        binary = pred >= 0.5
        target_b = target >= 0.5
        for channel in range(len(TARGET_NAMES)):
            self.intersections[channel] += float(np.count_nonzero(binary[channel] & target_b[channel]))
            self.unions[channel] += float(np.count_nonzero(binary[channel] | target_b[channel]))
        center = np.zeros_like(target_b[1], dtype=np.bool_)
        h, w = center.shape
        center[h // 2 :, w // 3 : 2 * w // 3] = True
        self.center_intersection += float(np.count_nonzero(center & target_b[1]))
        self.center_union += float(np.count_nonzero(center | target_b[1]))
        self.uncertainty_abs += float(np.mean(np.abs(uncertainty - target_uncertainty)))
        self.uncertainty_count += 1

    def to_metrics(self) -> dict[str, float]:
        ious = [float(self.intersections[i] / self.unions[i]) if self.unions[i] > 0 else 0.0 for i in range(len(TARGET_NAMES))]
        return {
            "free_iou": ious[0],
            "occupied_iou": ious[1],
            "unknown_iou": ious[2],
            "traversable_iou": ious[3],
            "risky_iou": ious[4],
            "uncertainty_ece_or_proxy": float(self.uncertainty_abs / max(self.uncertainty_count, 1)),
        }

    def center_prior_occupied_iou(self) -> float:
        return float(self.center_intersection / self.center_union) if self.center_union > 0.0 else 0.0


def _acceptance(manifest: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    train_ids = set(str(value) for value in manifest.get("train_route_ids", []))
    val_ids = set(str(value) for value in manifest.get("heldout_route_ids", []))
    checks = {
        "pack_not_synthetic_or_fixture": manifest.get("synthetic_or_fixture") is False,
        "real_source_route_count_gte_3": int(manifest.get("real_source_route_count", 0)) >= 3,
        "real_source_frame_count_gte_1000": int(manifest.get("real_source_frame_count", 0)) >= 1000,
        "route_heldout_split": manifest.get("route_held_out_split_basis") == "route_id" and not (train_ids & val_ids) and bool(val_ids),
        "heldout_dynamic_positive": int(metrics.get("dynamic_positive_frame_count", 0)) > 0,
        "occupied_iou_gte_0_35": float(metrics.get("occupied_iou", 0.0)) >= 0.35,
        "risky_iou_gte_0_20": float(metrics.get("risky_iou", 0.0)) >= 0.20,
        "dynamic_auprc_beats_previous_by_0_05": float(metrics.get("improvement_vs_previous_frame_copy", 0.0)) >= 0.05,
        "dynamic_f1_gte_0_20": float(metrics.get("dynamic_risk_f1_at_0_5", 0.0)) >= 0.20,
        "runtime_field_leakage_passed": metrics.get("runtime_field_leakage_passed") is True,
        "no_future_labels_used": metrics.get("no_future_labels_used") is True,
        "no_teacher_fields_at_runtime": metrics.get("no_teacher_fields_at_runtime") is True,
        "safety_flags_replay_only": metrics.get("replay_only") is True and metrics.get("not_executed") is True,
        "safety_flags_not_control": metrics.get("control_safe") is False
        and metrics.get("raw_pwm_emitted") is False
        and metrics.get("hardware_validated") is False,
    }
    accepted = all(checks.values())
    return {"accepted": bool(accepted), "checks": checks, "failed_checks": [key for key, value in checks.items() if not value]}


def _empty_report(manifest: dict[str, Any], *, split: str, reason: str) -> dict[str, Any]:
    metrics = {
        "occupied_iou": 0.0,
        "free_iou": 0.0,
        "unknown_iou": 0.0,
        "traversable_iou": 0.0,
        "risky_iou": 0.0,
        "dynamic_risk_auprc": 0.0,
        "dynamic_risk_f1_at_0_5": 0.0,
        "uncertainty_ece_or_proxy": 1.0,
        "route_heldout_count": 0,
        "dynamic_positive_frame_count": 0,
        "improvement_vs_center_prior": 0.0,
        "improvement_vs_previous_frame_copy": 0.0,
        "improvement_vs_static_memory_copy": 0.0,
        "runtime_field_leakage_passed": runtime_field_leakage_passed({}),
        "no_future_labels_used": False,
        "no_teacher_fields_at_runtime": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
    }
    return {
        "schema_version": "homebrain.eval_direct_bev_student_v0.v0",
        "split": split,
        "metrics": metrics,
        "acceptance": {"accepted": False, "failed_checks": [reason], "checks": {}},
        "accepted_real_rgbd_route_bev_student": False,
        "reason": reason,
        "pack_manifest_flags": {
            "synthetic_or_fixture": manifest.get("synthetic_or_fixture"),
            "real_source_route_count": manifest.get("real_source_route_count", 0),
            "real_source_frame_count": manifest.get("real_source_frame_count", 0),
        },
    }


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float32).reshape(-1) > 0.5
    s = np.asarray(score, dtype=np.float32).reshape(-1)
    positives = int(np.count_nonzero(y))
    if positives <= 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted, dtype=np.float64)
    precision = tp / np.maximum(np.arange(1, y_sorted.size + 1, dtype=np.float64), 1.0)
    return float(np.sum(precision[y_sorted]) / positives)


def _f1(target: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float32).reshape(-1) > 0.5
    p = np.asarray(pred, dtype=np.bool_).reshape(-1)
    tp = float(np.count_nonzero(y & p))
    fp = float(np.count_nonzero(~y & p))
    fn = float(np.count_nonzero(y & ~p))
    return float((2.0 * tp) / max(2.0 * tp + fp + fn, 1.0))


def _concat(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate([np.asarray(value, dtype=np.float32).reshape(-1) for value in values], axis=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate DirectBEVStudentV0 on a held-out real RGB-D route split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    eval_direct_bev_student_v0(
        checkpoint=args.checkpoint,
        pack_dir=args.pack,
        split=args.split,
        out_path=args.out,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
