from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import inspect

import numpy as np
import torch
from torch.utils.data import DataLoader

from homebrain.brain.homebrain_net_v0 import HOMEBRAIN_NET_V0_FORBIDDEN_RUNTIME_FIELDS, load_checkpoint
from homebrain.data.spatial_dataset import read_json, write_json
from homebrain.eval.eval_counterfactual_dynamic_bev_world_model_v0 import _evaluate_model as _evaluate_world_model
from homebrain.policies.runtime_decision import decide_trajectory
from homebrain.train.train_counterfactual_dynamic_bev_world_model_v0 import DynamicBEVWorldDataset
from homebrain.train.train_direct_bev_student_v0 import RealRGBDRouteBEVDataset, TARGET_NAMES


def eval_homebrain_net_v0(
    *,
    checkpoint: str | Path,
    real_pack: str | Path | None = None,
    world_pack: str | Path | None = None,
    split: str = "val",
    out_path: str | Path,
    device_name: str | None = None,
) -> Path:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    real_manifest = read_json(Path(real_pack) / "manifest.json") if real_pack is not None and (Path(real_pack) / "manifest.json").exists() else {}
    world_manifest = read_json(Path(world_pack) / "manifest.json") if world_pack is not None and (Path(world_pack) / "manifest.json").exists() else {}
    metrics: dict[str, Any] = {
        "pose": _empty_pose_metrics(),
        "depth": _empty_depth_metrics(),
        "occupancy": _empty_occupancy_metrics(),
        "dynamic": _empty_dynamic_metrics(),
        "world": _empty_world_metrics(),
    }
    if real_pack is not None and real_manifest.get("examples"):
        try:
            metrics.update(_eval_perception(model, real_pack, split=split, device=device))
        except ValueError as exc:
            metrics["perception_error"] = str(exc)
    if world_pack is not None and world_manifest.get("examples"):
        try:
            dataset = DynamicBEVWorldDataset(world_pack, split=split)
            loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=False)
            metrics["world"] = _evaluate_world_model(model.world_model, loader, device=device)
        except ValueError as exc:
            metrics["world_error"] = str(exc)
    runtime_leakage = runtime_field_leakage_passed(metadata)
    acceptance = _acceptance(real_manifest=real_manifest, world_manifest=world_manifest, metrics=metrics, runtime_leakage=runtime_leakage)
    report = {
        "schema_version": "homebrain.eval_homebrain_net_v0.v0",
        "checkpoint": Path(checkpoint).as_posix(),
        "real_pack": Path(real_pack).as_posix() if real_pack is not None else None,
        "world_pack": Path(world_pack).as_posix() if world_pack is not None else None,
        "split": split,
        "baselines": {
            "pose": ["identity", "odom_integration"],
            "depth": ["median_plane", "constant_depth"],
            "occupancy": ["static_copy"],
            "dynamic": ["previous_frame_copy", "zero_flow"],
            "world": ["static_copy", "previous_frame_copy", "center_prior", "transparent_scorer"],
        },
        "metrics": metrics,
        "runtime_field_leakage_passed": runtime_leakage,
        "accepted": bool(acceptance["accepted"]),
        "acceptance": acceptance,
        "safety": {
            "replay_only": metadata.get("replay_only") is True,
            "not_executed": metadata.get("not_executed") is True,
            "control_safe": metadata.get("control_safe") is False,
            "raw_pwm_emitted": metadata.get("raw_pwm_emitted") is False,
            "hardware_validated": metadata.get("hardware_validated") is False,
        },
    }
    write_json(out_path, report, pretty=True)
    return Path(out_path)


@torch.no_grad()
def _eval_perception(model: torch.nn.Module, pack: str | Path, *, split: str, device: torch.device) -> dict[str, Any]:
    dataset = RealRGBDRouteBEVDataset(pack, split=split, image_size=tuple(model.config.image_size), cache_in_memory=False)
    occ = _OccSums()
    depth_stats = _DepthSums()
    dynamic_targets: list[np.ndarray] = []
    dynamic_scores: list[np.ndarray] = []
    dynamic_prev: list[np.ndarray] = []
    flow_epe: list[float] = []
    zero_flow_epe: list[float] = []
    pose_pred: list[np.ndarray] = []
    pose_gt: list[np.ndarray] = []
    pose_odom: list[np.ndarray] = []
    prev_by_route: dict[str, dict[str, np.ndarray]] = {}
    memory_by_route: dict[str, Any] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        route_id = str(item["route_id"])
        previous = prev_by_route.get(route_id)
        pose_mask = item["pose_delta_prev_mask"][None].to(device) if "pose_delta_prev_mask" in item else None
        outputs = model(
            item["rgb"][None].to(device),
            depth=item["depth"][None].to(device),
            sensor_mask=item["sensor_mask"][None].to(device),
            pose_delta_prev=item["pose_delta_prev"][None].to(device),
            previous_action=item["previous_action"][None].to(device),
            memory_state=memory_by_route.get(route_id),
            pose_delta_to_current=item["pose_delta_prev"][None].to(device),
            pose_delta_to_current_mask=pose_mask,
            force_reset=torch.tensor([previous is None], dtype=torch.bool, device=device),
            enabled_heads={"pose", "bev_occupancy", "metric_depth", "dynamic", "memory"},
        )
        if "memory_state" in outputs:
            memory_by_route[route_id] = outputs["memory_state"].detach()
        pred_logits = outputs.get("fused_memory_bev_logits", outputs["bev_logits"])
        pred = torch.sigmoid(pred_logits)[0].cpu().numpy().astype(np.float32)
        target = item["targets"].numpy().astype(np.float32)
        static = target * 0.0 if previous is None else previous["target"]
        occ.add(pred, target, static)
        depth_stats.add(outputs["metric_depth_m"][0].cpu().numpy(), item["metric_depth"].numpy(), item["metric_depth_valid"].numpy())
        dyn_target = item["dynamic_occupancy"].numpy()[0]
        dyn_score = torch.sigmoid(outputs["dynamic_occupancy_logits"])[0, 0].cpu().numpy().astype(np.float32)
        dyn_prev = np.zeros_like(dyn_target) if previous is None else previous["dynamic"]
        dynamic_targets.append(dyn_target.reshape(-1))
        dynamic_scores.append(dyn_score.reshape(-1))
        dynamic_prev.append(dyn_prev.reshape(-1))
        flow_mask = item["bev_flow_valid"].numpy()[0] > 0.0
        if np.any(flow_mask):
            pred_flow = outputs["bev_flow_xy"][0].cpu().numpy().astype(np.float32)
            target_flow = item["bev_flow_xy"].numpy().astype(np.float32)
            flow_epe.extend(np.linalg.norm(pred_flow[:, flow_mask] - target_flow[:, flow_mask], axis=0).astype(float).tolist())
            zero_flow_epe.extend(np.linalg.norm(target_flow[:, flow_mask], axis=0).astype(float).tolist())
        if float(item["pose_delta_next_mask"].item()) > 0.0:
            pose_pred.append(outputs["pose_delta"][0].cpu().numpy().astype(np.float32))
            pose_gt.append(item["pose_delta_next"].numpy().astype(np.float32))
            pose_odom.append(item["pose_delta_prev"].numpy().astype(np.float32))
        prev_by_route[route_id] = {"target": target, "dynamic": dyn_target}
    y_dyn = _concat(dynamic_targets)
    s_dyn = _concat(dynamic_scores)
    p_dyn = _concat(dynamic_prev)
    pose = _pose_metrics(pose_pred, pose_gt, pose_odom)
    return {
        "pose": pose,
        "depth": depth_stats.metrics(),
        "occupancy": occ.metrics(),
        "dynamic": {
            "auprc": _average_precision(y_dyn, s_dyn),
            "f1": _f1(y_dyn, s_dyn >= 0.5),
            "previous_frame_copy_auprc": _average_precision(y_dyn, p_dyn),
            "improvement_vs_previous_frame_copy": _average_precision(y_dyn, s_dyn) - _average_precision(y_dyn, p_dyn),
            "flow_epe": float(np.mean(flow_epe)) if flow_epe else 0.0,
            "zero_flow_epe": float(np.mean(zero_flow_epe)) if zero_flow_epe else 0.0,
            "dynamic_positive_frame_count": int(sum(np.count_nonzero(v > 0.5) > 0 for v in dynamic_targets)),
        },
    }


def runtime_field_leakage_passed(metadata: dict[str, Any]) -> bool:
    runtime_inputs = set(str(value) for value in metadata.get("runtime_inputs", []))
    forbidden = set(HOMEBRAIN_NET_V0_FORBIDDEN_RUNTIME_FIELDS)
    if runtime_inputs & forbidden:
        return False
    decision_params = set(inspect.signature(decide_trajectory).parameters)
    return forbidden.isdisjoint(decision_params) and metadata.get("no_future_labels_used_at_runtime") is True and metadata.get("no_teacher_fields_at_runtime") is True


class _OccSums:
    def __init__(self) -> None:
        self.intersection = np.zeros((len(TARGET_NAMES),), dtype=np.float64)
        self.union = np.zeros((len(TARGET_NAMES),), dtype=np.float64)
        self.scores = [[] for _ in TARGET_NAMES]
        self.labels = [[] for _ in TARGET_NAMES]
        self.static_scores = [[] for _ in TARGET_NAMES]

    def add(self, pred: np.ndarray, target: np.ndarray, static: np.ndarray) -> None:
        for channel in range(len(TARGET_NAMES)):
            pb = pred[channel] >= 0.5
            tb = target[channel] >= 0.5
            self.intersection[channel] += float(np.count_nonzero(pb & tb))
            self.union[channel] += float(np.count_nonzero(pb | tb))
            self.scores[channel].extend(pred[channel].reshape(-1).astype(float).tolist())
            self.labels[channel].extend(tb.reshape(-1).astype(float).tolist())
            self.static_scores[channel].extend(static[channel].reshape(-1).astype(float).tolist())

    def metrics(self) -> dict[str, Any]:
        iou = {TARGET_NAMES[i].removeprefix("target_current_bev_") + "_iou": float(self.intersection[i] / self.union[i]) if self.union[i] else 0.0 for i in range(len(TARGET_NAMES))}
        auprc = [_average_precision(np.asarray(self.labels[i]), np.asarray(self.scores[i])) for i in range(len(TARGET_NAMES))]
        static = [_average_precision(np.asarray(self.labels[i]), np.asarray(self.static_scores[i])) for i in range(len(TARGET_NAMES))]
        return {**iou, "auprc_by_class": dict(zip(TARGET_NAMES, auprc)), "static_copy_auprc_by_class": dict(zip(TARGET_NAMES, static)), "mean_auprc": float(np.mean(auprc)), "static_copy_mean_auprc": float(np.mean(static)), "improvement_vs_static_copy": float(np.mean(auprc) - np.mean(static))}


class _DepthSums:
    def __init__(self) -> None:
        self.abs_rel: list[float] = []
        self.rmse: list[float] = []
        self.baseline_abs_rel: list[float] = []
        self.baseline_rmse: list[float] = []

    def add(self, pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> None:
        mask = valid.reshape(-1) > 0.0
        if not np.any(mask):
            return
        p = pred.reshape(-1)[mask]
        t = target.reshape(-1)[mask]
        baseline = np.full_like(t, float(np.median(t)))
        self.abs_rel.append(float(np.mean(np.abs(p - t) / np.maximum(t, 1.0e-3))))
        self.rmse.append(float(np.sqrt(np.mean((p - t) ** 2))))
        self.baseline_abs_rel.append(float(np.mean(np.abs(baseline - t) / np.maximum(t, 1.0e-3))))
        self.baseline_rmse.append(float(np.sqrt(np.mean((baseline - t) ** 2))))

    def metrics(self) -> dict[str, float]:
        return {
            "abs_rel": _mean(self.abs_rel),
            "rmse": _mean(self.rmse),
            "median_plane_abs_rel": _mean(self.baseline_abs_rel),
            "constant_depth_rmse": _mean(self.baseline_rmse),
            "improvement_vs_median_plane_abs_rel": _mean(self.baseline_abs_rel) - _mean(self.abs_rel),
            "improvement_vs_constant_depth_rmse": _mean(self.baseline_rmse) - _mean(self.rmse),
        }


def _pose_metrics(pred: list[np.ndarray], target: list[np.ndarray], odom: list[np.ndarray]) -> dict[str, float]:
    if not target:
        return _empty_pose_metrics()
    p = np.stack(pred)
    t = np.stack(target)
    o = np.stack(odom)
    ate = _trajectory_rmse(p, t)
    identity = _trajectory_rmse(np.zeros_like(t), t)
    odom_ate = _trajectory_rmse(o, t)
    return {"ate_rmse": ate, "rpe_rmse": float(np.sqrt(np.mean((p - t) ** 2))), "identity_ate_rmse": identity, "odom_integration_ate_rmse": odom_ate, "improvement_vs_identity": identity - ate, "improvement_vs_odom": odom_ate - ate}


def _trajectory_rmse(delta: np.ndarray, target: np.ndarray) -> float:
    pred_xy = np.cumsum(delta[:, :2], axis=0)
    gt_xy = np.cumsum(target[:, :2], axis=0)
    return float(np.sqrt(np.mean((pred_xy - gt_xy) ** 2)))


def _acceptance(*, real_manifest: dict[str, Any], world_manifest: dict[str, Any], metrics: dict[str, Any], runtime_leakage: bool) -> dict[str, Any]:
    manifest = world_manifest or real_manifest
    train = set(str(v) for v in manifest.get("train_route_ids", []))
    val = set(str(v) for v in manifest.get("val_route_ids", manifest.get("heldout_route_ids", [])))
    checks = {
        "synthetic_or_fixture_false": manifest.get("synthetic_or_fixture") is False,
        "real_source_route_count_gte_3": int(manifest.get("real_source_route_count", 0)) >= 3,
        "real_source_frame_count_gte_1000": int(manifest.get("real_source_frame_count", 0)) >= 1000,
        "route_heldout_no_overlap": bool(val) and not (train & val),
        "val_dynamic_positive_frame_count_gt_0": int(metrics.get("dynamic", {}).get("dynamic_positive_frame_count", manifest.get("dynamic_positive_frame_count", 0))) > 0,
        "runtime_field_leakage_passed": runtime_leakage,
        "pose_beats_odom": float(metrics.get("pose", {}).get("improvement_vs_odom", 0.0)) > 0.0,
        "depth_beats_constant": float(metrics.get("depth", {}).get("improvement_vs_constant_depth_rmse", 0.0)) > 0.0,
        "occupancy_beats_static_copy_by_0_05": float(metrics.get("occupancy", {}).get("improvement_vs_static_copy", 0.0)) >= 0.05,
        "dynamic_beats_previous_copy_by_0_05": float(metrics.get("dynamic", {}).get("improvement_vs_previous_frame_copy", 0.0)) >= 0.05,
        "flow_beats_zero_flow": float(metrics.get("dynamic", {}).get("zero_flow_epe", 0.0)) > float(metrics.get("dynamic", {}).get("flow_epe", 0.0)),
        "world_beats_static_copy_by_0_05": float(metrics.get("world", {}).get("improvement_vs_static_copy", 0.0)) >= 0.05,
        "world_candidate_auroc_gte_0_70": (metrics.get("world", {}).get("candidate_risk_auroc") or 0.0) >= 0.70,
    }
    return {"accepted": all(checks.values()), "checks": checks, "failed_checks": [key for key, value in checks.items() if not value]}


def _empty_pose_metrics() -> dict[str, float]:
    return {"ate_rmse": 0.0, "rpe_rmse": 0.0, "identity_ate_rmse": 0.0, "odom_integration_ate_rmse": 0.0, "improvement_vs_identity": 0.0, "improvement_vs_odom": 0.0}


def _empty_depth_metrics() -> dict[str, float]:
    return {"abs_rel": 0.0, "rmse": 0.0, "median_plane_abs_rel": 0.0, "constant_depth_rmse": 0.0, "improvement_vs_median_plane_abs_rel": 0.0, "improvement_vs_constant_depth_rmse": 0.0}


def _empty_occupancy_metrics() -> dict[str, Any]:
    return {"mean_auprc": 0.0, "static_copy_mean_auprc": 0.0, "improvement_vs_static_copy": 0.0}


def _empty_dynamic_metrics() -> dict[str, float]:
    return {"auprc": 0.0, "f1": 0.0, "previous_frame_copy_auprc": 0.0, "improvement_vs_previous_frame_copy": 0.0, "flow_epe": 0.0, "zero_flow_epe": 0.0, "dynamic_positive_frame_count": 0}


def _empty_world_metrics() -> dict[str, Any]:
    return {"improvement_vs_static_copy": 0.0, "improvement_vs_previous_frame_copy": 0.0, "candidate_risk_auroc": None, "candidate_top1_high_risk_rate": 1.0}


def _concat(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(value, dtype=np.float32).reshape(-1) for value in values], axis=0) if values else np.zeros((0,), dtype=np.float32)


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float32).reshape(-1) > 0.5
    s = np.asarray(score, dtype=np.float32).reshape(-1)
    positives = int(np.count_nonzero(y))
    if positives <= 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    precision = np.cumsum(y_sorted, dtype=np.float64) / np.arange(1, y_sorted.size + 1, dtype=np.float64)
    return float(np.sum(precision[y_sorted]) / positives)


def _f1(target: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(target).reshape(-1) > 0.5
    p = np.asarray(pred).reshape(-1) > 0
    tp, fp, fn = float(np.count_nonzero(y & p)), float(np.count_nonzero(~y & p)), float(np.count_nonzero(y & ~p))
    return float((2.0 * tp) / max(2.0 * tp + fp + fn, 1.0))


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate HomeBrainNetV0 on route-heldout packs.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--real-pack", default=None)
    parser.add_argument("--world-pack", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    eval_homebrain_net_v0(checkpoint=args.checkpoint, real_pack=args.real_pack, world_pack=args.world_pack, split=args.split, out_path=args.out, device_name=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
