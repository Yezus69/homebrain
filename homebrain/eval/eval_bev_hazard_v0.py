from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from homebrain.brain.direct_bev_student_v0 import load_checkpoint
from homebrain.data.spatial_dataset import load_example_npz, read_json, write_json
from homebrain.eval.hazard_eval_utils import (
    append_flat as _append_flat,
    average_precision as _average_precision,
    cell_values as _cell_values,
    concat as _concat,
    f1 as _f1,
    hazard_runtime_field_leakage_passed,
    mean as _mean,
    per_route_metrics as _per_route_metrics,
    recall as _recall,
)
from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.policies.trajectory_scorer import LocalBev, score_trajectories
from homebrain.train.train_direct_bev_student_v0 import RealRGBDRouteBEVDataset

HAZARD_EVAL_SCHEMA_VERSION = "homebrain.eval_bev_hazard_v0.v0"


def eval_bev_hazard_v0(
    *,
    checkpoint: str | Path,
    pack_dir: str | Path,
    split: str,
    out_path: str | Path,
    previous_checkpoint: str | Path | None = None,
    device_name: str | None = None,
) -> Path:
    pack_root = Path(pack_dir)
    manifest = read_json(pack_root / "manifest.json")
    if manifest.get("has_hazard_channel") is not True:
        report = _empty_report(
            checkpoint=checkpoint,
            pack_root=pack_root,
            split=split,
            manifest=manifest,
            reason="pack_has_hazard_channel_false",
        )
        write_json(out_path, report, pretty=True)
        return Path(out_path)

    records = [record for record in manifest.get("examples", []) if isinstance(record, dict) and str(record.get("split")) == split]
    if not records:
        report = _empty_report(
            checkpoint=checkpoint,
            pack_root=pack_root,
            split=split,
            manifest=manifest,
            reason=f"no_examples_for_split_{split}",
        )
        write_json(out_path, report, pretty=True)
        return Path(out_path)

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    dataset = RealRGBDRouteBEVDataset(pack_root, split=split, image_size=tuple(model.config.image_size))

    target_values: list[np.ndarray] = []
    pred_values: list[np.ndarray] = []
    risky_baseline_values: list[np.ndarray] = []
    depth_baseline_values: list[np.ndarray] = []
    thin_target_values: list[np.ndarray] = []
    thin_pred_values: list[np.ndarray] = []
    thin_depth_values: list[np.ndarray] = []
    free_nonhazard_pred_values: list[np.ndarray] = []
    per_route: dict[str, dict[str, list[np.ndarray]]] = {}
    selected_hazard_exposures: list[float] = []
    selected_depth_only_hazard_exposures: list[float] = []
    hazard_positive_frames = 0

    with torch.no_grad():
        for index, item in enumerate(dataset):
            record = dataset.records[index]
            arrays = load_example_npz(pack_root / str(record["example_path"]))
            target_hazard = np.asarray(arrays["target_bev_hazard"], dtype=np.float32)
            valid = np.asarray(arrays["target_bev_hazard_valid_mask"], dtype=np.float32) > 0.0
            if not np.any(valid):
                continue
            outputs = model(
                item["rgb"][None, ...].to(device),
                depth=item["depth"][None, ...].to(device),
                sensor_mask=item["sensor_mask"][None, ...].to(device),
                pose_delta_prev=item["pose_delta_prev"][None, ...].to(device),
                previous_action=item["previous_action"][None, ...].to(device),
            )
            pred_hazard = torch.sigmoid(outputs["hazard_logits"])[0, 0].detach().cpu().numpy().astype(np.float32)
            pred_bev = torch.sigmoid(outputs["bev_logits"])[0].detach().cpu().numpy().astype(np.float32)
            dynamic = torch.sigmoid(outputs["dynamic_risk_logits"])[0, 0].detach().cpu().numpy().astype(np.float32)
            risky_score = np.maximum(pred_bev[4], dynamic).astype(np.float32)
            depth_only_score = np.asarray(arrays["target_current_bev_occupied"], dtype=np.float32)
            free = np.asarray(arrays["target_current_bev_free"], dtype=np.float32)
            thin = valid & (target_hazard >= 0.5) & (free >= 0.5) & (depth_only_score < 0.5)
            nonhazard_free = valid & (target_hazard < 0.5) & (free >= 0.5)

            if int(np.count_nonzero(valid & (target_hazard >= 0.5))) > 0:
                hazard_positive_frames += 1
            _append_flat(target_values, target_hazard, valid)
            _append_flat(pred_values, pred_hazard, valid)
            _append_flat(risky_baseline_values, risky_score, valid)
            _append_flat(depth_baseline_values, depth_only_score, valid)
            _append_flat(thin_target_values, target_hazard, thin)
            _append_flat(thin_pred_values, pred_hazard, thin)
            _append_flat(thin_depth_values, depth_only_score, thin)
            _append_flat(free_nonhazard_pred_values, pred_hazard, nonhazard_free)
            route_id = str(record.get("route_id", ""))
            route_bucket = per_route.setdefault(route_id, {"target": [], "pred": [], "thin_target": [], "thin_pred": []})
            _append_flat(route_bucket["target"], target_hazard, valid)
            _append_flat(route_bucket["pred"], pred_hazard, valid)
            _append_flat(route_bucket["thin_target"], target_hazard, thin)
            _append_flat(route_bucket["thin_pred"], pred_hazard, thin)

            exposure, depth_exposure = _candidate_exposure_pair(
                arrays=arrays,
                predicted_hazard=pred_hazard,
                target_hazard=target_hazard,
                grid_shape=target_hazard.shape,
                meters_per_cell=float(manifest.get("meters_per_cell", 0.05)),
            )
            selected_hazard_exposures.append(exposure)
            selected_depth_only_hazard_exposures.append(depth_exposure)

    target = _concat(target_values)
    pred = _concat(pred_values)
    risky_baseline = _concat(risky_baseline_values)
    depth_baseline = _concat(depth_baseline_values)
    thin_target = _concat(thin_target_values)
    thin_pred = _concat(thin_pred_values)
    thin_depth = _concat(thin_depth_values)
    free_nonhazard_pred = _concat(free_nonhazard_pred_values)
    per_route_metrics = _per_route_metrics(per_route)
    previous_metrics = _previous_checkpoint_baseline(
        previous_checkpoint=previous_checkpoint,
        pack_root=pack_root,
        split=split,
        device=device,
    )
    metrics: dict[str, Any] = {
        "hazard_auprc": _average_precision(target, pred),
        "hazard_f1": _f1(target, pred >= 0.5),
        "hazard_recall_thin_floor_objects": _recall(thin_target, thin_pred >= 0.5),
        "depth_occupancy_only_hazard_auprc": _average_precision(target, depth_baseline),
        "depth_occupancy_only_thin_recall": _recall(thin_target, thin_depth >= 0.5),
        "risky_channel_only_hazard_auprc": _average_precision(target, risky_baseline),
        "risky_channel_only_f1": _f1(target, risky_baseline >= 0.5),
        "previous_accepted_checkpoint": str(previous_checkpoint) if previous_checkpoint is not None else None,
        "previous_accepted_checkpoint_metrics": previous_metrics,
        "free_space_false_positive_rate": float(np.mean(free_nonhazard_pred >= 0.5)) if free_nonhazard_pred.size else 0.0,
        "candidate_hazard_exposure_mean": _mean(selected_hazard_exposures),
        "candidate_hazard_exposure_vs_depth_only_scorer": _mean(selected_hazard_exposures)
        - _mean(selected_depth_only_hazard_exposures),
        "depth_only_scorer_candidate_hazard_exposure_mean": _mean(selected_depth_only_hazard_exposures),
        "route_heldout_count": int(len(set(str(record.get("route_id")) for record in records))),
        "per_route_hazard_metric_json": per_route_metrics,
        "worst_route_hazard_recall": min((float(route["hazard_recall_thin_floor_objects"]) for route in per_route_metrics.values()), default=0.0),
        "runtime_field_leakage_passed": hazard_runtime_field_leakage_passed(metadata),
        "weak_label_provenance_present": bool(manifest.get("hazard_weak_label") is True and manifest.get("control_safe") is False),
        "hazard_positive_frame_count": int(hazard_positive_frames),
        "real_source_route_count": int(manifest.get("real_source_route_count", 0) or 0),
        "hand_verified_hazard_positive_frame_count": int(manifest.get("hand_verified_hazard_positive_frame_count", 0) or 0),
        "synthetic_or_fixture": bool(manifest.get("synthetic_or_fixture", True)),
        "hazard_source_mock_used": bool(manifest.get("hazard_source_mock_used", False)),
        "hazard_source_synthetic_used": bool(manifest.get("hazard_source_synthetic_used", False)),
        "replay_only": metadata.get("replay_only"),
        "not_executed": metadata.get("not_executed"),
        "control_safe": metadata.get("control_safe"),
        "raw_pwm_emitted": metadata.get("raw_pwm_emitted"),
        "hardware_validated": metadata.get("hardware_validated"),
        "no_teacher_fields_at_runtime": metadata.get("no_teacher_fields_at_runtime"),
    }
    report = {
        "schema_version": HAZARD_EVAL_SCHEMA_VERSION,
        "checkpoint": Path(checkpoint).as_posix(),
        "pack": pack_root.as_posix(),
        "split": split,
        "metrics": metrics,
        "acceptance": _acceptance(manifest, metrics),
        "accepted_bev_hazard_v0": False,
    }
    report["accepted_bev_hazard_v0"] = bool(report["acceptance"]["accepted"])
    write_json(out_path, report, pretty=True)
    return Path(out_path)


def _candidate_exposure_pair(
    *,
    arrays: dict[str, np.ndarray],
    predicted_hazard: np.ndarray,
    target_hazard: np.ndarray,
    grid_shape: tuple[int, int],
    meters_per_cell: float,
) -> tuple[float, float]:
    candidates = generate_default_candidates(grid_shape=grid_shape, meters_per_cell=meters_per_cell, robot_radius_m=max(meters_per_cell, 0.15))
    free = np.asarray(arrays["target_current_bev_free"], dtype=np.float32)
    occupied = np.asarray(arrays["target_current_bev_occupied"], dtype=np.float32)
    unknown = np.asarray(arrays["target_current_bev_unknown"], dtype=np.float32)
    traversable = np.asarray(arrays["target_current_bev_traversable"], dtype=np.float32)
    risky = np.asarray(arrays["target_current_bev_risky"], dtype=np.float32)
    hazard_bev = LocalBev(
        free=free,
        occupied=occupied,
        unknown=unknown,
        traversable=traversable,
        risky=risky,
        hazard=predicted_hazard,
        confidence=np.ones_like(free),
        uncertainty=np.zeros_like(free),
        source="eval_bev_hazard_v0_predicted",
    )
    depth_only_bev = LocalBev(
        free=free,
        occupied=occupied,
        unknown=unknown,
        traversable=traversable,
        risky=risky,
        hazard=None,
        confidence=np.ones_like(free),
        uncertainty=np.zeros_like(free),
        source="eval_bev_hazard_v0_depth_only",
    )
    hazard_decision = score_trajectories(bev=hazard_bev, candidates=candidates)
    depth_decision = score_trajectories(bev=depth_only_bev, candidates=candidates)
    return (
        _selected_target_hazard_exposure(target_hazard, candidates, hazard_decision.selected_candidate_id),
        _selected_target_hazard_exposure(target_hazard, candidates, depth_decision.selected_candidate_id),
    )


def _selected_target_hazard_exposure(target_hazard: np.ndarray, candidates: list[Any], selected_id: str) -> float:
    for candidate in candidates:
        if candidate.id == selected_id:
            values = _cell_values(target_hazard, candidate.footprint_cells, default=1.0)
            return max(float(np.max(values)), float(np.mean(values)))
    return 1.0


def _acceptance(manifest: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    train_ids = set(str(value) for value in manifest.get("train_route_ids", []))
    val_ids = set(str(value) for value in manifest.get("heldout_route_ids", []))
    recall_margin = float(metrics.get("hazard_recall_thin_floor_objects", 0.0)) - float(
        metrics.get("depth_occupancy_only_thin_recall", 0.0)
    )
    checks = {
        "synthetic_or_fixture_false": metrics.get("synthetic_or_fixture") is False,
        "hazard_source_not_mock_or_synthetic": metrics.get("hazard_source_mock_used") is False
        and metrics.get("hazard_source_synthetic_used") is False,
        "real_source_route_count_gte_1": int(metrics.get("real_source_route_count", 0)) >= 1,
        "hand_verified_hazard_instances_exist": int(metrics.get("hand_verified_hazard_positive_frame_count", 0)) > 0,
        "hazard_positive_frame_count_gt_0": int(metrics.get("hazard_positive_frame_count", 0)) > 0,
        "route_heldout_split": manifest.get("route_held_out_split_basis") == "route_id" and bool(val_ids) and not (train_ids & val_ids),
        "hazard_recall_beats_depth_by_0_25": recall_margin >= 0.25,
        "free_space_false_positive_rate_lte_0_20": float(metrics.get("free_space_false_positive_rate", 1.0)) <= 0.20,
        "candidate_hazard_exposure_mean_lt_depth_only": float(metrics.get("candidate_hazard_exposure_vs_depth_only_scorer", 0.0)) < 0.0,
        "runtime_field_leakage_passed": metrics.get("runtime_field_leakage_passed") is True,
        "weak_label_provenance_present": metrics.get("weak_label_provenance_present") is True,
        "no_teacher_fields_at_runtime": metrics.get("no_teacher_fields_at_runtime") is True,
        "safety_flags_replay_only": metrics.get("replay_only") is True and metrics.get("not_executed") is True,
        "safety_flags_not_control": metrics.get("control_safe") is False
        and metrics.get("raw_pwm_emitted") is False
        and metrics.get("hardware_validated") is False,
    }
    accepted = all(checks.values())
    return {"accepted": bool(accepted), "checks": checks, "failed_checks": [key for key, value in checks.items() if not value]}


def _empty_report(
    *,
    checkpoint: str | Path,
    pack_root: Path,
    split: str,
    manifest: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    metrics = {
        "hazard_auprc": 0.0,
        "hazard_f1": 0.0,
        "hazard_recall_thin_floor_objects": 0.0,
        "depth_occupancy_only_thin_recall": 0.0,
        "risky_channel_only_hazard_auprc": 0.0,
        "free_space_false_positive_rate": 0.0,
        "candidate_hazard_exposure_mean": 0.0,
        "candidate_hazard_exposure_vs_depth_only_scorer": 0.0,
        "route_heldout_count": 0,
        "per_route_hazard_metric_json": {},
        "worst_route_hazard_recall": 0.0,
        "runtime_field_leakage_passed": hazard_runtime_field_leakage_passed({}),
        "weak_label_provenance_present": bool(manifest.get("hazard_weak_label") is True),
        "hazard_positive_frame_count": 0,
        "real_source_route_count": int(manifest.get("real_source_route_count", 0) or 0),
        "hand_verified_hazard_positive_frame_count": int(manifest.get("hand_verified_hazard_positive_frame_count", 0) or 0),
        "synthetic_or_fixture": bool(manifest.get("synthetic_or_fixture", True)),
        "hazard_source_mock_used": bool(manifest.get("hazard_source_mock_used", False)),
        "hazard_source_synthetic_used": bool(manifest.get("hazard_source_synthetic_used", False)),
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
        "no_teacher_fields_at_runtime": False,
    }
    return {
        "schema_version": HAZARD_EVAL_SCHEMA_VERSION,
        "checkpoint": Path(checkpoint).as_posix(),
        "pack": pack_root.as_posix(),
        "split": split,
        "metrics": metrics,
        "acceptance": {"accepted": False, "failed_checks": [reason], "checks": {}},
        "accepted_bev_hazard_v0": False,
        "reason": reason,
    }


def _previous_checkpoint_baseline(
    *,
    previous_checkpoint: str | Path | None,
    pack_root: Path,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    if previous_checkpoint is None:
        return {"provided": False, "hazard_auprc": 0.0, "hazard_f1": 0.0}
    try:
        model, _payload = load_checkpoint(previous_checkpoint, map_location=device)
        model.to(device).eval()
        dataset = RealRGBDRouteBEVDataset(pack_root, split=split, image_size=tuple(model.config.image_size))
        targets: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for index, item in enumerate(dataset):
                arrays = load_example_npz(pack_root / str(dataset.records[index]["example_path"]))
                if "target_bev_hazard" not in arrays:
                    continue
                valid = np.asarray(arrays.get("target_bev_hazard_valid_mask", np.ones_like(arrays["target_bev_hazard"])), dtype=np.float32) > 0.0
                outputs = model(
                    item["rgb"][None, ...].to(device),
                    depth=item["depth"][None, ...].to(device),
                    sensor_mask=item["sensor_mask"][None, ...].to(device),
                    pose_delta_prev=item["pose_delta_prev"][None, ...].to(device),
                    previous_action=item["previous_action"][None, ...].to(device),
                )
                pred = torch.sigmoid(outputs["hazard_logits"])[0, 0].detach().cpu().numpy().astype(np.float32)
                _append_flat(targets, np.asarray(arrays["target_bev_hazard"], dtype=np.float32), valid)
                _append_flat(scores, pred, valid)
        target = _concat(targets)
        score = _concat(scores)
        return {"provided": True, "hazard_auprc": _average_precision(target, score), "hazard_f1": _f1(target, score >= 0.5)}
    except Exception as exc:  # noqa: BLE001 - baseline failure should be visible, not fatal.
        return {"provided": True, "failed": True, "error": str(exc), "hazard_auprc": 0.0, "hazard_f1": 0.0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate DirectBEVStudentV0 hazard head on weak semantic BEV hazards.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    parser.add_argument("--previous-checkpoint", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    eval_bev_hazard_v0(
        checkpoint=args.checkpoint,
        pack_dir=args.pack,
        split=args.split,
        out_path=args.out,
        previous_checkpoint=args.previous_checkpoint,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
