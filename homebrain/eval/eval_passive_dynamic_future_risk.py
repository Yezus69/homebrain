from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from homebrain.brain.future_bev_rollout_v1 import load_checkpoint, score_local_bev_with_future_rollout
from homebrain.data.spatial_dataset import load_example_npz, read_json, write_json
from homebrain.eval.eval_future_bev_rollout_v1 import eval_future_bev_rollout_v1
from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.policies.runtime_decision import decide_trajectory
from homebrain.policies.trajectory_scorer import CoverageMemory, LocalBev, score_trajectories
from homebrain.teachers.passive_dynamic_teacher import (
    PASSIVE_DYNAMIC_FIXTURE_SCENARIOS,
    write_passive_dynamic_fixture_sequences,
)
from homebrain.train.future_bev_rollout_dataset import FutureBEVRolloutDataset
from homebrain.train.passive_dynamic_future_risk_pack import build_passive_dynamic_future_risk_pack
from homebrain.train.spatial_v0_common import batch_to_device
from homebrain.train.train_future_bev_rollout_v1 import _candidate_lower_score, _model_forward, train_future_bev_rollout_v1

PASSIVE_DYNAMIC_EVAL_SCHEMA_VERSION = "homebrain.passive_dynamic_future_risk_eval.v0"
PASSIVE_DYNAMIC_SAFETY = {
    "replay_only": True,
    "not_executed": True,
    "control_safe": False,
    "raw_pwm_emitted": False,
    "hardware_validated": False,
}


def run_passive_dynamic_future_risk_eval(
    *,
    out_dir: str | Path,
    input_dir: str | Path | None = None,
    max_steps: int = 400,
    hidden_channels: int = 16,
    batch_size: int = 8,
    device_name: str = "cpu",
    command: str | None = None,
) -> dict[str, Any]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    fixture_input = Path(input_dir) if input_dir is not None else output / "passive_dynamic_fixture_sequences"
    if input_dir is None:
        write_passive_dynamic_fixture_sequences(fixture_input)
    pack_dir = output / "passive_dynamic_future_risk_pack"
    build_passive_dynamic_future_risk_pack(
        input_dir=fixture_input,
        out_dir=pack_dir,
        horizons_s=(0.5, 1.0, 2.0),
        grid_shape=(16, 16),
        meters_per_cell=0.1,
        robot_radius_m=0.05,
        history_steps=3,
        frame_period_s=0.5,
    )
    train_dir = output / "future_rollout_train"
    train_metrics = train_future_bev_rollout_v1(
        rollout_pack=pack_dir,
        out_dir=train_dir,
        max_steps=max_steps,
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        tiny_overfit=False,
        device_name=device_name,
    )
    eval_metrics = eval_future_bev_rollout_v1(
        checkpoint=train_dir / "checkpoint.pt",
        rollout_pack=pack_dir,
        out_path=output / "future_rollout_eval.json",
        split="val",
        batch_size=batch_size,
        device_name=device_name,
    )
    prediction_metrics = _prediction_metrics(
        checkpoint=train_dir / "checkpoint.pt",
        pack_dir=pack_dir,
        split="val",
        device_name=device_name,
    )
    baseline_metrics = _baseline_selection_metrics(pack_dir, split="val")
    runtime_leakage_tests_passed = _runtime_leakage_contract_passed()
    accepted_probe = bool(
        prediction_metrics["candidate_future_risk_mse"]
        < prediction_metrics["copy_forward_candidate_future_risk_mse_baseline"]
        and prediction_metrics["candidate_future_risk_mse"]
        < prediction_metrics["static_memory_candidate_future_risk_mse_baseline"]
        and prediction_metrics["moving_obstacle_collision_recall"]
        > prediction_metrics["copy_forward_moving_obstacle_collision_recall"]
        and prediction_metrics["unsafe_selected_rate"] == 0.0
        and prediction_metrics["dominant_action_fraction"] < 0.95
        and runtime_leakage_tests_passed
        and _safety_ok(train_metrics, eval_metrics)
    )
    report: dict[str, Any] = {
        "schema_version": PASSIVE_DYNAMIC_EVAL_SCHEMA_VERSION,
        "goal": "Passive Dynamic Future-Risk Pack v0",
        "accepted_robot_brain": False,
        "accepted_passive_dynamic_future_risk_probe": accepted_probe,
        "input_dir": fixture_input.as_posix(),
        "routes": _route_summary(pack_dir),
        "fixtures": list(PASSIVE_DYNAMIC_FIXTURE_SCENARIOS),
        "metrics": {
            "future_risk_iou_or_proxy": eval_metrics.get("future_risk_iou_or_proxy"),
            "candidate_future_risk_mse": prediction_metrics["candidate_future_risk_mse"],
            "candidate_future_risk_mse_source": "future_risk_grid_projected_over_candidate_footprints",
            "candidate_risk_ranking_accuracy": prediction_metrics["candidate_risk_ranking_accuracy"],
            "unsafe_selected_rate": prediction_metrics["unsafe_selected_rate"],
            "future_collision_recall_on_moving_obstacles": prediction_metrics["moving_obstacle_collision_recall"],
            "action_entropy": prediction_metrics["action_entropy"],
            "dominant_action_fraction": prediction_metrics["dominant_action_fraction"],
            "improvement_vs_copy_forward": prediction_metrics["improvement_vs_copy_forward"],
            "improvement_vs_static_memory": prediction_metrics["improvement_vs_static_memory"],
            "selected_distribution": prediction_metrics["selected_distribution"],
            "train_loss_start": train_metrics.get("train_loss_start"),
            "train_loss_end": train_metrics.get("train_loss_end"),
            "val_loss": train_metrics.get("val_loss"),
        },
        "baselines": {
            "current_bev_copy_forward": {
                "candidate_future_risk_mse": prediction_metrics["copy_forward_candidate_future_risk_mse_baseline"],
                "moving_obstacle_collision_recall": prediction_metrics["copy_forward_moving_obstacle_collision_recall"],
            },
            "static_memory_copy_forward": {
                "candidate_future_risk_mse": prediction_metrics["static_memory_candidate_future_risk_mse_baseline"],
                "moving_obstacle_collision_recall": prediction_metrics["static_memory_moving_obstacle_collision_recall"],
            },
            **baseline_metrics,
        },
        "acceptance_checks": {
            "beats_copy_forward_candidate_future_risk_mse": prediction_metrics["candidate_future_risk_mse"]
            < prediction_metrics["copy_forward_candidate_future_risk_mse_baseline"],
            "beats_static_memory_candidate_future_risk_mse": prediction_metrics["candidate_future_risk_mse"]
            < prediction_metrics["static_memory_candidate_future_risk_mse_baseline"],
            "moving_obstacle_collision_recall_improves": prediction_metrics["moving_obstacle_collision_recall"]
            > prediction_metrics["copy_forward_moving_obstacle_collision_recall"],
            "unsafe_selected_rate_zero": prediction_metrics["unsafe_selected_rate"] == 0.0,
            "dominant_action_fraction_below_0_95": prediction_metrics["dominant_action_fraction"] < 0.95,
            "runtime_leakage_tests_pass": runtime_leakage_tests_passed,
        },
        "artifacts": [
            (pack_dir / "manifest.json").as_posix(),
            (train_dir / "checkpoint.pt").as_posix(),
            (train_dir / "train_metrics.json").as_posix(),
            (output / "future_rollout_eval.json").as_posix(),
            (output / "passive_dynamic_future_risk_report.json").as_posix(),
        ],
        "safety": dict(PASSIVE_DYNAMIC_SAFETY),
        "command": command,
        **PASSIVE_DYNAMIC_SAFETY,
    }
    write_json(output / "passive_dynamic_future_risk_report.json", report, pretty=True)
    return report


def _prediction_metrics(
    *,
    checkpoint: Path,
    pack_dir: Path,
    split: str,
    device_name: str,
) -> dict[str, Any]:
    device = torch.device(device_name)
    model, _payload = load_checkpoint(checkpoint, map_location=device)
    model.to(device)
    dataset = FutureBEVRolloutDataset(pack_dir, split=split)
    candidate_ids = list(dataset.candidate_ids)
    candidate_count = len(candidate_ids)
    mse_sum = copy_mse_sum = static_mse_sum = 0.0
    valid_total = 0.0
    moving_positive = moving_pred_positive = moving_copy_positive = moving_static_positive = 0.0
    selected_ids: list[str] = []
    unsafe_selected = 0.0
    selected_total = 0.0
    ranking_scores: list[float] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        batch = batch_to_device({key: value.unsqueeze(0) if torch.is_tensor(value) else value for key, value in sample.items()}, device)
        with torch.no_grad():
            outputs = _model_forward(model, batch)
        target_horizon = batch["candidate_horizon_future_risk"][0].detach().cpu().numpy().astype(np.float32)
        candidate_mask = batch["candidate_valid_mask"][0].detach().cpu().numpy().astype(np.float32)
        masks = batch["candidate_footprint_masks"][0].detach().cpu().numpy().astype(np.float32)
        future_risk_grid = torch.sigmoid(outputs["future_risk_logits"])[0, :, 0].detach().cpu().numpy().astype(np.float32)
        pred_horizon = _risk_values_from_future_grid(future_risk_grid, masks)
        current_risky = batch["current_bev"][0, 4].detach().cpu().numpy().astype(np.float32)
        memory_risky = batch["memory_bev"][0, 4].detach().cpu().numpy().astype(np.float32)
        copy_horizon = _risk_values_from_grid(current_risky, masks, target_horizon.shape[1])
        static_horizon = _risk_values_from_grid(memory_risky, masks, target_horizon.shape[1])
        valid = candidate_mask[:, None] > 0.0
        mse_sum += float(np.sum(((pred_horizon - target_horizon) ** 2) * valid))
        copy_mse_sum += float(np.sum(((copy_horizon - target_horizon) ** 2) * valid))
        static_mse_sum += float(np.sum(((static_horizon - target_horizon) ** 2) * valid))
        valid_total += float(np.sum(valid))
        ranking_scores.extend(_pairwise_ranking_accuracy(np.max(pred_horizon, axis=1), np.max(target_horizon, axis=1), candidate_mask))

        pred_cost = _candidate_lower_score(outputs, candidate_ids=candidate_ids)[0].detach().cpu().numpy().astype(np.float32)
        selected = _safe_selected_index(
            pred_cost=pred_cost,
            predicted_risk=np.max(pred_horizon, axis=1),
            candidate_ids=candidate_ids,
        )
        selected_ids.append(candidate_ids[selected])
        selected_total += 1.0
        future_collision = batch["candidate_future_collision"][0].detach().cpu().numpy().astype(np.float32)
        if candidate_ids[selected] != "stop" and float(future_collision[selected]) >= 0.5:
            unsafe_selected += 1.0

        scenario = str(sample.get("source_name", ""))
        moving = any(token in scenario for token in ("moving_blob", "person_pet", "ego_camera"))
        if moving:
            label_positive = (np.max(target_horizon, axis=1) >= 0.5) & (candidate_mask > 0.0)
            if np.any(label_positive):
                moving_positive += float(np.count_nonzero(label_positive))
                moving_pred_positive += float(np.count_nonzero((np.max(pred_horizon, axis=1) >= 0.5) & label_positive))
                moving_copy_positive += float(np.count_nonzero((np.max(copy_horizon, axis=1) >= 0.5) & label_positive))
                moving_static_positive += float(np.count_nonzero((np.max(static_horizon, axis=1) >= 0.5) & label_positive))

    model_mse = mse_sum / max(valid_total, 1.0)
    copy_mse = copy_mse_sum / max(valid_total, 1.0)
    static_mse = static_mse_sum / max(valid_total, 1.0)
    selected_distribution = _distribution(selected_ids)
    return {
        "candidate_future_risk_mse": float(model_mse),
        "copy_forward_candidate_future_risk_mse_baseline": float(copy_mse),
        "static_memory_candidate_future_risk_mse_baseline": float(static_mse),
        "candidate_risk_ranking_accuracy": _mean(ranking_scores),
        "moving_obstacle_collision_recall": float(moving_pred_positive / max(moving_positive, 1.0)),
        "copy_forward_moving_obstacle_collision_recall": float(moving_copy_positive / max(moving_positive, 1.0)),
        "static_memory_moving_obstacle_collision_recall": float(moving_static_positive / max(moving_positive, 1.0)),
        "unsafe_selected_rate": float(unsafe_selected / max(selected_total, 1.0)),
        "action_entropy": _entropy(selected_distribution),
        "dominant_action_fraction": _dominant_fraction(selected_distribution),
        "selected_distribution": selected_distribution,
        "improvement_vs_copy_forward": float(copy_mse - model_mse),
        "improvement_vs_static_memory": float(static_mse - model_mse),
    }


def _baseline_selection_metrics(pack_dir: Path, *, split: str) -> dict[str, dict[str, float]]:
    manifest = read_json(pack_dir / "manifest.json")
    grid_shape = (int(manifest["grid_shape"][0]), int(manifest["grid_shape"][1]))
    meters_per_cell = float(manifest["meters_per_cell"])
    robot_radius_m = float(manifest["robot_radius_m"])
    candidates = generate_default_candidates(
        grid_shape=grid_shape,
        meters_per_cell=meters_per_cell,
        robot_radius_m=robot_radius_m,
    )
    candidate_ids = [candidate.id for candidate in candidates]
    records = [record for record in manifest["examples"] if isinstance(record, dict) and record.get("split") == split]
    results = {"stop_only": [], "transparent_scorer": [], "unknown_is_dangerous_scorer": []}
    for record in records:
        example = load_example_npz(pack_dir / str(record["example_path"]))
        oracle_cost = np.asarray(example["candidate_oracle_cost"], dtype=np.float32)
        future_collision = np.asarray(example["candidate_future_collision"], dtype=np.float32)
        stop_index = candidate_ids.index("stop")
        results["stop_only"].append(_selection_record(stop_index, oracle_cost, future_collision, candidate_ids))
        transparent = score_trajectories(
            bev=_local_bev(example, unknown_is_dangerous=False),
            candidates=candidates,
            coverage_memory=CoverageMemory(grid_shape, meters_per_cell=meters_per_cell),
        )
        transparent_index = candidate_ids.index(transparent.selected_candidate_id)
        results["transparent_scorer"].append(_selection_record(transparent_index, oracle_cost, future_collision, candidate_ids))
        conservative = score_trajectories(
            bev=_local_bev(example, unknown_is_dangerous=True),
            candidates=candidates,
            coverage_memory=CoverageMemory(grid_shape, meters_per_cell=meters_per_cell),
        )
        conservative_index = candidate_ids.index(conservative.selected_candidate_id)
        results["unknown_is_dangerous_scorer"].append(_selection_record(conservative_index, oracle_cost, future_collision, candidate_ids))
    return {name: _aggregate_selection(values) for name, values in results.items()}


def _local_bev(example: dict[str, np.ndarray], *, unknown_is_dangerous: bool) -> LocalBev:
    free = np.asarray(example["current_bev_free"], dtype=np.float32)
    occupied = np.asarray(example["current_bev_occupied"], dtype=np.float32)
    unknown = np.asarray(example["current_bev_unknown"], dtype=np.float32)
    risky = np.asarray(example["current_bev_risky"], dtype=np.float32)
    if unknown_is_dangerous:
        occupied = np.maximum(occupied, unknown)
        risky = np.maximum(risky, unknown)
    return LocalBev(
        free=free,
        occupied=occupied,
        unknown=unknown,
        traversable=np.asarray(example["current_bev_traversable"], dtype=np.float32),
        risky=risky,
        confidence=np.asarray(example["current_bev_confidence"], dtype=np.float32),
        uncertainty=np.asarray(example["uncertainty_map"], dtype=np.float32),
        source="passive_dynamic_eval",
    )


def _risk_values_from_grid(grid: np.ndarray, masks: np.ndarray, horizon_count: int) -> np.ndarray:
    values = []
    for mask in masks:
        denom = max(float(np.sum(mask)), 1.0)
        values.append(float(np.sum(grid * mask) / denom))
    return np.repeat(np.asarray(values, dtype=np.float32)[:, None], horizon_count, axis=1)


def _risk_values_from_future_grid(future_grid: np.ndarray, masks: np.ndarray) -> np.ndarray:
    values: list[list[float]] = []
    for mask in masks:
        denom = max(float(np.sum(mask)), 1.0)
        values.append([float(np.sum(future_grid[horizon] * mask) / denom) for horizon in range(future_grid.shape[0])])
    return np.asarray(values, dtype=np.float32)


def _safe_selected_index(*, pred_cost: np.ndarray, predicted_risk: np.ndarray, candidate_ids: list[str]) -> int:
    gated = np.asarray(pred_cost, dtype=np.float32).copy()
    unsafe = np.asarray(predicted_risk, dtype=np.float32) >= 0.5
    for index, candidate_id in enumerate(candidate_ids):
        if candidate_id != "stop" and unsafe[index]:
            gated[index] += np.float32(1000.0)
    return int(np.argmin(gated))


def _pairwise_ranking_accuracy(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> list[float]:
    valid = np.where(np.asarray(mask, dtype=np.float32) > 0.0)[0]
    correct = 0
    total = 0
    for left_pos, left in enumerate(valid):
        for right in valid[left_pos + 1 :]:
            target_diff = float(target[left] - target[right])
            if abs(target_diff) < 1.0e-6:
                continue
            pred_diff = float(pred[left] - pred[right])
            correct += int((target_diff > 0.0 and pred_diff > 0.0) or (target_diff < 0.0 and pred_diff < 0.0))
            total += 1
    return [float(correct / total)] if total > 0 else []


def _selection_record(index: int, oracle_cost: np.ndarray, future_collision: np.ndarray, candidate_ids: list[str]) -> dict[str, Any]:
    oracle = int(np.argmin(oracle_cost))
    candidate_id = candidate_ids[index]
    return {
        "candidate_id": candidate_id,
        "oracle_match": index == oracle,
        "unsafe_selected": candidate_id != "stop" and float(future_collision[index]) >= 0.5,
        "oracle_cost": float(oracle_cost[index]),
    }


def _aggregate_selection(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"oracle_match_fraction": 0.0, "unsafe_selected_rate": 0.0, "mean_oracle_cost": 0.0}
    return {
        "oracle_match_fraction": float(sum(1 for item in records if item["oracle_match"]) / len(records)),
        "unsafe_selected_rate": float(sum(1 for item in records if item["unsafe_selected"]) / len(records)),
        "mean_oracle_cost": float(np.mean([float(item["oracle_cost"]) for item in records])),
    }


def _runtime_leakage_contract_passed() -> bool:
    forbidden = {"future_bev", "future_risk", "candidate_oracle_cost", "teacher_artifacts", "route_ground_truth"}
    return forbidden.isdisjoint(inspect.signature(decide_trajectory).parameters) and forbidden.isdisjoint(
        inspect.signature(score_local_bev_with_future_rollout).parameters
    )


def _route_summary(pack_dir: Path) -> list[JsonDict]:
    manifest = read_json(pack_dir / "manifest.json")
    return [
        {
            "sequence_id": str(record.get("source_name", "")),
            "split": str(record.get("split", "")),
            "scenario": str(record.get("scenario", "")),
            "frame_count": int(record.get("frame_count", 0)),
        }
        for record in manifest.get("sources", [])
        if isinstance(record, dict)
    ]


def _safety_ok(*metrics: dict[str, Any]) -> bool:
    return all(
        item.get("replay_only") is True
        and item.get("not_executed") is True
        and item.get("control_safe") is False
        and item.get("raw_pwm_emitted") is not True
        for item in metrics
    )


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
        entropy -= p * float(np.log(p))
    return float(entropy)


def _dominant_fraction(distribution: dict[str, int]) -> float:
    total = sum(distribution.values())
    if total <= 0:
        return 0.0
    return float(max(distribution.values()) / total)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run passive dynamic FutureBEV future-risk eval.")
    parser.add_argument("--out", required=True, help="Output directory for pack, checkpoint, metrics, and report.")
    parser.add_argument("--input", default=None, help="Optional existing passive dynamic sequence root.")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    command = " ".join(["eval_passive_dynamic_future_risk", *(argv or [])])
    report = run_passive_dynamic_future_risk_eval(
        out_dir=args.out,
        input_dir=args.input,
        max_steps=args.max_steps,
        hidden_channels=args.hidden_channels,
        batch_size=args.batch_size,
        device_name=args.device,
        command=command,
    )
    print(json.dumps({"report": report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
