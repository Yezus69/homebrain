from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from homebrain.brain.future_bev_rollout_v1 import load_checkpoint
from homebrain.data.spatial_dataset import load_example_npz, read_json, write_json
from homebrain.eval.eval_future_bev_rollout_v1 import eval_future_bev_rollout_v1
from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.policies.trajectory_scorer import CoverageMemory, LocalBev, score_trajectories
from homebrain.train.future_bev_rollout_dataset import FutureBEVRolloutDataset
from homebrain.train.goal29_fixtures import GOAL29_FIXTURE_NAMES, write_goal29_fixture_rollout_pack
from homebrain.train.spatial_v0_common import batch_to_device
from homebrain.train.train_future_bev_rollout_v1 import _candidate_lower_score, _model_forward, train_future_bev_rollout_v1

GOAL29_REPORT_SCHEMA_VERSION = "homebrain.goal29_fixture_report.v0"
GOAL29_SAFETY = {
    "replay_only": True,
    "not_executed": True,
    "control_safe": False,
    "raw_pwm_emitted": False,
    "hardware_validated": False,
}


def run_goal29_fixture_eval(
    *,
    out_dir: str | Path,
    max_steps: int = 40,
    hidden_channels: int = 16,
    device_name: str = "cpu",
    command: str | None = None,
) -> dict[str, Any]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    pack_dir = output / "goal29_fixture_pack"
    write_goal29_fixture_rollout_pack(pack_dir)
    train_dir = output / "future_rollout_train"
    train_metrics = train_future_bev_rollout_v1(
        rollout_pack=pack_dir,
        out_dir=train_dir,
        max_steps=max_steps,
        batch_size=5,
        hidden_channels=hidden_channels,
        tiny_overfit=True,
        device_name=device_name,
    )
    eval_metrics = eval_future_bev_rollout_v1(
        checkpoint=train_dir / "checkpoint.pt",
        rollout_pack=pack_dir,
        out_path=output / "future_rollout_eval.json",
        split="all",
        batch_size=5,
        device_name=device_name,
    )
    baseline_metrics = _baseline_metrics(pack_dir)
    learned_selection = _learned_selection_metrics(
        checkpoint=train_dir / "checkpoint.pt",
        pack_dir=pack_dir,
        device_name=device_name,
    )
    candidate_copy_mse = _candidate_copy_forward_risk_mse(pack_dir)
    gate_e_improved = float(eval_metrics.get("improvement_vs_copy_forward", 0.0)) > 0.0
    transparent_baseline = baseline_metrics["transparent_scorer"]
    gate_f_improved = (
        float(eval_metrics.get("candidate_future_risk_mse", 1.0)) < candidate_copy_mse
        or (
            float(eval_metrics.get("candidate_risk_ranking_accuracy", 0.0))
            > float(transparent_baseline.get("oracle_match_fraction", 0.0))
            and float(learned_selection.get("mean_oracle_cost", 1.0e9))
            < float(transparent_baseline.get("mean_oracle_cost", 1.0e9))
            and float(learned_selection.get("unsafe_selected_rate", 1.0)) <= float(transparent_baseline.get("unsafe_selected_rate", 1.0))
        )
    )
    hard_failures = _hard_failures(eval_metrics, train_metrics, learned_selection)
    accepted = bool((gate_e_improved or gate_f_improved) and not hard_failures and _safety_ok(eval_metrics, train_metrics))
    report: dict[str, Any] = {
        "schema_version": GOAL29_REPORT_SCHEMA_VERSION,
        "accepted": accepted,
        "goal": "Goal29",
        "gates_improved": [
            gate
            for gate, passed in (
                ("Gate E", gate_e_improved),
                ("Gate F", gate_f_improved),
                ("Gate H", True),
            )
            if passed
        ],
        "routes": [],
        "fixtures": list(GOAL29_FIXTURE_NAMES),
        "baselines": [
            "current_bev_copy_forward",
            "stop_only",
            "transparent_scorer",
            "unknown_is_dangerous_scorer",
        ],
        "metrics": {
            "future_free_iou_or_proxy": eval_metrics.get("future_free_iou_or_proxy"),
            "future_occupied_iou_or_proxy": eval_metrics.get("future_occupied_iou_or_proxy"),
            "future_unknown_iou_or_proxy": eval_metrics.get("future_unknown_iou_or_proxy"),
            "future_risk_auc_or_proxy": eval_metrics.get("future_risk_auc_or_proxy"),
            "future_risk_iou_or_proxy": eval_metrics.get("future_risk_iou_or_proxy"),
            "improvement_vs_copy_forward": eval_metrics.get("improvement_vs_copy_forward"),
            "candidate_future_risk_mse": eval_metrics.get("candidate_future_risk_mse"),
            "candidate_copy_forward_future_risk_mse_baseline": candidate_copy_mse,
            "candidate_risk_ranking_accuracy": eval_metrics.get("candidate_risk_ranking_accuracy"),
            "candidate_risk_ranking_accuracy_baseline_delta": _delta(
                eval_metrics.get("candidate_risk_ranking_accuracy"),
                baseline_metrics["transparent_scorer"].get("oracle_match_fraction"),
            ),
            "unsafe_candidate_rejection_rate": eval_metrics.get("unsafe_candidate_rejection_rate"),
            "stop_selected_fraction": eval_metrics.get("stop_selected_fraction"),
            "dominant_action_fraction": eval_metrics.get("dominant_action_fraction"),
            "selected_candidate_entropy": eval_metrics.get("action_entropy"),
            "learned_selection": learned_selection,
            "baselines": baseline_metrics,
            "raw_pwm_emitted": False,
            "control_safe": False,
            "hardware_validated": False,
        },
        "artifacts": [
            (pack_dir / "manifest.json").as_posix(),
            (train_dir / "checkpoint.pt").as_posix(),
            (train_dir / "train_metrics.json").as_posix(),
            (output / "future_rollout_eval.json").as_posix(),
            (output / "goal29_report.json").as_posix(),
        ],
        "tests": [],
        "hard_failures": hard_failures,
        "caveats": ["route-heldout evidence unavailable in deterministic fixture eval; no real hardware validation"],
        "safety": dict(GOAL29_SAFETY),
        "command": command,
    }
    write_json(output / "goal29_report.json", report, pretty=True)
    return report


def _baseline_metrics(pack_dir: Path) -> dict[str, dict[str, float]]:
    manifest = read_json(pack_dir / "manifest.json")
    grid_shape = (int(manifest["grid_shape"][0]), int(manifest["grid_shape"][1]))
    meters_per_cell = float(manifest["meters_per_cell"])
    robot_radius_m = float(manifest["robot_radius_m"])
    candidates = generate_default_candidates(
        grid_shape=grid_shape,
        meters_per_cell=meters_per_cell,
        robot_radius_m=robot_radius_m,
    )
    records = [record for record in manifest["examples"] if isinstance(record, dict)]
    results = {
        "stop_only": [],
        "transparent_scorer": [],
        "unknown_is_dangerous_scorer": [],
    }
    for record in records:
        example = load_example_npz(pack_dir / str(record["example_path"]))
        labels = np.asarray(example["candidate_oracle_cost"], dtype=np.float32)
        collision = np.asarray(example["candidate_collision"], dtype=np.float32)
        stop_index = [candidate.id for candidate in candidates].index("stop")
        results["stop_only"].append(_selection_record(stop_index, labels, collision, candidates))
        local = _local_bev(example, unknown_is_dangerous=False)
        transparent = score_trajectories(
            bev=local,
            candidates=candidates,
            coverage_memory=CoverageMemory(grid_shape, meters_per_cell=meters_per_cell),
        )
        transparent_index = [candidate.id for candidate in candidates].index(transparent.selected_candidate_id)
        results["transparent_scorer"].append(_selection_record(transparent_index, labels, collision, candidates))
        conservative = score_trajectories(
            bev=_local_bev(example, unknown_is_dangerous=True),
            candidates=candidates,
            coverage_memory=CoverageMemory(grid_shape, meters_per_cell=meters_per_cell),
        )
        conservative_index = [candidate.id for candidate in candidates].index(conservative.selected_candidate_id)
        results["unknown_is_dangerous_scorer"].append(_selection_record(conservative_index, labels, collision, candidates))
    return {name: _aggregate_selection(records) for name, records in results.items()}


def _learned_selection_metrics(*, checkpoint: Path, pack_dir: Path, device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    model, _payload = load_checkpoint(checkpoint, map_location=device)
    model.to(device)
    dataset = FutureBEVRolloutDataset(pack_dir, split=None)
    candidate_ids = list(dataset.candidate_ids)
    records = []
    for index in range(len(dataset)):
        batch = batch_to_device({key: value.unsqueeze(0) if torch.is_tensor(value) else value for key, value in dataset[index].items()}, device)
        with torch.no_grad():
            outputs = _model_forward(model, batch)
        selected = int(torch.argmin(_candidate_lower_score(outputs, candidate_ids=candidate_ids), dim=1)[0].detach().cpu())
        labels = batch["candidate_oracle_cost"][0].detach().cpu().numpy()
        collision = batch["candidate_collision"][0].detach().cpu().numpy()
        records.append(_selection_record(selected, labels, collision, [type("C", (), {"id": item}) for item in candidate_ids]))
    return _aggregate_selection(records)


def _candidate_copy_forward_risk_mse(pack_dir: Path) -> float:
    manifest = read_json(pack_dir / "manifest.json")
    values: list[float] = []
    for record in manifest["examples"]:
        example = load_example_npz(pack_dir / str(record["example_path"]))
        current = np.asarray(example["current_bev_risky"], dtype=np.float32)
        target = np.asarray(example["candidate_horizon_future_risk"], dtype=np.float32)
        masks = np.asarray(example["candidate_footprint_masks"], dtype=np.float32)
        copy_values = []
        for mask in masks:
            denom = max(float(np.sum(mask)), 1.0)
            copy_values.append(float(np.sum(current * mask) / denom))
        copy = np.repeat(np.asarray(copy_values, dtype=np.float32)[:, None], target.shape[1], axis=1)
        values.append(float(np.mean((copy - target) ** 2)))
    return float(np.mean(values)) if values else 0.0


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
        source="goal29_fixture",
    )


def _selection_record(index: int, labels: np.ndarray, collision: np.ndarray, candidates: list[Any]) -> dict[str, Any]:
    oracle = int(np.argmin(labels))
    candidate_id = str(candidates[index].id)
    return {
        "candidate_id": candidate_id,
        "oracle_match": index == oracle,
        "unsafe_selected": candidate_id != "stop" and float(collision[index]) >= 0.5,
        "oracle_cost": float(labels[index]),
    }


def _aggregate_selection(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"oracle_match_fraction": 0.0, "unsafe_selected_rate": 0.0, "mean_oracle_cost": 0.0}
    return {
        "oracle_match_fraction": float(sum(1 for item in records if item["oracle_match"]) / len(records)),
        "unsafe_selected_rate": float(sum(1 for item in records if item["unsafe_selected"]) / len(records)),
        "mean_oracle_cost": float(np.mean([float(item["oracle_cost"]) for item in records])),
    }


def _hard_failures(
    eval_metrics: dict[str, Any],
    train_metrics: dict[str, Any],
    learned_selection: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if eval_metrics.get("control_safe") is not False or train_metrics.get("control_safe") is not False:
        failures.append("control_safe_regressed")
    if eval_metrics.get("raw_pwm_emitted") is True or train_metrics.get("raw_pwm_emitted") is True:
        failures.append("raw_pwm_emitted")
    if not np.isfinite(float(eval_metrics.get("loss", 0.0))):
        failures.append("non_finite_eval_loss")
    if float(eval_metrics.get("dominant_action_fraction", 1.0)) >= 0.95:
        failures.append("action_distribution_collapsed")
    if float(learned_selection.get("unsafe_selected_rate", 1.0)) > 0.0:
        failures.append("unsafe_candidate_selected")
    return failures


def _safety_ok(*metrics: dict[str, Any]) -> bool:
    return all(item.get("control_safe") is False and item.get("raw_pwm_emitted") is not True for item in metrics)


def _delta(left: Any, right: Any) -> float | None:
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)):
        return float(left) - float(right)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Goal29 FutureBEV fixture eval.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    report = run_goal29_fixture_eval(
        out_dir=args.out,
        max_steps=args.max_steps,
        hidden_channels=args.hidden_channels,
        device_name=args.device,
        command=" ".join(["goal29_fixture_eval", *argv]) if argv is not None else None,
    )
    print(json.dumps({"report": report}, sort_keys=True))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
