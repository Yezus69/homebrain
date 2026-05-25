from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from homebrain.data.spatial_dataset import write_json
from homebrain.policies.trajectory_scorer_net_v0 import TRAJECTORY_SCORER_V0_CHECKPOINT_VERSION

CALIBRATION_SCHEMA_VERSION = "homebrain.trajectory_scorer_logit_bias_calibration.v0"
LEFT_RIGHT_PAIRS: tuple[tuple[str, str], ...] = (
    ("arc_left_small", "arc_right_small"),
    ("arc_left_medium", "arc_right_medium"),
    ("rotate_left", "rotate_right"),
)


def calibrate_trajectory_scorer_bias(
    *,
    checkpoint: str | Path,
    train_predictions: str | Path,
    out_checkpoint: str | Path,
    out_report: str | Path,
    validation_predictions: str | Path | None = None,
    dominant_max: float = 0.75,
    iterations: int = 60_000,
    bias_range: float = 3.0,
    max_pair_bias_delta: float | None = None,
    pair_bias_penalty: float = 0.0,
    seed: int = 2201,
) -> dict[str, Any]:
    ids, train_logits, train_labels = _prediction_arrays(train_predictions)
    if len(ids) == 0:
        raise ValueError("prediction file has no candidate ids")
    bias, train_metrics = _fit_bias(
        train_logits,
        train_labels,
        candidate_count=len(ids),
        dominant_max=dominant_max,
        iterations=iterations,
        bias_range=bias_range,
        max_pair_bias_delta=max_pair_bias_delta,
        pair_bias_penalty=pair_bias_penalty,
        seed=seed,
        candidate_ids=ids,
    )
    val_metrics = None
    if validation_predictions is not None:
        val_ids, val_logits, val_labels = _prediction_arrays(validation_predictions)
        if val_ids != ids:
            raise ValueError("validation candidate ids do not match train predictions")
        val_metrics = _evaluate_bias(val_logits, val_labels, bias)

    report: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "checkpoint": Path(checkpoint).as_posix(),
        "out_checkpoint": Path(out_checkpoint).as_posix(),
        "train_predictions": Path(train_predictions).as_posix(),
        "validation_predictions": Path(validation_predictions).as_posix() if validation_predictions is not None else None,
        "candidate_ids": ids,
        "logit_bias_by_candidate_id": {candidate_id: float(bias[index]) for index, candidate_id in enumerate(ids)},
        "dominant_max": float(dominant_max),
        "iterations": int(iterations),
        "bias_range": float(bias_range),
        "max_pair_bias_delta": float(max_pair_bias_delta) if max_pair_bias_delta is not None else None,
        "pair_bias_penalty": float(pair_bias_penalty),
        "pair_bias_stats": _pair_bias_stats(ids, bias),
        "seed": int(seed),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "product_training_approved": False,
    }

    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if payload.get("checkpoint_version") != TRAJECTORY_SCORER_V0_CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {payload.get('checkpoint_version')!r}")
    metadata = dict(payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {})
    metadata.update(
        {
            "logit_bias_by_candidate_id": report["logit_bias_by_candidate_id"],
            "logit_bias_calibration": {
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "train_predictions": Path(train_predictions).as_posix(),
                "validation_predictions": Path(validation_predictions).as_posix()
                if validation_predictions is not None
                else None,
                "dominant_max": float(dominant_max),
                "iterations": int(iterations),
                "bias_range": float(bias_range),
                "max_pair_bias_delta": float(max_pair_bias_delta) if max_pair_bias_delta is not None else None,
                "pair_bias_penalty": float(pair_bias_penalty),
                "pair_bias_stats": report["pair_bias_stats"],
                "seed": int(seed),
                "train_metrics": train_metrics,
                "validation_metrics": val_metrics,
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
                "product_training_approved": False,
            },
        }
    )
    metrics = dict(payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {})
    metrics["logit_bias_calibration"] = metadata["logit_bias_calibration"]
    payload["metadata"] = metadata
    payload["metrics"] = metrics
    out_path = Path(out_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    write_json(out_report, report, pretty=True)
    return report


def _prediction_arrays(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"prediction file is empty: {path}")
    ids = [str(item["candidate_id"]) for item in rows[0]["candidate_scores"]]
    logits = np.zeros((len(rows), len(ids)), dtype=np.float32)
    labels = np.zeros((len(rows),), dtype=np.int64)
    for row_index, row in enumerate(rows):
        scores = row.get("candidate_scores")
        if not isinstance(scores, list) or len(scores) != len(ids):
            raise ValueError(f"candidate_scores mismatch at row {row_index}")
        label = str(row.get("expert_selected_candidate_id"))
        if label not in ids:
            raise ValueError(f"expert label {label!r} missing from candidate ids at row {row_index}")
        labels[row_index] = ids.index(label)
        for candidate_index, score in enumerate(scores):
            if str(score.get("candidate_id")) != ids[candidate_index]:
                raise ValueError(f"candidate order mismatch at row {row_index}")
            raw = score.get("raw_learned_logit", score.get("learned_logit"))
            logits[row_index, candidate_index] = float(raw)
    return ids, logits, labels


def _fit_bias(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    candidate_count: int,
    dominant_max: float,
    iterations: int,
    bias_range: float,
    max_pair_bias_delta: float | None,
    pair_bias_penalty: float,
    seed: int,
    candidate_ids: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = random.Random(seed)
    best_bias = np.zeros((candidate_count,), dtype=np.float32)
    best_metrics = _evaluate_bias(logits, labels, best_bias)
    best_pair_stats = _pair_bias_stats(candidate_ids, best_bias)
    best_score = _objective(
        best_metrics,
        dominant_max,
        pair_stats=best_pair_stats,
        pair_bias_penalty=pair_bias_penalty,
    )
    for _ in range(max(0, int(iterations))):
        bias = np.asarray([rng.uniform(-bias_range, bias_range) for _ in range(candidate_count)], dtype=np.float32)
        pair_stats = _pair_bias_stats(candidate_ids, bias)
        if max_pair_bias_delta is not None and float(pair_stats["max_abs_pair_bias_delta"]) > float(
            max_pair_bias_delta
        ):
            continue
        metrics = _evaluate_bias(logits, labels, bias)
        score = _objective(metrics, dominant_max, pair_stats=pair_stats, pair_bias_penalty=pair_bias_penalty)
        if score > best_score:
            best_score = score
            best_bias = bias
            best_metrics = metrics
    best_metrics = dict(best_metrics)
    best_metrics["objective"] = float(best_score)
    return best_bias, best_metrics


def _evaluate_bias(logits: np.ndarray, labels: np.ndarray, bias: np.ndarray) -> dict[str, Any]:
    selected = np.argmax(logits + bias[None, :], axis=1)
    counts_array = np.bincount(selected, minlength=logits.shape[1]).astype(np.int64)
    total = int(selected.shape[0])
    dominant = float(np.max(counts_array) / max(total, 1))
    entropy = 0.0
    for count in counts_array:
        if count > 0:
            probability = float(count / max(total, 1))
            entropy -= probability * float(np.log2(probability))
    return {
        "example_count": total,
        "top1_action_agreement": float(np.mean(selected == labels)) if total else 0.0,
        "dominant_action_fraction": dominant,
        "action_entropy": float(entropy),
        "selected_counts": [int(value) for value in counts_array.tolist()],
    }


def _pair_bias_stats(candidate_ids: list[str], bias: np.ndarray) -> dict[str, Any]:
    pair_deltas: dict[str, float] = {}
    abs_values: list[float] = []
    for left_id, right_id in LEFT_RIGHT_PAIRS:
        if left_id not in candidate_ids or right_id not in candidate_ids:
            continue
        delta = float(bias[candidate_ids.index(left_id)] - bias[candidate_ids.index(right_id)])
        pair_deltas[f"{left_id}_vs_{right_id}"] = delta
        abs_values.append(abs(delta))
    return {
        "pair_bias_deltas": pair_deltas,
        "max_abs_pair_bias_delta": float(max(abs_values) if abs_values else 0.0),
        "sum_abs_pair_bias_delta": float(sum(abs_values)),
    }


def _objective(
    metrics: dict[str, Any],
    dominant_max: float,
    *,
    pair_stats: dict[str, Any] | None = None,
    pair_bias_penalty: float = 0.0,
) -> float:
    agreement = float(metrics["top1_action_agreement"])
    entropy = float(metrics["action_entropy"])
    dominant = float(metrics["dominant_action_fraction"])
    pair_penalty = 0.0
    if pair_stats is not None and pair_bias_penalty > 0.0:
        pair_penalty = float(pair_bias_penalty) * float(pair_stats["sum_abs_pair_bias_delta"])
    if dominant <= dominant_max:
        return agreement + 0.02 * entropy - pair_penalty
    return agreement - 3.0 * (dominant - dominant_max) - pair_penalty


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit fixed per-candidate logit-bias calibration for a scorer checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-predictions", required=True)
    parser.add_argument("--validation-predictions", default=None)
    parser.add_argument("--out-checkpoint", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--dominant-max", type=float, default=0.75)
    parser.add_argument("--iterations", type=int, default=60_000)
    parser.add_argument("--bias-range", type=float, default=3.0)
    parser.add_argument("--max-pair-bias-delta", type=float, default=None)
    parser.add_argument("--pair-bias-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2201)
    args = parser.parse_args(argv)
    report = calibrate_trajectory_scorer_bias(
        checkpoint=args.checkpoint,
        train_predictions=args.train_predictions,
        validation_predictions=args.validation_predictions,
        out_checkpoint=args.out_checkpoint,
        out_report=args.out_report,
        dominant_max=args.dominant_max,
        iterations=args.iterations,
        bias_range=args.bias_range,
        max_pair_bias_delta=args.max_pair_bias_delta,
        pair_bias_penalty=args.pair_bias_penalty,
        seed=args.seed,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
