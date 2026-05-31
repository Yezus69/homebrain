from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from homebrain.eval.eval_direct_bev_student_v0 import runtime_field_leakage_passed
from homebrain.runtime.counterfactual_world_model_scorer import score_candidates_with_world_model
from homebrain.runtime.direct_bev_runtime import predict_local_bev_from_rgbd


def hazard_runtime_field_leakage_passed(metadata: dict[str, Any] | None = None) -> bool:
    forbidden = {
        "hazard_boxes",
        "hazard_masks",
        "teacher_hazard_artifacts",
        "open_vocab_detector",
        "detector",
        "target_bev_hazard",
        "future_labels",
        "future_frames",
    }
    runtime_params = set(inspect.signature(predict_local_bev_from_rgbd).parameters)
    scorer_params = set(inspect.signature(score_candidates_with_world_model).parameters)
    if (runtime_params | scorer_params) & forbidden:
        return False
    return runtime_field_leakage_passed(metadata)


def per_route_metrics(per_route: dict[str, dict[str, list[np.ndarray]]]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for route_id, values in sorted(per_route.items()):
        target = concat(values["target"])
        pred = concat(values["pred"])
        thin_target = concat(values["thin_target"])
        thin_pred = concat(values["thin_pred"])
        metrics[route_id] = {
            "hazard_auprc": average_precision(target, pred),
            "hazard_f1": f1(target, pred >= 0.5),
            "hazard_recall_thin_floor_objects": recall(thin_target, thin_pred >= 0.5),
        }
    return metrics


def append_flat(out: list[np.ndarray], values: np.ndarray, mask: np.ndarray) -> None:
    if np.any(mask):
        out.append(np.asarray(values, dtype=np.float32)[mask].reshape(-1))


def concat(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate([np.asarray(value, dtype=np.float32).reshape(-1) for value in values], axis=0)


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float32).reshape(-1) >= 0.5
    s = np.asarray(score, dtype=np.float32).reshape(-1)
    positives = int(np.count_nonzero(y))
    if positives <= 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted, dtype=np.float64)
    precision = tp / np.maximum(np.arange(1, y_sorted.size + 1, dtype=np.float64), 1.0)
    return float(np.sum(precision[y_sorted]) / positives)


def f1(target: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float32).reshape(-1) >= 0.5
    p = np.asarray(pred, dtype=np.bool_).reshape(-1)
    tp = float(np.count_nonzero(y & p))
    fp = float(np.count_nonzero(~y & p))
    fn = float(np.count_nonzero(y & ~p))
    return float((2.0 * tp) / max(2.0 * tp + fp + fn, 1.0))


def recall(target: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float32).reshape(-1) >= 0.5
    p = np.asarray(pred, dtype=np.bool_).reshape(-1)
    positives = float(np.count_nonzero(y))
    if positives <= 0.0:
        return 0.0
    return float(np.count_nonzero(y & p) / positives)


def cell_values(array: np.ndarray, cells: tuple[tuple[int, int], ...], *, default: float) -> np.ndarray:
    if not cells:
        return np.asarray([default], dtype=np.float32)
    values: list[float] = []
    for row, col in cells:
        if 0 <= row < array.shape[0] and 0 <= col < array.shape[1]:
            values.append(float(array[row, col]))
        else:
            values.append(default)
    if not values:
        values.append(default)
    return np.asarray(values, dtype=np.float32)


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))
