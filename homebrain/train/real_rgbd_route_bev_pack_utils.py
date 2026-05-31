from __future__ import annotations

import numpy as np

from homebrain.data.tum_rgbd_route import RouteFrame, pose_delta
from homebrain.teachers.rgbd_bev_teacher import RGBDBEVLabels, warp_bev_by_pose_delta


def bev_flow_label(
    *,
    labels: RGBDBEVLabels,
    previous_labels: RGBDBEVLabels | None,
    pose_delta_prev: tuple[float, float, float] | None,
    meters_per_cell: float,
) -> tuple[np.ndarray, np.ndarray]:
    shape = labels.current_bev_occupied.shape
    flow = np.zeros((2, *shape), dtype=np.float32)
    if previous_labels is None or pose_delta_prev is None:
        return flow, np.zeros(shape, dtype=np.float32)
    previous_occupied = warp_bev_by_pose_delta(
        previous_labels.current_bev_occupied,
        pose_delta_prev,
        meters_per_cell=meters_per_cell,
    )
    residual = np.abs(labels.current_bev_occupied.astype(np.float32) - previous_occupied.astype(np.float32))
    moving = (labels.dynamic_residual_risk > 0.05) | (residual > 0.5)
    previous_center = _weighted_center(previous_occupied * moving)
    current_center = _weighted_center(labels.current_bev_occupied * moving)
    if previous_center is None or current_center is None or not np.any(moving):
        return flow, moving.astype(np.float32)
    flow[0, moving] = np.float32(current_center[1] - previous_center[1])
    flow[1, moving] = np.float32(current_center[0] - previous_center[0])
    return flow, moving.astype(np.float32)


def next_pose_delta(frames: list[RouteFrame], index: int) -> tuple[float, float, float] | None:
    if index + 1 >= len(frames):
        return None
    return pose_delta(frames[index].pose, frames[index + 1].pose)


def pack_acceptance_reasons(
    *,
    route_count: int,
    val_route_ids: list[str],
    route_dynamic_frames: dict[str, int],
    train_ids: list[str],
) -> list[str]:
    reasons: list[str] = []
    if route_count < 3:
        reasons.append("real_source_route_count_lt_3")
    if not val_route_ids:
        reasons.append("no_heldout_val_route")
    if any(route_id in train_ids for route_id in val_route_ids):
        reasons.append("train_val_route_overlap")
    if sum(route_dynamic_frames.get(route_id, 0) for route_id in val_route_ids) <= 0:
        reasons.append("heldout_val_route_has_zero_dynamic_positive_frames")
    return reasons


def _weighted_center(grid: np.ndarray) -> tuple[float, float] | None:
    weight = np.clip(np.asarray(grid, dtype=np.float32), 0.0, 1.0)
    total = float(weight.sum())
    if total <= 1.0e-6:
        return None
    rows, cols = np.indices(weight.shape, dtype=np.float32)
    return (float((rows * weight).sum() / total), float((cols * weight).sum() / total))
