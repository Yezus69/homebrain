from __future__ import annotations

import argparse
import json
from math import atan2, cos, sin
from pathlib import Path
import sys
from typing import Any

import numpy as np

from homebrain.brain.modeld import write_spatial_model_outputs
from homebrain.data.spatial_dataset import write_deterministic_npz
from homebrain.data.spatial_dataset import read_json
from homebrain.eval.closed_loop_replay_report import build_closed_loop_replay_report, write_report
from homebrain.messages.schema import BrainOutputEvent, Event, FrameEvent, OdomEvent, PoseEvent
from homebrain.replay.segment_log import read_events


RUNTIME_OPENLORIS_REPLAY_SCHEMA_VERSION = "homebrain.runtime.openloris_replay.v0"


def run_openloris_runtime_replay(
    *,
    log_dir: str | Path,
    out_dir: str | Path,
    checkpoint: str | Path,
    feature_dir: str | Path | None = None,
    trajectory_scorer_checkpoint: str | Path | None = None,
    future_rollout_checkpoint: str | Path | None = None,
    device_name: str | None = None,
    v1_pose_warp_source: str = "odom",
    v1_policy_bev_source: str = "memory",
    runtime_feature_source: str = "dino",
    future_rollout_selection_mode: str = "guided_transparent",
    run_transparent_baseline: bool = True,
    allow_non_openloris: bool = False,
    command: str | None = None,
) -> dict[str, Any]:
    route = Path(log_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    route_metadata = _route_metadata(route)
    if not allow_non_openloris and route_metadata.get("source_type") != "openloris_scene":
        raise ValueError(
            f"runtime OpenLORIS replay requires route_metadata.source_type='openloris_scene', "
            f"got {route_metadata.get('source_type')!r}"
        )
    device = device_name or _preferred_device()
    modeld_out = output / "online_modeld"
    write_spatial_model_outputs(
        route,
        modeld_out,
        checkpoint=checkpoint,
        feature_dir=feature_dir,
        trajectory_scorer_checkpoint=trajectory_scorer_checkpoint,
        future_rollout_checkpoint=future_rollout_checkpoint,
        device_name=device,
        v1_pose_warp_source=v1_pose_warp_source,
        v1_policy_bev_source=v1_policy_bev_source,
        runtime_feature_source=runtime_feature_source,
        future_rollout_selection_mode=future_rollout_selection_mode,
    )
    baseline_out: Path | None = None
    baseline_reason = "disabled"
    primary_has_model_scorer = trajectory_scorer_checkpoint is not None or future_rollout_checkpoint is not None
    if run_transparent_baseline and primary_has_model_scorer:
        baseline_out = output / "transparent_baseline_modeld"
        baseline_reason = "primary_uses_model_scorer"
        write_spatial_model_outputs(
            route,
            baseline_out,
            checkpoint=checkpoint,
            feature_dir=feature_dir,
            trajectory_scorer_checkpoint=None,
            future_rollout_checkpoint=None,
            device_name=device,
            v1_pose_warp_source=v1_pose_warp_source,
            v1_policy_bev_source=v1_policy_bev_source,
            runtime_feature_source=runtime_feature_source,
            future_rollout_selection_mode="argmin",
        )
    elif run_transparent_baseline:
        baseline_reason = "primary_already_uses_transparent_scorer"

    report = build_closed_loop_replay_report(log_dir=modeld_out, compare_log_dir=baseline_out)
    route_frame_count = _metadata_int(route_metadata, "frame_count", "imported_frame_count")
    frame_event_count = int(report.get("frame_count", 0))
    if frame_event_count == 0 and route_frame_count > 0:
        report["frame_count"] = route_frame_count
    report.update(
        {
            "schema_version": RUNTIME_OPENLORIS_REPLAY_SCHEMA_VERSION,
            "route_log": route.as_posix(),
            "modeld_log": modeld_out.as_posix(),
            "transparent_baseline_enabled": bool(baseline_out is not None),
            "transparent_baseline_reason": baseline_reason,
            "transparent_baseline_modeld_log": baseline_out.as_posix() if baseline_out is not None else None,
            "frame_event_count": frame_event_count,
            "route_frame_count": route_frame_count,
            "checkpoint": Path(checkpoint).as_posix(),
            "features": Path(feature_dir).as_posix() if feature_dir is not None else None,
            "trajectory_scorer_checkpoint": Path(trajectory_scorer_checkpoint).as_posix()
            if trajectory_scorer_checkpoint is not None
            else None,
            "future_rollout_checkpoint": Path(future_rollout_checkpoint).as_posix()
            if future_rollout_checkpoint is not None
            else None,
            "runtime_feature_source": runtime_feature_source,
            "future_rollout_selection_mode": future_rollout_selection_mode,
            "requires_precomputed_dino_runtime": runtime_feature_source == "dino",
            "direct_rgbd_runtime_path": runtime_feature_source == "direct_rgbd",
            "runtime_api": "Brain.step",
            "runtime_api_step_count": int(report.get("decision_count", 0)),
            "latency_step_p50_ms": report.get("latency_end_to_end_p50_ms"),
            "latency_step_p95_ms": report.get("latency_end_to_end_p95_ms"),
            "memory_update_latency_p95_ms": report.get("latency_memory_update_p95_ms"),
            "action_entropy": report.get("selected_candidate_entropy"),
            "dominant_action_fraction": report.get("selected_candidate_dominant_fraction"),
            "teacher_runtime_dependency": runtime_feature_source == "dino",
            "future_or_groundtruth_runtime_dependency": float(
                report.get("route_pose_leakage_ablation_fraction", 0.0)
            )
            > 0.0,
            "device": device,
            "gpu_inventory": _gpu_inventory(),
            "online_replay_as_live": True,
            "spatial_memory_online": True,
            "cmd_vel_proposal_only": True,
            "cmd_vel_executed": False,
            "raw_pwm_emitted": False,
            "route_metadata": {
                "source_type": route_metadata.get("source_type"),
                "sequence": route_metadata.get("sequence"),
                "scene": route_metadata.get("scene"),
                "robot_frame_truth": route_metadata.get("robot_frame_truth"),
                "has_depth": route_metadata.get("has_depth"),
                "has_groundtruth_pose": route_metadata.get("has_groundtruth_pose"),
                "has_odometry": route_metadata.get("has_odometry"),
                "has_camera_to_base_transform": route_metadata.get("has_camera_to_base_transform"),
                "control_safe": route_metadata.get("control_safe"),
                "license_name": route_metadata.get("license_name"),
                "license_review_status": route_metadata.get("license_review_status"),
            },
            "runtime_command": command,
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "product_training_approved": False,
        }
    )
    scene_memory = write_runtime_scene_memory_artifacts(
        route_dir=route,
        modeld_log_dir=modeld_out,
        out_dir=output / "scene_memory",
        name=route_metadata.get("sequence") or route.name,
    )
    report.update(
        {
            "scene_memory": scene_memory,
            "scene_memory_artifact": scene_memory.get("scene_memory_npz"),
            "scene_pose_trace_artifact": scene_memory.get("scene_pose_trace_artifact"),
            "scene_memory_visual": scene_memory.get("scene_memory_visual"),
            "current_local_bev_artifact": scene_memory.get("current_local_bev_artifact"),
            "scene_memory_used_for_policy": str(report.get("policy_bev_source")) == "memory",
            "scene_bev_free_nonzero_fraction": scene_memory.get("scene_bev_free_nonzero_fraction"),
            "scene_bev_occupied_nonzero_fraction": scene_memory.get("scene_bev_occupied_nonzero_fraction"),
            "coverage_memory_cells_seen": scene_memory.get("coverage_memory_cells_seen"),
            "future_prediction_horizon_count": scene_memory.get("future_prediction_horizon_count"),
            "observed_cell_ratio": scene_memory.get("observed_cell_ratio"),
            "scene_unknown_mean": scene_memory.get("scene_unknown_mean"),
            "current_unknown_mean": scene_memory.get("current_unknown_mean"),
            "unknown_reduction_vs_current": scene_memory.get("unknown_reduction_vs_current"),
            "pose_ate_rmse_m_or_proxy": scene_memory.get("pose_ate_rmse_m_or_proxy"),
            "pose_rpe_translation_rmse_m_or_proxy": scene_memory.get("pose_rpe_translation_rmse_m_or_proxy"),
            "pose_metric_proxy": scene_memory.get("pose_metric_proxy"),
            "pose_metric_source": scene_memory.get("pose_metric_source"),
            "pose_metric_independent_groundtruth": scene_memory.get("pose_metric_independent_groundtruth"),
        }
    )
    write_report(
        report,
        out_json=output / "runtime_report.json",
        out_md=output / "runtime_report.md",
    )
    return report


def write_runtime_scene_memory_artifacts(
    *,
    route_dir: str | Path,
    modeld_log_dir: str | Path,
    out_dir: str | Path,
    name: object,
) -> dict[str, Any]:
    route = Path(route_dir)
    modeld = Path(modeld_log_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    route_events = read_events(route)
    model_events = read_events(modeld)
    outputs = [event for event in model_events if isinstance(event, BrainOutputEvent)]
    records = _scene_records(modeld, outputs)
    if not records:
        empty = output / "scene_memory_empty.npz"
        write_deterministic_npz(empty, {"empty": np.asarray([1], dtype=np.int64)})
        report = {
            "schema_version": "homebrain.runtime.scene_memory.v0",
            "route": route.as_posix(),
            "modeld_log": modeld.as_posix(),
            "scene_memory_npz": empty.as_posix(),
            "step_count": 0,
            "control_safe": False,
            "replay_only": True,
        }
        _write_json(output / "scene_memory_report.json", report)
        return report

    meters_per_cell = _meters_per_cell(outputs)
    height, width = records[-1]["memory_free"].shape
    poses = np.asarray([record["pose"] for record in records], dtype=np.float32)
    x_min, y_min, grid_h, grid_w = _scene_grid_bounds(
        poses=poses,
        local_shape=(height, width),
        meters_per_cell=meters_per_cell,
    )
    sums = {
        "free": np.zeros((grid_h, grid_w), dtype=np.float32),
        "occupied": np.zeros((grid_h, grid_w), dtype=np.float32),
        "unknown": np.zeros((grid_h, grid_w), dtype=np.float32),
        "traversable": np.zeros((grid_h, grid_w), dtype=np.float32),
        "risky": np.zeros((grid_h, grid_w), dtype=np.float32),
        "uncertainty": np.zeros((grid_h, grid_w), dtype=np.float32),
        "seen": np.zeros((grid_h, grid_w), dtype=np.float32),
        "future_free": np.zeros((grid_h, grid_w), dtype=np.float32),
        "future_unknown": np.zeros((grid_h, grid_w), dtype=np.float32),
        "future_occupied": np.zeros((grid_h, grid_w), dtype=np.float32),
    }
    counts = np.zeros((grid_h, grid_w), dtype=np.float32)
    future_counts = np.zeros((grid_h, grid_w), dtype=np.float32)
    selected_overlay = np.zeros((grid_h, grid_w), dtype=np.float32)
    pose_overlay = np.zeros((grid_h, grid_w), dtype=np.float32)

    last_future = None
    last_current = None
    last_pose = None
    for record in records:
        output_event = record["event"]
        pose = record["pose"]
        grid_rows, grid_cols = _local_to_scene_indices(
            local_shape=(height, width),
            pose=pose,
            meters_per_cell=meters_per_cell,
            x_min=x_min,
            y_min=y_min,
            scene_shape=(grid_h, grid_w),
        )
        valid = (grid_rows >= 0) & (grid_rows < grid_h) & (grid_cols >= 0) & (grid_cols < grid_w)
        indices = (grid_rows[valid], grid_cols[valid])
        for key in ("free", "occupied", "unknown", "traversable", "risky"):
            np.add.at(sums[key], indices, record[f"memory_{key}"][valid])
        np.add.at(sums["uncertainty"], indices, record["uncertainty"][valid])
        np.add.at(sums["seen"], indices, (record["memory_unknown"][valid] < 0.75).astype(np.float32))
        np.add.at(counts, indices, 1.0)

        future = record.get("future_bev")
        if future is not None:
            latest = np.asarray(future[-1], dtype=np.float32)
            np.add.at(sums["future_free"], indices, latest[0][valid])
            np.add.at(sums["future_unknown"], indices, latest[2][valid])
            np.add.at(sums["future_occupied"], indices, latest[1][valid])
            np.add.at(future_counts, indices, 1.0)
            last_future = future
        last_current = record["current_stack"]
        last_pose = pose
        _add_selected_overlay(
            selected_overlay,
            output_event,
            pose=pose,
            meters_per_cell=meters_per_cell,
            x_min=x_min,
            y_min=y_min,
        )
        _add_pose_overlay(
            pose_overlay,
            pose=pose,
            meters_per_cell=meters_per_cell,
            x_min=x_min,
            y_min=y_min,
        )

    denom = np.maximum(counts, np.float32(1.0))
    scene = {
        key: (value / denom).astype(np.float32)
        for key, value in sums.items()
        if key not in {"future_free", "future_unknown", "future_occupied"}
    }
    scene["seen"] = np.clip(scene["seen"], 0.0, 1.0)
    future_denom = np.maximum(future_counts, np.float32(1.0))
    future_free = (sums["future_free"] / future_denom).astype(np.float32)
    future_unknown = (sums["future_unknown"] / future_denom).astype(np.float32)
    future_occupied = (sums["future_occupied"] / future_denom).astype(np.float32)
    observed_cell_ratio = float(np.count_nonzero(counts > 0.0) / max(counts.size, 1))
    scene_unknown_values = scene["unknown"][counts > 0.0]
    scene_unknown_mean = float(np.mean(scene_unknown_values)) if scene_unknown_values.size else 0.0
    current_only_unknown = np.ones((grid_h, grid_w), dtype=np.float32)
    if last_current is not None and last_pose is not None:
        grid_rows, grid_cols = _local_to_scene_indices(
            local_shape=(height, width),
            pose=last_pose,
            meters_per_cell=meters_per_cell,
            x_min=x_min,
            y_min=y_min,
            scene_shape=(grid_h, grid_w),
        )
        valid = (grid_rows >= 0) & (grid_rows < grid_h) & (grid_cols >= 0) & (grid_cols < grid_w)
        current_only_unknown[grid_rows[valid], grid_cols[valid]] = last_current[2][valid]
    current_unknown_values = current_only_unknown[counts > 0.0]
    current_unknown_mean = float(np.mean(current_unknown_values)) if current_unknown_values.size else 0.0
    pose_metrics = _pose_metrics(route_events, outputs)
    safe_name = _safe_id(str(name))
    npz_path = output / f"{safe_name}_scene_memory.npz"
    pose_trace_path = output / f"{safe_name}_pose_trace.npz"
    arrays = {
        "scene_bev_free_prob": scene["free"].astype(np.float32),
        "scene_bev_occupied_prob": scene["occupied"].astype(np.float32),
        "scene_bev_unknown_prob": scene["unknown"].astype(np.float32),
        "scene_bev_traversable_prob": scene["traversable"].astype(np.float32),
        "scene_bev_risky_prob": scene["risky"].astype(np.float32),
        "uncertainty_unknown_map": scene["uncertainty"].astype(np.float32),
        "coverage_seen_map": scene["seen"].astype(np.float32),
        "current_local_bev": last_current.astype(np.float32) if last_current is not None else np.zeros((5, height, width), dtype=np.float32),
        "robot_pose_trace": poses.astype(np.float32),
        "selected_candidate_trajectory_overlay": np.clip(selected_overlay, 0.0, 1.0).astype(np.float32),
        "predicted_future_free_overlay": future_free.astype(np.float32),
        "predicted_future_unknown_overlay": future_unknown.astype(np.float32),
        "predicted_future_occupied_overlay": future_occupied.astype(np.float32),
        "predicted_future_bev": last_future.astype(np.float32)
        if last_future is not None
        else np.zeros((0, 5, height, width), dtype=np.float32),
        "pose_trace_overlay": np.clip(pose_overlay, 0.0, 1.0).astype(np.float32),
        "scene_grid_origin_xy_m": np.asarray([x_min, y_min], dtype=np.float32),
        "meters_per_cell": np.asarray([meters_per_cell], dtype=np.float32),
    }
    write_deterministic_npz(npz_path, arrays)
    write_deterministic_npz(
        pose_trace_path,
        {
            "robot_pose_trace": poses.astype(np.float32),
            "pose_trace_overlay": np.clip(pose_overlay, 0.0, 1.0).astype(np.float32),
            "scene_grid_origin_xy_m": np.asarray([x_min, y_min], dtype=np.float32),
            "meters_per_cell": np.asarray([meters_per_cell], dtype=np.float32),
        },
    )
    visual = _scene_memory_contact_sheet(
        scene=scene,
        current=last_current,
        future_free=future_free,
        future_unknown=future_unknown,
        future_occupied=future_occupied,
        selected_overlay=np.clip(selected_overlay, 0.0, 1.0),
        pose_overlay=np.clip(pose_overlay, 0.0, 1.0),
    )
    ppm_path = output / f"{safe_name}_scene_memory.ppm"
    _write_ppm(ppm_path, visual)
    report = {
        "schema_version": "homebrain.runtime.scene_memory.v0",
        "route": route.as_posix(),
        "modeld_log": modeld.as_posix(),
        "scene_memory_npz": npz_path.as_posix(),
        "scene_pose_trace_artifact": pose_trace_path.as_posix(),
        "scene_memory_visual": ppm_path.as_posix(),
        "current_local_bev_artifact": outputs[-1].local_bev_ref,
        "step_count": len(records),
        "runtime_api_step_count": len(records),
        "meters_per_cell": meters_per_cell,
        "scene_grid_shape": [grid_h, grid_w],
        "pose_trace_count": int(poses.shape[0]),
        "observed_cell_ratio": observed_cell_ratio,
        "scene_bev_free_nonzero_fraction": float(np.count_nonzero(scene["free"] > 0.05) / max(scene["free"].size, 1)),
        "scene_bev_occupied_nonzero_fraction": float(
            np.count_nonzero(scene["occupied"] > 0.05) / max(scene["occupied"].size, 1)
        ),
        "current_unknown_mean": current_unknown_mean,
        "scene_unknown_mean": scene_unknown_mean,
        "unknown_reduction_vs_current": float(current_unknown_mean - scene_unknown_mean),
        "coverage_memory_cells_seen": int(np.count_nonzero(scene["seen"] > 0.0)),
        "selected_candidate_overlay_nonzero_count": int(np.count_nonzero(selected_overlay > 0.0)),
        "predicted_future_overlay_available": bool(last_future is not None),
        "future_prediction_horizon_count": int(last_future.shape[0]) if last_future is not None else 0,
        "frame_count_with_future_prediction": int(np.count_nonzero(future_counts > 0.0)),
        "frame_count_with_scene_memory": len(records),
        "teacher_runtime_dependency": any(
            isinstance(event.debug, dict) and event.debug.get("teacher_runtime_dependency") is True for event in outputs
        ),
        "future_or_groundtruth_runtime_dependency": any(
            isinstance(event.debug, dict)
            and event.debug.get("future_or_groundtruth_runtime_dependency") is True
            for event in outputs
        ),
        "replay_only": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        **pose_metrics,
    }
    _write_json(output / "scene_memory_report.json", report)
    return report


def _route_metadata(route: Path) -> dict[str, Any]:
    path = route / "route_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"missing route metadata: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"route metadata must be a JSON object: {path}")
    return data


def _frame_lookup(events: list[Event]) -> dict[tuple[str, str, int], FrameEvent]:
    return {
        (event.sequence_id, event.camera_id, int(event.frame_id)): event
        for event in events
        if isinstance(event, FrameEvent)
    }


def _scene_records(modeld: Path, outputs: list[BrainOutputEvent]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for event in outputs:
        if event.local_bev_ref is None or not isinstance(event.debug, dict):
            continue
        artifact_path = modeld / event.local_bev_ref
        if not artifact_path.exists():
            continue
        pose = event.debug.get("scene_pose_estimate")
        if not isinstance(pose, dict):
            continue
        try:
            pose_tuple = (
                float(pose.get("x_m", 0.0)),
                float(pose.get("y_m", 0.0)),
                float(pose.get("yaw_rad", 0.0)),
            )
        except (TypeError, ValueError):
            continue
        with np.load(artifact_path, allow_pickle=False) as data:
            memory_free = _array_or_zeros(data, "memory_bev_free_prob")
            memory_occupied = _array_or_zeros(data, "memory_bev_occupied_prob", shape=memory_free.shape)
            memory_unknown = _array_or_zeros(data, "memory_bev_unknown_prob", shape=memory_free.shape)
            memory_traversable = _array_or_zeros(data, "memory_bev_traversable_prob", shape=memory_free.shape)
            memory_risky = _array_or_zeros(data, "memory_bev_risky_prob", shape=memory_free.shape)
            current_stack = np.stack(
                [
                    _array_or_zeros(data, "current_bev_free_prob", shape=memory_free.shape),
                    _array_or_zeros(data, "current_bev_occupied_prob", shape=memory_free.shape),
                    _array_or_zeros(data, "current_bev_unknown_prob", shape=memory_free.shape),
                    _array_or_zeros(data, "current_bev_traversable_prob", shape=memory_free.shape),
                    _array_or_zeros(data, "current_bev_risky_prob", shape=memory_free.shape),
                ],
                axis=0,
            ).astype(np.float32)
            future = np.asarray(data["future_rollout_future_bev_prob"], dtype=np.float32) if "future_rollout_future_bev_prob" in data else None
            records.append(
                {
                    "event": event,
                    "pose": pose_tuple,
                    "memory_free": memory_free,
                    "memory_occupied": memory_occupied,
                    "memory_unknown": memory_unknown,
                    "memory_traversable": memory_traversable,
                    "memory_risky": memory_risky,
                    "uncertainty": _array_or_zeros(data, "uncertainty_grid", shape=memory_free.shape),
                    "current_stack": current_stack,
                    "future_bev": future,
                }
            )
    return records


def _array_or_zeros(data: Any, name: str, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    if name in data:
        return np.asarray(data[name], dtype=np.float32)
    if shape is None:
        shape = (1, 1)
    return np.zeros(shape, dtype=np.float32)


def _meters_per_cell(outputs: list[BrainOutputEvent]) -> float:
    for event in outputs:
        if not isinstance(event.debug, dict):
            continue
        coverage = event.debug.get("coverage_memory")
        if isinstance(coverage, dict) and isinstance(coverage.get("meters_per_cell"), (int, float)):
            value = float(coverage["meters_per_cell"])
            if value > 0.0:
                return value
    return 0.05


def _scene_grid_bounds(
    *,
    poses: np.ndarray,
    local_shape: tuple[int, int],
    meters_per_cell: float,
) -> tuple[float, float, int, int]:
    height, width = local_shape
    forward_m = height * meters_per_cell
    lateral_m = width * meters_per_cell / 2.0
    xs = poses[:, 0] if poses.size else np.asarray([0.0], dtype=np.float32)
    ys = poses[:, 1] if poses.size else np.asarray([0.0], dtype=np.float32)
    margin = max(forward_m, lateral_m) + 0.5
    x_min = float(np.floor((float(np.min(xs)) - margin) / meters_per_cell) * meters_per_cell)
    x_max = float(np.ceil((float(np.max(xs)) + margin) / meters_per_cell) * meters_per_cell)
    y_min = float(np.floor((float(np.min(ys)) - margin) / meters_per_cell) * meters_per_cell)
    y_max = float(np.ceil((float(np.max(ys)) + margin) / meters_per_cell) * meters_per_cell)
    grid_h = max(4, int(np.ceil((x_max - x_min) / meters_per_cell)) + 1)
    grid_w = max(4, int(np.ceil((y_max - y_min) / meters_per_cell)) + 1)
    return x_min, y_min, grid_h, grid_w


def _local_to_scene_indices(
    *,
    local_shape: tuple[int, int],
    pose: tuple[float, float, float],
    meters_per_cell: float,
    x_min: float,
    y_min: float,
    scene_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = local_shape
    rows, cols = np.indices((height, width), dtype=np.float32)
    forward = (np.float32(height - 1) - rows + np.float32(0.5)) * np.float32(meters_per_cell)
    left = (cols - np.float32(width) / np.float32(2.0) + np.float32(0.5)) * np.float32(meters_per_cell)
    x_m, y_m, yaw = pose
    world_x = np.float32(x_m) + np.float32(cos(yaw)) * forward - np.float32(sin(yaw)) * left
    world_y = np.float32(y_m) + np.float32(sin(yaw)) * forward + np.float32(cos(yaw)) * left
    grid_rows = np.rint((world_x - np.float32(x_min)) / np.float32(meters_per_cell)).astype(np.int64)
    grid_cols = np.rint((world_y - np.float32(y_min)) / np.float32(meters_per_cell)).astype(np.int64)
    _ = scene_shape
    return grid_rows, grid_cols


def _add_selected_overlay(
    overlay: np.ndarray,
    event: BrainOutputEvent,
    *,
    pose: tuple[float, float, float],
    meters_per_cell: float,
    x_min: float,
    y_min: float,
) -> None:
    selected = event.selected_trajectory_id
    candidate = None
    for item in event.candidate_trajectories or []:
        if str(item.get("id", item.get("trajectory_id", ""))) == selected:
            candidate = item
            break
    if not isinstance(candidate, dict):
        return
    cells = candidate.get("footprint_cells")
    if not isinstance(cells, list):
        return
    local_shape = _candidate_local_shape(event)
    if local_shape is None:
        return
    height, width = local_shape
    for cell in cells:
        if not isinstance(cell, list) or len(cell) != 2:
            continue
        row, col = int(cell[0]), int(cell[1])
        if row < 0 or col < 0 or row >= height or col >= width:
            continue
        grid_rows, grid_cols = _local_to_scene_indices(
            local_shape=(height, width),
            pose=pose,
            meters_per_cell=meters_per_cell,
            x_min=x_min,
            y_min=y_min,
            scene_shape=overlay.shape,
        )
        scene_row = int(grid_rows[row, col])
        scene_col = int(grid_cols[row, col])
        if 0 <= scene_row < overlay.shape[0] and 0 <= scene_col < overlay.shape[1]:
            overlay[scene_row, scene_col] = 1.0


def _candidate_local_shape(event: BrainOutputEvent) -> tuple[int, int] | None:
    if not isinstance(event.debug, dict):
        return None
    value = event.debug.get("local_bev_shape")
    if isinstance(value, list) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return None


def _add_pose_overlay(
    overlay: np.ndarray,
    *,
    pose: tuple[float, float, float],
    meters_per_cell: float,
    x_min: float,
    y_min: float,
) -> None:
    row = int(round((pose[0] - x_min) / meters_per_cell))
    col = int(round((pose[1] - y_min) / meters_per_cell))
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            rr = row + dr
            cc = col + dc
            if 0 <= rr < overlay.shape[0] and 0 <= cc < overlay.shape[1]:
                overlay[rr, cc] = 1.0


def _pose_metrics(route_events: list[Event], outputs: list[BrainOutputEvent]) -> dict[str, Any]:
    pose_samples = {
        (event.sequence_id, event.timestamp_ns): event
        for event in route_events
        if isinstance(event, PoseEvent)
    }
    pose_sample_kind = "groundtruth_pose"
    pose_metric_independent_groundtruth = True
    if not pose_samples:
        pose_samples = {
            (event.sequence_id, event.timestamp_ns): event
            for event in route_events
            if isinstance(event, OdomEvent)
        }
        pose_sample_kind = "runtime_odometry_proxy"
        pose_metric_independent_groundtruth = False
    matched_estimates: list[tuple[float, float, float]] = []
    matched_truth: list[tuple[float, float, float]] = []
    for event in outputs:
        if not isinstance(event.debug, dict):
            continue
        estimate = event.debug.get("scene_pose_estimate")
        truth = _nearest_pose_sample(pose_samples, event.sequence_id, event.timestamp_ns)
        if not isinstance(estimate, dict) or truth is None:
            continue
        matched_estimates.append(
            (
                float(estimate.get("x_m", 0.0)),
                float(estimate.get("y_m", 0.0)),
                float(estimate.get("yaw_rad", 0.0)),
            )
        )
        matched_truth.append(
            (
                float(truth.position_m[0]),
                float(truth.position_m[1]),
                _yaw_from_quaternion_xyzw(truth.orientation_xyzw),
            )
        )
    if len(matched_estimates) < 2:
        return {
            "pose_metric_available": False,
            "pose_metric_proxy": "pose_trace_available_no_groundtruth_match",
            "pose_ate_rmse_m_or_proxy": 0.0,
            "pose_rpe_translation_rmse_m_or_proxy": 0.0,
            "pose_label_match_count": len(matched_estimates),
            "pose_metric_source": pose_sample_kind,
            "pose_metric_independent_groundtruth": pose_metric_independent_groundtruth,
        }
    est = np.asarray(matched_estimates, dtype=np.float32)
    truth = np.asarray(matched_truth, dtype=np.float32)
    est_rel = _relative_pose_trace(est)
    truth_rel = _relative_pose_trace(truth)
    ate = np.sqrt(np.sum((est_rel[:, :2] - truth_rel[:, :2]) ** 2, axis=1))
    est_step = np.diff(est_rel[:, :2], axis=0)
    truth_step = np.diff(truth_rel[:, :2], axis=0)
    rpe = np.sqrt(np.sum((est_step - truth_step) ** 2, axis=1))
    return {
        "pose_metric_available": True,
        "pose_metric_proxy": "scene_pose_trace_vs_pose_source_relative_trace_eval_only",
        "pose_metric_source": pose_sample_kind,
        "pose_metric_independent_groundtruth": pose_metric_independent_groundtruth,
        "pose_ate_rmse_m_or_proxy": float(np.sqrt(np.mean(ate**2))),
        "pose_rpe_translation_rmse_m_or_proxy": float(np.sqrt(np.mean(rpe**2))),
        "pose_label_match_count": int(len(matched_estimates)),
    }


def _nearest_pose_sample(
    samples: dict[tuple[str, int], PoseEvent | OdomEvent],
    sequence_id: str,
    timestamp_ns: int,
) -> PoseEvent | OdomEvent | None:
    exact = samples.get((sequence_id, timestamp_ns))
    if exact is not None:
        return exact
    candidates = [
        (abs(sample_timestamp - timestamp_ns), sample)
        for (sample_sequence, sample_timestamp), sample in samples.items()
        if sample_sequence == sequence_id
    ]
    if not candidates:
        return None
    distance, sample = min(candidates, key=lambda item: item[0])
    return sample if distance <= 100_000_000 else None


def _relative_pose_trace(trace: np.ndarray) -> np.ndarray:
    anchor = trace[0]
    yaw0 = float(anchor[2])
    result = np.zeros_like(trace, dtype=np.float32)
    for index, pose in enumerate(trace):
        dx = float(pose[0] - anchor[0])
        dy = float(pose[1] - anchor[1])
        result[index, 0] = cos(yaw0) * dx + sin(yaw0) * dy
        result[index, 1] = -sin(yaw0) * dx + cos(yaw0) * dy
        result[index, 2] = _wrap_angle(float(pose[2] - yaw0))
    return result


def _scene_memory_contact_sheet(
    *,
    scene: dict[str, np.ndarray],
    current: np.ndarray | None,
    future_free: np.ndarray,
    future_unknown: np.ndarray,
    future_occupied: np.ndarray,
    selected_overlay: np.ndarray,
    pose_overlay: np.ndarray,
) -> np.ndarray:
    scene_rgb = _bev_rgb(
        free=scene["free"],
        occupied=scene["occupied"],
        unknown=scene["unknown"],
        risky=scene["risky"],
        selected=selected_overlay,
        pose=pose_overlay,
        size=(256, 256),
    )
    if current is None:
        current_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    else:
        current_rgb = _bev_rgb(
            free=current[0],
            occupied=current[1],
            unknown=current[2],
            risky=current[4],
            selected=None,
            pose=None,
            size=(256, 256),
        )
    future_rgb = _bev_rgb(
        free=future_free,
        occupied=future_occupied,
        unknown=future_unknown,
        risky=future_occupied,
        selected=selected_overlay,
        pose=pose_overlay,
        size=(256, 256),
    )
    seen_rgb = _gray_rgb(scene["seen"], size=(256, 256))
    return np.concatenate(
        [
            np.concatenate([scene_rgb, current_rgb], axis=1),
            np.concatenate([future_rgb, seen_rgb], axis=1),
        ],
        axis=0,
    )


def _bev_rgb(
    *,
    free: np.ndarray,
    occupied: np.ndarray,
    unknown: np.ndarray,
    risky: np.ndarray,
    selected: np.ndarray | None,
    pose: np.ndarray | None,
    size: tuple[int, int],
) -> np.ndarray:
    image = np.zeros((*free.shape, 3), dtype=np.uint8)
    image[:] = np.asarray([42, 42, 42], dtype=np.uint8)
    image[unknown >= 0.5] = np.asarray([70, 70, 70], dtype=np.uint8)
    image[free >= 0.5] = np.asarray([60, 170, 105], dtype=np.uint8)
    image[occupied >= 0.5] = np.asarray([210, 70, 55], dtype=np.uint8)
    image[risky >= 0.5] = np.asarray([205, 60, 150], dtype=np.uint8)
    if selected is not None:
        image[selected > 0.0] = np.asarray([60, 130, 230], dtype=np.uint8)
    if pose is not None:
        image[pose > 0.0] = np.asarray([245, 220, 70], dtype=np.uint8)
    return _resize_rgb_nearest(image, size)


def _gray_rgb(array: np.ndarray, *, size: tuple[int, int]) -> np.ndarray:
    clipped = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    image = (clipped[..., None] * np.asarray([180, 210, 230], dtype=np.float32)).astype(np.uint8)
    return _resize_rgb_nearest(image, size)


def _resize_rgb_nearest(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    rows = np.linspace(0, image.shape[0] - 1, size[1]).round().astype(np.int64)
    cols = np.linspace(0, image.shape[1] - 1, size[0]).round().astype(np.int64)
    return image[rows[:, None], cols[None, :]].astype(np.uint8)


def _write_ppm(path: Path, image: np.ndarray) -> None:
    array = np.asarray(image, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{array.shape[1]} {array.shape[0]}\n255\n".encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(array.tobytes())


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, sort_keys=True, indent=2)
        handle.write("\n")


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower() or "route"


def _yaw_from_quaternion_xyzw(quaternion: tuple[float, float, float, float]) -> float:
    qx, qy, qz, qw = (float(value) for value in quaternion)
    return atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _wrap_angle(value: float) -> float:
    while value > np.pi:
        value -= 2.0 * float(np.pi)
    while value < -np.pi:
        value += 2.0 * float(np.pi)
    return float(value)


def _metadata_int(metadata: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return 0


def _preferred_device() -> str:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    inventory = _gpu_inventory()
    for item in inventory:
        if "4090" in str(item.get("name", "")):
            return f"cuda:{item['index']}"
    return "cuda:0"


def _gpu_inventory() -> list[dict[str, Any]]:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return []
    if not torch.cuda.is_available():
        return []
    return [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "memory_gb": round(torch.cuda.get_device_properties(index).total_memory / (1024.0**3), 3),
        }
        for index in range(torch.cuda.device_count())
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OpenLORIS replay through the online HomeBrain runtime path.")
    parser.add_argument("--log", required=True, help="Input OpenLORIS HomeBrain route log.")
    parser.add_argument("--out", required=True, help="Output runtime replay directory.")
    parser.add_argument("--checkpoint", required=True, help="SpatialMemoryNet checkpoint.")
    parser.add_argument("--features", default=None, help="DINO feature artifact directory.")
    parser.add_argument("--trajectory-scorer-checkpoint", default=None)
    parser.add_argument("--future-rollout-checkpoint", default=None)
    parser.add_argument("--device", default=None, help="Torch device. Defaults to the first RTX 4090 when available.")
    parser.add_argument(
        "--v1-pose-warp-source",
        choices=("odom", "odom_or_route_pose", "route_pose", "route_pose_ablation", "predicted_pose", "none"),
        default="odom",
    )
    parser.add_argument("--v1-policy-bev-source", choices=("current", "memory"), default="memory")
    parser.add_argument(
        "--runtime-feature-source",
        choices=("dino", "direct_rgbd"),
        default="dino",
        help="Use precomputed DINO features or direct current RGB-D runtime features.",
    )
    parser.add_argument(
        "--future-rollout-selection-mode",
        choices=("argmin", "guided_transparent"),
        default="guided_transparent",
    )
    parser.add_argument(
        "--skip-transparent-baseline",
        action="store_true",
        help="Do not run the transparent heuristic baseline comparison pass.",
    )
    parser.add_argument("--allow-non-openloris", action="store_true")
    args = parser.parse_args(argv)
    command = "python -m homebrain.runtime.replay_openloris_brain " + " ".join(sys.argv[1:])
    report = run_openloris_runtime_replay(
        log_dir=args.log,
        out_dir=args.out,
        checkpoint=args.checkpoint,
        feature_dir=args.features,
        trajectory_scorer_checkpoint=args.trajectory_scorer_checkpoint,
        future_rollout_checkpoint=args.future_rollout_checkpoint,
        device_name=args.device,
        v1_pose_warp_source=args.v1_pose_warp_source,
        v1_policy_bev_source=args.v1_policy_bev_source,
        runtime_feature_source=args.runtime_feature_source,
        future_rollout_selection_mode=args.future_rollout_selection_mode,
        run_transparent_baseline=not args.skip_transparent_baseline,
        allow_non_openloris=args.allow_non_openloris,
        command=command,
    )
    print(
        json.dumps(
            {
                "runtime_report": (Path(args.out) / "runtime_report.json").as_posix(),
                "decision_count": report.get("decision_count"),
                "comparison": report.get("comparison"),
                "latency_end_to_end_p95_ms": report.get("latency_end_to_end_p95_ms"),
                "cmd_vel_proposal_count": report.get("cmd_vel_proposal_count"),
                "control_safe": report.get("control_safe"),
            },
            sort_keys=True,
        )
    )
    return 0 if int(report.get("decision_count", 0)) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
