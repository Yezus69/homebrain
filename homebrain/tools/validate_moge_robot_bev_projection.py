from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import itertools
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np

from homebrain.artifacts.io import (
    load_array_artifact_optional,
    load_array_artifact_required,
    read_json_object,
    write_json_object,
)
from homebrain.data.spatial_io import (
    load_spatial_example_arrays,
    scene_frames_by_id,
    spatial_examples_by_id,
)
from homebrain.datasets.openloris_scene import OPENLORIS_ROUTE_ASSOCIATIONS_FILE
from homebrain.datasets.tum_rgbd import TUM_RGBD_ROUTE_ASSOCIATIONS_FILE
from homebrain.geometry.bev_projector import robot_points_to_bev_arrays
from homebrain.messages.schema import JsonDict, deterministic_json
from homebrain.teachers.scene_teacher import load_scene_teacher_manifest
from homebrain.visualization.panels import join_with_gap, resize_nearest, write_ppm

SCHEMA_VERSION = "homebrain.moge_robot_bev_projection_gate.v0"
AGGREGATE_SCHEMA_VERSION = "homebrain.moge_robot_bev_projection_gate_aggregate.v0"
DEFAULT_GRID_CELLS = 32
DEFAULT_METERS_PER_CELL = 0.05
DEFAULT_FLOOR_HEIGHT_TOL_M = 0.08
DEFAULT_OBSTACLE_HEIGHT_MIN_M = 0.08
DEFAULT_OBSTACLE_DILATION_CELLS = 1
NEXT_REVIEW_ONLY = "review_only_do_not_train"
NEXT_LOCAL_REVIEW = "local_replay_moge_bev_candidate_review"
SAFETY_FLAGS: JsonDict = {
    "replay_only": True,
    "not_executed": True,
    "control_safe": False,
    "product_training_approved": False,
    "action_supervision_ok": False,
    "raw_pwm_emitted": False,
    "cmd_vel_emitted": False,
    "moge_robot_frame_truth": False,
}
AXIS_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class ConventionCandidate:
    name: str
    permutation: tuple[int, int, int]
    signs: tuple[int, int, int]

    def to_json(self) -> JsonDict:
        return {
            "name": self.name,
            "mapping": {
                "camera_x": _signed_axis(self.signs[0], self.permutation[0]),
                "camera_y": _signed_axis(self.signs[1], self.permutation[1]),
                "camera_z": _signed_axis(self.signs[2], self.permutation[2]),
            },
            "permutation": [int(value) for value in self.permutation],
            "signs": [int(value) for value in self.signs],
        }


@dataclass
class MetricAccumulator:
    frame_count: int = 0
    cell_count: int = 0
    source_sample_count: int = 0
    source_valid_point_count: int = 0
    free_intersection: int = 0
    free_union: int = 0
    obstacle_intersection: int = 0
    obstacle_union: int = 0
    unknown_intersection: int = 0
    unknown_union: int = 0
    true_obstacle_count: int = 0
    true_free_count: int = 0
    predicted_obstacle_on_true_obstacle: int = 0
    predicted_free_on_true_obstacle: int = 0
    predicted_obstacle_on_true_free: int = 0
    unknown_ratio_delta_sum: float = 0.0
    unknown_ratio_abs_delta_sum: float = 0.0
    predicted_unknown_ratio_sum: float = 0.0
    target_unknown_ratio_sum: float = 0.0
    projected_confidence_sum: float = 0.0
    projected_confidence_frame_count: int = 0

    def update(
        self,
        *,
        predicted: dict[str, np.ndarray],
        target: dict[str, np.ndarray],
        source_sample_count: int,
        source_valid_point_count: int,
    ) -> None:
        pred_free = np.asarray(predicted["bev_free"], dtype=np.float32) > np.float32(0.5)
        pred_obstacle = np.asarray(predicted["bev_obstacle"], dtype=np.float32) > np.float32(0.5)
        pred_unknown = np.asarray(predicted["bev_unknown"], dtype=np.float32) > np.float32(0.5)
        true_free = np.asarray(target["bev_free"], dtype=np.float32) > np.float32(0.5)
        true_obstacle = np.asarray(target["bev_obstacle"], dtype=np.float32) > np.float32(0.5)
        true_unknown = np.asarray(target["bev_unknown"], dtype=np.float32) > np.float32(0.5)
        if not (pred_free.shape == pred_obstacle.shape == pred_unknown.shape == true_free.shape == true_obstacle.shape == true_unknown.shape):
            raise ValueError(
                "predicted and target BEV masks must have matching shapes; "
                f"got {pred_free.shape}, {true_free.shape}"
            )

        self.frame_count += 1
        self.cell_count += int(pred_free.size)
        self.source_sample_count += int(source_sample_count)
        self.source_valid_point_count += int(source_valid_point_count)
        self.free_intersection += int(np.count_nonzero(pred_free & true_free))
        self.free_union += int(np.count_nonzero(pred_free | true_free))
        self.obstacle_intersection += int(np.count_nonzero(pred_obstacle & true_obstacle))
        self.obstacle_union += int(np.count_nonzero(pred_obstacle | true_obstacle))
        self.unknown_intersection += int(np.count_nonzero(pred_unknown & true_unknown))
        self.unknown_union += int(np.count_nonzero(pred_unknown | true_unknown))
        self.true_obstacle_count += int(np.count_nonzero(true_obstacle))
        self.true_free_count += int(np.count_nonzero(true_free))
        self.predicted_obstacle_on_true_obstacle += int(np.count_nonzero(pred_obstacle & true_obstacle))
        self.predicted_free_on_true_obstacle += int(np.count_nonzero(pred_free & true_obstacle))
        self.predicted_obstacle_on_true_free += int(np.count_nonzero(pred_obstacle & true_free))

        pred_unknown_ratio = _mask_ratio(pred_unknown)
        true_unknown_ratio = _mask_ratio(true_unknown)
        delta = pred_unknown_ratio - true_unknown_ratio
        self.unknown_ratio_delta_sum += float(delta)
        self.unknown_ratio_abs_delta_sum += float(abs(delta))
        self.predicted_unknown_ratio_sum += float(pred_unknown_ratio)
        self.target_unknown_ratio_sum += float(true_unknown_ratio)
        confidence = np.asarray(predicted.get("bev_confidence", np.zeros(pred_free.shape)), dtype=np.float32)
        self.projected_confidence_sum += float(np.nanmean(confidence)) if confidence.size else 0.0
        self.projected_confidence_frame_count += 1

    def to_metrics(self) -> JsonDict:
        free_iou = _ratio_or_none(self.free_intersection, self.free_union)
        obstacle_iou = _ratio_or_none(self.obstacle_intersection, self.obstacle_union)
        unknown_iou = _ratio_or_none(self.unknown_intersection, self.unknown_union)
        obstacle_recall = _ratio_or_none(self.predicted_obstacle_on_true_obstacle, self.true_obstacle_count)
        false_free_over_obstacle = _ratio_or_none(self.predicted_free_on_true_obstacle, self.true_obstacle_count)
        false_obstacle_over_free = _ratio_or_none(self.predicted_obstacle_on_true_free, self.true_free_count)
        unknown_ratio_delta = _mean_or_none(self.unknown_ratio_delta_sum, self.frame_count)
        unknown_ratio_abs_delta = _mean_or_none(self.unknown_ratio_abs_delta_sum, self.frame_count)
        score = _calibration_score(
            free_iou=free_iou,
            obstacle_iou=obstacle_iou,
            unknown_iou=unknown_iou,
            obstacle_recall=obstacle_recall,
            false_free_over_obstacle_rate=false_free_over_obstacle,
            false_obstacle_over_free_rate=false_obstacle_over_free,
            unknown_ratio_abs_delta=unknown_ratio_abs_delta,
        )
        return {
            "frame_count": int(self.frame_count),
            "cell_count": int(self.cell_count),
            "free_iou": free_iou,
            "obstacle_iou": obstacle_iou,
            "unknown_iou": unknown_iou,
            "obstacle_recall": obstacle_recall,
            "false_free_over_obstacle_rate": false_free_over_obstacle,
            "false_obstacle_over_free_rate": false_obstacle_over_free,
            "unknown_ratio_delta": unknown_ratio_delta,
            "unknown_ratio_abs_delta": unknown_ratio_abs_delta,
            "predicted_unknown_ratio_mean": _mean_or_none(self.predicted_unknown_ratio_sum, self.frame_count),
            "target_unknown_ratio_mean": _mean_or_none(self.target_unknown_ratio_sum, self.frame_count),
            "confidence_valid_ratio": _ratio_or_none(self.source_valid_point_count, self.source_sample_count),
            "projected_confidence_mean": _mean_or_none(
                self.projected_confidence_sum,
                self.projected_confidence_frame_count,
            ),
            "score": float(score),
        }


@dataclass
class CandidateAccumulators:
    all_frames: MetricAccumulator = field(default_factory=MetricAccumulator)
    calibration: MetricAccumulator = field(default_factory=MetricAccumulator)
    heldout: MetricAccumulator = field(default_factory=MetricAccumulator)


def validate_moge_robot_bev_projection(
    *,
    route_dir: str | Path,
    scene_teacher_dir: str | Path,
    spatial_pack_dir: str | Path,
    out_json: str | Path,
    out_md: str | Path,
    out_viz: str | Path,
    max_frames: int | None = None,
    command: str | None = None,
) -> JsonDict:
    started = time.perf_counter()
    route_root = Path(route_dir)
    scene_root = Path(scene_teacher_dir)
    spatial_root = Path(spatial_pack_dir)
    output_json = Path(out_json)
    output_md = Path(out_md)
    output_viz = _resolve_viz_path(out_viz)

    missing = _missing_inputs(route_root=route_root, scene_root=scene_root, spatial_root=spatial_root)
    if missing:
        report = _blocked_report(
            route_root=route_root,
            scene_root=scene_root,
            spatial_root=spatial_root,
            blockers=missing,
            runtime_sec=float(time.perf_counter() - started),
            command=command,
        )
        write_json_object(output_json, report)
        _write_markdown(output_md, report)
        return report

    try:
        route_metadata = read_json_object(route_root / "route_metadata.json")
        associations, associations_path = _load_route_associations(route_root, route_metadata)
        scene_manifest = load_scene_teacher_manifest(scene_root)
        spatial_manifest = read_json_object(spatial_root / "manifest.json")
    except Exception as exc:  # noqa: BLE001 - report exact input failure.
        report = _blocked_report(
            route_root=route_root,
            scene_root=scene_root,
            spatial_root=spatial_root,
            blockers=[f"input_load_failed:{exc}"],
            runtime_sec=float(time.perf_counter() - started),
            command=command,
        )
        write_json_object(output_json, report)
        _write_markdown(output_md, report)
        return report

    scene_frames = scene_frames_by_id(scene_manifest)
    spatial_examples = spatial_examples_by_id(spatial_manifest)
    matched_ids = sorted(set(scene_frames).intersection(spatial_examples))
    if max_frames is not None:
        matched_ids = matched_ids[: max(0, int(max_frames))]
    calibration_ids, heldout_ids = _split_frame_ids(matched_ids)
    split_by_id = {frame_id: "calibration" for frame_id in calibration_ids}
    split_by_id.update({frame_id: "heldout" for frame_id in heldout_ids})

    route_projection = _route_projection_context(route_metadata, associations, associations_path)
    grid_config = _grid_config(spatial_manifest)
    candidates = convention_candidates()
    accs = {candidate.name: CandidateAccumulators() for candidate in candidates}
    frame_errors: list[str] = []
    projected_frame_count = 0

    for frame_id in matched_ids:
        scene_frame = scene_frames[frame_id]
        spatial_example = spatial_examples[frame_id]
        try:
            source = _load_scene_frame_source(scene_root, scene_frame)
            target = load_spatial_example_arrays(spatial_root, spatial_example, missing_ok=False, require_confidence=True)
            if target is None:
                raise ValueError(f"spatial example {frame_id} could not be loaded")
            camera_to_base = _camera_to_base_for_frame(
                associations=associations,
                frame_id=frame_id,
                fallback=route_projection["camera_to_base"],
            )
            for candidate in candidates:
                predicted, projection_stats = _project_convention_to_bev(
                    source=source,
                    camera_to_base=camera_to_base,
                    candidate=candidate,
                    grid_config=grid_config,
                )
                candidate_acc = accs[candidate.name]
                candidate_acc.all_frames.update(
                    predicted=predicted,
                    target=target,
                    source_sample_count=int(projection_stats["source_sample_count"]),
                    source_valid_point_count=int(projection_stats["source_valid_point_count"]),
                )
                split = split_by_id.get(frame_id)
                if split == "calibration":
                    candidate_acc.calibration.update(
                        predicted=predicted,
                        target=target,
                        source_sample_count=int(projection_stats["source_sample_count"]),
                        source_valid_point_count=int(projection_stats["source_valid_point_count"]),
                    )
                elif split == "heldout":
                    candidate_acc.heldout.update(
                        predicted=predicted,
                        target=target,
                        source_sample_count=int(projection_stats["source_sample_count"]),
                        source_valid_point_count=int(projection_stats["source_valid_point_count"]),
                    )
            projected_frame_count += 1
        except Exception as exc:  # noqa: BLE001 - keep per-route report falsifiable.
            frame_errors.append(f"frame_{frame_id}_failed:{exc}")

    candidate_reports = _candidate_reports(candidates, accs)
    best_candidate = _select_best_candidate(candidate_reports)
    hard_blockers = _hard_blockers(
        route_projection=route_projection,
        projected_frame_count=projected_frame_count,
        frame_errors=frame_errors,
        scene_manifest=scene_manifest,
    )
    next_allowed_use = _next_allowed_use(
        best_candidate=best_candidate,
        hard_blockers=hard_blockers,
        scene_manifest=scene_manifest,
    )
    recommendation = _recommendation(next_allowed_use, best_candidate)

    if best_candidate is not None and projected_frame_count > 0:
        try:
            _write_projection_viz(
                route_root=route_root,
                scene_root=scene_root,
                spatial_root=spatial_root,
                scene_frames=scene_frames,
                spatial_examples=spatial_examples,
                frame_ids=matched_ids[:8],
                candidate=_candidate_by_name(candidates, str(best_candidate["name"])),
                grid_config=grid_config,
                associations=associations,
                route_camera_to_base=route_projection["camera_to_base"],
                out_path=output_viz,
            )
        except Exception as exc:  # noqa: BLE001
            frame_errors.append(f"viz_failed:{exc}")

    calibration_metrics = best_candidate.get("calibration_split_metrics") if isinstance(best_candidate, dict) else _empty_metrics()
    heldout_metrics = best_candidate.get("heldout_split_metrics") if isinstance(best_candidate, dict) else _empty_metrics()
    route_summary: JsonDict = {
        "route": route_root.as_posix(),
        "scene": route_metadata.get("scene"),
        "sequence": route_metadata.get("sequence"),
        "matched_frame_count": int(len(matched_ids)),
        "projected_frame_count": int(projected_frame_count),
        "candidate_convention_count": int(len(candidates)),
        "best_convention_name": best_candidate.get("name") if isinstance(best_candidate, dict) else None,
        "best_calibration_score": calibration_metrics.get("score") if isinstance(calibration_metrics, dict) else None,
        "heldout_free_iou": heldout_metrics.get("free_iou") if isinstance(heldout_metrics, dict) else None,
        "heldout_obstacle_iou": heldout_metrics.get("obstacle_iou") if isinstance(heldout_metrics, dict) else None,
        "heldout_obstacle_recall": heldout_metrics.get("obstacle_recall") if isinstance(heldout_metrics, dict) else None,
        "heldout_false_free_over_obstacle_rate": heldout_metrics.get("false_free_over_obstacle_rate") if isinstance(heldout_metrics, dict) else None,
        "next_allowed_use": next_allowed_use,
        **SAFETY_FLAGS,
    }

    report: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "goal": "18A MoGe-to-robot-BEV projection convention gate on OpenLORIS",
        "inputs": {
            "route": route_root.as_posix(),
            "scene_teacher": scene_root.as_posix(),
            "spatial_pack": spatial_root.as_posix(),
        },
        "route": {
            "source_type": route_metadata.get("source_type"),
            "scene": route_metadata.get("scene"),
            "sequence": route_metadata.get("sequence"),
            "license_name": route_metadata.get("license_name"),
            "license_review_status": route_metadata.get("license_review_status"),
            "associations_path": associations_path.as_posix() if associations_path is not None else None,
            "measured_camera_to_base": bool(route_projection["measured_camera_to_base"]),
            "camera_to_base_source": route_projection["camera_to_base_source"],
            "route_has_robot_frame_truth": bool(route_projection["route_has_robot_frame_truth"]),
            "openloris_local_research_replay_only": route_metadata.get("source_type") == "openloris_scene",
        },
        "scene_teacher": {
            "teacher_name": scene_manifest.get("teacher_name"),
            "backend": scene_manifest.get("backend"),
            "model_id": scene_manifest.get("model_id"),
            "real_perception": bool(scene_manifest.get("real_perception", False)),
            "mock": bool(scene_manifest.get("mock", False)),
            "synthetic": bool(scene_manifest.get("synthetic", False)),
            "scale_status": scene_manifest.get("scale_status"),
            "moge_robot_frame_truth": False,
            "action_supervision_ok": False,
        },
        "spatial_pack": {
            "example_count": int(spatial_manifest.get("example_count", len(spatial_examples))),
            "robot_frame_truth": bool(spatial_manifest.get("robot_frame_truth", False)),
            "dataset_frame_type": spatial_manifest.get("dataset_frame_type"),
            "grid_config": grid_config,
        },
        "matched_frame_count": int(len(matched_ids)),
        "projected_frame_count": int(projected_frame_count),
        "matched_frame_ids_sample": matched_ids[:10],
        "candidate_convention_count": int(len(candidates)),
        "convention_candidates": candidate_reports,
        "selected_convention": best_candidate,
        "best_convention_name": best_candidate.get("name") if isinstance(best_candidate, dict) else None,
        "calibration_split": {
            "strategy": "first_half_sorted_frame_ids",
            "frame_count": int(len(calibration_ids)),
            "frame_ids_sample": calibration_ids[:10],
        },
        "heldout_split": {
            "strategy": "remaining_sorted_frame_ids",
            "frame_count": int(len(heldout_ids)),
            "frame_ids_sample": heldout_ids[:10],
        },
        "calibration_split_metrics": calibration_metrics,
        "heldout_split_metrics": heldout_metrics,
        "per_route_summary": route_summary,
        "aggregate_summary": route_summary,
        "next_allowed_use": next_allowed_use,
        "recommendation": recommendation,
        "hard_blockers": hard_blockers,
        "frame_errors": frame_errors[:20],
        "frame_error_count": int(len(frame_errors)),
        **SAFETY_FLAGS,
        "safety_flags": dict(SAFETY_FLAGS),
        "hard_constraints": {
            "trained_model": False,
            "spatial_memory_training": False,
            "trajectory_scorer_training": False,
            "policy_or_control_loop_run": False,
            "raw_pwm_emitted": False,
            "cmd_vel_emitted": False,
            "moge_promoted_to_robot_frame_truth": False,
        },
        "artifacts": {
            "out_json": output_json.as_posix(),
            "out_md": output_md.as_posix(),
            "out_viz": output_viz.as_posix(),
        },
        "commands_run": [command] if command else [],
        "runtime_sec": float(time.perf_counter() - started),
    }
    write_json_object(output_json, report)
    _write_markdown(output_md, report)
    return report


def write_aggregate_report(
    *,
    reports: Iterable[JsonDict],
    out_json: str | Path,
    out_md: str | Path,
    command: str | None = None,
) -> JsonDict:
    route_reports = list(reports)
    route_summaries = [_route_summary_from_report(report) for report in route_reports]
    matched_counts = [int(item.get("matched_frame_count", 0)) for item in route_summaries]
    heldout_false_free = [
        float(item["heldout_false_free_over_obstacle_rate"])
        for item in route_summaries
        if _is_number(item.get("heldout_false_free_over_obstacle_rate"))
    ]
    heldout_obstacle_recall = [
        float(item["heldout_obstacle_recall"])
        for item in route_summaries
        if _is_number(item.get("heldout_obstacle_recall"))
    ]
    local_review_routes = [item for item in route_summaries if item.get("next_allowed_use") == NEXT_LOCAL_REVIEW]
    review_only_routes = [item for item in route_summaries if item.get("next_allowed_use") != NEXT_LOCAL_REVIEW]
    next_allowed_use = NEXT_LOCAL_REVIEW if route_summaries and not review_only_routes else NEXT_REVIEW_ONLY
    report: JsonDict = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "goal": "18A MoGe-to-robot-BEV projection convention gate on OpenLORIS",
        "route_count": int(len(route_summaries)),
        "matched_frame_count": int(sum(matched_counts)),
        "local_replay_candidate_route_count": int(len(local_review_routes)),
        "review_only_route_count": int(len(review_only_routes)),
        "best_convention_distribution": _distribution(
            str(item.get("best_convention_name"))
            for item in route_summaries
            if item.get("best_convention_name") is not None
        ),
        "aggregate_summary": {
            "matched_frame_count": int(sum(matched_counts)),
            "heldout_false_free_over_obstacle_rate_mean": _mean_list(heldout_false_free),
            "heldout_obstacle_recall_mean": _mean_list(heldout_obstacle_recall),
            "all_routes_share_local_replay_candidate": bool(route_summaries and not review_only_routes),
        },
        "route_summaries": route_summaries,
        "next_allowed_use": next_allowed_use,
        "recommendation": (
            "build_local_replay_moge_spatial_train_pack_candidate_generator"
            if next_allowed_use == NEXT_LOCAL_REVIEW
            else "stop_and_repair_projection_or_teacher_assumptions_before_candidate_generation"
        ),
        **SAFETY_FLAGS,
        "safety_flags": dict(SAFETY_FLAGS),
        "hard_constraints": {
            "trained_model": False,
            "spatial_memory_training": False,
            "trajectory_scorer_training": False,
            "policy_or_control_loop_run": False,
            "raw_pwm_emitted": False,
            "cmd_vel_emitted": False,
            "moge_promoted_to_robot_frame_truth": False,
        },
        "commands_run": [command] if command else [],
    }
    write_json_object(out_json, report)
    _write_aggregate_markdown(Path(out_md), report)
    return report


def convention_candidates() -> list[ConventionCandidate]:
    candidates: list[ConventionCandidate] = []
    for permutation in itertools.permutations((0, 1, 2)):
        for signs in itertools.product((1, -1), repeat=3):
            mapping = [
                f"cam_{AXIS_NAMES[out_axis]}={_signed_axis(signs[out_axis], permutation[out_axis])}"
                for out_axis in range(3)
            ]
            candidates.append(
                ConventionCandidate(
                    name="__".join(mapping),
                    permutation=tuple(int(value) for value in permutation),
                    signs=tuple(int(value) for value in signs),
                )
            )
    return candidates


def _project_convention_to_bev(
    *,
    source: JsonDict,
    camera_to_base: np.ndarray,
    candidate: ConventionCandidate,
    grid_config: JsonDict,
) -> tuple[dict[str, np.ndarray], JsonDict]:
    point_map = np.asarray(source["point_map"], dtype=np.float32)
    if point_map.ndim != 3 or point_map.shape[-1] != 3:
        raise ValueError(f"point_map must have shape HxWx3, got {point_map.shape}")
    height, width = point_map.shape[:2]
    stride = max(1, int(min(height, width) // 96))
    sampled = point_map[::stride, ::stride, :].reshape(-1, 3)
    depth = _sample_optional_2d(source.get("depth"), stride=stride, shape=(height, width), default=1.0)
    confidence = _sample_optional_2d(source.get("confidence"), stride=stride, shape=(height, width), default=1.0)
    validity = _sample_optional_2d(source.get("validity_mask"), stride=stride, shape=(height, width), default=1.0) > 0.0

    camera_points = sampled[:, candidate.permutation] * np.asarray(candidate.signs, dtype=np.float32)[None, :]
    finite = (
        np.isfinite(camera_points).all(axis=1)
        & np.isfinite(depth)
        & (depth > np.float32(0.0))
        & np.isfinite(confidence)
        & validity
        & (camera_points[:, 2] > np.float32(0.0))
    )
    valid_camera = camera_points[finite].astype(np.float32)
    valid_confidence = np.clip(confidence[finite], np.float32(0.0), np.float32(1.0)).astype(np.float32)
    if valid_camera.size:
        rotation = np.asarray(camera_to_base[:3, :3], dtype=np.float32)
        translation = np.asarray(camera_to_base[:3, 3], dtype=np.float32)
        base_points = valid_camera @ rotation.T + translation[None, :]
    else:
        base_points = np.zeros((0, 3), dtype=np.float32)

    arrays, stats, _raw_obstacle = robot_points_to_bev_arrays(
        forward_m=base_points[:, 0],
        left_m=base_points[:, 1],
        height_m=base_points[:, 2],
        confidence=valid_confidence,
        grid_cells=int(grid_config["grid_cells"]),
        meters_per_cell=float(grid_config["meters_per_cell"]),
        grid_size_m=float(grid_config["grid_size_m"]),
        floor_height_tol_m=float(grid_config["floor_height_tol_m"]),
        obstacle_height_min_m=float(grid_config["obstacle_height_min_m"]),
        obstacle_dilation_cells=int(grid_config["obstacle_dilation_cells"]),
        sampled_pixel_count=int(sampled.shape[0]),
        valid_point_count=int(np.count_nonzero(finite)),
    )
    stats.update(
        {
            "source_sample_count": int(sampled.shape[0]),
            "source_valid_point_count": int(np.count_nonzero(finite)),
            "pixel_stride": int(stride),
        }
    )
    return arrays, stats


def _candidate_reports(
    candidates: list[ConventionCandidate],
    accs: dict[str, CandidateAccumulators],
) -> list[JsonDict]:
    reports: list[JsonDict] = []
    for candidate in candidates:
        candidate_acc = accs[candidate.name]
        reports.append(
            {
                **candidate.to_json(),
                "all_metrics": candidate_acc.all_frames.to_metrics(),
                "calibration_split_metrics": candidate_acc.calibration.to_metrics(),
                "heldout_split_metrics": candidate_acc.heldout.to_metrics(),
            }
        )
    return reports


def _select_best_candidate(candidate_reports: list[JsonDict]) -> JsonDict | None:
    if not candidate_reports:
        return None
    return max(
        candidate_reports,
        key=lambda item: (
            _metric_value(item.get("calibration_split_metrics"), "score"),
            _metric_value(item.get("calibration_split_metrics"), "obstacle_iou"),
            _metric_value(item.get("calibration_split_metrics"), "free_iou"),
            str(item.get("name")),
        ),
    )


def _next_allowed_use(
    *,
    best_candidate: JsonDict | None,
    hard_blockers: list[str],
    scene_manifest: JsonDict,
) -> str:
    if hard_blockers:
        return NEXT_REVIEW_ONLY
    if best_candidate is None:
        return NEXT_REVIEW_ONLY
    if scene_manifest.get("mock") is True or scene_manifest.get("synthetic") is True:
        return NEXT_REVIEW_ONLY
    heldout = best_candidate.get("heldout_split_metrics")
    if not isinstance(heldout, dict) or int(heldout.get("frame_count", 0)) <= 0:
        return NEXT_REVIEW_ONLY
    free_iou = _none_as_zero(heldout.get("free_iou"))
    obstacle_iou = _none_as_zero(heldout.get("obstacle_iou"))
    obstacle_recall = _none_as_zero(heldout.get("obstacle_recall"))
    false_free = _none_as_one(heldout.get("false_free_over_obstacle_rate"))
    useful_geometry = (free_iou >= 0.05 or obstacle_iou >= 0.03) and obstacle_recall >= 0.05
    safety_not_catastrophic = false_free <= 0.25
    if useful_geometry and safety_not_catastrophic:
        return NEXT_LOCAL_REVIEW
    return NEXT_REVIEW_ONLY


def _recommendation(next_allowed_use: str, best_candidate: JsonDict | None) -> str:
    if next_allowed_use == NEXT_LOCAL_REVIEW:
        return (
            "One fixed convention produced nonzero held-out robot-frame-ish BEV agreement. "
            "Use only for local replay candidate review; do not train or claim robot-frame truth."
        )
    if best_candidate is None:
        return "No convention could be evaluated; repair inputs before any candidate generation."
    return (
        "No convention produced usable held-out safety metrics. Stop before candidate generation "
        "and repair projection, calibration, or teacher assumptions."
    )


def _hard_blockers(
    *,
    route_projection: JsonDict,
    projected_frame_count: int,
    frame_errors: list[str],
    scene_manifest: JsonDict,
) -> list[str]:
    blockers: list[str] = []
    if route_projection.get("measured_camera_to_base") is not True:
        blockers.append("missing_measured_camera_to_base_transform")
    if projected_frame_count <= 0:
        blockers.append("no_frames_projected")
    if scene_manifest.get("control_safe") is True:
        blockers.append("scene_teacher_claims_control_safe")
    if scene_manifest.get("product_training_approved") is True:
        blockers.append("scene_teacher_claims_product_training_approved")
    if scene_manifest.get("robot_frame_truth") is True:
        blockers.append("scene_teacher_claims_robot_frame_truth")
    if scene_manifest.get("action_supervision_ok") is True:
        blockers.append("scene_teacher_claims_action_supervision")
    if frame_errors and projected_frame_count <= 0:
        blockers.append("all_matched_frames_failed_projection")
    return sorted(set(blockers))


def _route_projection_context(route_metadata: JsonDict, associations: JsonDict, associations_path: Path | None) -> JsonDict:
    camera_to_base = _valid_matrix_or_none(associations.get("camera_to_base"))
    camera_to_base_source = associations.get("camera_to_base_source") or route_metadata.get("extrinsics_source")
    if camera_to_base is None:
        for frame in associations.get("frames", []):
            if isinstance(frame, dict):
                camera_to_base = _valid_matrix_or_none(frame.get("camera_to_base"))
                if camera_to_base is not None:
                    camera_to_base_source = frame.get("camera_to_base_source") or camera_to_base_source
                    break
    measured = bool(
        camera_to_base is not None
        and route_metadata.get("has_camera_to_base_transform") is True
        and route_metadata.get("review_assumed_extrinsics") is not True
        and route_metadata.get("extrinsics_source") not in {"assumed_review_only", "missing", "missing_not_supplied"}
    )
    base_pose_or_odom = bool(
        route_metadata.get("has_robot_base_pose") is True
        or route_metadata.get("has_groundtruth_pose") is True
        or route_metadata.get("has_odometry") is True
        or route_metadata.get("has_wheel_odometry") is True
    )
    return {
        "associations_path": associations_path.as_posix() if associations_path is not None else None,
        "camera_to_base": camera_to_base,
        "camera_to_base_source": camera_to_base_source,
        "measured_camera_to_base": measured,
        "base_pose_or_odom": base_pose_or_odom,
        "route_has_robot_frame_truth": bool(route_metadata.get("robot_frame_truth") is True and measured and base_pose_or_odom),
    }


def _camera_to_base_for_frame(
    *,
    associations: JsonDict,
    frame_id: int,
    fallback: np.ndarray | None,
) -> np.ndarray:
    frames = associations.get("frames")
    if isinstance(frames, list):
        for frame in frames:
            if isinstance(frame, dict) and _int_or_none(frame.get("frame_id")) == frame_id:
                matrix = _valid_matrix_or_none(frame.get("camera_to_base"))
                if matrix is not None:
                    return matrix
                break
    if fallback is not None:
        return fallback
    raise ValueError("missing camera_to_base transform")


def _load_route_associations(route_root: Path, route_metadata: JsonDict) -> tuple[JsonDict, Path | None]:
    candidates: list[Path] = []
    metadata_file = route_metadata.get("associations_file")
    if isinstance(metadata_file, str) and metadata_file:
        candidates.append(route_root / metadata_file)
    candidates.extend([route_root / OPENLORIS_ROUTE_ASSOCIATIONS_FILE, route_root / TUM_RGBD_ROUTE_ASSOCIATIONS_FILE])
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return read_json_object(path), path
    return {}, None


def _grid_config(spatial_manifest: JsonDict) -> JsonDict:
    camera_config = spatial_manifest.get("camera_config")
    config = dict(camera_config) if isinstance(camera_config, dict) else {}
    grid_shape = spatial_manifest.get("grid_shape")
    grid_cells = _int_or_none(config.get("grid_cell_count"))
    if grid_cells is None and isinstance(grid_shape, list) and grid_shape:
        grid_cells = _int_or_none(grid_shape[0])
    if grid_cells is None:
        grid_cells = DEFAULT_GRID_CELLS
    meters_per_cell = _float_or_default(config.get("meters_per_cell"), DEFAULT_METERS_PER_CELL)
    obstacle_dilation = _int_or_none(config.get("obstacle_dilation_cells"))
    if obstacle_dilation is None:
        obstacle_dilation = DEFAULT_OBSTACLE_DILATION_CELLS
    return {
        "grid_cells": int(grid_cells),
        "meters_per_cell": float(meters_per_cell),
        "grid_size_m": float(_float_or_default(config.get("grid_size_m"), int(grid_cells) * meters_per_cell)),
        "floor_height_tol_m": float(_float_or_default(config.get("floor_height_tol_m"), DEFAULT_FLOOR_HEIGHT_TOL_M)),
        "obstacle_height_min_m": float(
            _float_or_default(config.get("obstacle_height_min_m"), DEFAULT_OBSTACLE_HEIGHT_MIN_M)
        ),
        "obstacle_dilation_cells": int(obstacle_dilation),
    }


def _load_scene_frame_source(scene_root: Path, frame: JsonDict) -> JsonDict:
    artifacts = frame.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("scene frame is missing artifacts")
    point_map = load_array_artifact_required(scene_root, artifacts, "point_map")
    depth = load_array_artifact_optional(scene_root, artifacts, "depth", missing_ok=False)
    confidence = load_array_artifact_optional(scene_root, artifacts, "confidence", missing_ok=False)
    validity_mask = load_array_artifact_optional(scene_root, artifacts, "validity_mask", missing_ok=False)
    metadata_path = frame.get("metadata_path")
    metadata: JsonDict = {}
    if isinstance(metadata_path, str) and (scene_root / metadata_path).exists():
        metadata = read_json_object(scene_root / metadata_path)
    return {
        "point_map": point_map,
        "depth": depth,
        "confidence": confidence,
        "validity_mask": validity_mask,
        "metadata": metadata,
    }


def _sample_optional_2d(
    value: object,
    *,
    stride: int,
    shape: tuple[int, int],
    default: float,
) -> np.ndarray:
    if value is None:
        return np.full(((shape[0] + stride - 1) // stride, (shape[1] + stride - 1) // stride), np.float32(default)).reshape(-1)
    array = np.asarray(value, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"expected 2D artifact, got {array.shape}")
    if array.shape != shape:
        array = resize_nearest(array, shape)
    return array[::stride, ::stride].reshape(-1).astype(np.float32)


def _split_frame_ids(frame_ids: list[int]) -> tuple[list[int], list[int]]:
    if not frame_ids:
        return [], []
    calibration_count = max(1, len(frame_ids) // 2)
    if len(frame_ids) > 1:
        calibration_count = min(calibration_count, len(frame_ids) - 1)
    return frame_ids[:calibration_count], frame_ids[calibration_count:]


def _write_projection_viz(
    *,
    route_root: Path,
    scene_root: Path,
    spatial_root: Path,
    scene_frames: dict[int, JsonDict],
    spatial_examples: dict[int, JsonDict],
    frame_ids: list[int],
    candidate: ConventionCandidate,
    grid_config: JsonDict,
    associations: JsonDict,
    route_camera_to_base: np.ndarray | None,
    out_path: Path,
) -> None:
    _ = route_root
    rows: list[np.ndarray] = []
    for frame_id in frame_ids:
        if frame_id not in scene_frames or frame_id not in spatial_examples:
            continue
        source = _load_scene_frame_source(scene_root, scene_frames[frame_id])
        target = load_spatial_example_arrays(
            spatial_root,
            spatial_examples[frame_id],
            missing_ok=False,
            require_confidence=True,
        )
        if target is None:
            raise ValueError(f"spatial example {frame_id} could not be loaded")
        camera_to_base = _camera_to_base_for_frame(
            associations=associations,
            frame_id=frame_id,
            fallback=route_camera_to_base,
        )
        predicted, _stats = _project_convention_to_bev(
            source=source,
            camera_to_base=camera_to_base,
            candidate=candidate,
            grid_config=grid_config,
        )
        target_rgb = resize_nearest(_bev_label_rgb(target), (96, 96))
        predicted_rgb = resize_nearest(_bev_label_rgb(predicted), (96, 96))
        error_rgb = resize_nearest(_bev_error_rgb(predicted, target), (96, 96))
        rows.append(join_with_gap([target_rgb, predicted_rgb, error_rgb], gap=4, axis=1))
    if rows:
        sheet = join_with_gap(rows, gap=4, axis=0)
    else:
        sheet = np.full((96, 292, 3), 255, dtype=np.uint8)
    write_ppm(out_path, sheet)


def _bev_label_rgb(arrays: dict[str, np.ndarray]) -> np.ndarray:
    free = np.asarray(arrays["bev_free"], dtype=np.float32) > np.float32(0.5)
    obstacle = np.asarray(arrays["bev_obstacle"], dtype=np.float32) > np.float32(0.5)
    unknown = np.asarray(arrays["bev_unknown"], dtype=np.float32) > np.float32(0.5)
    rgb = np.full((*free.shape, 3), 40, dtype=np.uint8)
    rgb[unknown] = (150, 150, 150)
    rgb[free] = (40, 180, 80)
    rgb[obstacle] = (210, 60, 60)
    return rgb


def _bev_error_rgb(predicted: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> np.ndarray:
    pred_free = np.asarray(predicted["bev_free"], dtype=np.float32) > np.float32(0.5)
    pred_obstacle = np.asarray(predicted["bev_obstacle"], dtype=np.float32) > np.float32(0.5)
    true_free = np.asarray(target["bev_free"], dtype=np.float32) > np.float32(0.5)
    true_obstacle = np.asarray(target["bev_obstacle"], dtype=np.float32) > np.float32(0.5)
    rgb = np.full((*pred_free.shape, 3), 210, dtype=np.uint8)
    rgb[pred_free & true_free] = (40, 180, 80)
    rgb[pred_obstacle & true_obstacle] = (210, 60, 60)
    rgb[pred_free & true_obstacle] = (240, 40, 220)
    rgb[pred_obstacle & true_free] = (40, 80, 240)
    return rgb


def _write_markdown(path: str | Path, report: JsonDict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    selected = report.get("selected_convention") if isinstance(report.get("selected_convention"), dict) else {}
    heldout = report.get("heldout_split_metrics") if isinstance(report.get("heldout_split_metrics"), dict) else {}
    lines = [
        "# Goal 18A MoGe Robot BEV Projection Gate",
        "",
        "## Result",
        f"- next_allowed_use: `{report.get('next_allowed_use')}`",
        f"- matched_frame_count: `{report.get('matched_frame_count')}`",
        f"- candidate_convention_count: `{report.get('candidate_convention_count')}`",
        f"- best_convention: `{selected.get('name')}`",
        f"- moge_robot_frame_truth: `false`",
        f"- action_supervision_ok: `false`",
        f"- control_safe: `false`",
        "",
        "## Held-Out Metrics",
        f"- free_iou: `{heldout.get('free_iou')}`",
        f"- obstacle_iou: `{heldout.get('obstacle_iou')}`",
        f"- obstacle_recall: `{heldout.get('obstacle_recall')}`",
        f"- false_free_over_obstacle_rate: `{heldout.get('false_free_over_obstacle_rate')}`",
        f"- false_obstacle_over_free_rate: `{heldout.get('false_obstacle_over_free_rate')}`",
        f"- unknown_ratio_delta: `{heldout.get('unknown_ratio_delta')}`",
        f"- confidence_valid_ratio: `{heldout.get('confidence_valid_ratio')}`",
        "",
        "## Recommendation",
        str(report.get("recommendation")),
        "",
        "## Safety",
        "- Replay-only projection validation. No model was trained, no action supervision was created, and no cmd_vel/raw PWM was emitted.",
    ]
    if report.get("hard_blockers"):
        lines.extend(["", "## Blockers", *[f"- `{item}`" for item in report["hard_blockers"]]])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_aggregate_markdown(path: Path, report: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Goal 18A Aggregate Projection Gate",
        "",
        "## Result",
        f"- next_allowed_use: `{report.get('next_allowed_use')}`",
        f"- route_count: `{report.get('route_count')}`",
        f"- matched_frame_count: `{report.get('matched_frame_count')}`",
        f"- recommendation: `{report.get('recommendation')}`",
        "",
        "## Routes",
    ]
    for item in report.get("route_summaries", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            f"- `{item.get('sequence') or item.get('route')}`: best=`{item.get('best_convention_name')}`, "
            f"heldout_false_free_over_obstacle=`{item.get('heldout_false_free_over_obstacle_rate')}`, "
            f"next=`{item.get('next_allowed_use')}`"
        )
    lines.extend(
        [
            "",
            "## Safety",
            "- Replay-only projection validation. Control-safe and product-training flags remain false.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _blocked_report(
    *,
    route_root: Path,
    scene_root: Path,
    spatial_root: Path,
    blockers: list[str],
    runtime_sec: float,
    command: str | None,
) -> JsonDict:
    return {
        "schema_version": SCHEMA_VERSION,
        "goal": "18A MoGe-to-robot-BEV projection convention gate on OpenLORIS",
        "inputs": {
            "route": route_root.as_posix(),
            "scene_teacher": scene_root.as_posix(),
            "spatial_pack": spatial_root.as_posix(),
        },
        "matched_frame_count": 0,
        "projected_frame_count": 0,
        "candidate_convention_count": len(convention_candidates()),
        "convention_candidates": [],
        "selected_convention": None,
        "best_convention_name": None,
        "calibration_split_metrics": _empty_metrics(),
        "heldout_split_metrics": _empty_metrics(),
        "per_route_summary": {},
        "aggregate_summary": {},
        "next_allowed_use": NEXT_REVIEW_ONLY,
        "recommendation": "Projection gate inputs are missing or unreadable; repair prerequisites before any candidate generation.",
        "hard_blockers": blockers,
        **SAFETY_FLAGS,
        "safety_flags": dict(SAFETY_FLAGS),
        "commands_run": [command] if command else [],
        "runtime_sec": runtime_sec,
    }


def _missing_inputs(*, route_root: Path, scene_root: Path, spatial_root: Path) -> list[str]:
    missing: list[str] = []
    if not route_root.exists():
        missing.append(f"missing_route:{route_root.as_posix()}")
    if route_root.exists() and not (route_root / "route_metadata.json").exists():
        missing.append(f"missing_route_metadata:{(route_root / 'route_metadata.json').as_posix()}")
    if not scene_root.exists():
        missing.append(f"missing_scene_teacher:{scene_root.as_posix()}")
    if scene_root.exists() and not (scene_root / "scene_teacher_manifest.json").exists():
        missing.append(f"missing_scene_teacher_manifest:{(scene_root / 'scene_teacher_manifest.json').as_posix()}")
    if not spatial_root.exists():
        missing.append(f"missing_spatial_pack:{spatial_root.as_posix()}")
    if spatial_root.exists() and not (spatial_root / "manifest.json").exists():
        missing.append(f"missing_spatial_pack_manifest:{(spatial_root / 'manifest.json').as_posix()}")
    return missing


def _route_summary_from_report(report: JsonDict) -> JsonDict:
    summary = report.get("per_route_summary")
    if isinstance(summary, dict) and summary:
        return dict(summary)
    heldout = report.get("heldout_split_metrics") if isinstance(report.get("heldout_split_metrics"), dict) else {}
    route = report.get("route") if isinstance(report.get("route"), dict) else {}
    return {
        "route": report.get("inputs", {}).get("route") if isinstance(report.get("inputs"), dict) else None,
        "scene": route.get("scene"),
        "sequence": route.get("sequence"),
        "matched_frame_count": int(report.get("matched_frame_count", 0)),
        "candidate_convention_count": int(report.get("candidate_convention_count", 0)),
        "best_convention_name": report.get("best_convention_name"),
        "heldout_free_iou": heldout.get("free_iou"),
        "heldout_obstacle_iou": heldout.get("obstacle_iou"),
        "heldout_obstacle_recall": heldout.get("obstacle_recall"),
        "heldout_false_free_over_obstacle_rate": heldout.get("false_free_over_obstacle_rate"),
        "next_allowed_use": report.get("next_allowed_use"),
        **SAFETY_FLAGS,
    }


def _candidate_by_name(candidates: list[ConventionCandidate], name: str) -> ConventionCandidate:
    for candidate in candidates:
        if candidate.name == name:
            return candidate
    raise KeyError(name)


def _signed_axis(sign: int, axis_index: int) -> str:
    prefix = "+" if int(sign) >= 0 else "-"
    return f"{prefix}moge_{AXIS_NAMES[int(axis_index)]}"


def _calibration_score(
    *,
    free_iou: float | None,
    obstacle_iou: float | None,
    unknown_iou: float | None,
    obstacle_recall: float | None,
    false_free_over_obstacle_rate: float | None,
    false_obstacle_over_free_rate: float | None,
    unknown_ratio_abs_delta: float | None,
) -> float:
    return float(
        _none_as_zero(obstacle_iou) * 2.0
        + _none_as_zero(free_iou)
        + _none_as_zero(unknown_iou) * 0.25
        + _none_as_zero(obstacle_recall) * 0.5
        - _none_as_zero(false_free_over_obstacle_rate) * 2.0
        - _none_as_zero(false_obstacle_over_free_rate)
        - _none_as_zero(unknown_ratio_abs_delta) * 0.1
    )


def _empty_metrics() -> JsonDict:
    return MetricAccumulator().to_metrics()


def _valid_matrix_or_none(value: object) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(isinstance(row, list) and len(row) == 4 for row in value):
        return None
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    return matrix


def _resolve_viz_path(path: str | Path) -> Path:
    target = Path(path)
    if target.suffix:
        return target
    return target / "moge_robot_bev_projection_review.ppm"


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _mean_or_none(total: float, count: int) -> float | None:
    if count <= 0:
        return None
    return float(total / count)


def _mask_ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask) / max(int(mask.size), 1))


def _metric_value(metrics: object, key: str) -> float:
    if isinstance(metrics, dict) and _is_number(metrics.get(key)):
        return float(metrics[key])
    return -1.0


def _none_as_zero(value: object) -> float:
    return float(value) if _is_number(value) else 0.0


def _none_as_one(value: object) -> float:
    return float(value) if _is_number(value) else 1.0


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return None


def _float_or_default(value: object, default: float) -> float:
    if _is_number(value):
        return float(value)
    return float(default)


def _mean_list(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _distribution(values: Iterable[str]) -> JsonDict:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _load_reports(paths: Iterable[str]) -> list[JsonDict]:
    return [read_json_object(path) for path in paths]


def _command() -> str:
    return "python -m homebrain.tools.validate_moge_robot_bev_projection " + " ".join(str(value) for value in sys.argv[1:])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate fixed MoGe point-map conventions against robot-frame SpatialTrainPack BEV labels."
    )
    parser.add_argument("--route", help="Input OpenLORIS route directory.")
    parser.add_argument("--scene-teacher", help="Input MoGe SceneTeacherPack directory.")
    parser.add_argument("--spatial-pack", help="Input robot-frame SpatialTrainPack directory.")
    parser.add_argument("--out-json", required=True, help="Output report JSON.")
    parser.add_argument("--out-md", required=True, help="Output report Markdown.")
    parser.add_argument("--out-viz", help="Output PPM review sheet or directory.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--aggregate-from", nargs="+", default=None, help="Aggregate per-route projection reports.")
    args = parser.parse_args(argv)
    command = _command()
    if args.aggregate_from:
        report = write_aggregate_report(
            reports=_load_reports(args.aggregate_from),
            out_json=args.out_json,
            out_md=args.out_md,
            command=command,
        )
        print(deterministic_json({"route_count": report["route_count"], "next_allowed_use": report["next_allowed_use"]}))
        return 0
    missing_args = [
        name
        for name, value in (
            ("--route", args.route),
            ("--scene-teacher", args.scene_teacher),
            ("--spatial-pack", args.spatial_pack),
            ("--out-viz", args.out_viz),
        )
        if not value
    ]
    if missing_args:
        parser.error("projection mode requires " + ", ".join(missing_args))
    report = validate_moge_robot_bev_projection(
        route_dir=args.route,
        scene_teacher_dir=args.scene_teacher,
        spatial_pack_dir=args.spatial_pack,
        out_json=args.out_json,
        out_md=args.out_md,
        out_viz=args.out_viz,
        max_frames=args.max_frames,
        command=command,
    )
    print(
        deterministic_json(
            {
                "next_allowed_use": report["next_allowed_use"],
                "matched_frame_count": report["matched_frame_count"],
                "best_convention_name": report["best_convention_name"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
