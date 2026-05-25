from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from homebrain.artifacts.io import load_array_artifact_optional, read_json_object, write_json_object
from homebrain.data.spatial_io import load_spatial_example_arrays, scene_frames_by_id, spatial_examples_by_id
from homebrain.messages.schema import JsonDict, deterministic_json
from homebrain.teachers.scene_teacher import load_scene_teacher_manifest
from homebrain.visualization.panels import (
    gray_rgb,
    join_with_gap,
    normalize_gray,
    resize_nearest,
    thumbnail_shape,
    tint,
    write_ppm,
)

COMPARE_SCHEMA_VERSION = "homebrain.moge_scene_spatial_pack_comparison.v0"
AGGREGATE_SCHEMA_VERSION = "homebrain.goal17a_moge_openloris_aggregate.v0"
NEXT_ALLOWED_USES = {"review_only", "single_frame_geometry_pretrain_candidate", "blocked"}
VALID_RATIO_MIN = 0.95


def compare_moge_scene_to_spatial_pack(
    *,
    route_dir: str | Path,
    scene_teacher_dir: str | Path,
    spatial_pack_dir: str | Path,
    out_json: str | Path,
    out_md: str | Path,
    out_viz: str | Path,
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
        scene_manifest = load_scene_teacher_manifest(scene_root)
        spatial_manifest = read_json_object(spatial_root / "manifest.json")
    except Exception as exc:  # noqa: BLE001 - the comparison report should preserve the exact load failure.
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
    matched_scene_frames = [scene_frames[frame_id] for frame_id in matched_ids]
    matched_spatial_examples = [spatial_examples[frame_id] for frame_id in matched_ids]

    route_truth = _route_robot_frame_truth(route_metadata=route_metadata, spatial_manifest=spatial_manifest)
    spatial_ratios_all = _spatial_label_ratios(spatial_root=spatial_root, examples=list(spatial_examples.values()))
    spatial_ratios_matched = _spatial_label_ratios(spatial_root=spatial_root, examples=matched_spatial_examples)
    moge_stats = _moge_stats(scene_root=scene_root, frames=matched_scene_frames)
    image_plane_proxy = _image_plane_depth_agreement_proxy(
        route_root=route_root,
        route_metadata=route_metadata,
        scene_root=scene_root,
        frames=matched_scene_frames,
    )
    qa_summary = _qa_summary(scene_root)
    audit_summary = _audit_summary(scene_root)

    safety_violations = _scene_safety_violations(scene_manifest)
    mock = bool(scene_manifest.get("mock", False))
    synthetic = bool(scene_manifest.get("synthetic", False))
    real_perception = bool(scene_manifest.get("real_perception", False))
    hard_blockers = _hard_blockers(
        matched_frame_count=len(matched_ids),
        route_has_robot_frame_truth=route_truth["route_has_robot_frame_truth"],
        spatial_pack_robot_frame_truth_ratio=route_truth["spatial_pack_robot_frame_truth_ratio"],
        safety_violations=safety_violations,
        moge_depth_valid_ratio=moge_stats["moge_depth_valid_ratio"],
        moge_confidence_valid_ratio=moge_stats["moge_confidence_valid_ratio"],
    )
    next_allowed_use = _next_allowed_use(
        hard_blockers=hard_blockers,
        mock=mock,
        synthetic=synthetic,
        real_perception=real_perception,
        image_plane_proxy=image_plane_proxy,
    )

    report: JsonDict = {
        "schema_version": COMPARE_SCHEMA_VERSION,
        "goal": "17A validate real MoGe against OpenLORIS robot-frame geometry before training",
        "inputs": {
            "route": route_root.as_posix(),
            "scene_teacher": scene_root.as_posix(),
            "spatial_pack": spatial_root.as_posix(),
        },
        "route": {
            "source_type": route_metadata.get("source_type"),
            "sequence": route_metadata.get("sequence"),
            "scene": route_metadata.get("scene"),
            "license_name": route_metadata.get("license_name"),
            "license_review_status": route_metadata.get("license_review_status"),
            "openloris_local_research_replay_only": route_metadata.get("source_type") == "openloris_scene",
            **route_truth,
        },
        "scene_teacher": {
            "teacher_name": scene_manifest.get("teacher_name"),
            "backend": scene_manifest.get("backend"),
            "model_id": scene_manifest.get("model_id"),
            "frame_count": int(scene_manifest.get("frame_count", len(scene_frames))),
            "real_perception": real_perception,
            "mock": mock,
            "synthetic": synthetic,
            "mock_or_synthetic": bool(mock or synthetic),
            "scale_status": scene_manifest.get("scale_status"),
            "moge_robot_frame_truth": False,
            "action_supervision_ok": False,
            "direct_bev_conversion_attempted": False,
            "projection_blocked_reason": _projection_blocked_reason(route_truth),
        },
        "qa_summary": qa_summary,
        "signal_audit_summary": audit_summary,
        "matched_frame_count": len(matched_ids),
        "matched_frame_ids_sample": matched_ids[:10],
        "spatial_pack_example_count": int(spatial_manifest.get("example_count", len(spatial_examples))),
        "spatial_pack_matched_example_count": len(matched_spatial_examples),
        "spatial_pack_label_ratios": {
            "all_examples": spatial_ratios_all,
            "matched_examples": spatial_ratios_matched,
        },
        **moge_stats,
        "image_plane_agreement_proxy": image_plane_proxy,
        "moge_projected_proxy_statistics": moge_stats["moge_projected_proxy_statistics"],
        "route_has_robot_frame_truth": route_truth["route_has_robot_frame_truth"],
        "moge_robot_frame_truth": False,
        "action_supervision_ok": False,
        "next_allowed_use": next_allowed_use,
        "recommendations": _recommendations(next_allowed_use=next_allowed_use, hard_blockers=hard_blockers),
        "hard_blockers": hard_blockers,
        "safety_flags": {
            "replay_only": scene_manifest.get("replay_only") is True,
            "not_executed": scene_manifest.get("not_executed") is True,
            "control_safe": False,
            "product_training_approved": False,
            "cmd_vel_emitted": False,
            "raw_pwm_emitted": False,
        },
        "hard_constraints": {
            "trained_model": False,
            "spatial_memory_training": False,
            "trajectory_scorer_training": False,
            "policy_or_control_loop_run": False,
            "cmd_vel_emitted": False,
            "raw_pwm_emitted": False,
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
    if report["next_allowed_use"] not in NEXT_ALLOWED_USES:
        raise AssertionError(f"invalid next_allowed_use: {report['next_allowed_use']}")
    _write_comparison_viz(
        route_root=route_root,
        scene_root=scene_root,
        spatial_root=spatial_root,
        frames=matched_scene_frames,
        spatial_by_id=spatial_examples,
        out_path=output_viz,
    )
    write_json_object(output_json, report)
    _write_markdown(output_md, report)
    return report


def write_aggregate_report(
    *,
    comparison_reports: Iterable[JsonDict],
    out_json: str | Path,
    out_md: str | Path,
    command: str | None = None,
) -> JsonDict:
    reports = list(comparison_reports)
    route_summaries = [_aggregate_route_summary(report) for report in reports]
    candidate_routes = [
        item for item in route_summaries if item.get("next_allowed_use") == "single_frame_geometry_pretrain_candidate"
    ]
    blocked_routes = [item for item in route_summaries if item.get("next_allowed_use") == "blocked"]
    qa_pass_routes = [item for item in route_summaries if item.get("qa_structural_pass") is True]
    real_routes = [item for item in route_summaries if item.get("real_moge_ran") is True]
    report: JsonDict = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "goal": "17A validate real MoGe against OpenLORIS robot-frame geometry before training",
        "route_count": len(route_summaries),
        "real_moge_route_count": len(real_routes),
        "qa_structural_pass_count": len(qa_pass_routes),
        "candidate_route_count": len(candidate_routes),
        "blocked_route_count": len(blocked_routes),
        "partial_route_success_or_failure": bool(blocked_routes and len(blocked_routes) < len(route_summaries)),
        "did_real_moge_run_on_public_robot_frame_routes": bool(real_routes),
        "did_qa_pass_structurally": bool(route_summaries and len(qa_pass_routes) == len(route_summaries)),
        "does_moge_look_useful_as_single_frame_geometry_supervision": bool(candidate_routes),
        "should_proceed_to_moge_spatial_train_pack_candidate": bool(candidate_routes and not blocked_routes),
        "aggregate_recommendation": _aggregate_recommendation(candidate_routes=candidate_routes, blocked_routes=blocked_routes),
        "still_missing_before_temporal_memory_training": [
            "MoGe output is not robot-frame truth.",
            "Single-frame MoGe has no temporal pose, odometry, or point-track evidence.",
            "Direct robot-frame BEV conversion remains blocked until projection conventions and transforms are validated.",
            "OpenLORIS remains local research/replay only with product-training approval pending.",
            "Action supervision remains false and no policy/control loop was run.",
        ],
        "route_summaries": route_summaries,
        "safety_flags": {
            "control_safe": False,
            "product_training_approved": False,
            "cmd_vel_emitted": False,
            "raw_pwm_emitted": False,
        },
        "hard_constraints": {
            "trained_model": False,
            "spatial_memory_training": False,
            "trajectory_scorer_training": False,
            "policy_or_control_loop_run": False,
            "moge_promoted_to_robot_frame_truth": False,
        },
        "commands_run": [command] if command else [],
    }
    write_json_object(out_json, report)
    _write_aggregate_markdown(Path(out_md), report)
    return report


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
        "schema_version": COMPARE_SCHEMA_VERSION,
        "goal": "17A validate real MoGe against OpenLORIS robot-frame geometry before training",
        "inputs": {
            "route": route_root.as_posix(),
            "scene_teacher": scene_root.as_posix(),
            "spatial_pack": spatial_root.as_posix(),
        },
        "matched_frame_count": 0,
        "moge_depth_valid_ratio": 0.0,
        "moge_confidence_valid_ratio": 0.0,
        "spatial_pack_example_count": 0,
        "spatial_pack_label_ratios": {
            "all_examples": _empty_ratio_summary(),
            "matched_examples": _empty_ratio_summary(),
        },
        "route_has_robot_frame_truth": False,
        "moge_robot_frame_truth": False,
        "action_supervision_ok": False,
        "next_allowed_use": "blocked",
        "projection_blocked_reason": "comparison inputs are missing or unreadable",
        "hard_blockers": blockers,
        "recommendations": [
            "Restore the exact missing route, SceneTeacherPack, and SpatialTrainPack prerequisites before rerunning Goal 17A.",
            "Do not redownload public data unless an existing setup path and license note already allow it.",
        ],
        "safety_flags": {
            "control_safe": False,
            "product_training_approved": False,
            "cmd_vel_emitted": False,
            "raw_pwm_emitted": False,
        },
        "hard_constraints": {
            "trained_model": False,
            "spatial_memory_training": False,
            "trajectory_scorer_training": False,
            "policy_or_control_loop_run": False,
            "moge_promoted_to_robot_frame_truth": False,
        },
        "commands_run": [command] if command else [],
        "runtime_sec": runtime_sec,
    }


def _route_robot_frame_truth(*, route_metadata: JsonDict, spatial_manifest: JsonDict) -> JsonDict:
    has_camera_to_base = route_metadata.get("has_camera_to_base_transform") is True
    base_pose_or_odom = bool(
        route_metadata.get("has_robot_base_pose") is True
        or route_metadata.get("has_groundtruth_pose") is True
        or route_metadata.get("has_odometry") is True
        or route_metadata.get("has_wheel_odometry") is True
    )
    measured = bool(
        has_camera_to_base
        and base_pose_or_odom
        and route_metadata.get("review_assumed_extrinsics") is not True
        and route_metadata.get("extrinsics_source") not in {"assumed_review_only", "missing", "missing_not_supplied"}
    )
    example_truth_values = [
        example.get("robot_frame_truth") is True
        for example in spatial_manifest.get("examples", [])
        if isinstance(example, dict)
    ]
    spatial_truth_ratio = _mean([1.0 if value else 0.0 for value in example_truth_values])
    return {
        "route_has_robot_frame_truth": bool(route_metadata.get("robot_frame_truth") is True and measured),
        "route_robot_frame_truth_claim": route_metadata.get("robot_frame_truth") is True,
        "measured_camera_to_base": has_camera_to_base,
        "base_pose_or_odom": base_pose_or_odom,
        "spatial_pack_robot_frame_truth_ratio": spatial_truth_ratio,
    }


def _spatial_label_ratios(*, spatial_root: Path, examples: list[JsonDict]) -> JsonDict:
    if not examples:
        return _empty_ratio_summary()
    free: list[float] = []
    obstacle: list[float] = []
    unknown: list[float] = []
    confidence: list[float] = []
    loaded_npz_count = 0
    for example in examples:
        stats = example.get("source_bev_stats") if isinstance(example.get("source_bev_stats"), dict) else {}
        if all(_is_number(stats.get(key)) for key in ("free_ratio", "obstacle_ratio", "unknown_ratio")):
            free.append(float(stats["free_ratio"]))
            obstacle.append(float(stats["obstacle_ratio"]))
            unknown.append(float(stats["unknown_ratio"]))
            if _is_number(stats.get("confidence_mean")):
                confidence.append(float(stats["confidence_mean"]))
            continue
        arrays = load_spatial_example_arrays(spatial_root, example, missing_ok=True)
        if arrays is None:
            continue
        loaded_npz_count += 1
        free.append(_array_ratio(arrays.get("bev_free")))
        obstacle.append(_array_ratio(arrays.get("bev_obstacle")))
        unknown.append(_array_ratio(arrays.get("bev_unknown")))
        if arrays.get("bev_confidence") is not None:
            confidence.append(float(np.nanmean(np.asarray(arrays["bev_confidence"], dtype=np.float32))))
    return {
        "example_count": len(examples),
        "usable_example_count": len(free),
        "free_ratio_mean": _mean(free),
        "obstacle_ratio_mean": _mean(obstacle),
        "unknown_ratio_mean": _mean(unknown),
        "confidence_mean": _mean(confidence),
        "ratios_sum_mean": _mean([a + b + c for a, b, c in zip(free, obstacle, unknown)]),
        "loaded_npz_count": loaded_npz_count,
    }


def _moge_stats(*, scene_root: Path, frames: list[JsonDict]) -> JsonDict:
    depth_valid = 0
    depth_total = 0
    confidence_valid = 0
    confidence_total = 0
    point_total = 0
    point_finite = 0
    point_z_positive = 0
    depth_values: list[np.ndarray] = []
    z_values: list[np.ndarray] = []
    missing_depth_count = 0
    missing_confidence_count = 0
    point_map_frame_count = 0
    for frame in frames:
        artifacts = frame.get("artifacts") if isinstance(frame.get("artifacts"), dict) else {}
        depth = load_array_artifact_optional(scene_root, artifacts, "depth", missing_ok=True)
        if depth is None:
            missing_depth_count += 1
        else:
            depth_hw = np.asarray(depth, dtype=np.float32).squeeze()
            if depth_hw.ndim == 2:
                valid = np.isfinite(depth_hw) & (depth_hw > np.float32(0.0))
                depth_valid += int(valid.sum())
                depth_total += int(valid.size)
                depth_values.append(_sample_1d(depth_hw[valid], max_count=4096))
        confidence = load_array_artifact_optional(scene_root, artifacts, "confidence", missing_ok=True)
        if confidence is None:
            missing_confidence_count += 1
        else:
            conf_hw = np.asarray(confidence, dtype=np.float32).squeeze()
            if conf_hw.ndim == 2:
                valid = np.isfinite(conf_hw)
                confidence_valid += int(valid.sum())
                confidence_total += int(valid.size)
        point_map = load_array_artifact_optional(scene_root, artifacts, "point_map", missing_ok=True)
        if point_map is not None:
            points = np.asarray(point_map, dtype=np.float32)
            if points.ndim == 3 and points.shape[-1] == 3:
                finite = np.isfinite(points).all(axis=2)
                z = points[:, :, 2]
                z_positive = finite & (z > np.float32(0.0))
                point_map_frame_count += 1
                point_finite += int(finite.sum())
                point_total += int(finite.size)
                point_z_positive += int(z_positive.sum())
                z_values.append(_sample_1d(z[z_positive], max_count=4096))
    depth_sample = _concat_samples(depth_values)
    z_sample = _concat_samples(z_values)
    return {
        "moge_depth_valid_ratio": _safe_ratio(depth_valid, depth_total),
        "moge_confidence_valid_ratio": _safe_ratio(confidence_valid, confidence_total),
        "moge_missing_depth_frame_count": missing_depth_count,
        "moge_missing_confidence_frame_count": missing_confidence_count,
        "moge_depth_statistics_m": _quantile_summary(depth_sample),
        "moge_projected_proxy_statistics": {
            "projection_available": point_map_frame_count > 0,
            "projection_type": "camera_frame_point_map_proxy_only",
            "point_map_frame_count": point_map_frame_count,
            "point_map_valid_ratio": _safe_ratio(point_finite, point_total),
            "point_map_positive_z_ratio": _safe_ratio(point_z_positive, point_total),
            "point_z_statistics_m": _quantile_summary(z_sample),
        },
    }


def _image_plane_depth_agreement_proxy(
    *,
    route_root: Path,
    route_metadata: JsonDict,
    scene_root: Path,
    frames: list[JsonDict],
) -> JsonDict:
    depth_scale = float(route_metadata.get("depth_scale", 1000.0)) if _is_number(route_metadata.get("depth_scale")) else 1000.0
    if depth_scale <= 0.0:
        depth_scale = 1000.0
    raw_abs_rel: list[np.ndarray] = []
    scaled_abs_rel: list[np.ndarray] = []
    moge_samples: list[np.ndarray] = []
    rgbd_samples: list[np.ndarray] = []
    scale_factors: list[float] = []
    source_depth_frame_count = 0
    for frame in frames:
        frame_id = _int_or_none(frame.get("frame_id"))
        if frame_id is None:
            continue
        source_depth = _load_route_depth(route_root, frame_id=frame_id, depth_scale=depth_scale)
        if source_depth is None:
            continue
        artifacts = frame.get("artifacts") if isinstance(frame.get("artifacts"), dict) else {}
        moge_depth = load_array_artifact_optional(scene_root, artifacts, "depth", missing_ok=True)
        if moge_depth is None:
            continue
        moge_hw = np.asarray(moge_depth, dtype=np.float32).squeeze()
        if moge_hw.ndim != 2:
            continue
        if moge_hw.shape != source_depth.shape:
            moge_hw = resize_nearest(moge_hw, source_depth.shape)
        valid = (
            np.isfinite(source_depth)
            & np.isfinite(moge_hw)
            & (source_depth > np.float32(0.0))
            & (moge_hw > np.float32(0.0))
        )
        if int(valid.sum()) < 16:
            continue
        source_depth_frame_count += 1
        truth = _sample_1d(source_depth[valid], max_count=2048)
        pred = _sample_1d(moge_hw[valid], max_count=2048)
        count = min(truth.size, pred.size)
        if count == 0:
            continue
        truth = truth[:count]
        pred = pred[:count]
        scale = _median_scale(pred=pred, truth=truth)
        scale_factors.append(scale)
        raw_abs_rel.append(np.abs(pred - truth) / np.maximum(truth, np.float32(1e-6)))
        scaled_abs_rel.append(np.abs(pred * np.float32(scale) - truth) / np.maximum(truth, np.float32(1e-6)))
        moge_samples.append(pred)
        rgbd_samples.append(truth)
    if source_depth_frame_count == 0:
        return {
            "available": False,
            "source": "route RGB-D depth image",
            "reason": "no matched route depth frames could be decoded",
            "not_action_supervision": True,
        }
    raw = _concat_samples(raw_abs_rel)
    scaled = _concat_samples(scaled_abs_rel)
    moge = _concat_samples(moge_samples)
    truth = _concat_samples(rgbd_samples)
    return {
        "available": True,
        "source": "route RGB-D depth image",
        "not_action_supervision": True,
        "matched_depth_frame_count": source_depth_frame_count,
        "sample_count": int(min(moge.size, truth.size)),
        "raw_abs_rel_median": _nan_float(np.nanmedian(raw)) if raw.size else None,
        "raw_abs_rel_mean": _nan_float(np.nanmean(raw)) if raw.size else None,
        "median_scaled_abs_rel_median": _nan_float(np.nanmedian(scaled)) if scaled.size else None,
        "median_scaled_abs_rel_mean": _nan_float(np.nanmean(scaled)) if scaled.size else None,
        "median_scale_factor_moge_to_rgbd": _nan_float(np.nanmedian(np.asarray(scale_factors, dtype=np.float32))),
        "pearson_corr": _pearson(moge, truth),
    }


def _qa_summary(scene_root: Path) -> JsonDict:
    qa_path = scene_root.parent / f"{scene_root.name}_qa.json"
    if not qa_path.exists():
        return {"available": False, "path": qa_path.as_posix(), "structural_pass": False}
    try:
        qa = read_json_object(qa_path)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "path": qa_path.as_posix(), "structural_pass": False, "error": str(exc)}
    missing = int(qa.get("missing_artifact_count", 0))
    shape = int(qa.get("artifact_shape_error_count", 0))
    depth_valid = float(qa.get("depth_valid_ratio", 0.0)) if _is_number(qa.get("depth_valid_ratio")) else 0.0
    confidence_valid = (
        float(qa.get("confidence_valid_ratio", 0.0)) if _is_number(qa.get("confidence_valid_ratio")) else 0.0
    )
    return {
        "available": True,
        "path": qa_path.as_posix(),
        "structural_pass": bool(missing == 0 and shape == 0 and depth_valid >= VALID_RATIO_MIN),
        "frame_count": qa.get("frame_count"),
        "missing_artifact_count": missing,
        "artifact_shape_error_count": shape,
        "depth_valid_ratio": depth_valid,
        "confidence_valid_ratio": confidence_valid,
        "pose_valid_ratio": qa.get("pose_valid_ratio"),
        "scale_status": qa.get("scale_status"),
        "mock": qa.get("mock"),
        "synthetic": qa.get("synthetic"),
        "real_perception": qa.get("real_perception"),
    }


def _audit_summary(scene_root: Path) -> JsonDict:
    audit_path = scene_root.parent / f"{scene_root.name}_signal_audit.json"
    if not audit_path.exists():
        return {"available": False, "path": audit_path.as_posix()}
    try:
        audit = read_json_object(audit_path)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "path": audit_path.as_posix(), "error": str(exc)}
    return {
        "available": True,
        "path": audit_path.as_posix(),
        "next_allowed_use": audit.get("next_allowed_use"),
        "hard_blockers": audit.get("hard_blockers", []),
        "single_frame_geometry_pretrain_candidate": audit.get("single_frame_geometry_pretrain_candidate"),
        "temporal_memory_pretrain_candidate": audit.get("temporal_memory_pretrain_candidate"),
        "robot_frame_truth": audit.get("robot_frame_truth"),
        "action_supervision_ok": audit.get("action_supervision_ok"),
    }


def _hard_blockers(
    *,
    matched_frame_count: int,
    route_has_robot_frame_truth: bool,
    spatial_pack_robot_frame_truth_ratio: float | None,
    safety_violations: list[str],
    moge_depth_valid_ratio: float,
    moge_confidence_valid_ratio: float,
) -> list[str]:
    blockers: list[str] = []
    if matched_frame_count <= 0:
        blockers.append("no_matched_scene_teacher_spatial_pack_frames")
    if not route_has_robot_frame_truth:
        blockers.append("route_lacks_robot_frame_truth")
    if spatial_pack_robot_frame_truth_ratio is None or spatial_pack_robot_frame_truth_ratio < 0.5:
        blockers.append("spatial_pack_lacks_robot_frame_truth")
    if moge_depth_valid_ratio < VALID_RATIO_MIN:
        blockers.append("moge_depth_validity_below_gate")
    if moge_confidence_valid_ratio < VALID_RATIO_MIN:
        blockers.append("moge_confidence_validity_below_gate")
    blockers.extend(safety_violations)
    return sorted(set(blockers))


def _next_allowed_use(
    *,
    hard_blockers: list[str],
    mock: bool,
    synthetic: bool,
    real_perception: bool,
    image_plane_proxy: JsonDict,
) -> str:
    if hard_blockers:
        return "blocked"
    if mock or synthetic or not real_perception:
        return "review_only"
    if image_plane_proxy.get("available") is True:
        scaled = image_plane_proxy.get("median_scaled_abs_rel_median")
        if _is_number(scaled) and float(scaled) > 0.75:
            return "review_only"
    return "single_frame_geometry_pretrain_candidate"


def _recommendations(*, next_allowed_use: str, hard_blockers: list[str]) -> list[str]:
    if next_allowed_use == "blocked":
        return [
            "Repair the listed blockers before treating MoGe output as a spatial supervision candidate.",
            "Keep OpenLORIS local research/replay only and do not redownload or invent missing public-route semantics.",
        ]
    if next_allowed_use == "review_only":
        return [
            "Use this comparison for visual and structural review only.",
            "Do not pack MoGe as SpatialTrainPack supervision until real non-synthetic geometry clears this gate.",
        ]
    return [
        "Proceed only to a local/replay MoGe SpatialTrainPack candidate gate for single-frame geometry.",
        "Do not treat MoGe as robot-frame truth, action supervision, temporal memory evidence, or control safety.",
    ]


def _projection_blocked_reason(route_truth: JsonDict) -> str:
    if route_truth.get("route_has_robot_frame_truth") is True:
        return (
            "Direct robot-frame BEV conversion is blocked for this comparison because the SceneTeacherPack "
            "does not claim robot-frame truth and single-frame MoGe point maps have no teacher temporal "
            "pose/odometry evidence. Route RGB-D BEV is used only as an external comparison anchor."
        )
    return (
        "Direct robot-frame BEV conversion is blocked because the route/spatial pack does not provide "
        "verified robot-frame truth for this comparison."
    )


def _scene_safety_violations(manifest: JsonDict) -> list[str]:
    violations: list[str] = []
    if manifest.get("control_safe") is True:
        violations.append("scene_teacher_control_safe_claim_present")
    if manifest.get("product_training_approved") is True:
        violations.append("scene_teacher_product_training_claim_present")
    if manifest.get("raw_pwm_emitted") is True:
        violations.append("scene_teacher_raw_pwm_claim_present")
    if manifest.get("robot_frame_truth") is True:
        violations.append("scene_teacher_robot_frame_truth_claim_present")
    if manifest.get("action_supervision_ok") is True:
        violations.append("scene_teacher_action_supervision_claim_present")
    return violations


def _write_comparison_viz(
    *,
    route_root: Path,
    scene_root: Path,
    spatial_root: Path,
    frames: list[JsonDict],
    spatial_by_id: dict[int, JsonDict],
    out_path: Path,
    max_frames: int = 6,
) -> None:
    selected = frames[:max_frames]
    rows: list[np.ndarray] = []
    for frame in selected:
        frame_id = _int_or_none(frame.get("frame_id"))
        artifacts = frame.get("artifacts") if isinstance(frame.get("artifacts"), dict) else {}
        depth = load_array_artifact_optional(scene_root, artifacts, "depth", missing_ok=True)
        if depth is None:
            continue
        depth_hw = np.asarray(depth, dtype=np.float32).squeeze()
        if depth_hw.ndim != 2:
            continue
        shape = thumbnail_shape(depth_hw)
        panels = [
            _source_rgb_panel(route_root=route_root, frame=frame, shape=shape),
            gray_rgb(resize_nearest(normalize_gray(depth_hw), shape)),
            _confidence_panel(scene_root=scene_root, artifacts=artifacts, shape=shape),
            _route_depth_panel(route_root=route_root, frame_id=frame_id, shape=shape),
            _spatial_bev_panel(spatial_root=spatial_root, example=spatial_by_id.get(frame_id) if frame_id is not None else None),
        ]
        rows.append(join_with_gap(panels, gap=2, axis=1))
    if not rows:
        rows.append(np.full((32, 32, 3), 230, dtype=np.uint8))
    write_ppm(out_path, join_with_gap(rows, gap=2, axis=0))


def _source_rgb_panel(*, route_root: Path, frame: JsonDict, shape: tuple[int, int]) -> np.ndarray:
    data_ref = frame.get("source_data_ref")
    if not isinstance(data_ref, str):
        return np.full((shape[0], shape[1], 3), 230, dtype=np.uint8)
    image = _read_rgb_image(route_root / data_ref)
    if image is None:
        return np.full((shape[0], shape[1], 3), 230, dtype=np.uint8)
    return resize_nearest(np.asarray(image, dtype=np.uint8), shape)


def _confidence_panel(*, scene_root: Path, artifacts: JsonDict, shape: tuple[int, int]) -> np.ndarray:
    confidence = load_array_artifact_optional(scene_root, artifacts, "confidence", missing_ok=True)
    if confidence is None:
        return np.full((shape[0], shape[1], 3), 230, dtype=np.uint8)
    return tint(resize_nearest(normalize_gray(np.asarray(confidence, dtype=np.float32).squeeze()), shape), (50, 180, 80))


def _route_depth_panel(*, route_root: Path, frame_id: int | None, shape: tuple[int, int]) -> np.ndarray:
    if frame_id is None:
        return np.full((shape[0], shape[1], 3), 230, dtype=np.uint8)
    depth = _load_route_depth(route_root, frame_id=frame_id, depth_scale=1000.0)
    if depth is None:
        return np.full((shape[0], shape[1], 3), 230, dtype=np.uint8)
    return gray_rgb(resize_nearest(normalize_gray(depth), shape))


def _spatial_bev_panel(*, spatial_root: Path, example: JsonDict | None, size: int = 96) -> np.ndarray:
    if example is None:
        return np.full((size, size, 3), 230, dtype=np.uint8)
    arrays = load_spatial_example_arrays(spatial_root, example, missing_ok=True)
    if arrays is None:
        return np.full((size, size, 3), 230, dtype=np.uint8)
    free = np.asarray(arrays.get("bev_free", np.zeros((32, 32))), dtype=np.float32)
    obstacle = np.asarray(arrays.get("bev_obstacle", np.zeros_like(free)), dtype=np.float32)
    unknown = np.asarray(arrays.get("bev_unknown", np.zeros_like(free)), dtype=np.float32)
    rgb = np.zeros((*free.shape, 3), dtype=np.uint8)
    rgb[:, :, :] = np.clip(unknown[:, :, None] * 150, 0, 150).astype(np.uint8)
    rgb[:, :, 1] = np.maximum(rgb[:, :, 1], np.clip(free * 210, 0, 210).astype(np.uint8))
    rgb[:, :, 0] = np.maximum(rgb[:, :, 0], np.clip(obstacle * 230, 0, 230).astype(np.uint8))
    return resize_nearest(rgb, (size, size))


def _load_route_depth(route_root: Path, *, frame_id: int, depth_scale: float) -> np.ndarray | None:
    path = route_root / "depth" / f"depth_{frame_id:06d}.png"
    if not path.exists():
        return None
    depth = _read_depth_image(path)
    if depth is None:
        return None
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        return None
    return values / np.float32(depth_scale)


def _read_rgb_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix in {".ppm", ".pgm"}:
        return _read_netpbm(path)
    try:
        import cv2  # type: ignore

        image = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:  # noqa: BLE001
        pass
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception:  # noqa: BLE001
        return None


def _read_depth_image(path: Path) -> np.ndarray | None:
    try:
        import cv2  # type: ignore

        image = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)
        if image is not None:
            return np.asarray(image)
    except Exception:  # noqa: BLE001
        pass
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            return np.asarray(image)
    except Exception:  # noqa: BLE001
        return None


def _read_netpbm(path: Path) -> np.ndarray | None:
    data = path.read_bytes()
    index = 0

    def read_token() -> bytes:
        nonlocal index
        while index < len(data):
            char = data[index : index + 1]
            index += 1
            if char == b"#":
                while index < len(data) and data[index : index + 1] not in {b"\n", b"\r"}:
                    index += 1
                continue
            if char.isspace():
                continue
            token = bytearray(char)
            while index < len(data) and not data[index : index + 1].isspace():
                token.extend(data[index : index + 1])
                index += 1
            return bytes(token)
        raise ValueError("unexpected EOF in Netpbm header")

    try:
        magic = read_token()
        width = int(read_token())
        height = int(read_token())
        max_value = int(read_token())
    except Exception:
        return None
    if max_value <= 0 or max_value > 255:
        return None
    while index < len(data) and data[index : index + 1].isspace():
        index += 1
    channels = 3 if magic == b"P6" else 1 if magic == b"P5" else 0
    if channels == 0:
        return None
    raw = np.frombuffer(data[index : index + width * height * channels], dtype=np.uint8)
    if raw.size != width * height * channels:
        return None
    if channels == 3:
        return raw.reshape((height, width, 3)).copy()
    gray = raw.reshape((height, width)).copy()
    return np.stack([gray, gray, gray], axis=2)


def _aggregate_route_summary(report: JsonDict) -> JsonDict:
    scene = report.get("scene_teacher", {}) if isinstance(report.get("scene_teacher"), dict) else {}
    qa = report.get("qa_summary", {}) if isinstance(report.get("qa_summary"), dict) else {}
    route = report.get("route", {}) if isinstance(report.get("route"), dict) else {}
    return {
        "route": report.get("inputs", {}).get("route") if isinstance(report.get("inputs"), dict) else None,
        "scene": route.get("scene"),
        "sequence": route.get("sequence"),
        "real_moge_ran": bool(scene.get("backend") == "real" and scene.get("real_perception") is True),
        "qa_structural_pass": qa.get("structural_pass") is True,
        "matched_frame_count": report.get("matched_frame_count", 0),
        "moge_depth_valid_ratio": report.get("moge_depth_valid_ratio"),
        "moge_confidence_valid_ratio": report.get("moge_confidence_valid_ratio"),
        "route_has_robot_frame_truth": report.get("route_has_robot_frame_truth") is True,
        "moge_robot_frame_truth": False,
        "action_supervision_ok": False,
        "next_allowed_use": report.get("next_allowed_use"),
        "hard_blockers": report.get("hard_blockers", []),
    }


def _aggregate_recommendation(*, candidate_routes: list[JsonDict], blocked_routes: list[JsonDict]) -> str:
    if blocked_routes and not candidate_routes:
        return "stop_until_blockers_are_repaired"
    if blocked_routes:
        return "partial_success_review_blocked_routes_before_candidate"
    if candidate_routes:
        return "proceed_to_local_replay_single_frame_geometry_candidate_gate"
    return "review_only_do_not_pack_yet"


def _empty_ratio_summary() -> JsonDict:
    return {
        "example_count": 0,
        "usable_example_count": 0,
        "free_ratio_mean": None,
        "obstacle_ratio_mean": None,
        "unknown_ratio_mean": None,
        "confidence_mean": None,
        "ratios_sum_mean": None,
        "loaded_npz_count": 0,
    }


def _array_ratio(value: np.ndarray | None) -> float:
    if value is None:
        return 0.0
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return 0.0
    return float(np.nanmean(array > np.float32(0.5)))


def _quantile_summary(values: np.ndarray) -> JsonDict:
    if values.size == 0:
        return {"count": 0, "p10": None, "median": None, "p90": None, "mean": None}
    return {
        "count": int(values.size),
        "p10": _nan_float(np.nanpercentile(values, 10)),
        "median": _nan_float(np.nanmedian(values)),
        "p90": _nan_float(np.nanpercentile(values, 90)),
        "mean": _nan_float(np.nanmean(values)),
    }


def _sample_1d(values: np.ndarray, *, max_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size <= max_count:
        return array
    indices = np.linspace(0, array.size - 1, max_count).round().astype(np.int64)
    return array[indices]


def _concat_samples(samples: list[np.ndarray]) -> np.ndarray:
    kept = [np.asarray(sample, dtype=np.float32).reshape(-1) for sample in samples if sample.size]
    if not kept:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(kept)


def _median_scale(*, pred: np.ndarray, truth: np.ndarray) -> float:
    pred_med = float(np.nanmedian(pred))
    truth_med = float(np.nanmedian(truth))
    if not math.isfinite(pred_med) or pred_med <= 1e-6 or not math.isfinite(truth_med):
        return 1.0
    return float(truth_med / pred_med)


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    count = min(a.size, b.size)
    if count < 2:
        return None
    x = np.asarray(a[:count], dtype=np.float64)
    y = np.asarray(b[:count], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return _nan_float(np.corrcoef(x, y)[0, 1])


def _write_markdown(path: str | Path, report: JsonDict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scene = report.get("scene_teacher", {}) if isinstance(report.get("scene_teacher"), dict) else {}
    labels = report.get("spatial_pack_label_ratios", {})
    matched = labels.get("matched_examples", {}) if isinstance(labels, dict) else {}
    lines = [
        "# MoGe Scene To SpatialPack Comparison",
        "",
        "## Result",
        f"- next_allowed_use: `{report.get('next_allowed_use')}`",
        f"- matched_frame_count: `{report.get('matched_frame_count')}`",
        f"- route_has_robot_frame_truth: `{str(report.get('route_has_robot_frame_truth')).lower()}`",
        f"- moge_robot_frame_truth: `false`",
        f"- action_supervision_ok: `false`",
        "",
        "## MoGe",
        f"- backend: `{scene.get('backend')}`",
        f"- real_perception: `{str(scene.get('real_perception')).lower()}`",
        f"- mock_or_synthetic: `{str(scene.get('mock_or_synthetic')).lower()}`",
        f"- depth_valid_ratio: `{report.get('moge_depth_valid_ratio')}`",
        f"- confidence_valid_ratio: `{report.get('moge_confidence_valid_ratio')}`",
        "",
        "## Spatial Pack",
        f"- example_count: `{report.get('spatial_pack_example_count')}`",
        f"- matched_free_ratio_mean: `{matched.get('free_ratio_mean')}`",
        f"- matched_obstacle_ratio_mean: `{matched.get('obstacle_ratio_mean')}`",
        f"- matched_unknown_ratio_mean: `{matched.get('unknown_ratio_mean')}`",
        "",
        "## Projection",
        f"- direct_bev_conversion_attempted: `false`",
        f"- projection_blocked_reason: {scene.get('projection_blocked_reason', report.get('projection_blocked_reason'))}",
        "",
        "## Blockers",
        *([f"- `{item}`" for item in report.get("hard_blockers", [])] if report.get("hard_blockers") else ["- `none`"]),
        "",
        "## Safety",
        "- No model was trained, no policy/scorer loop was run, and no cmd_vel/raw PWM was emitted.",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_aggregate_markdown(path: Path, report: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Goal 17A MoGe OpenLORIS Teacher Quality Report",
        "",
        "## Questions",
        f"- Did real MoGe run on public robot-frame routes? `{str(report['did_real_moge_run_on_public_robot_frame_routes']).lower()}`",
        f"- Did QA pass structurally? `{str(report['did_qa_pass_structurally']).lower()}`",
        f"- Does MoGe look useful as single-frame geometry supervision? `{str(report['does_moge_look_useful_as_single_frame_geometry_supervision']).lower()}`",
        f"- Should proceed to a MoGe SpatialTrainPack candidate? `{str(report['should_proceed_to_moge_spatial_train_pack_candidate']).lower()}`",
        f"- Aggregate recommendation: `{report['aggregate_recommendation']}`",
        "",
        "## Still Missing Before Temporal Memory Training",
        *[f"- {item}" for item in report["still_missing_before_temporal_memory_training"]],
        "",
        "## Routes",
    ]
    for item in report["route_summaries"]:
        lines.append(
            f"- `{item.get('sequence') or item.get('route')}`: next_allowed_use=`{item.get('next_allowed_use')}`, "
            f"matched=`{item.get('matched_frame_count')}`, qa_structural_pass=`{str(item.get('qa_structural_pass')).lower()}`"
        )
    lines.extend(
        [
            "",
            "## Safety",
            "- No model was trained, no policy/scorer loop was run, MoGe was not promoted to robot-frame truth, and no cmd_vel/raw PWM was emitted.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _resolve_viz_path(path: str | Path) -> Path:
    target = Path(path)
    if target.suffix:
        return target
    return target / "comparison_review.ppm"


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _nan_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(out):
        return None
    return out


def _load_reports(paths: Iterable[str]) -> list[JsonDict]:
    return [read_json_object(path) for path in paths]


def _compare_command(args: argparse.Namespace) -> str:
    return "python -m homebrain.tools.compare_moge_scene_to_spatial_pack " + " ".join(
        _iterable_strings(sys.argv[1:])
    )


def _iterable_strings(values: Iterable[str]) -> list[str]:
    return [str(value) for value in values]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a real MoGe SceneTeacherPack with a robot-frame SpatialTrainPack."
    )
    parser.add_argument("--route", help="Input HomeBrain route directory.")
    parser.add_argument("--scene-teacher", help="Input real MoGe SceneTeacherPack directory.")
    parser.add_argument("--spatial-pack", help="Input robot-frame SpatialTrainPack directory.")
    parser.add_argument("--out-json", required=True, help="Output report JSON.")
    parser.add_argument("--out-md", required=True, help="Output report Markdown.")
    parser.add_argument("--out-viz", help="Output comparison contact-sheet PPM or directory.")
    parser.add_argument(
        "--aggregate-from",
        nargs="+",
        default=None,
        help="Aggregate existing per-route comparison JSON reports instead of running one comparison.",
    )
    args = parser.parse_args(argv)
    command = _compare_command(args)
    if args.aggregate_from:
        report = write_aggregate_report(
            comparison_reports=_load_reports(args.aggregate_from),
            out_json=args.out_json,
            out_md=args.out_md,
            command=command,
        )
        print(deterministic_json({"route_count": report["route_count"], "aggregate_recommendation": report["aggregate_recommendation"]}))
        return 0 if report["blocked_route_count"] == 0 else 1
    missing_args = [
        name
        for name, value in (("--route", args.route), ("--scene-teacher", args.scene_teacher), ("--spatial-pack", args.spatial_pack), ("--out-viz", args.out_viz))
        if not value
    ]
    if missing_args:
        parser.error("comparison mode requires " + ", ".join(missing_args))
    report = compare_moge_scene_to_spatial_pack(
        route_dir=args.route,
        scene_teacher_dir=args.scene_teacher,
        spatial_pack_dir=args.spatial_pack,
        out_json=args.out_json,
        out_md=args.out_md,
        out_viz=args.out_viz,
        command=command,
    )
    print(deterministic_json({"next_allowed_use": report["next_allowed_use"], "matched_frame_count": report["matched_frame_count"]}))
    return 0 if report["next_allowed_use"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
