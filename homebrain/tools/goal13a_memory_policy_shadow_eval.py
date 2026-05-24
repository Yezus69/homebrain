from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from homebrain.brain.spatial_memory_v1 import load_checkpoint
from homebrain.data.spatial_dataset import read_json, write_json
from homebrain.messages.schema import BrainOutputEvent, JsonDict, deterministic_json
from homebrain.policies.build_action_label_pack import (
    COLLISION_THRESHOLD,
    UNKNOWN_BLOCK_THRESHOLD,
    ExpertCandidateLabel,
    expert_labels_for_frame,
)
from homebrain.policies.candidate_trajectories import CandidateTrajectory, generate_default_candidates
from homebrain.policies.run_trajectory_scorer import _record_from_model_event
from homebrain.policies.trajectory_scorer import CoverageMemory, LocalBev, TrajectoryDecision, score_trajectories
from homebrain.policies.trajectory_scorer_net_v0 import (
    CANDIDATE_FEATURE_NAMES,
    load_trajectory_scorer_checkpoint,
    score_local_bev_with_model,
)
from homebrain.replay.segment_log import read_events
from homebrain.train.spatial_dataset import (
    BEV_OUTPUT_CHANNELS,
    SpatialPackSpec,
    dataset_manifest_hashes,
    load_spatial_dataset_manifest,
)
from homebrain.train.spatial_temporal_dataset import SpatialTemporalMultiPackDataset, SpatialTemporalTrainDataset
from homebrain.train.spatial_v0_common import batch_to_device
from homebrain.train.spatial_v1_hard_eval import (
    OCCLUSION_MODES,
    _forward_deployment,
    _forward_with_observation_mask,
    _future_reveal_masks,
    deterministic_occlusion_keep_mask,
)


GOAL13A_REPORT_SCHEMA_VERSION = "homebrain.goal13a_memory_policy_shadow_eval_report.v0"
GOAL13A_DECISION_SCHEMA_VERSION = "homebrain.goal13a_shadow_trajectory_decision.v0"
DEFAULT_DATASET_MANIFEST = "runs/goal12b_spatial_memory_v1/spatial_dataset_manifest_goal12b.json"
DEFAULT_V1_CHECKPOINT = "runs/goal12b_spatial_memory_v1/v1_window4_route_pose_warm_start/checkpoint.pt"
DEFAULT_V0_MODELD_ROOT = "runs/goal11b_nightly/modeld_best_spatial"
DEFAULT_ACTION_LABEL_PACK = "runs/goal11b_nightly/action_label_pack_v4"
DEFAULT_GOAL12C_REPORT = "runs/goal12c_spatial_memory_v1_hard_validation_report.json"
DEFAULT_TRUE_ROUTE_OUT_ROOT = "runs/goal12c_true_route_out"
COLLAPSE_ENTROPY_MIN = 1.0
COLLAPSE_DOMINANT_MAX = 0.65
STOP_FRACTION_TOLERANCE = 0.05
METRIC_TOLERANCE = 1.0e-9
BEV_DECISION_SOURCES: tuple[str, ...] = (
    "oracle_bev",
    "v0_current_bev",
    "v1_current_bev",
    "v1_memory_bev",
)


@dataclass(frozen=True)
class ShadowFrame:
    source_name: str
    sequence_id: str
    camera_id: str
    frame_id: int
    timestamp_ns: int
    pose_delta: tuple[float, float, float] | None
    oracle_bev: LocalBev
    v1_current_bev: LocalBev
    v1_memory_bev: LocalBev
    v0_current_bev: LocalBev | None
    future_motion_label: str | None
    subset_reason: str

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.source_name, self.sequence_id, self.camera_id, self.frame_id)


def run_goal13a_shadow_eval(
    *,
    dataset_manifest: str | Path = DEFAULT_DATASET_MANIFEST,
    v1_checkpoint: str | Path = DEFAULT_V1_CHECKPOINT,
    v0_modeld_root: str | Path | None = DEFAULT_V0_MODELD_ROOT,
    action_label_pack: str | Path | None = DEFAULT_ACTION_LABEL_PACK,
    goal12c_report: str | Path | None = DEFAULT_GOAL12C_REPORT,
    true_route_out_root: str | Path = DEFAULT_TRUE_ROUTE_OUT_ROOT,
    out_json: str | Path = "runs/goal13a_memory_policy_shadow_eval_report.json",
    out_md: str | Path = "runs/goal13a_memory_policy_shadow_eval_report.md",
    decisions_jsonl: str | Path = "runs/goal13a_memory_policy_shadow_eval_decisions.jsonl",
    contact_sheet: str | Path = "runs/goal13a_memory_policy_shadow_eval_worst_disagreements.ppm",
    scorer_checkpoint: str | Path | None = None,
    device_name: str | None = None,
    batch_size: int = 32,
    split: str = "val",
    future_horizon: int = 3,
    max_frames_per_mode: int | None = None,
    command: str | None = None,
) -> JsonDict:
    started = time.perf_counter()
    dataset_manifest = Path(dataset_manifest)
    v1_checkpoint = Path(v1_checkpoint)
    specs = load_spatial_dataset_manifest(dataset_manifest)
    model, payload = load_checkpoint(v1_checkpoint, map_location="cpu")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    window_length = int(metadata.get("window_length", 4))
    sensor_context_mode = str(metadata.get("sensor_context_mode", "masks"))
    missing_pose_behavior = str(metadata.get("missing_pose_behavior", model.config.missing_pose_behavior))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    learned_scorer = None
    if scorer_checkpoint is not None:
        learned_scorer, _scorer_payload = load_trajectory_scorer_checkpoint(scorer_checkpoint, map_location=device)
        learned_scorer.to(device)
        learned_scorer.eval()

    dataset = _build_temporal_dataset(
        specs=specs,
        split=split,
        window_length=window_length,
        sensor_context_mode=sensor_context_mode,
        missing_pose_behavior=missing_pose_behavior,
    )
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    meters_per_cell, robot_radius_m = _candidate_geometry(specs[0], model)
    candidates = generate_default_candidates(
        grid_shape=dataset.grid_shape,
        meters_per_cell=meters_per_cell,
        robot_radius_m=robot_radius_m,
    )
    v0_map = _load_v0_modeld_map(v0_modeld_root) if v0_modeld_root is not None and Path(v0_modeld_root).exists() else {}
    future_map = _load_future_motion_map(action_label_pack) if action_label_pack is not None and Path(action_label_pack).exists() else {}

    normal_frames = _collect_shadow_frames(
        model=model,
        loader=loader,
        device=device,
        mode="normal",
        candidates=candidates,
        v0_map=v0_map,
        future_map=future_map,
        future_horizon=future_horizon,
        max_frames=max_frames_per_mode,
    )
    normal_records = score_shadow_frames(
        normal_frames,
        candidates=candidates,
        meters_per_cell=meters_per_cell,
        mode="normal",
        scorer_model=learned_scorer,
        scorer_device=device,
    )
    hidden_frames = _collect_shadow_frames(
        model=model,
        loader=loader,
        device=device,
        mode="hidden_cell",
        candidates=candidates,
        v0_map=v0_map,
        future_map=future_map,
        future_horizon=future_horizon,
        max_frames=max_frames_per_mode,
    )
    hidden_records = score_shadow_frames(
        hidden_frames,
        candidates=candidates,
        meters_per_cell=meters_per_cell,
        mode="hidden_cell",
        scorer_model=learned_scorer,
        scorer_device=device,
    )

    occlusion_records_by_mode: dict[str, list[JsonDict]] = {}
    for occlusion_mode in OCCLUSION_MODES:
        occlusion_frames = _collect_shadow_frames(
            model=model,
            loader=loader,
            device=device,
            mode="occlusion",
            candidates=candidates,
            v0_map=v0_map,
            future_map=future_map,
            future_horizon=future_horizon,
            occlusion_mode=str(occlusion_mode),
            max_frames=max_frames_per_mode,
        )
        occlusion_records_by_mode[str(occlusion_mode)] = score_shadow_frames(
            occlusion_frames,
            candidates=candidates,
            meters_per_cell=meters_per_cell,
            mode=f"occlusion:{occlusion_mode}",
            scorer_model=learned_scorer,
            scorer_device=device,
        )
    all_occlusion_records = [
        record for records in occlusion_records_by_mode.values() for record in records
    ]

    route_out_metrics = _run_route_out_shadow_eval(
        true_route_out_root=true_route_out_root,
        v0_modeld_root=v0_modeld_root,
        action_label_pack=action_label_pack,
        device_name=device_name,
        batch_size=batch_size,
        split=split,
        future_horizon=future_horizon,
        max_frames_per_mode=max_frames_per_mode,
        scorer_checkpoint=scorer_checkpoint,
    )

    normal_metrics = summarize_decision_records(normal_records)
    hidden_metrics = summarize_decision_records(hidden_records)
    occlusion_metrics = {
        "aggregate": summarize_decision_records(all_occlusion_records),
        "modes": {
            mode: summarize_decision_records(records)
            for mode, records in sorted(occlusion_records_by_mode.items())
        },
    }
    if learned_scorer is None:
        gate = memory_action_benefit_gate(
            normal_metrics=normal_metrics,
            hidden_metrics=hidden_metrics,
            occlusion_metrics=occlusion_metrics,
            route_out_metrics=route_out_metrics,
        )
    else:
        gate = learned_scorer_memory_action_gate(
            normal_metrics=normal_metrics,
            route_out_metrics=route_out_metrics,
        )
    blockers = gate["failure_reasons"] if not gate["memory_action_benefit_pass"] else ["none"]
    if gate["memory_action_benefit_pass"]:
        next_goal = "Use this shadow-eval evidence to design a learned memory-aware trajectory scorer, still replay-only and not control-safe."
    else:
        next_goal = "Do not train a learned memory scorer yet; inspect the failing action gate and add data or scorer calibration only where the shadow eval shows why."

    decisions_path = Path(decisions_jsonl)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    all_records = normal_records + hidden_records + all_occlusion_records
    with decisions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in all_records:
            handle.write(deterministic_json(record))
            handle.write("\n")
    contact_records = _worst_disagreement_records(all_records, limit=24)
    _write_worst_contact_sheet(Path(contact_sheet), contact_records, candidates=candidates)

    goal12c_summary = _load_optional_json(goal12c_report)
    data_sources = {
        "dataset_manifest": dataset_manifest.as_posix(),
        "spatial_packs": [spec.dataset_dir.as_posix() for spec in specs],
        "feature_artifacts": [spec.feature_dir.as_posix() for spec in specs],
        "v0_modeld_root": Path(v0_modeld_root).as_posix() if v0_modeld_root is not None else None,
        "v0_modeld_frame_count": len(v0_map),
        "action_label_pack": Path(action_label_pack).as_posix() if action_label_pack is not None else None,
        "future_motion_label_count": len(future_map),
        "scorer_checkpoint": Path(scorer_checkpoint).as_posix() if scorer_checkpoint is not None else None,
        "learned_trajectory_scorer_used": bool(learned_scorer is not None),
        "goal12c_report": Path(goal12c_report).as_posix() if goal12c_report is not None else None,
        "goal12c_pass": goal12c_summary.get("pass_fail_gates", {}).get("goal12c_hard_validation_pass")
        if isinstance(goal12c_summary.get("pass_fail_gates"), dict)
        else None,
    }
    report = {
        "schema_version": GOAL13A_REPORT_SCHEMA_VERSION,
        "goal": "13A SpatialMemoryV1 memory-to-trajectory shadow evaluation",
        "data_sources": data_sources,
        "checkpoints": {
            "v1_checkpoint": v1_checkpoint.as_posix(),
            "scorer_checkpoint": Path(scorer_checkpoint).as_posix() if scorer_checkpoint is not None else None,
            "v1_checkpoint_metadata": {
                "model_name": payload.get("model_name"),
                "window_length": window_length,
                "sensor_context_mode": sensor_context_mode,
                "missing_pose_behavior": missing_pose_behavior,
            },
        },
        "data_manifest_hashes": dataset_manifest_hashes(dataset_manifest=dataset_manifest, pack_specs=specs),
        "candidate_config": {
            "candidate_count": len(candidates),
            "candidate_ids": [candidate.id for candidate in candidates],
            "meters_per_cell": meters_per_cell,
            "robot_radius_m": robot_radius_m,
        },
        "normal_metrics": normal_metrics,
        "hidden_cell_metrics": hidden_metrics,
        "occlusion_metrics": occlusion_metrics,
        "route_out_metrics": route_out_metrics,
        "pass_fail_gates": gate,
        "blockers": blockers,
        "artifacts_created": {
            "json_report": Path(out_json).as_posix(),
            "markdown_report": Path(out_md).as_posix(),
            "decisions_jsonl": decisions_path.as_posix(),
            "worst_disagreement_contact_sheet": Path(contact_sheet).as_posix(),
        },
        "worst_disagreement_examples": _worst_example_summaries(contact_records),
        "safety_flags": {
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "product_training_approved": False,
            "runtime_dependency": False,
            "cmd_vel_emitted": False,
            "raw_pwm_emitted": False,
            "memory_bev_treated_as_oracle": False,
            "learned_memory_scorer_trained": False,
            "learned_trajectory_scorer_used": bool(learned_scorer is not None),
        },
        "commands_run": [command] if command else [],
        "runtime_sec": float(time.perf_counter() - started),
        "recommended_next_goal": next_goal,
    }
    write_json(out_json, report, pretty=True)
    _write_markdown(out_md, report)
    return report


def score_shadow_frames(
    frames: list[ShadowFrame],
    *,
    candidates: list[CandidateTrajectory],
    meters_per_cell: float,
    mode: str,
    scorer_model: torch.nn.Module | None = None,
    scorer_device: torch.device | None = None,
) -> list[JsonDict]:
    coverage: dict[str, CoverageMemory] = {
        source: CoverageMemory(frames[0].oracle_bev.shape, meters_per_cell=meters_per_cell)
        for source in BEV_DECISION_SOURCES
        if frames
    }
    records: list[JsonDict] = []
    for index, frame in enumerate(sorted(frames, key=_shadow_sort_key)):
        source_bevs: dict[str, LocalBev | None] = {
            "oracle_bev": frame.oracle_bev,
            "v0_current_bev": frame.v0_current_bev,
            "v1_current_bev": frame.v1_current_bev,
            "v1_memory_bev": frame.v1_memory_bev,
        }
        oracle_labels = expert_labels_for_frame(bev=frame.oracle_bev, candidates=candidates)
        decisions: dict[str, JsonDict] = {}
        raw_decisions: dict[str, TrajectoryDecision] = {}
        for source, bev in source_bevs.items():
            if bev is None:
                continue
            if source == "v1_memory_bev" and bev.source == "oracle_bev":
                raise ValueError("v1_memory_bev cannot be treated as oracle")
            memory = coverage[source]
            memory.align_with_pose_delta(frame.pose_delta)
            if scorer_model is not None and source == "oracle_bev" and frame.future_motion_label is not None:
                summary = _bc_oracle_decision_summary(
                    candidates=candidates,
                    oracle_labels=oracle_labels,
                    future_motion_label=frame.future_motion_label,
                )
            elif scorer_model is None:
                decision = score_trajectories(bev=bev, candidates=candidates, coverage_memory=memory)
                summary = _decision_summary(
                    decision=decision,
                    oracle_labels=oracle_labels,
                    future_motion_label=frame.future_motion_label,
                )
            else:
                summary = _learned_decision_summary(
                    scorer_model=scorer_model,
                    scorer_device=scorer_device or torch.device("cpu"),
                    bev=bev,
                    candidates=candidates,
                    coverage_memory=memory,
                    oracle_labels=oracle_labels,
                    future_motion_label=frame.future_motion_label,
                )
            memory.update_current_frame(bev)
            if scorer_model is None:
                raw_decisions[source] = decision  # type: ignore[assignment]
            decisions[source] = summary
        oracle_action = decisions["oracle_bev"]["selected_candidate_id"]
        for source, summary in decisions.items():
            summary["agreement_with_oracle_action"] = bool(summary["selected_candidate_id"] == oracle_action)
        comparison = _memory_vs_current(decisions)
        records.append(
            {
                "schema_version": GOAL13A_DECISION_SCHEMA_VERSION,
                "mode": mode,
                "frame_index": index,
                "source_name": frame.source_name,
                "sequence_id": frame.sequence_id,
                "camera_id": frame.camera_id,
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "subset_reason": frame.subset_reason,
                "candidate_count": len(candidates),
                "future_motion_label": frame.future_motion_label,
                "decisions": decisions,
                "memory_vs_current": comparison,
                "cmd_vel": None,
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
                "product_training_approved": False,
                "raw_pwm_emitted": False,
            }
        )
    return records


def summarize_decision_records(records: list[JsonDict]) -> JsonDict:
    per_source = {
        source: _metrics_for_source(records, source)
        for source in BEV_DECISION_SOURCES
    }
    per_route: dict[str, JsonDict] = {}
    route_names = sorted({str(record["source_name"]) for record in records})
    for route in route_names:
        route_records = [record for record in records if record["source_name"] == route]
        per_route[route] = {
            source: _metrics_for_source(route_records, source)
            for source in BEV_DECISION_SOURCES
        }
        per_route[route]["memory_vs_current"] = _memory_comparison_metrics(route_records)
    result = {
        "schema_version": "homebrain.goal13a_mode_shadow_metrics.v0",
        "frame_count": len(records),
        "per_source_metrics": per_source,
        "per_route_metrics": per_route,
        "memory_vs_current": _memory_comparison_metrics(records),
        "collapse_thresholds": {
            "action_entropy_min": COLLAPSE_ENTROPY_MIN,
            "dominant_action_fraction_max": COLLAPSE_DOMINANT_MAX,
        },
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
    }
    return result


def memory_action_benefit_gate(
    *,
    normal_metrics: JsonDict,
    hidden_metrics: JsonDict,
    occlusion_metrics: JsonDict,
    route_out_metrics: JsonDict,
) -> JsonDict:
    reasons: list[str] = []
    normal_current, normal_memory = _current_memory_metrics(normal_metrics)
    if normal_current is None or normal_memory is None:
        reasons.append("normal metrics missing v1_current_bev or v1_memory_bev")
    else:
        improves_action_metric = bool(
            _less(normal_memory, normal_current, "unsafe_selected_rate")
            or _less(normal_memory, normal_current, "unknown_penalty_mean")
            or _greater(normal_memory, normal_current, "agreement_with_oracle_action")
            or (
                int(normal_memory.get("future_motion_label_valid_count", 0)) > 0
                and _greater(normal_memory, normal_current, "agreement_with_future_motion_label")
            )
        )
        if not improves_action_metric:
            reasons.append(
                "v1_memory_bev did not reduce unsafe_selected_rate/unknown_penalty_mean or improve oracle/future-motion agreement"
            )
        if _greater(normal_memory, normal_current, "collision_proxy_rate"):
            reasons.append("normal v1_memory_bev increased collision_proxy_rate versus v1_current_bev")
        if bool(normal_memory.get("goal11b_distribution_collapse_flag")):
            reasons.append("normal v1_memory_bev crosses Goal 11B collapse threshold")
        if _stop_increase_fails(normal_current, normal_memory):
            reasons.append("normal v1_memory_bev stop_fraction increased beyond tolerance without all motion candidates unsafe")

    for scope_name, metrics in (
        ("hidden_cell", hidden_metrics),
        ("occlusion_aggregate", occlusion_metrics.get("aggregate", {}) if isinstance(occlusion_metrics, dict) else {}),
    ):
        current, memory = _current_memory_metrics(metrics)
        if current is None or memory is None:
            continue
        if _greater(memory, current, "collision_proxy_rate"):
            reasons.append(f"{scope_name} v1_memory_bev increased collision_proxy_rate versus v1_current_bev")
        if bool(memory.get("goal11b_distribution_collapse_flag")):
            reasons.append(f"{scope_name} v1_memory_bev crosses Goal 11B collapse threshold")
        if _stop_increase_fails(current, memory):
            reasons.append(f"{scope_name} v1_memory_bev stop_fraction increased beyond tolerance")

    route_out_available = bool(route_out_metrics.get("available")) and not bool(route_out_metrics.get("blocked"))
    fold_deltas = [
        float(fold.get("memory_quality_delta", 0.0))
        for fold in route_out_metrics.get("folds", [])
        if isinstance(fold, dict)
    ]
    if not route_out_available or not fold_deltas:
        reasons.append("true route-out shadow metrics unavailable")
    else:
        if not any(delta > METRIC_TOLERANCE for delta in fold_deltas):
            reasons.append("v1_memory_bev did not improve any true route-out fold")
        if any(delta < -METRIC_TOLERANCE for delta in fold_deltas):
            reasons.append("v1_memory_bev worsened at least one true route-out fold")

    return {
        "memory_action_benefit_pass": not reasons,
        "failure_reasons": reasons,
        "comparison_source": "v1_memory_bev_vs_v1_current_bev",
        "collision_proxy_tolerance": METRIC_TOLERANCE,
        "stop_fraction_tolerance": STOP_FRACTION_TOLERANCE,
        "goal11b_collapse_thresholds": {
            "action_entropy_min": COLLAPSE_ENTROPY_MIN,
            "dominant_action_fraction_max": COLLAPSE_DOMINANT_MAX,
        },
        "route_out_memory_quality_deltas": fold_deltas,
        "learned_memory_scorer_trained": False,
    }


def learned_scorer_memory_action_gate(
    *,
    normal_metrics: JsonDict,
    route_out_metrics: JsonDict,
) -> JsonDict:
    reasons: list[str] = []
    current, memory = _current_memory_metrics(normal_metrics)
    comparison = normal_metrics.get("memory_vs_current", {}) if isinstance(normal_metrics.get("memory_vs_current"), dict) else {}
    changed_fraction = float(comparison.get("memory_vs_current_action_changed_fraction", 0.0))
    current_future = 0.0
    memory_future = 0.0
    if current is None or memory is None:
        reasons.append("normal metrics missing v1_current_bev or v1_memory_bev")
    else:
        current_future = float(current.get("agreement_with_future_motion_label", 0.0))
        memory_future = float(memory.get("agreement_with_future_motion_label", 0.0))
        if changed_fraction < 0.05:
            reasons.append("memory changed v1_current decisions on fewer than 5% of normal held-out frames")
        if memory_future <= current_future + METRIC_TOLERANCE:
            reasons.append("memory did not improve agreement with future_motion labels on normal held-out frames")
        if _greater(memory, current, "collision_proxy_rate"):
            reasons.append("normal v1_memory_bev increased collision_proxy_rate versus v1_current_bev")

    route_out_future_deltas: list[float] = []
    for fold in route_out_metrics.get("folds", []):
        if not isinstance(fold, dict):
            continue
        metrics = fold.get("metrics")
        if not isinstance(metrics, dict):
            continue
        fold_current, fold_memory = _current_memory_metrics(metrics)
        if fold_current is None or fold_memory is None:
            continue
        route_out_future_deltas.append(
            float(fold_memory.get("agreement_with_future_motion_label", 0.0))
            - float(fold_current.get("agreement_with_future_motion_label", 0.0))
        )

    return {
        "memory_action_benefit_pass": not reasons,
        "failure_reasons": reasons,
        "comparison_source": "v1_memory_bev_vs_v1_current_bev",
        "gate_mode": "learned_v5_future_motion_behavior_cloning",
        "required_memory_action_changed_fraction": 0.05,
        "normal_memory_action_changed_fraction": changed_fraction,
        "normal_current_future_motion_agreement": current_future,
        "normal_memory_future_motion_agreement": memory_future,
        "normal_future_motion_agreement_delta": float(memory_future - current_future),
        "route_out_future_motion_agreement_deltas": route_out_future_deltas,
        "learned_memory_scorer_trained": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "product_training_approved": False,
    }


def _collect_shadow_frames(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    candidates: list[CandidateTrajectory],
    v0_map: dict[tuple[str, str, str, int], LocalBev],
    future_map: dict[tuple[str, str, str, int], str],
    future_horizon: int,
    occlusion_mode: str | None = None,
    max_frames: int | None = None,
) -> list[ShadowFrame]:
    del candidates
    frames: list[ShadowFrame] = []
    seen: set[tuple[str, str, str, int]] = set()
    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device)
        outputs = _forward_deployment(model, batch, pose_mode="route_pose")
        if mode == "normal":
            include = _include_final_steps(raw_batch)
            subset_reason = "final_step_deployment_window"
        elif mode == "hidden_cell":
            include = _include_hidden_steps(batch, outputs, future_horizon=future_horizon, meters_per_cell=float(model.config.meters_per_cell))
            subset_reason = "goal12c_future_hidden_cell_subset"
        elif mode == "occlusion":
            if occlusion_mode is None:
                raise ValueError("occlusion mode requires occlusion_mode")
            base_update = (outputs["update_mask"] > 0.5).to(dtype=batch["features"].dtype)
            keep = deterministic_occlusion_keep_mask(base_update, mode=occlusion_mode, frame_ids=batch["frame_id"])
            occluded_update = base_update * keep
            outputs = _forward_with_observation_mask(
                model,
                batch,
                observation_mask=occluded_update,
                pose_mode="route_pose",
            )
            include = _include_occluded_steps(base_update, occluded_update)
            subset_reason = f"goal12c_occlusion_{occlusion_mode}"
        else:
            raise ValueError(f"unsupported mode: {mode}")
        for batch_index, step_index in include:
            frame = _shadow_frame_from_batch(
                raw_batch=raw_batch,
                outputs=outputs,
                batch_index=batch_index,
                step_index=step_index,
                v0_map=v0_map,
                future_map=future_map,
                subset_reason=subset_reason,
            )
            if frame.key in seen:
                continue
            seen.add(frame.key)
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                return frames
    return frames


def _include_final_steps(raw_batch: dict[str, Any]) -> list[tuple[int, int]]:
    frame_ids = raw_batch["frame_id"]
    batch_size = int(frame_ids.shape[0])
    final_step = int(frame_ids.shape[1]) - 1
    return [(batch_index, final_step) for batch_index in range(batch_size)]


def _include_hidden_steps(
    batch: dict[str, Any],
    outputs: dict[str, Any],
    *,
    future_horizon: int,
    meters_per_cell: float,
) -> list[tuple[int, int]]:
    current_trusted = outputs["update_mask"] > 0.5
    includes: set[tuple[int, int]] = set()
    for reveal in _future_reveal_masks(batch, future_horizon=future_horizon, meters_per_cell=meters_per_cell):
        step = int(reveal["step"])
        future_observed = reveal["observed"] > 0.5
        valid = reveal["valid"].view(-1, 1, 1, 1)
        hidden_cells = future_observed & (~current_trusted[:, step]) & valid
        for batch_index in range(int(hidden_cells.shape[0])):
            if int(torch.count_nonzero(hidden_cells[batch_index]).detach().cpu()) > 0:
                includes.add((batch_index, step))
    return sorted(includes)


def _include_occluded_steps(base_update: torch.Tensor, occluded_update: torch.Tensor) -> list[tuple[int, int]]:
    dropped = (base_update > 0.5) & (occluded_update <= 0.0)
    includes: list[tuple[int, int]] = []
    batch_size, steps = int(dropped.shape[0]), int(dropped.shape[1])
    for batch_index in range(batch_size):
        for step_index in range(steps):
            if int(torch.count_nonzero(dropped[batch_index, step_index]).detach().cpu()) > 0:
                includes.append((batch_index, step_index))
    return includes


def _shadow_frame_from_batch(
    *,
    raw_batch: dict[str, Any],
    outputs: dict[str, Any],
    batch_index: int,
    step_index: int,
    v0_map: dict[tuple[str, str, str, int], LocalBev],
    future_map: dict[tuple[str, str, str, int], str],
    subset_reason: str,
) -> ShadowFrame:
    source_name = _string_at(raw_batch["source_name"], batch_index)
    sequence_id = _string_at(raw_batch["sequence_id"], batch_index)
    camera_id = _string_at(raw_batch["camera_id"], batch_index)
    frame_id = int(raw_batch["frame_id"][batch_index, step_index].item())
    timestamp_ns = int(raw_batch["timestamp_ns"][batch_index, step_index].item())
    key = (source_name, sequence_id, camera_id, frame_id)
    pose_delta = None
    if float(raw_batch["pose_delta_to_current_mask"][batch_index, step_index].reshape(-1)[0].item()) > 0.0:
        pose_delta = tuple(
            float(value)
            for value in raw_batch["pose_delta_to_current"][batch_index, step_index].detach().cpu().numpy().reshape(3)
        )
    oracle = _bev_from_channels(
        raw_batch["bev_labels"][batch_index, step_index],
        source="oracle_bev",
        confidence=torch.clamp(raw_batch["bev_label_mask"][batch_index, step_index, 0], 0.0, 1.0),
    )
    uncertainty = outputs.get("uncertainty_grid")
    uncertainty_grid = uncertainty[batch_index, step_index, 0] if isinstance(uncertainty, torch.Tensor) else None
    current = _bev_from_logits(
        outputs["current_bev_logits"][batch_index, step_index],
        source="v1_current_bev",
        uncertainty=uncertainty_grid,
    )
    memory = _bev_from_logits(
        outputs["fused_memory_bev_logits"][batch_index, step_index],
        source="v1_memory_bev",
        uncertainty=uncertainty_grid,
    )
    return ShadowFrame(
        source_name=source_name,
        sequence_id=sequence_id,
        camera_id=camera_id,
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        pose_delta=pose_delta,
        oracle_bev=oracle,
        v1_current_bev=current,
        v1_memory_bev=memory,
        v0_current_bev=v0_map.get(key),
        future_motion_label=future_map.get(key),
        subset_reason=subset_reason,
    )


def _run_route_out_shadow_eval(
    *,
    true_route_out_root: str | Path,
    v0_modeld_root: str | Path | None,
    action_label_pack: str | Path | None,
    device_name: str | None,
    batch_size: int,
    split: str,
    future_horizon: int,
    max_frames_per_mode: int | None,
    scorer_checkpoint: str | Path | None = None,
) -> JsonDict:
    root = Path(true_route_out_root)
    if not root.exists():
        return {
            "available": False,
            "blocked": True,
            "reason": f"true route-out root does not exist: {root.as_posix()}",
            "folds": [],
        }
    v0_map = _load_v0_modeld_map(v0_modeld_root) if v0_modeld_root is not None and Path(v0_modeld_root).exists() else {}
    future_map = _load_future_motion_map(action_label_pack) if action_label_pack is not None and Path(action_label_pack).exists() else {}
    scorer_model = None
    scorer_device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if scorer_checkpoint is not None:
        scorer_model, _payload = load_trajectory_scorer_checkpoint(scorer_checkpoint, map_location=scorer_device)
        scorer_model.to(scorer_device)
        scorer_model.eval()
    folds: list[JsonDict] = []
    errors: list[JsonDict] = []
    for heldout_manifest in sorted(root.glob("*/heldout_manifest.json")):
        fold_root = heldout_manifest.parent
        checkpoint = fold_root / "v1_window4" / "checkpoint.pt"
        if not checkpoint.exists():
            errors.append({"fold": fold_root.name, "error": "missing v1_window4 checkpoint"})
            continue
        try:
            specs = load_spatial_dataset_manifest(heldout_manifest)
            model, payload = load_checkpoint(checkpoint, map_location="cpu")
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            window_length = int(metadata.get("window_length", 4))
            sensor_context_mode = str(metadata.get("sensor_context_mode", "masks"))
            missing_pose_behavior = str(metadata.get("missing_pose_behavior", model.config.missing_pose_behavior))
            device = scorer_device
            model.to(device)
            model.eval()
            dataset = _build_temporal_dataset(
                specs=specs,
                split=split,
                window_length=window_length,
                sensor_context_mode=sensor_context_mode,
                missing_pose_behavior=missing_pose_behavior,
            )
            loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
            meters_per_cell, robot_radius_m = _candidate_geometry(specs[0], model)
            candidates = generate_default_candidates(
                grid_shape=dataset.grid_shape,
                meters_per_cell=meters_per_cell,
                robot_radius_m=robot_radius_m,
            )
            frames = _collect_shadow_frames(
                model=model,
                loader=loader,
                device=device,
                mode="normal",
                candidates=candidates,
                v0_map=v0_map,
                future_map=future_map,
                future_horizon=future_horizon,
                max_frames=max_frames_per_mode,
            )
            records = score_shadow_frames(
                frames,
                candidates=candidates,
                meters_per_cell=meters_per_cell,
                mode=f"route_out:{fold_root.name}",
                scorer_model=scorer_model,
                scorer_device=scorer_device,
            )
            metrics = summarize_decision_records(records)
            comparison = metrics["memory_vs_current"]
            folds.append(
                {
                    "heldout_route": specs[0].source_name if specs else fold_root.name,
                    "fold_root": fold_root.as_posix(),
                    "checkpoint": checkpoint.as_posix(),
                    "frame_count": len(records),
                    "metrics": metrics,
                    "memory_quality_delta": float(comparison.get("memory_oracle_score_mean_delta", 0.0)),
                    "memory_improved_decision_fraction": comparison.get("memory_improved_decision_fraction"),
                    "memory_worsened_decision_fraction": comparison.get("memory_worsened_decision_fraction"),
                }
            )
        except Exception as exc:  # noqa: BLE001 - report partial route-out evidence.
            errors.append({"fold": fold_root.name, "error": str(exc)})
    return {
        "schema_version": "homebrain.goal13a_true_route_out_shadow_metrics.v0",
        "available": bool(folds),
        "blocked": bool(errors) or not bool(folds),
        "fold_count": len(folds),
        "folds": folds,
        "errors": errors,
        "true_leave_one_route_out": bool(folds),
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
    }


def _metrics_for_source(records: list[JsonDict], source: str) -> JsonDict:
    items = [
        record["decisions"][source]
        for record in records
        if isinstance(record.get("decisions"), dict) and source in record["decisions"]
    ]
    selected = [str(item["selected_candidate_id"]) for item in items]
    distribution = _distribution(selected)
    dominant = max(distribution.values(), default=0) / max(len(items), 1)
    entropy = _entropy(distribution)
    stop_count = sum(1 for value in selected if value == "stop")
    future_valid = [item for item in items if item.get("future_motion_label_valid") is True]
    future_agree = [item for item in future_valid if item.get("agreement_with_future_motion_label") is True]
    collapse_reasons = _collapse_reasons(entropy=entropy, dominant_action_fraction=dominant)
    return {
        "frame_count": len(items),
        "selected_action_distribution": distribution,
        "action_entropy": entropy,
        "dominant_action_fraction": float(dominant),
        "stop_fraction": float(stop_count / max(len(items), 1)),
        "selected_motion_fraction": float((len(items) - stop_count) / max(len(items), 1)),
        "unsafe_selected_rate": _fraction(bool(item["unsafe_selected"]) for item in items),
        "collision_proxy_rate": _fraction(bool(item["collision_proxy_positive"]) for item in items),
        "unknown_penalty_mean": _mean(float(item["oracle_unknown_penalty"]) for item in items),
        "coverage_gain_mean": _mean(float(item["oracle_coverage_gain"]) for item in items),
        "oracle_total_score_mean": _mean(float(item["oracle_total_expert_score"]) for item in items),
        "source_unknown_penalty_mean": _mean(float(item["selected_score"]["unknown_penalty"]) for item in items),
        "source_coverage_gain_mean": _mean(float(item["selected_score"]["coverage_gain_proxy"]) for item in items),
        "agreement_with_oracle_action": _fraction(bool(item["agreement_with_oracle_action"]) for item in items),
        "agreement_with_future_motion_label": _fraction(
            bool(item["agreement_with_future_motion_label"]) for item in future_valid
        ),
        "future_motion_label_valid_count": len(future_valid),
        "all_motion_candidates_risky_fraction": _fraction(
            bool(item["all_motion_candidates_risky"]) for item in items
        ),
        "goal11b_distribution_collapse_flag": bool(collapse_reasons),
        "goal11b_distribution_collapse_reasons": collapse_reasons,
    }


def _memory_comparison_metrics(records: list[JsonDict]) -> JsonDict:
    comparisons = [
        record["memory_vs_current"]
        for record in records
        if isinstance(record.get("memory_vs_current"), dict) and record["memory_vs_current"].get("available") is True
    ]
    changed = [bool(item["action_changed"]) for item in comparisons]
    improved = [bool(item["memory_improved"]) for item in comparisons]
    worsened = [bool(item["memory_worsened"]) for item in comparisons]
    deltas = [float(item["memory_oracle_score_delta"]) for item in comparisons]
    return {
        "comparison_count": len(comparisons),
        "memory_vs_current_action_changed_fraction": _fraction(changed),
        "memory_improved_decision_fraction": _fraction(improved),
        "memory_worsened_decision_fraction": _fraction(worsened),
        "memory_oracle_score_mean_delta": _mean(deltas),
        "memory_better_or_equal_fraction": _fraction(delta >= -METRIC_TOLERANCE for delta in deltas),
    }


def _decision_summary(
    *,
    decision: TrajectoryDecision,
    oracle_labels: tuple[ExpertCandidateLabel, ...],
    future_motion_label: str | None,
) -> JsonDict:
    selected = decision.selected_candidate_id
    oracle_label = _label_for_candidate(oracle_labels, selected)
    selected_score = decision.selected_score()
    unsafe = oracle_label.collision_proxy >= COLLISION_THRESHOLD or oracle_label.unknown_penalty >= UNKNOWN_BLOCK_THRESHOLD
    return {
        "selected_candidate_id": selected,
        "selected_score": selected_score.to_dict(),
        "scores": [score.to_dict() for score in decision.scores],
        "oracle_collision_proxy": float(oracle_label.collision_proxy),
        "oracle_unknown_penalty": float(oracle_label.unknown_penalty),
        "oracle_coverage_gain": float(oracle_label.coverage_gain),
        "oracle_total_expert_score": float(oracle_label.total_expert_score),
        "collision_proxy_positive": bool(oracle_label.collision_proxy >= COLLISION_THRESHOLD),
        "unsafe_selected": bool(unsafe),
        "all_motion_candidates_risky": bool(decision.all_motion_candidates_risky),
        "future_motion_label_valid": future_motion_label is not None,
        "agreement_with_future_motion_label": bool(future_motion_label is not None and selected == future_motion_label),
    }


def _learned_decision_summary(
    *,
    scorer_model: torch.nn.Module,
    scorer_device: torch.device,
    bev: LocalBev,
    candidates: list[CandidateTrajectory],
    coverage_memory: CoverageMemory,
    oracle_labels: tuple[ExpertCandidateLabel, ...],
    future_motion_label: str | None,
) -> JsonDict:
    logits, selected, features = score_local_bev_with_model(
        model=scorer_model,  # type: ignore[arg-type]
        bev=bev,
        candidates=candidates,
        coverage_memory=coverage_memory,
        device=scorer_device,
    )
    scores = _learned_score_records(candidates=candidates, logits=logits, candidate_features=features)
    selected_score = next(score for score in scores if score["candidate_id"] == selected)
    oracle_label = _label_for_candidate(oracle_labels, selected)
    unsafe = oracle_label.collision_proxy >= COLLISION_THRESHOLD or oracle_label.unknown_penalty >= UNKNOWN_BLOCK_THRESHOLD
    motion_scores = [score for score in scores if score["candidate_id"] != "stop"]
    all_motion_risky = bool(motion_scores) and all(float(score["risk_score"]) >= COLLISION_THRESHOLD for score in motion_scores)
    return {
        "selected_candidate_id": selected,
        "selected_score": selected_score,
        "scores": scores,
        "oracle_collision_proxy": float(oracle_label.collision_proxy),
        "oracle_unknown_penalty": float(oracle_label.unknown_penalty),
        "oracle_coverage_gain": float(oracle_label.coverage_gain),
        "oracle_total_expert_score": float(oracle_label.total_expert_score),
        "collision_proxy_positive": bool(oracle_label.collision_proxy >= COLLISION_THRESHOLD),
        "unsafe_selected": bool(unsafe),
        "all_motion_candidates_risky": bool(all_motion_risky),
        "future_motion_label_valid": future_motion_label is not None,
        "agreement_with_future_motion_label": bool(future_motion_label is not None and selected == future_motion_label),
        "learned_scorer_used": True,
    }


def _bc_oracle_decision_summary(
    *,
    candidates: list[CandidateTrajectory],
    oracle_labels: tuple[ExpertCandidateLabel, ...],
    future_motion_label: str,
) -> JsonDict:
    candidate_ids = {candidate.id for candidate in candidates}
    selected = future_motion_label if future_motion_label in candidate_ids else candidates[0].id
    oracle_label = _label_for_candidate(oracle_labels, selected)
    unsafe = oracle_label.collision_proxy >= COLLISION_THRESHOLD or oracle_label.unknown_penalty >= UNKNOWN_BLOCK_THRESHOLD
    scores: list[JsonDict] = []
    for index, candidate in enumerate(candidates):
        label = _label_for_candidate(oracle_labels, candidate.id)
        total = 0.0 if candidate.id == selected else 1.0 + 0.001 * index
        scores.append(
            {
                "candidate_id": candidate.id,
                "risk_score": float(label.collision_proxy),
                "unknown_penalty": float(label.unknown_penalty),
                "uncertainty_penalty": float(label.uncertainty_penalty),
                "coverage_gain_proxy": float(label.coverage_gain),
                "smoothness_penalty": float(label.smoothness_cost),
                "total_score": float(total),
                "risky": bool(label.collision_proxy >= COLLISION_THRESHOLD),
                "label_source": "future_motion_behavior_cloning",
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
                "product_training_approved": False,
            }
        )
    selected_score = next(score for score in scores if score["candidate_id"] == selected)
    return {
        "selected_candidate_id": selected,
        "selected_score": selected_score,
        "scores": scores,
        "oracle_collision_proxy": float(oracle_label.collision_proxy),
        "oracle_unknown_penalty": float(oracle_label.unknown_penalty),
        "oracle_coverage_gain": float(oracle_label.coverage_gain),
        "oracle_total_expert_score": float(oracle_label.total_expert_score),
        "collision_proxy_positive": bool(oracle_label.collision_proxy >= COLLISION_THRESHOLD),
        "unsafe_selected": bool(unsafe),
        "all_motion_candidates_risky": False,
        "future_motion_label_valid": True,
        "agreement_with_future_motion_label": True,
        "label_source": "future_motion_behavior_cloning",
    }


def _learned_score_records(
    *,
    candidates: list[CandidateTrajectory],
    logits: np.ndarray,
    candidate_features: np.ndarray,
) -> list[JsonDict]:
    feature_index = {name: index for index, name in enumerate(CANDIDATE_FEATURE_NAMES)}
    records: list[JsonDict] = []
    for index, candidate in enumerate(candidates):
        risk = float(candidate_features[index, feature_index["occupied_max"]])
        unknown = float(candidate_features[index, feature_index["unknown_mean"]])
        uncertainty = float(candidate_features[index, feature_index["uncertainty_mean"]])
        coverage = float(candidate_features[index, feature_index["coverage_gain_fraction"]])
        smoothness = float(candidate_features[index, feature_index["smoothness_norm"]])
        logit = float(logits[index])
        records.append(
            {
                "candidate_id": candidate.id,
                "risk_score": risk,
                "unknown_penalty": unknown,
                "uncertainty_penalty": uncertainty,
                "coverage_gain_proxy": coverage,
                "smoothness_penalty": smoothness,
                "total_score": float(-logit),
                "learned_logit": logit,
                "higher_is_better": True,
                "risky": bool(risk >= COLLISION_THRESHOLD),
                "scorer": "trajectory_scorer_net_v1",
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
                "product_training_approved": False,
            }
        )
    return records


def _memory_vs_current(decisions: dict[str, JsonDict]) -> JsonDict:
    current = decisions.get("v1_current_bev")
    memory = decisions.get("v1_memory_bev")
    if current is None or memory is None:
        return {"available": False}
    delta = float(current["oracle_total_expert_score"]) - float(memory["oracle_total_expert_score"])
    return {
        "available": True,
        "current_action": current["selected_candidate_id"],
        "memory_action": memory["selected_candidate_id"],
        "action_changed": bool(current["selected_candidate_id"] != memory["selected_candidate_id"]),
        "memory_oracle_score_delta": delta,
        "memory_improved": bool(delta > METRIC_TOLERANCE),
        "memory_worsened": bool(delta < -METRIC_TOLERANCE),
    }


def _bev_from_logits(
    logits: torch.Tensor,
    *,
    source: str,
    uncertainty: torch.Tensor | None,
) -> LocalBev:
    probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    uncertainty_np = uncertainty.detach().cpu().numpy().astype(np.float32) if uncertainty is not None else None
    confidence = (np.float32(1.0) - uncertainty_np).astype(np.float32) if uncertainty_np is not None else None
    return LocalBev(
        free=probs[BEV_OUTPUT_CHANNELS.index("free")],
        occupied=probs[BEV_OUTPUT_CHANNELS.index("occupied")],
        unknown=probs[BEV_OUTPUT_CHANNELS.index("unknown")],
        traversable=probs[BEV_OUTPUT_CHANNELS.index("traversable")],
        risky=probs[BEV_OUTPUT_CHANNELS.index("risky")],
        confidence=confidence,
        uncertainty=uncertainty_np,
        source=source,
    )


def _bev_from_channels(
    channels: torch.Tensor,
    *,
    source: str,
    confidence: torch.Tensor | None = None,
) -> LocalBev:
    arrays = channels.detach().cpu().numpy().astype(np.float32)
    confidence_np = confidence.detach().cpu().numpy().astype(np.float32) if confidence is not None else None
    return LocalBev(
        free=arrays[BEV_OUTPUT_CHANNELS.index("free")],
        occupied=arrays[BEV_OUTPUT_CHANNELS.index("occupied")],
        unknown=arrays[BEV_OUTPUT_CHANNELS.index("unknown")],
        traversable=arrays[BEV_OUTPUT_CHANNELS.index("traversable")],
        risky=arrays[BEV_OUTPUT_CHANNELS.index("risky")],
        confidence=confidence_np,
        uncertainty=(np.float32(1.0) - confidence_np).astype(np.float32) if confidence_np is not None else None,
        source=source,
    )


def _load_v0_modeld_map(root: str | Path | None) -> dict[tuple[str, str, str, int], LocalBev]:
    if root is None:
        return {}
    base = Path(root)
    if not base.exists():
        return {}
    records: dict[tuple[str, str, str, int], LocalBev] = {}
    for event_root in _event_roots(base):
        source_name = _source_name_from_event_root(event_root)
        for event in read_events(event_root):
            if not isinstance(event, BrainOutputEvent) or not isinstance(event.local_bev_ref, str):
                continue
            if not isinstance(event.debug, dict) or not isinstance(event.debug.get("input_frame_id"), int):
                continue
            artifact_path = event_root / event.local_bev_ref
            if not artifact_path.exists():
                continue
            try:
                record = _record_from_model_event(event, event_root, bev_source="model")
            except Exception:  # noqa: BLE001 - skip non-v0 modeld outputs here.
                continue
            camera_id = str(event.debug.get("camera_id", record.camera_id))
            key = (source_name, event.sequence_id, camera_id, int(event.debug["input_frame_id"]))
            bev = LocalBev(
                free=record.bev.free,
                occupied=record.bev.occupied,
                unknown=record.bev.unknown,
                traversable=record.bev.traversable,
                risky=record.bev.risky,
                confidence=record.bev.confidence,
                uncertainty=record.bev.uncertainty,
                source="v0_current_bev",
            )
            records[key] = bev
    return records


def _event_roots(root: Path) -> list[Path]:
    if (root / "manifest.json").exists():
        return [root]
    return sorted(path.parent for path in root.rglob("manifest.json"))


def _source_name_from_event_root(root: Path) -> str:
    name = root.name
    aliases = {
        "cafe1_1_2": "cafe1-1_2",
        "office1_1_7": "office1-1_7",
        "corridor1_1": "corridor1-1",
    }
    return aliases.get(name, name.replace("_", "-"))


def _load_future_motion_map(action_pack: str | Path | None) -> dict[tuple[str, str, str, int], str]:
    if action_pack is None:
        return {}
    root = Path(action_pack)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = read_json(manifest_path)
    examples = manifest.get("examples")
    if not isinstance(examples, list):
        return {}
    result: dict[tuple[str, str, str, int], str] = {}
    for record in examples:
        if not isinstance(record, dict):
            continue
        candidate = record.get("future_motion_primary_candidate_id")
        if not isinstance(candidate, str) or not candidate or candidate == "missing":
            continue
        key = (
            str(record.get("source_name", "")),
            str(record.get("sequence_id", "")),
            str(record.get("camera_id", "")),
            int(record.get("frame_id", -1)),
        )
        if key[0] and key[1] and key[2] and key[3] >= 0:
            result.setdefault(key, candidate)
    return result


def _build_temporal_dataset(
    *,
    specs: list[SpatialPackSpec],
    split: str,
    window_length: int,
    sensor_context_mode: str,
    missing_pose_behavior: str,
) -> SpatialTemporalMultiPackDataset | SpatialTemporalTrainDataset:
    if len(specs) == 1:
        spec = specs[0]
        return SpatialTemporalTrainDataset(
            spec.dataset_dir,
            feature_dir=spec.feature_dir,
            source_name=spec.source_name,
            split=split,
            window_length=window_length,
            sensor_context_mode=sensor_context_mode,
            missing_pose_behavior=missing_pose_behavior,  # type: ignore[arg-type]
        )
    return SpatialTemporalMultiPackDataset(
        specs,
        split=split,
        window_length=window_length,
        sensor_context_mode=sensor_context_mode,
        missing_pose_behavior=missing_pose_behavior,  # type: ignore[arg-type]
    )


def _candidate_geometry(spec: SpatialPackSpec, model: torch.nn.Module) -> tuple[float, float]:
    manifest = read_json(spec.dataset_dir / "manifest.json")
    camera_config = manifest.get("camera_config")
    meters = None
    radius = None
    if isinstance(camera_config, dict):
        meters = camera_config.get("meters_per_cell")
        radius = camera_config.get("robot_radius_m")
    if not isinstance(meters, (int, float)) or isinstance(meters, bool) or float(meters) <= 0.0:
        meters = getattr(model.config, "meters_per_cell", 0.05)
    if not isinstance(radius, (int, float)) or isinstance(radius, bool) or float(radius) < 0.0:
        radius = 0.18
    return float(meters), float(radius)


def _label_for_candidate(labels: tuple[ExpertCandidateLabel, ...], candidate_id: str) -> ExpertCandidateLabel:
    for label in labels:
        if label.candidate_id == candidate_id:
            return label
    raise ValueError(f"missing oracle label for candidate {candidate_id}")


def _current_memory_metrics(metrics: JsonDict) -> tuple[JsonDict | None, JsonDict | None]:
    per_source = metrics.get("per_source_metrics")
    if not isinstance(per_source, dict):
        return None, None
    current = per_source.get("v1_current_bev")
    memory = per_source.get("v1_memory_bev")
    return current if isinstance(current, dict) else None, memory if isinstance(memory, dict) else None


def _less(left: JsonDict, right: JsonDict, key: str) -> bool:
    return float(left.get(key, 0.0)) < float(right.get(key, 0.0)) - METRIC_TOLERANCE


def _greater(left: JsonDict, right: JsonDict, key: str) -> bool:
    return float(left.get(key, 0.0)) > float(right.get(key, 0.0)) + METRIC_TOLERANCE


def _stop_increase_fails(current: JsonDict, memory: JsonDict) -> bool:
    stop_increase = float(memory.get("stop_fraction", 0.0)) - float(current.get("stop_fraction", 0.0))
    if stop_increase <= STOP_FRACTION_TOLERANCE:
        return False
    all_motion_risky_fraction = float(memory.get("all_motion_candidates_risky_fraction", 0.0))
    return all_motion_risky_fraction + STOP_FRACTION_TOLERANCE < float(memory.get("stop_fraction", 0.0))


def _collapse_reasons(*, entropy: float, dominant_action_fraction: float) -> list[str]:
    reasons: list[str] = []
    if entropy < COLLAPSE_ENTROPY_MIN:
        reasons.append(f"action_entropy<{COLLAPSE_ENTROPY_MIN}")
    if dominant_action_fraction > COLLAPSE_DOMINANT_MAX:
        reasons.append(f"dominant_action_fraction>{COLLAPSE_DOMINANT_MAX}")
    return reasons


def _shadow_sort_key(frame: ShadowFrame) -> tuple[str, str, str, int, int]:
    return (frame.source_name, frame.sequence_id, frame.camera_id, frame.timestamp_ns, frame.frame_id)


def _string_at(value: Any, index: int) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return str(value[index])
    return str(value)


def _distribution(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        probability = count / total
        if probability > 0.0:
            value -= probability * math.log2(probability)
    return float(value)


def _fraction(values: Iterable[bool]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(sum(1 for value in values_list if value) / len(values_list))


def _mean(values: Iterable[float]) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))


def _load_optional_json(path: str | Path | None) -> JsonDict:
    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    return read_json(target)


def _worst_disagreement_records(records: list[JsonDict], *, limit: int) -> list[JsonDict]:
    disagreements = [
        record
        for record in records
        if isinstance(record.get("memory_vs_current"), dict)
        and record["memory_vs_current"].get("available") is True
        and record["memory_vs_current"].get("action_changed") is True
    ]
    disagreements.sort(key=lambda record: float(record["memory_vs_current"].get("memory_oracle_score_delta", 0.0)))
    return disagreements[:limit]


def _worst_example_summaries(records: list[JsonDict]) -> list[JsonDict]:
    return [
        {
            "mode": record["mode"],
            "source_name": record["source_name"],
            "frame_id": record["frame_id"],
            "current_action": record["memory_vs_current"].get("current_action"),
            "memory_action": record["memory_vs_current"].get("memory_action"),
            "memory_oracle_score_delta": record["memory_vs_current"].get("memory_oracle_score_delta"),
            "oracle_action": record["decisions"]["oracle_bev"]["selected_candidate_id"],
        }
        for record in records
    ]


def _write_worst_contact_sheet(
    path: Path,
    records: list[JsonDict],
    *,
    candidates: list[CandidateTrajectory],
) -> None:
    tiles: list[np.ndarray] = []
    for record in records:
        decisions = record.get("decisions", {})
        oracle = decisions.get("oracle_bev", {}) if isinstance(decisions, dict) else {}
        current = decisions.get("v1_current_bev", {}) if isinstance(decisions, dict) else {}
        memory = decisions.get("v1_memory_bev", {}) if isinstance(decisions, dict) else {}
        tiles.append(_decision_tile(candidates, oracle_action=str(oracle.get("selected_candidate_id", "")), current_action=str(current.get("selected_candidate_id", "")), memory_action=str(memory.get("selected_candidate_id", ""))))
    _write_contact_sheet(path, tiles)


def _decision_tile(
    candidates: list[CandidateTrajectory],
    *,
    oracle_action: str,
    current_action: str,
    memory_action: str,
) -> np.ndarray:
    shape = (96, 96, 3)
    tile = np.zeros(shape, dtype=np.uint8)
    ids = [candidate.id for candidate in candidates]
    cell_h = max(1, shape[0] // max(len(ids), 1))
    for index, candidate_id in enumerate(ids):
        row0 = index * cell_h
        row1 = shape[0] if index == len(ids) - 1 else min(shape[0], row0 + cell_h)
        color = np.asarray([40, 40, 40], dtype=np.uint8)
        if candidate_id == oracle_action:
            color = np.maximum(color, np.asarray([0, 160, 40], dtype=np.uint8))
        if candidate_id == current_action:
            color = np.maximum(color, np.asarray([190, 80, 20], dtype=np.uint8))
        if candidate_id == memory_action:
            color = np.maximum(color, np.asarray([40, 120, 210], dtype=np.uint8))
        tile[row0:row1, :, :] = color
    return tile


def _write_contact_sheet(path: Path, tiles: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not tiles:
        path.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
        return
    tile_h, tile_w, _channels = tiles[0].shape
    cols = min(6, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // cols
        col = index % cols
        sheet[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    path.write_bytes(f"P6\n{sheet.shape[1]} {sheet.shape[0]}\n255\n".encode("ascii") + sheet.tobytes())


def _write_markdown(path: str | Path, report: JsonDict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    gates = report["pass_fail_gates"]
    normal = report["normal_metrics"]["per_source_metrics"]
    memory = normal["v1_memory_bev"]
    current = normal["v1_current_bev"]
    lines = [
        "# Goal 13A Memory Policy Shadow Eval",
        "",
        "## Sources",
        f"- dataset_manifest: `{report['data_sources']['dataset_manifest']}`",
        f"- v1_checkpoint: `{report['checkpoints']['v1_checkpoint']}`",
        f"- v0_modeld_frame_count: `{report['data_sources']['v0_modeld_frame_count']}`",
        f"- future_motion_label_count: `{report['data_sources']['future_motion_label_count']}`",
        "",
        "## Normal Metrics",
        f"- v1_current unsafe/collision/unknown/oracle_agree: `{current['unsafe_selected_rate']}` / `{current['collision_proxy_rate']}` / `{current['unknown_penalty_mean']}` / `{current['agreement_with_oracle_action']}`",
        f"- v1_memory unsafe/collision/unknown/oracle_agree: `{memory['unsafe_selected_rate']}` / `{memory['collision_proxy_rate']}` / `{memory['unknown_penalty_mean']}` / `{memory['agreement_with_oracle_action']}`",
        f"- memory action changed: `{report['normal_metrics']['memory_vs_current']['memory_vs_current_action_changed_fraction']}`",
        f"- memory improved/worsened: `{report['normal_metrics']['memory_vs_current']['memory_improved_decision_fraction']}` / `{report['normal_metrics']['memory_vs_current']['memory_worsened_decision_fraction']}`",
        "",
        "## Hard Modes",
        f"- hidden frames: `{report['hidden_cell_metrics']['frame_count']}`",
        f"- occlusion aggregate frames: `{report['occlusion_metrics']['aggregate']['frame_count']}`",
        f"- route-out folds: `{report['route_out_metrics'].get('fold_count', 0)}`",
        "",
        "## Gates",
        f"- memory_action_benefit_pass: `{str(gates['memory_action_benefit_pass']).lower()}`",
        f"- failure_reasons: `{', '.join(gates['failure_reasons']) if gates['failure_reasons'] else 'none'}`",
        "",
        "## Safety",
        "- Replay/eval shadow decisions only. `control_safe=false`, `not_executed=true`, no `cmd_vel`, and no learned memory scorer was trained.",
        f"- recommended_next_goal: {report['recommended_next_goal']}",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Goal 13A memory-aware trajectory shadow evaluation.")
    parser.add_argument("--dataset-manifest", default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--v1-checkpoint", default=DEFAULT_V1_CHECKPOINT)
    parser.add_argument("--v0-modeld-root", default=DEFAULT_V0_MODELD_ROOT)
    parser.add_argument("--action-label-pack", default=DEFAULT_ACTION_LABEL_PACK)
    parser.add_argument("--goal12c-report", default=DEFAULT_GOAL12C_REPORT)
    parser.add_argument("--true-route-out-root", default=DEFAULT_TRUE_ROUTE_OUT_ROOT)
    parser.add_argument("--out-json", default="runs/goal13a_memory_policy_shadow_eval_report.json")
    parser.add_argument("--out-md", default="runs/goal13a_memory_policy_shadow_eval_report.md")
    parser.add_argument("--decisions-jsonl", default="runs/goal13a_memory_policy_shadow_eval_decisions.jsonl")
    parser.add_argument("--contact-sheet", default="runs/goal13a_memory_policy_shadow_eval_worst_disagreements.ppm")
    parser.add_argument("--scorer-checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", default="val")
    parser.add_argument("--future-horizon", type=int, default=3)
    parser.add_argument("--max-frames-per-mode", type=int, default=None)
    args = parser.parse_args(argv)
    command = "python -m homebrain.tools.goal13a_memory_policy_shadow_eval " + " ".join(sys.argv[1:])
    report = run_goal13a_shadow_eval(
        dataset_manifest=args.dataset_manifest,
        v1_checkpoint=args.v1_checkpoint,
        v0_modeld_root=args.v0_modeld_root,
        action_label_pack=args.action_label_pack,
        goal12c_report=args.goal12c_report,
        true_route_out_root=args.true_route_out_root,
        out_json=args.out_json,
        out_md=args.out_md,
        decisions_jsonl=args.decisions_jsonl,
        contact_sheet=args.contact_sheet,
        scorer_checkpoint=args.scorer_checkpoint,
        device_name=args.device,
        batch_size=args.batch_size,
        split=args.split,
        future_horizon=args.future_horizon,
        max_frames_per_mode=args.max_frames_per_mode,
        command=command,
    )
    print(json.dumps(report["pass_fail_gates"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
