from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.brain.direct_rgbd_features import (
    DIRECT_RGBD_FEATURE_DIM,
    has_direct_rgbd_feature_artifacts,
    write_direct_rgbd_feature_artifacts,
)
from homebrain.brain.modeld import RUNTIME_FEATURE_SOURCES, V1_POLICY_BEV_SOURCES, V1_POSE_WARP_SOURCES
from homebrain.data.pack_spatial_dataset import pack_spatial_dataset
from homebrain.data.qa_spatial_dataset import qa_spatial_dataset, write_qa_metrics
from homebrain.data.spatial_dataset import read_json, write_json
from homebrain.datasets.openloris_scene import discover_sequence_root, sequence_scene
from homebrain.datasets.openloris_to_route import openloris_to_route
from homebrain.datasets.setup_openloris_scene import setup_openloris_scene
from homebrain.eval.eval_future_bev_rollout_v1 import eval_future_bev_rollout_v1
from homebrain.geometry.qa_robot_frame_bev import qa_robot_frame_bev
from homebrain.geometry.robot_rgbd_to_bev import robot_rgbd_to_bev
from homebrain.geometry.validate_bev import validate_bev_artifacts, write_bev_metrics
from homebrain.geometry.visualize_bev import visualize_bev
from homebrain.messages.schema import BrainOutputEvent, FrameEvent
from homebrain.policies.build_action_label_pack_v5 import build_action_label_pack_v5
from homebrain.policies.train_trajectory_scorer_v1 import train_trajectory_scorer_v1
from homebrain.replay.segment_log import read_events
from homebrain.runtime.replay_openloris_brain import run_openloris_runtime_replay
from homebrain.teachers.artifacts import file_sha256
from homebrain.teachers.dino_teacher import DINO_DEFAULT_MODEL_ID, run_dino_teacher
from homebrain.train.build_future_bev_rollout_pack import build_future_bev_rollout_pack
from homebrain.train.eval_spatial_v1 import eval_spatial_v1
from homebrain.train.train_future_bev_rollout_v1 import train_future_bev_rollout_v1

PIPELINE_SCHEMA_VERSION = "homebrain.openloris_route_heldout_milestone.v0"
DEFAULT_SEQUENCES = ("cafe1-1_2", "corridor1-1", "office1-1_7")


@dataclass(frozen=True)
class RouteArtifact:
    sequence: str
    scene: str
    route_id: str
    source_dir: Path
    route_dir: Path
    bev_dir: Path
    dino_dir: Path
    direct_rgbd_feature_dir: Path
    spatial_pack_dir: Path
    bev_validation_path: Path
    spatial_qa_path: Path
    robot_frame_bev_qa_path: Path
    visualization_manifest: Path


def run_openloris_route_heldout_milestone(
    *,
    openloris_root: str | Path,
    out_dir: str | Path,
    sequences: list[str],
    heldout_sequence: str | None = None,
    runtime_heldout_sequences: list[str] | None = None,
    max_frames: int = 160,
    download_missing: bool = False,
    dino_device: str | None = None,
    dino_image_size: int = 224,
    spatial_steps: int = 80,
    spatial_window_length: int = 4,
    spatial_batch_size: int = 4,
    max_spatial_folds: int | None = None,
    future_steps: int = 80,
    future_batch_size: int = 4,
    future_max_examples: int | None = None,
    trajectory_steps: int = 160,
    trajectory_batch_size: int = 32,
    action_label_horizon_s: float | None = None,
    runtime_max_frames: int | None = None,
    runtime_feature_source: str = "dino",
    student_feature_source: str = "dino",
    future_rollout_selection_mode: str = "guided_transparent",
    runtime_policy_source: str = "trajectory_scorer",
    run_direct_rgbd_runtime_slice: bool = True,
    build_dino_features: bool = True,
    pose_warp_source: str = "odom",
    policy_bev_source: str = "scene",
    command: str | None = None,
) -> dict[str, Any]:
    if len(sequences) < 3:
        raise ValueError("the milestone requires at least three OpenLORIS sequences")
    if max_frames < spatial_window_length + 4:
        raise ValueError("max_frames must leave enough temporal windows for training/eval")
    if runtime_feature_source not in RUNTIME_FEATURE_SOURCES:
        raise ValueError(f"runtime_feature_source must be one of {RUNTIME_FEATURE_SOURCES}")
    if student_feature_source not in {"dino", "direct_rgbd"}:
        raise ValueError("student_feature_source must be 'dino' or 'direct_rgbd'")
    if runtime_policy_source not in {"trajectory_scorer", "future_rollout", "auto"}:
        raise ValueError("runtime_policy_source must be 'trajectory_scorer', 'future_rollout', or 'auto'")
    if action_label_horizon_s is not None and float(action_label_horizon_s) <= 0.0:
        raise ValueError("action_label_horizon_s must be positive when supplied")
    if pose_warp_source not in V1_POSE_WARP_SOURCES:
        raise ValueError(f"pose_warp_source must be one of {V1_POSE_WARP_SOURCES}")
    if policy_bev_source not in V1_POLICY_BEV_SOURCES:
        raise ValueError(f"policy_bev_source must be one of {V1_POLICY_BEV_SOURCES}")

    root = Path(openloris_root)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    need_dino_features = (
        bool(build_dino_features)
        or student_feature_source == "dino"
        or runtime_feature_source == "dino"
    )
    route_artifacts = [
        _prepare_route(
            openloris_root=root,
            out_dir=output,
            sequence=sequence,
            max_frames=max_frames,
            download_missing=download_missing,
            dino_device=dino_device or _preferred_4090_device() or _preferred_cuda_device() or "cpu",
            dino_image_size=dino_image_size,
            build_dino_features=need_dino_features,
            student_feature_source=student_feature_source,
        )
        for sequence in sequences
    ]
    scenes = sorted({artifact.scene for artifact in route_artifacts})
    if len(scenes) < 3:
        raise ValueError(f"expected routes from at least three scenes, got {scenes}")

    heldout = heldout_sequence or route_artifacts[min(1, len(route_artifacts) - 1)].sequence
    if heldout not in {artifact.sequence for artifact in route_artifacts}:
        raise ValueError(f"heldout_sequence {heldout!r} is not in staged sequences")
    runtime_heldouts = _dedupe_sequences(runtime_heldout_sequences or [heldout])
    if heldout not in runtime_heldouts:
        runtime_heldouts.insert(0, heldout)
    available_sequences = {artifact.sequence for artifact in route_artifacts}
    missing_runtime_heldouts = [sequence for sequence in runtime_heldouts if sequence not in available_sequences]
    if missing_runtime_heldouts:
        raise ValueError(f"runtime heldout sequences are not staged: {missing_runtime_heldouts}")

    manifests_dir = output / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    spatial_fold_count = max(_resolve_fold_count(max_spatial_folds, route_artifacts), len(runtime_heldouts))
    heldout_order = runtime_heldouts + [
        artifact.sequence for artifact in route_artifacts if artifact.sequence not in set(runtime_heldouts)
    ]
    heldout_order = heldout_order[:spatial_fold_count]
    fold_records = []
    spatial_devices = _parallel_4090_devices()
    if not spatial_devices:
        spatial_devices = [_preferred_cuda_device() or "cpu"]

    train_processes: list[tuple[subprocess.Popen[str], Path, str, str]] = []
    for index, fold_heldout in enumerate(heldout_order):
        train_manifest = _write_spatial_manifest(
            manifests_dir / f"spatial_train_except_{_safe_id(fold_heldout)}.json",
            route_artifacts=[artifact for artifact in route_artifacts if artifact.sequence != fold_heldout],
            heldout_sequence=fold_heldout,
            split_role="train",
            feature_source=student_feature_source,
        )
        heldout_manifest = _write_spatial_manifest(
            manifests_dir / f"spatial_heldout_{_safe_id(fold_heldout)}.json",
            route_artifacts=[artifact for artifact in route_artifacts if artifact.sequence == fold_heldout],
            heldout_sequence=fold_heldout,
            split_role="heldout",
            feature_source=student_feature_source,
        )
        train_dir = output / "train" / f"spatial_v1_except_{_safe_id(fold_heldout)}"
        device = spatial_devices[index % len(spatial_devices)]
        train_processes.append(
            (
                _launch_spatial_train(
                    dataset_manifest=train_manifest,
                    out_dir=train_dir,
                    max_steps=spatial_steps,
                    window_length=spatial_window_length,
                    batch_size=spatial_batch_size,
                    device=device,
                ),
                heldout_manifest,
                fold_heldout,
                device,
            )
        )

    for process, heldout_manifest, fold_heldout, device in train_processes:
        stdout, stderr = process.communicate()
        train_dir = output / "train" / f"spatial_v1_except_{_safe_id(fold_heldout)}"
        _write_text(train_dir / "stdout.txt", stdout)
        _write_text(train_dir / "stderr.txt", stderr)
        if process.returncode != 0:
            raise RuntimeError(f"SpatialMemoryNetV1 training failed for heldout {fold_heldout}; see {train_dir}")
        checkpoint = train_dir / "checkpoint.pt"
        eval_path = output / "eval" / f"spatial_v1_heldout_{_safe_id(fold_heldout)}.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_metrics = eval_spatial_v1(
            checkpoint=checkpoint,
            dataset_manifest=heldout_manifest,
            split=None,
            out_path=eval_path,
            device_name=device,
            batch_size=spatial_batch_size,
            window_length=spatial_window_length,
            command=_command_text(
                [
                    sys.executable,
                    "-m",
                    "homebrain.train.eval_spatial_v1",
                    "--checkpoint",
                    checkpoint.as_posix(),
                    "--dataset-manifest",
                    heldout_manifest.as_posix(),
                    "--split",
                    "",
                    "--out",
                    eval_path.as_posix(),
                    "--device",
                    device,
                ]
            ),
        )
        fold_records.append(
            {
                "heldout_sequence": fold_heldout,
                "device": device,
                "train_manifest": (manifests_dir / f"spatial_train_except_{_safe_id(fold_heldout)}.json").as_posix(),
                "heldout_manifest": heldout_manifest.as_posix(),
                "checkpoint": checkpoint.as_posix(),
                "train_metrics": (train_dir / "train_metrics.json").as_posix(),
                "eval_metrics": eval_path.as_posix(),
                "current_bev_iou_or_proxy": eval_metrics.get("current_bev_iou_or_proxy"),
                "fused_memory_bev_iou_or_proxy": eval_metrics.get("fused_memory_bev_iou_or_proxy"),
                "memory_warp_valid_fraction": eval_metrics.get("memory_warp_valid_fraction"),
                "control_safe": False,
            }
        )

    primary_fold = next(record for record in fold_records if record["heldout_sequence"] == heldout)
    heldout_artifact = next(artifact for artifact in route_artifacts if artifact.sequence == heldout)
    train_artifacts = [artifact for artifact in route_artifacts if artifact.sequence != heldout]
    future_train_pack = output / "future" / f"future_pack_train_except_{_safe_id(heldout)}"
    future_heldout_pack = output / "future" / f"future_pack_heldout_{_safe_id(heldout)}"
    build_future_bev_rollout_pack(
        sources=[artifact.spatial_pack_dir for artifact in train_artifacts],
        feature_dirs=[_feature_dir_for_artifact(artifact, student_feature_source) for artifact in train_artifacts],
        out_dir=future_train_pack,
        max_examples=future_max_examples,
        sampling_strategy="route_balanced",
        fail_on_degenerate_labels=True,
        reject_degenerate_groups=True,
    )
    build_future_bev_rollout_pack(
        sources=[heldout_artifact.spatial_pack_dir],
        feature_dirs=[_feature_dir_for_artifact(heldout_artifact, student_feature_source)],
        out_dir=future_heldout_pack,
        max_examples=future_max_examples,
        sampling_strategy="route_balanced",
        fail_on_degenerate_labels=True,
        reject_degenerate_groups=True,
    )
    future_train_manifest = read_json(future_train_pack / "manifest.json")
    future_heldout_manifest = read_json(future_heldout_pack / "manifest.json")
    future_device = _secondary_4090_device(spatial_devices) or spatial_devices[0]
    future_train_dir = output / "train" / f"future_bev_rollout_except_{_safe_id(heldout)}"
    future_metrics = train_future_bev_rollout_v1(
        rollout_pack=future_train_pack,
        out_dir=future_train_dir,
        max_steps=future_steps,
        batch_size=future_batch_size,
        device_name=future_device,
        command=_command_text(
            [
                sys.executable,
                "-m",
                "homebrain.train.train_future_bev_rollout_v1",
                "--rollout-pack",
                future_train_pack.as_posix(),
                "--out",
                future_train_dir.as_posix(),
                "--max-steps",
                str(future_steps),
                "--batch-size",
                str(future_batch_size),
                "--device",
                future_device,
            ]
        ),
    )
    future_eval_path = output / "eval" / f"future_bev_rollout_heldout_{_safe_id(heldout)}.json"
    future_eval_metrics = eval_future_bev_rollout_v1(
        checkpoint=future_train_dir / "checkpoint.pt",
        rollout_pack=future_heldout_pack,
        split=None,
        out_path=future_eval_path,
        batch_size=future_batch_size,
        device_name=future_device,
        command=_command_text(
            [
                sys.executable,
                "-m",
                "homebrain.eval.eval_future_bev_rollout_v1",
                "--checkpoint",
                (future_train_dir / "checkpoint.pt").as_posix(),
                "--rollout-pack",
                future_heldout_pack.as_posix(),
                "--split",
                "all",
                "--out",
                future_eval_path.as_posix(),
                "--device",
                future_device,
            ]
        ),
    )

    action_pack = output / "policy" / f"action_v5_train_except_{_safe_id(heldout)}"
    build_action_label_pack_v5(
        sources=[artifact.spatial_pack_dir for artifact in train_artifacts],
        out_dir=action_pack,
        horizon_s=action_label_horizon_s,
    )
    action_manifest = read_json(action_pack / "manifest.json")
    action_selected_distribution = _json_dict(action_manifest.get("selected_distribution"))
    trajectory_label_degenerate = len([key for key, value in action_selected_distribution.items() if int(value or 0) > 0]) <= 1
    trajectory_label_entropy = _entropy_counts(action_selected_distribution)
    trajectory_label_dominant_fraction = _dominant_fraction_counts(action_selected_distribution)
    trajectory_train_dir = output / "train" / f"trajectory_scorer_v1_except_{_safe_id(heldout)}"
    trajectory_metrics = train_trajectory_scorer_v1(
        action_pack=action_pack,
        out_dir=trajectory_train_dir,
        max_steps=trajectory_steps,
        batch_size=trajectory_batch_size,
        device_name=future_device,
        bev_source="oracle",
        class_balanced_loss=True,
        mirror_left_right=True,
    )
    resolved_runtime_policy_source = (
        "future_rollout" if runtime_policy_source == "auto" and trajectory_label_degenerate else runtime_policy_source
    )
    trajectory_checkpoint_for_runtime = (
        trajectory_train_dir / "checkpoint.pt" if resolved_runtime_policy_source == "trajectory_scorer" else None
    )

    runtime_route = (
        heldout_artifact.route_dir
        if runtime_feature_source == "dino"
        else _runtime_route_slice(heldout_artifact, output, runtime_max_frames)
    )
    runtime_out = output / "runtime" / f"heldout_{_safe_id(heldout)}_{runtime_feature_source}"
    runtime_report = run_openloris_runtime_replay(
        log_dir=runtime_route,
        out_dir=runtime_out,
        checkpoint=Path(str(primary_fold["checkpoint"])),
        feature_dir=heldout_artifact.dino_dir if runtime_feature_source == "dino" else None,
        trajectory_scorer_checkpoint=trajectory_checkpoint_for_runtime,
        future_rollout_checkpoint=future_train_dir / "checkpoint.pt",
        device_name=future_device,
        v1_pose_warp_source=pose_warp_source,
        v1_policy_bev_source=policy_bev_source,
        runtime_feature_source=runtime_feature_source,
        future_rollout_selection_mode=future_rollout_selection_mode,
        command=command,
    )
    contact_sheet = write_runtime_failure_contact_sheet(
        route_dir=runtime_route,
        modeld_log_dir=runtime_out / "online_modeld",
        out_dir=output / "contact_sheets",
        name=f"heldout_{_safe_id(heldout)}_{runtime_feature_source}",
    )
    runtime_route_records: list[dict[str, Any]] = [
        {
            "heldout_sequence": heldout,
            "route": runtime_route.as_posix(),
            "runtime_report": (runtime_out / "runtime_report.json").as_posix(),
            "scene": heldout_artifact.scene,
            "report": runtime_report,
            "future_eval_metrics": future_eval_metrics,
            "future_checkpoint": (future_train_dir / "checkpoint.pt").as_posix(),
            "spatial_checkpoint": primary_fold["checkpoint"],
            "contact_sheet": contact_sheet.as_posix(),
        }
    ]

    for extra_heldout in [sequence for sequence in runtime_heldouts if sequence != heldout]:
        extra_fold = next(record for record in fold_records if record["heldout_sequence"] == extra_heldout)
        extra_heldout_artifact = next(artifact for artifact in route_artifacts if artifact.sequence == extra_heldout)
        extra_train_artifacts = [artifact for artifact in route_artifacts if artifact.sequence != extra_heldout]
        extra_future_train_pack = output / "future" / f"future_pack_train_except_{_safe_id(extra_heldout)}"
        extra_future_heldout_pack = output / "future" / f"future_pack_heldout_{_safe_id(extra_heldout)}"
        build_future_bev_rollout_pack(
            sources=[artifact.spatial_pack_dir for artifact in extra_train_artifacts],
            feature_dirs=[_feature_dir_for_artifact(artifact, student_feature_source) for artifact in extra_train_artifacts],
            out_dir=extra_future_train_pack,
            max_examples=future_max_examples,
            sampling_strategy="route_balanced",
            fail_on_degenerate_labels=True,
            reject_degenerate_groups=True,
        )
        build_future_bev_rollout_pack(
            sources=[extra_heldout_artifact.spatial_pack_dir],
            feature_dirs=[_feature_dir_for_artifact(extra_heldout_artifact, student_feature_source)],
            out_dir=extra_future_heldout_pack,
            max_examples=future_max_examples,
            sampling_strategy="route_balanced",
            fail_on_degenerate_labels=True,
            reject_degenerate_groups=True,
        )
        extra_future_train_dir = output / "train" / f"future_bev_rollout_except_{_safe_id(extra_heldout)}"
        train_future_bev_rollout_v1(
            rollout_pack=extra_future_train_pack,
            out_dir=extra_future_train_dir,
            max_steps=future_steps,
            batch_size=future_batch_size,
            device_name=future_device,
            command=_command_text(
                [
                    sys.executable,
                    "-m",
                    "homebrain.train.train_future_bev_rollout_v1",
                    "--rollout-pack",
                    extra_future_train_pack.as_posix(),
                    "--out",
                    extra_future_train_dir.as_posix(),
                    "--max-steps",
                    str(future_steps),
                    "--batch-size",
                    str(future_batch_size),
                    "--device",
                    future_device,
                ]
            ),
        )
        extra_future_eval_path = output / "eval" / f"future_bev_rollout_heldout_{_safe_id(extra_heldout)}.json"
        extra_future_eval_metrics = eval_future_bev_rollout_v1(
            checkpoint=extra_future_train_dir / "checkpoint.pt",
            rollout_pack=extra_future_heldout_pack,
            split=None,
            out_path=extra_future_eval_path,
            batch_size=future_batch_size,
            device_name=future_device,
            command=_command_text(
                [
                    sys.executable,
                    "-m",
                    "homebrain.eval.eval_future_bev_rollout_v1",
                    "--checkpoint",
                    (extra_future_train_dir / "checkpoint.pt").as_posix(),
                    "--rollout-pack",
                    extra_future_heldout_pack.as_posix(),
                    "--split",
                    "all",
                    "--out",
                    extra_future_eval_path.as_posix(),
                    "--device",
                    future_device,
                ]
            ),
        )
        extra_runtime_route = (
            extra_heldout_artifact.route_dir
            if runtime_feature_source == "dino"
            else _runtime_route_slice(extra_heldout_artifact, output, runtime_max_frames)
        )
        extra_runtime_out = output / "runtime" / f"heldout_{_safe_id(extra_heldout)}_{runtime_feature_source}"
        extra_runtime_report = run_openloris_runtime_replay(
            log_dir=extra_runtime_route,
            out_dir=extra_runtime_out,
            checkpoint=Path(str(extra_fold["checkpoint"])),
            feature_dir=extra_heldout_artifact.dino_dir if runtime_feature_source == "dino" else None,
            trajectory_scorer_checkpoint=None,
            future_rollout_checkpoint=extra_future_train_dir / "checkpoint.pt",
            device_name=future_device,
            v1_pose_warp_source=pose_warp_source,
            v1_policy_bev_source=policy_bev_source,
            runtime_feature_source=runtime_feature_source,
            future_rollout_selection_mode=future_rollout_selection_mode,
            command=command,
        )
        extra_contact_sheet = write_runtime_failure_contact_sheet(
            route_dir=extra_runtime_route,
            modeld_log_dir=extra_runtime_out / "online_modeld",
            out_dir=output / "contact_sheets",
            name=f"heldout_{_safe_id(extra_heldout)}_{runtime_feature_source}",
        )
        runtime_route_records.append(
            {
                "heldout_sequence": extra_heldout,
                "route": extra_runtime_route.as_posix(),
                "runtime_report": (extra_runtime_out / "runtime_report.json").as_posix(),
                "scene": extra_heldout_artifact.scene,
                "report": extra_runtime_report,
                "future_eval_metrics": extra_future_eval_metrics,
                "future_checkpoint": (extra_future_train_dir / "checkpoint.pt").as_posix(),
                "spatial_checkpoint": extra_fold["checkpoint"],
                "contact_sheet": extra_contact_sheet.as_posix(),
            }
        )

    direct_runtime_record: dict[str, Any] | None = None
    if run_direct_rgbd_runtime_slice and runtime_feature_source != "direct_rgbd":
        direct_route = _runtime_route_slice(heldout_artifact, output, min(runtime_max_frames or 48, 48))
        direct_out = output / "runtime" / f"heldout_{_safe_id(heldout)}_direct_rgbd_slice"
        direct_report = run_openloris_runtime_replay(
            log_dir=direct_route,
            out_dir=direct_out,
            checkpoint=Path(str(primary_fold["checkpoint"])),
            feature_dir=None,
            trajectory_scorer_checkpoint=trajectory_checkpoint_for_runtime,
            future_rollout_checkpoint=future_train_dir / "checkpoint.pt",
            device_name=future_device,
            v1_pose_warp_source=pose_warp_source,
            v1_policy_bev_source=policy_bev_source,
            runtime_feature_source="direct_rgbd",
            future_rollout_selection_mode=future_rollout_selection_mode,
            command=(
                "direct RGB-D runtime smoke slice from "
                "homebrain.tools.run_openloris_route_heldout_milestone"
            ),
        )
        direct_runtime_record = {
            "runtime_report": (direct_out / "runtime_report.json").as_posix(),
            "decision_count": direct_report.get("decision_count"),
            "selected_candidate_entropy": direct_report.get("selected_candidate_entropy"),
            "selected_candidate_dominant_fraction": direct_report.get("selected_candidate_dominant_fraction"),
            "cmd_vel_proposal_count": direct_report.get("cmd_vel_proposal_count"),
            "latency_end_to_end_p95_ms": direct_report.get("latency_end_to_end_p95_ms"),
            "latency_step_p50_ms": direct_report.get("latency_step_p50_ms"),
            "latency_step_p95_ms": direct_report.get("latency_step_p95_ms"),
            "requires_precomputed_dino_runtime": direct_report.get("requires_precomputed_dino_runtime"),
            "direct_rgbd_runtime_path": direct_report.get("direct_rgbd_runtime_path"),
            "teacher_runtime_dependency": direct_report.get("teacher_runtime_dependency"),
            "future_or_groundtruth_runtime_dependency": direct_report.get("future_or_groundtruth_runtime_dependency"),
            "control_safe": False,
        }

    runtime_learned_policy = (
        _distribution_has(runtime_report.get("trajectory_scorer_mode_distribution"), "learned")
        or _distribution_has(runtime_report.get("trajectory_scorer_mode_distribution"), "future_rollout")
        or runtime_report.get("trajectory_scorer_mode") in {"learned", "future_rollout"}
    )
    temporal_diversity_prior_fraction = float(runtime_report.get("temporal_diversity_prior_enabled_fraction") or 0.0)
    accepted_policy_uses_guided_transparent = bool(
        future_rollout_selection_mode == "guided_transparent"
        and resolved_runtime_policy_source == "future_rollout"
    )
    accepted_policy_uses_handcrafted_diversity_prior = temporal_diversity_prior_fraction > 0.0
    scene_memory_artifact_exists = _path_exists(runtime_report.get("scene_memory_artifact"))
    scene_pose_trace_artifact_exists = _path_exists(runtime_report.get("scene_pose_trace_artifact"))
    physical_geometry_visual_exists = _path_exists(runtime_report.get("scene_memory_visual"))
    future_state_visual_exists = (
        _path_exists(runtime_report.get("scene_memory_visual"))
        and int(runtime_report.get("future_prediction_horizon_count") or 0) >= 2
    )
    runtime_reports = [record["report"] for record in runtime_route_records]
    runtime_future_evals = [record["future_eval_metrics"] for record in runtime_route_records]
    runtime_route_summaries = [
        {
            "heldout_sequence": record["heldout_sequence"],
            "scene": record.get("scene"),
            "route": record["route"],
            "runtime_report": record["runtime_report"],
            "spatial_checkpoint": record["spatial_checkpoint"],
            "future_checkpoint": record["future_checkpoint"],
            "contact_sheet": record["contact_sheet"],
            "decision_count": record["report"].get("decision_count"),
            "runtime_api_step_count": record["report"].get("runtime_api_step_count"),
            "selected_candidate_entropy": record["report"].get("selected_candidate_entropy"),
            "selected_candidate_dominant_fraction": record["report"].get("selected_candidate_dominant_fraction"),
            "unsafe_selected_rate": record["report"].get("unsafe_selected_rate"),
            "unknown_reduction_vs_current": record["report"].get("unknown_reduction_vs_current"),
            "coverage_memory_cells_seen": record["report"].get("coverage_memory_cells_seen"),
            "latency_step_p50_ms": record["report"].get("latency_step_p50_ms"),
            "latency_step_p95_ms": record["report"].get("latency_step_p95_ms"),
            "scene_state_online": record["report"].get("scene_state_online"),
            "scene_memory_used_for_policy": record["report"].get("scene_memory_used_for_policy"),
            "scene_map_created_only_posthoc": record["report"].get("scene_map_created_only_posthoc"),
            "scene_memory_artifact": record["report"].get("scene_memory_artifact"),
            "scene_pose_trace_artifact": record["report"].get("scene_pose_trace_artifact"),
            "physical_geometry_visual": record["report"].get("scene_memory_visual"),
            "future_state_visual": record["report"].get("scene_memory_visual"),
            "future_prediction_horizon_count": record["report"].get("future_prediction_horizon_count"),
            "future_free_iou_or_proxy": record["future_eval_metrics"].get("future_free_iou_or_proxy"),
            "future_occupied_iou_or_proxy": record["future_eval_metrics"].get("future_occupied_iou_or_proxy"),
            "future_unknown_iou_or_proxy": record["future_eval_metrics"].get("future_unknown_iou_or_proxy"),
            "teacher_runtime_dependency": record["report"].get("teacher_runtime_dependency"),
            "future_or_groundtruth_runtime_dependency": record["report"].get("future_or_groundtruth_runtime_dependency"),
            "route_pose_leakage_ablation_fraction": record["report"].get("route_pose_leakage_ablation_fraction"),
            "raw_pwm_emitted": record["report"].get("raw_pwm_emitted"),
            "cmd_vel_proposal_count": record["report"].get("cmd_vel_proposal_count"),
            "pose_metric_source": record["report"].get("pose_metric_source"),
            "pose_metric_independent_groundtruth": record["report"].get("pose_metric_independent_groundtruth"),
            "pose_ate_rmse_m_or_proxy": record["report"].get("pose_ate_rmse_m_or_proxy"),
            "pose_rpe_translation_rmse_m_or_proxy": record["report"].get("pose_rpe_translation_rmse_m_or_proxy"),
        }
        for record in runtime_route_records
    ]
    aggregate_pose_sources = sorted(
        {
            str(report.get("pose_metric_source"))
            for report in runtime_reports
            if report.get("pose_metric_source") is not None
        }
    )
    aggregate_pose_source = aggregate_pose_sources[0] if len(aggregate_pose_sources) == 1 else "mixed"
    runtime_scenes = sorted(
        {
            str(record.get("scene"))
            for record in runtime_route_summaries
            if record.get("scene") is not None
        }
    )
    independent_pose_route_count = sum(
        1 for record in runtime_route_summaries if record.get("pose_metric_independent_groundtruth") is True
    )

    summary = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "command": command,
        "openloris_root": root.as_posix(),
        "out_dir": output.as_posix(),
        "route_count": len(route_artifacts),
        "scenes": scenes,
        "routes": [_route_record(artifact) for artifact in route_artifacts],
        "heldout_sequence": heldout,
        "heldout_sequences": [record["heldout_sequence"] for record in runtime_route_records],
        "runtime_routes_evaluated": len(runtime_route_records),
        "runtime_scenes_evaluated": len(runtime_scenes),
        "runtime_scene_names": runtime_scenes,
        "runtime_route_reports": runtime_route_summaries,
        "max_frames_per_route": max_frames,
        "student_feature_source": student_feature_source,
        "runtime_feature_source": runtime_feature_source,
        "runtime_policy_bev_source": policy_bev_source,
        "runtime_policy_source_requested": runtime_policy_source,
        "runtime_policy_source_resolved": resolved_runtime_policy_source,
        "real_public_data_only": True,
        "mock_synthetic_or_fake_used_for_metrics": False,
        "dino_backend": "real" if need_dino_features else "not_built",
        "dino_model_id": DINO_DEFAULT_MODEL_ID if need_dino_features else None,
        "spatial_leave_one_route_out_folds": fold_records,
        "spatial_primary_checkpoint": primary_fold["checkpoint"],
        "spatial_primary_eval": primary_fold["eval_metrics"],
        "future_rollout": {
            "train_pack": future_train_pack.as_posix(),
            "heldout_pack": future_heldout_pack.as_posix(),
            "train_pack_example_count": future_train_manifest.get("example_count"),
            "heldout_pack_example_count": future_heldout_manifest.get("example_count"),
            "train_pack_rejected_degenerate_example_count": future_train_manifest.get(
                "rejected_degenerate_example_count"
            ),
            "heldout_pack_rejected_degenerate_example_count": future_heldout_manifest.get(
                "rejected_degenerate_example_count"
            ),
            "train_pack_label_qa_status": future_train_manifest.get("label_qa_status"),
            "heldout_pack_label_qa_status": future_heldout_manifest.get("label_qa_status"),
            "checkpoint": (future_train_dir / "checkpoint.pt").as_posix(),
            "train_metrics": (future_train_dir / "train_metrics.json").as_posix(),
            "eval_metrics": future_eval_path.as_posix(),
            "used_as_runtime_selector": resolved_runtime_policy_source == "future_rollout",
            "future_free_iou_or_proxy": future_eval_metrics.get("future_free_iou_or_proxy"),
            "future_occupied_iou_or_proxy": future_eval_metrics.get("future_occupied_iou_or_proxy"),
            "future_unknown_iou_or_proxy": future_eval_metrics.get("future_unknown_iou_or_proxy"),
            "candidate_collision_accuracy": future_eval_metrics.get("candidate_collision_accuracy"),
            "candidate_future_collision_positive_rate": future_eval_metrics.get(
                "candidate_future_collision_positive_rate"
            ),
            "candidate_new_area_gain_ranking_quality": future_eval_metrics.get(
                "candidate_new_area_gain_ranking_quality"
            ),
            "action_entropy": future_eval_metrics.get("action_entropy"),
            "dominant_action_fraction": future_eval_metrics.get("dominant_action_fraction"),
            "control_safe": False,
        },
        "learned_trajectory_policy": {
            "action_pack": action_pack.as_posix(),
            "checkpoint": (trajectory_train_dir / "checkpoint.pt").as_posix(),
            "train_metrics": (trajectory_train_dir / "train_metrics.json").as_posix(),
            "used_as_runtime_selector": resolved_runtime_policy_source == "trajectory_scorer",
            "label_distribution": action_selected_distribution,
            "label_degenerate": trajectory_label_degenerate,
            "label_action_entropy": trajectory_label_entropy,
            "label_dominant_action_fraction": trajectory_label_dominant_fraction,
            "label_horizon_s": (
                float(action_label_horizon_s)
                if action_label_horizon_s is not None
                else action_manifest.get("bc_labeling", {}).get("horizon_s")
            ),
            "train_top1_action_agreement": trajectory_metrics.get("train_top1_action_agreement"),
            "val_top1_action_agreement": trajectory_metrics.get("val_top1_action_agreement"),
            "val_rank_correlation_or_proxy": trajectory_metrics.get("val_rank_correlation_or_proxy"),
            "bev_source": trajectory_metrics.get("bev_source"),
            "class_balanced_loss": trajectory_metrics.get("class_balanced_loss"),
            "mirror_left_right": trajectory_metrics.get("mirror_left_right"),
            "control_safe": False,
        },
        "runtime_replay": {
            "route": runtime_route.as_posix(),
            "runtime_report": (runtime_out / "runtime_report.json").as_posix(),
            "decision_count": runtime_report.get("decision_count"),
            "selected_candidate_entropy": runtime_report.get("selected_candidate_entropy"),
            "selected_candidate_dominant_fraction": runtime_report.get("selected_candidate_dominant_fraction"),
            "unsafe_selected_rate": runtime_report.get("unsafe_selected_rate"),
            "latency_end_to_end_p50_ms": runtime_report.get("latency_end_to_end_p50_ms"),
            "latency_end_to_end_p95_ms": runtime_report.get("latency_end_to_end_p95_ms"),
            "comparison": runtime_report.get("comparison"),
            "route_pose_leakage_ablation_fraction": runtime_report.get("route_pose_leakage_ablation_fraction"),
            "cmd_vel_proposal_count": runtime_report.get("cmd_vel_proposal_count"),
            "cmd_vel_non_null_count": runtime_report.get("cmd_vel_non_null_count"),
            "raw_pwm_emitted": runtime_report.get("raw_pwm_emitted", False),
            "runtime_api_step_count": runtime_report.get("runtime_api_step_count"),
            "scene_memory_used_for_policy": runtime_report.get("scene_memory_used_for_policy"),
            "scene_memory_artifact": runtime_report.get("scene_memory_artifact"),
            "scene_pose_trace_artifact": runtime_report.get("scene_pose_trace_artifact"),
            "scene_memory_visual": runtime_report.get("scene_memory_visual"),
            "physical_geometry_visual": runtime_report.get("scene_memory_visual"),
            "future_state_visual": runtime_report.get("scene_memory_visual"),
            "scene_bev_free_nonzero_fraction": runtime_report.get("scene_bev_free_nonzero_fraction"),
            "scene_bev_occupied_nonzero_fraction": runtime_report.get("scene_bev_occupied_nonzero_fraction"),
            "coverage_memory_cells_seen": runtime_report.get("coverage_memory_cells_seen"),
            "future_prediction_horizon_count": runtime_report.get("future_prediction_horizon_count"),
            "observed_cell_ratio": runtime_report.get("observed_cell_ratio"),
            "unknown_reduction_vs_current": runtime_report.get("unknown_reduction_vs_current"),
            "pose_ate_rmse_m_or_proxy": runtime_report.get("pose_ate_rmse_m_or_proxy"),
            "pose_rpe_translation_rmse_m_or_proxy": runtime_report.get("pose_rpe_translation_rmse_m_or_proxy"),
            "pose_metric_proxy": runtime_report.get("pose_metric_proxy"),
            "pose_metric_source": runtime_report.get("pose_metric_source"),
            "latency_step_p50_ms": runtime_report.get("latency_step_p50_ms"),
            "latency_step_p95_ms": runtime_report.get("latency_step_p95_ms"),
            "teacher_runtime_dependency": runtime_report.get("teacher_runtime_dependency"),
            "future_or_groundtruth_runtime_dependency": runtime_report.get("future_or_groundtruth_runtime_dependency"),
            "future_rollout_selection_mode": future_rollout_selection_mode,
            "runtime_policy_source": resolved_runtime_policy_source,
            "control_safe": False,
            "contact_sheet": contact_sheet.as_posix(),
        },
        "direct_rgbd_runtime_slice": direct_runtime_record,
        "gpu_inventory": _gpu_inventory(),
        "parallel_spatial_fold_devices": spatial_devices,
        "future_device": future_device,
        "acceptance": {
            "new_pipeline_command": True,
            "three_openloris_routes_different_scenes": len(route_artifacts) >= 3 and len(scenes) >= 3,
            "spatial_checkpoint_exists": Path(str(primary_fold["checkpoint"])).exists(),
            "future_rollout_checkpoint_exists": (future_train_dir / "checkpoint.pt").exists(),
            "heldout_runtime_report_non_empty": int(runtime_report.get("decision_count", 0)) > 0,
            "no_fake_synthetic_mock_metrics": True,
            "runtime_no_route_pose_leakage": float(runtime_report.get("route_pose_leakage_ablation_fraction", 0.0)) == 0.0,
            "bounded_cmd_vel_proposals_only": bool(runtime_report.get("cmd_vel_proposal_only", False))
            and bool(runtime_report.get("raw_pwm_emitted", False)) is False,
            "control_safe": False,
        },
        "control_safe": False,
        "replay_only": True,
        "not_executed": True,
        "product_training_approved": False,
        "raw_pwm_emitted": False,
    }
    accepted_runtime = summary["runtime_replay"]
    route_report_records = runtime_route_summaries
    scene_memory_artifacts_exist_all = all(_path_exists(record.get("scene_memory_artifact")) for record in route_report_records)
    scene_pose_trace_artifacts_exist_all = all(
        _path_exists(record.get("scene_pose_trace_artifact")) for record in route_report_records
    )
    physical_geometry_visuals_exist_all = all(
        _path_exists(record.get("physical_geometry_visual")) for record in route_report_records
    )
    future_state_visuals_exist_all = all(
        _path_exists(record.get("future_state_visual"))
        and int(record.get("future_prediction_horizon_count") or 0) >= 2
        for record in route_report_records
    )
    runtime_modes = sorted(
        {
            str(report.get("trajectory_scorer_mode"))
            for report in runtime_reports
            if report.get("trajectory_scorer_mode") is not None
        }
    )
    aggregate_runtime_mode = runtime_modes[0] if len(runtime_modes) == 1 else "mixed"
    summary.update(
        {
            "runtime_api_step_count": int(_sum_metric(route_report_records, "runtime_api_step_count")),
            "scene_state_online": _all_true(route_report_records, "scene_state_online"),
            "scene_map_created_only_posthoc": not _all_false(route_report_records, "scene_map_created_only_posthoc"),
            "current_bev_iou_or_proxy": _min_metric(fold_records, "current_bev_iou_or_proxy"),
            "fused_memory_bev_iou_or_proxy": _min_metric(fold_records, "fused_memory_bev_iou_or_proxy"),
            "fused_scene_bev_iou_or_proxy": _min_metric(fold_records, "fused_memory_bev_iou_or_proxy"),
            "scene_memory_used_for_policy": _all_true(route_report_records, "scene_memory_used_for_policy"),
            "scene_memory_artifact_exists": scene_memory_artifacts_exist_all,
            "scene_pose_trace_artifact_exists": scene_pose_trace_artifacts_exist_all,
            "physical_geometry_visual_exists": physical_geometry_visuals_exist_all,
            "future_state_visual_exists": future_state_visuals_exist_all,
            "unknown_reduction_vs_current": _min_metric(route_report_records, "unknown_reduction_vs_current"),
            "coverage_memory_cells_seen": int(_sum_metric(route_report_records, "coverage_memory_cells_seen")),
            "future_prediction_horizon_count": int(_min_metric(route_report_records, "future_prediction_horizon_count")),
            "future_free_iou_or_proxy": _min_metric(runtime_future_evals, "future_free_iou_or_proxy"),
            "future_occupied_iou_or_proxy": _min_metric(runtime_future_evals, "future_occupied_iou_or_proxy"),
            "pose_metric_proxy": runtime_report.get("pose_metric_proxy"),
            "pose_metric_source": aggregate_pose_source,
            "pose_metric_sources_by_route": {
                str(record["heldout_sequence"]): record.get("pose_metric_source")
                for record in route_report_records
            },
            "pose_metric_independent_groundtruth_route_count": int(independent_pose_route_count),
            "pose_metric_independent_groundtruth": all(
                record.get("pose_metric_independent_groundtruth") is True for record in route_report_records
            ),
            "pose_ate_rmse_m_or_proxy": _max_metric(route_report_records, "pose_ate_rmse_m_or_proxy"),
            "pose_rpe_translation_rmse_m_or_proxy": _max_metric(route_report_records, "pose_rpe_translation_rmse_m_or_proxy"),
            "pose_warp_valid_fraction": _min_metric(runtime_reports, "pose_warp_valid_fraction"),
            "future_prediction_metric_proxy": _min_metric(runtime_future_evals, "future_unknown_iou_or_proxy"),
            "future_unknown_iou_or_proxy": _min_metric(runtime_future_evals, "future_unknown_iou_or_proxy"),
            "action_entropy": _min_metric(route_report_records, "selected_candidate_entropy"),
            "dominant_action_fraction": _max_metric(route_report_records, "selected_candidate_dominant_fraction"),
            "unsafe_selected_rate": _max_metric(route_report_records, "unsafe_selected_rate"),
            "cmd_vel_proposal_count": int(_sum_metric(route_report_records, "cmd_vel_proposal_count")),
            "latency_step_p50_ms": _max_metric(route_report_records, "latency_step_p50_ms"),
            "latency_step_p95_ms": _max_metric(route_report_records, "latency_step_p95_ms"),
            "teacher_runtime_dependency": not _all_false(route_report_records, "teacher_runtime_dependency"),
            "future_or_groundtruth_runtime_dependency": not _all_false(
                route_report_records, "future_or_groundtruth_runtime_dependency"
            ),
            "route_pose_leakage_ablation_fraction": _max_metric(
                route_report_records, "route_pose_leakage_ablation_fraction"
            ),
            "raw_pwm_emitted": not _all_false(route_report_records, "raw_pwm_emitted"),
            "future_rollout_selection_mode": future_rollout_selection_mode,
            "runtime_policy_source_requested": runtime_policy_source,
            "runtime_policy_source_resolved": resolved_runtime_policy_source,
            "accepted_policy_uses_guided_transparent": accepted_policy_uses_guided_transparent,
            "accepted_policy_uses_handcrafted_diversity_prior": accepted_policy_uses_handcrafted_diversity_prior,
            "trajectory_scorer_mode": aggregate_runtime_mode,
            "trajectory_scorer_mode_distribution": runtime_report.get("trajectory_scorer_mode_distribution"),
            "future_rollout_policy_selection_role": runtime_report.get("future_rollout_policy_selection_role"),
            "temporal_diversity_prior_enabled_fraction": temporal_diversity_prior_fraction,
            "trajectory_label_degenerate": bool(trajectory_label_degenerate),
            "trajectory_label_action_entropy": float(trajectory_label_entropy),
            "trajectory_label_dominant_action_fraction": float(trajectory_label_dominant_fraction),
            "action_label_horizon_s": float(action_label_horizon_s) if action_label_horizon_s is not None else None,
        }
    )
    gate7_checks = {
        "real_public_data_only": summary.get("real_public_data_only") is True,
        "runtime_routes_evaluated_ge_2": int(summary.get("runtime_routes_evaluated") or 0) >= 2,
        "runtime_api_step_count_gt_0": int(summary.get("runtime_api_step_count") or 0) > 0,
        "scene_state_online": summary.get("scene_state_online") is True,
        "scene_memory_used_for_policy": summary.get("scene_memory_used_for_policy") is True,
        "scene_map_created_only_posthoc_false": summary.get("scene_map_created_only_posthoc") is False,
        "scene_memory_artifact_exists": scene_memory_artifacts_exist_all,
        "scene_pose_trace_artifact_exists": scene_pose_trace_artifacts_exist_all,
        "physical_geometry_visual_exists": physical_geometry_visuals_exist_all,
        "future_state_visual_exists": future_state_visuals_exist_all,
        "teacher_runtime_dependency_false": summary.get("teacher_runtime_dependency") is False,
        "future_or_groundtruth_runtime_dependency_false": summary.get("future_or_groundtruth_runtime_dependency") is False,
        "route_pose_leakage_ablation_fraction_eq_0": float(summary.get("route_pose_leakage_ablation_fraction") or 0.0)
        == 0.0,
        "raw_pwm_emitted_false": summary.get("raw_pwm_emitted") is False,
        "control_safe_false": summary.get("control_safe") is False,
        "accepted_policy_uses_guided_transparent_false": summary.get("accepted_policy_uses_guided_transparent")
        is False,
        "accepted_policy_uses_handcrafted_diversity_prior_false": summary.get(
            "accepted_policy_uses_handcrafted_diversity_prior"
        )
        is False,
        "action_entropy_gt_0": float(summary.get("action_entropy") or 0.0) > 0.0,
        "dominant_action_fraction_lt_1": float(summary.get("dominant_action_fraction") or 1.0) < 1.0,
        "learned_policy_runtime_selector": summary.get("trajectory_scorer_mode") in {"learned", "future_rollout"},
        "unsafe_selected_rate_eq_0": summary.get("unsafe_selected_rate") is not None
        and float(summary.get("unsafe_selected_rate")) == 0.0,
        "unknown_reduction_vs_current_gt_0": float(summary.get("unknown_reduction_vs_current") or 0.0) > 0.0,
        "coverage_memory_cells_seen_gt_0": int(summary.get("coverage_memory_cells_seen") or 0) > 0,
        "future_prediction_horizon_count_ge_2": int(summary.get("future_prediction_horizon_count") or 0) >= 2,
        "future_free_iou_or_proxy_gt_0": float(summary.get("future_free_iou_or_proxy") or 0.0) > 0.0,
        "future_occupied_iou_or_proxy_gt_0": float(summary.get("future_occupied_iou_or_proxy") or 0.0) > 0.0,
        "latency_step_p95_ms_le_100": float(summary.get("latency_step_p95_ms") or 1.0e9) <= 100.0,
    }
    caveat_repair_checks = {
        **gate7_checks,
        "pose_metric_source_learned_visual_memory": summary.get("pose_metric_source") == "learned_visual_memory",
        "pose_metric_has_independent_groundtruth_route": int(
            summary.get("pose_metric_independent_groundtruth_route_count") or 0
        )
        > 0,
        "trajectory_bc_labels_non_degenerate": summary.get("trajectory_label_degenerate") is False,
        "trajectory_bc_label_entropy_gt_0": float(summary.get("trajectory_label_action_entropy") or 0.0) > 0.0,
        "runtime_routes_evaluated_ge_3": int(summary.get("runtime_routes_evaluated") or 0) >= 3,
        "runtime_scenes_evaluated_ge_3": int(summary.get("runtime_scenes_evaluated") or 0) >= 3,
    }
    summary["acceptance"].update(gate7_checks)
    summary["acceptance"]["caveat_repair_checks"] = caveat_repair_checks
    summary["acceptance"]["non_collapsed_action_distribution"] = (
        bool(gate7_checks["action_entropy_gt_0"]) and bool(gate7_checks["dominant_action_fraction_lt_1"])
    )
    summary["acceptance"]["accepted_goal27_gate7"] = all(bool(value) for value in gate7_checks.values())
    summary["accepted_goal27_gate7"] = bool(summary["acceptance"]["accepted_goal27_gate7"])
    summary["acceptance"]["accepted_goal28_scene_brain"] = all(bool(value) for value in gate7_checks.values())
    summary["accepted_goal28_scene_brain"] = bool(summary["acceptance"]["accepted_goal28_scene_brain"])
    summary["acceptance"]["accepted_goal28_caveat_repair"] = all(
        bool(value) for value in caveat_repair_checks.values()
    )
    summary["accepted_goal28_caveat_repair"] = bool(summary["acceptance"]["accepted_goal28_caveat_repair"])
    summary["hard_gate_failures"] = [key for key, value in gate7_checks.items() if not bool(value)]
    summary["caveat_repair_failures"] = [key for key, value in caveat_repair_checks.items() if not bool(value)]
    summary_path = output / "milestone_report.json"
    write_json(summary_path, summary, pretty=True)
    _write_markdown_report(output / "milestone_report.md", summary)
    return summary


def _prepare_route(
    *,
    openloris_root: Path,
    out_dir: Path,
    sequence: str,
    max_frames: int,
    download_missing: bool,
    dino_device: str,
    dino_image_size: int,
    build_dino_features: bool,
    student_feature_source: str,
) -> RouteArtifact:
    metadata_path, return_code = setup_openloris_scene(
        out_dir=openloris_root,
        sequence=sequence,
        download=download_missing,
    )
    if return_code != 0:
        raise RuntimeError(f"OpenLORIS setup blocked for {sequence}; see {metadata_path}")
    source_dir = discover_sequence_root(openloris_root / sequence)
    scene = sequence_scene(source_dir)
    route_id = _safe_id(sequence)
    route_dir = out_dir / "routes" / f"openloris_{route_id}_route"
    openloris_to_route(
        source_dir=source_dir,
        out_dir=route_dir,
        max_frames=max_frames,
        require_depth=True,
        require_robot_frame_calibration=True,
        include_imu=False,
    )
    bev_dir = route_dir / "geometry" / "robot_rgbd_bev"
    robot_rgbd_to_bev(log_dir=route_dir, out_dir=bev_dir)
    bev_validation_path = out_dir / "qa" / f"bev_validate_{route_id}.json"
    bev_validation_path.parent.mkdir(parents=True, exist_ok=True)
    write_bev_metrics(validate_bev_artifacts(bev_dir).to_metrics(), bev_validation_path)
    visualization_manifest = visualize_bev(bev_dir, out_dir / "bev_visuals" / route_id)

    direct_rgbd_feature_dir = route_dir / "teacher_artifacts" / "direct_rgbd_student_features"
    if not has_direct_rgbd_feature_artifacts(
        direct_rgbd_feature_dir,
        expected_frame_count=max_frames,
        feature_dim=DIRECT_RGBD_FEATURE_DIM,
    ):
        if direct_rgbd_feature_dir.exists():
            shutil.rmtree(direct_rgbd_feature_dir)
        write_direct_rgbd_feature_artifacts(
            log_dir=route_dir,
            out_dir=direct_rgbd_feature_dir,
            feature_dim=DIRECT_RGBD_FEATURE_DIM,
        )
    if not has_direct_rgbd_feature_artifacts(
        direct_rgbd_feature_dir,
        expected_frame_count=max_frames,
        feature_dim=DIRECT_RGBD_FEATURE_DIM,
    ):
        raise RuntimeError(f"direct RGB-D feature artifact QA failed for {sequence}: {direct_rgbd_feature_dir}")

    dino_dir = route_dir / "teacher_artifacts" / "dino"
    if build_dino_features and not _has_real_dino(dino_dir, expected_frame_count=max_frames):
        if dino_dir.exists():
            shutil.rmtree(dino_dir)
        run_dino_teacher(
            route_dir,
            dino_dir,
            backend_name="real",
            model_id=DINO_DEFAULT_MODEL_ID,
            device=dino_device,
            image_size=dino_image_size,
        )
        _empty_cuda_cache()
    if build_dino_features:
        _require_real_dino(dino_dir)
    elif student_feature_source == "dino":
        raise RuntimeError("student_feature_source=dino requires build_dino_features=true")
    pack_feature_artifacts = direct_rgbd_feature_dir if student_feature_source == "direct_rgbd" else dino_dir
    spatial_pack_dir = out_dir / "spatial_packs" / f"openloris_{route_id}_spatial_pack"
    pack_spatial_dataset(
        log_dir=route_dir,
        bev_dir=bev_dir,
        teacher_artifacts=pack_feature_artifacts,
        out_dir=spatial_pack_dir,
    )
    spatial_qa_path = out_dir / "qa" / f"spatial_qa_{route_id}.json"
    spatial_qa = qa_spatial_dataset(spatial_pack_dir)
    write_qa_metrics(spatial_qa, spatial_qa_path)
    if not bool(spatial_qa.get("structurally_trainable", False)):
        raise RuntimeError(f"SpatialTrainPack is structurally blocked for {sequence}: {spatial_qa_path}")
    robot_frame_bev_qa_path = out_dir / "qa" / f"robot_frame_bev_qa_{route_id}.json"
    robot_qa = qa_robot_frame_bev(bev_dir=bev_dir, spatial_pack=spatial_pack_dir)
    write_json(robot_frame_bev_qa_path, robot_qa, pretty=True)
    if int(robot_qa.get("frame_count", 0)) <= 0:
        raise RuntimeError(f"Robot-frame BEV QA found no frames for {sequence}: {robot_frame_bev_qa_path}")
    return RouteArtifact(
        sequence=sequence,
        scene=scene,
        route_id=route_id,
        source_dir=source_dir,
        route_dir=route_dir,
        bev_dir=bev_dir,
        dino_dir=dino_dir,
        direct_rgbd_feature_dir=direct_rgbd_feature_dir,
        spatial_pack_dir=spatial_pack_dir,
        bev_validation_path=bev_validation_path,
        spatial_qa_path=spatial_qa_path,
        robot_frame_bev_qa_path=robot_frame_bev_qa_path,
        visualization_manifest=visualization_manifest,
    )


def _write_spatial_manifest(
    path: Path,
    *,
    route_artifacts: list[RouteArtifact],
    heldout_sequence: str,
    split_role: str,
    feature_source: str = "dino",
) -> Path:
    data = {
        "schema_version": "homebrain.openloris_route_heldout_spatial_manifest.v0",
        "split_policy": "leave_one_route_out",
        "split_role": split_role,
        "heldout_sequence": heldout_sequence,
        "feature_source": feature_source,
        "public_robot_data_only": True,
        "fake_synthetic_or_mock": False,
        "control_safe": False,
        "packs": [
            {
                "source_name": artifact.sequence,
                "scene": artifact.scene,
                "dataset": artifact.spatial_pack_dir.as_posix(),
                "features": _feature_dir_for_artifact(artifact, feature_source).as_posix(),
            }
            for artifact in route_artifacts
        ],
    }
    write_json(path, data, pretty=True)
    return path


def _feature_dir_for_artifact(artifact: RouteArtifact, feature_source: str) -> Path:
    if feature_source == "direct_rgbd":
        return artifact.direct_rgbd_feature_dir
    if feature_source == "dino":
        return artifact.dino_dir
    raise ValueError(f"unsupported feature_source {feature_source!r}")


def _launch_spatial_train(
    *,
    dataset_manifest: Path,
    out_dir: Path,
    max_steps: int,
    window_length: int,
    batch_size: int,
    device: str,
) -> subprocess.Popen[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "homebrain.train.train_spatial_v1",
        "--dataset-manifest",
        dataset_manifest.as_posix(),
        "--out",
        out_dir.as_posix(),
        "--max-steps",
        str(max_steps),
        "--window-length",
        str(window_length),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--sensor-context-mode",
        "masks",
        "--missing-pose-behavior",
        "reset",
    ]
    _write_text(out_dir / "command.txt", _command_text(command) + "\n")
    return subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _runtime_route_slice(artifact: RouteArtifact, out_dir: Path, max_frames: int | None) -> Path:
    if max_frames is None:
        return artifact.route_dir
    sliced = out_dir / "runtime_routes" / f"{artifact.route_id}_first_{max_frames}"
    openloris_to_route(
        source_dir=artifact.source_dir,
        out_dir=sliced,
        max_frames=max_frames,
        require_depth=True,
        require_robot_frame_calibration=True,
        include_imu=False,
    )
    return sliced


def write_runtime_failure_contact_sheet(
    *,
    route_dir: str | Path,
    modeld_log_dir: str | Path,
    out_dir: str | Path,
    name: str,
    max_items: int = 12,
) -> Path:
    route = Path(route_dir)
    modeld = Path(modeld_log_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    route_events = read_events(route)
    frames = {
        (event.sequence_id, event.camera_id, int(event.frame_id)): event
        for event in route_events
        if isinstance(event, FrameEvent)
    }
    outputs = [event for event in read_events(modeld) if isinstance(event, BrainOutputEvent)]
    ranked = sorted(outputs, key=_failure_score, reverse=True)[:max_items]
    tiles = []
    selected_records = []
    for event in ranked:
        if not isinstance(event.debug, dict):
            continue
        frame_id = int(event.debug.get("input_frame_id", -1))
        camera_id = str(event.debug.get("camera_id", ""))
        frame = frames.get((event.sequence_id, camera_id, frame_id))
        if frame is None or event.local_bev_ref is None:
            continue
        rgb = _load_rgb_for_contact_sheet(route / frame.data_ref, size=(96, 96))
        artifact = np.load(modeld / event.local_bev_ref)
        bev = _bev_rgb_from_npz(artifact, size=(96, 96))
        tile = np.concatenate([rgb, bev], axis=1)
        tiles.append(tile)
        selected_records.append(
            {
                "sequence_id": event.sequence_id,
                "camera_id": camera_id,
                "frame_id": frame_id,
                "selected_candidate_id": event.selected_trajectory_id,
                "failure_score": _failure_score(event),
                "local_bev_ref": event.local_bev_ref,
            }
        )
    if not tiles:
        sheet = np.zeros((96, 192, 3), dtype=np.uint8)
    else:
        cols = min(3, len(tiles))
        rows = int(math.ceil(len(tiles) / cols))
        sheet = np.zeros((rows * 96, cols * 192, 3), dtype=np.uint8)
        for index, tile in enumerate(tiles):
            row = index // cols
            col = index % cols
            sheet[row * 96 : (row + 1) * 96, col * 192 : (col + 1) * 192] = tile
    ppm_path = output / f"{name}_failure_contact_sheet.ppm"
    _write_ppm(ppm_path, sheet)
    manifest = {
        "schema_version": "homebrain.runtime_failure_contact_sheet.v0",
        "route": route.as_posix(),
        "modeld_log": modeld.as_posix(),
        "contact_sheet": ppm_path.as_posix(),
        "tile_layout": "rgb_left_bev_right",
        "item_count": len(selected_records),
        "frames": selected_records,
        "control_safe": False,
        "replay_only": True,
    }
    write_json(output / f"{name}_failure_contact_sheet.json", manifest, pretty=True)
    return ppm_path


def _failure_score(event: BrainOutputEvent) -> float:
    if not isinstance(event.debug, dict):
        return 0.0
    selected = event.debug.get("selected_candidate_metrics")
    score = 0.0
    if isinstance(selected, dict):
        for key in (
            "risk_score",
            "unknown_penalty",
            "uncertainty_penalty",
            "future_collision_probability",
            "future_unknown_exposure",
        ):
            value = selected.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                score += float(value)
    return float(score)


def _load_rgb_for_contact_sheet(path: Path, *, size: tuple[int, int]) -> np.ndarray:
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Pillow is required for contact sheets") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize(size), dtype=np.uint8)


def _bev_rgb_from_npz(artifact: Any, *, size: tuple[int, int]) -> np.ndarray:
    free = _first_npz_array(artifact, ("memory_bev_free_prob", "current_bev_free_prob", "bev_free_prob"))
    occupied = _first_npz_array(artifact, ("memory_bev_occupied_prob", "current_bev_occupied_prob", "bev_occupied_prob"))
    unknown = _first_npz_array(artifact, ("memory_bev_unknown_prob", "current_bev_unknown_prob", "bev_unknown_prob"))
    if free is None or occupied is None or unknown is None:
        return np.zeros((*size, 3), dtype=np.uint8)
    label = np.zeros((*free.shape, 3), dtype=np.uint8)
    label[unknown >= 0.5] = np.asarray([32, 32, 32], dtype=np.uint8)
    label[free >= 0.5] = np.asarray([45, 180, 100], dtype=np.uint8)
    label[occupied >= 0.5] = np.asarray([220, 70, 55], dtype=np.uint8)
    return _resize_rgb_nearest(label, size)


def _first_npz_array(artifact: Any, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in artifact:
            return np.asarray(artifact[name], dtype=np.float32)
    return None


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


def _require_real_dino(dino_dir: Path) -> None:
    manifest = read_json(dino_dir / "teacher_manifest.json")
    if manifest.get("teacher_name") != "dino":
        raise ValueError(f"expected DINO teacher manifest in {dino_dir}")
    if manifest.get("backend") != "real" or manifest.get("mock") is not False or manifest.get("synthetic") is not False:
        raise ValueError(f"milestone refuses non-real DINO artifacts: {dino_dir}")
    if manifest.get("real_perception") is not True:
        raise ValueError(f"DINO artifacts must be marked real_perception=true: {dino_dir}")


def _has_real_dino(dino_dir: Path, *, expected_frame_count: int) -> bool:
    manifest_path = dino_dir / "teacher_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = read_json(manifest_path)
    except Exception:  # noqa: BLE001
        return False
    if manifest.get("teacher_name") != "dino":
        return False
    if manifest.get("backend") != "real" or manifest.get("mock") is not False or manifest.get("synthetic") is not False:
        return False
    if manifest.get("real_perception") is not True:
        return False
    frame_count = int(manifest.get("frame_count", 0))
    if frame_count != expected_frame_count:
        return False
    return all((dino_dir / item["feature_file"]).exists() for item in manifest.get("features", []))


def _route_record(artifact: RouteArtifact) -> dict[str, Any]:
    dino_manifest = artifact.dino_dir / "teacher_manifest.json"
    direct_manifest = artifact.direct_rgbd_feature_dir / "teacher_manifest.json"
    spatial_qa = read_json(artifact.spatial_qa_path)
    robot_qa = read_json(artifact.robot_frame_bev_qa_path)
    return {
        "sequence": artifact.sequence,
        "scene": artifact.scene,
        "route": artifact.route_dir.as_posix(),
        "bev": artifact.bev_dir.as_posix(),
        "dino": artifact.dino_dir.as_posix(),
        "dino_manifest_sha256": file_sha256(dino_manifest) if dino_manifest.exists() else None,
        "direct_rgbd_features": artifact.direct_rgbd_feature_dir.as_posix(),
        "direct_rgbd_feature_manifest_sha256": file_sha256(direct_manifest),
        "spatial_pack": artifact.spatial_pack_dir.as_posix(),
        "bev_validation": artifact.bev_validation_path.as_posix(),
        "spatial_qa": artifact.spatial_qa_path.as_posix(),
        "spatial_structurally_trainable": bool(spatial_qa.get("structurally_trainable", False)),
        "spatial_trainable_candidate": bool(spatial_qa.get("trainable_candidate", False)),
        "spatial_quality_status": spatial_qa.get("quality_status"),
        "spatial_quarantine_reasons": spatial_qa.get("quarantine_reasons", []),
        "robot_frame_bev_qa": artifact.robot_frame_bev_qa_path.as_posix(),
        "robot_frame_bev_qa_pass": bool(robot_qa.get("qa_pass", False)),
        "robot_frame_bev_blockers": robot_qa.get("blockers", []),
        "visualization_manifest": artifact.visualization_manifest.as_posix(),
        "control_safe": False,
    }


def _resolve_fold_count(max_spatial_folds: int | None, route_artifacts: list[RouteArtifact]) -> int:
    if max_spatial_folds is not None:
        return max(1, min(int(max_spatial_folds), len(route_artifacts)))
    return max(1, min(len(route_artifacts), max(2, len(_parallel_4090_devices()))))


def _parallel_4090_devices() -> list[str]:
    return [f"cuda:{item['index']}" for item in _gpu_inventory() if "4090" in str(item.get("name", ""))]


def _preferred_4090_device() -> str | None:
    devices = _parallel_4090_devices()
    return devices[0] if devices else None


def _secondary_4090_device(existing: list[str]) -> str | None:
    for device in _parallel_4090_devices():
        if device not in existing[:1]:
            return device
    return existing[0] if existing else None


def _preferred_cuda_device() -> str | None:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    return "cuda:0" if torch.cuda.is_available() else None


def _empty_cuda_cache() -> None:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()


def _command_text(command: list[str]) -> str:
    return " ".join(command)


def _path_exists(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).exists()


def _distribution_has(value: Any, key: str) -> bool:
    return isinstance(value, dict) and int(value.get(key, 0) or 0) > 0


def _metric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)):
            values.append(float(value))
    return values


def _min_metric(records: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = _metric_values(records, key)
    return float(min(values)) if values else float(default)


def _max_metric(records: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = _metric_values(records, key)
    return float(max(values)) if values else float(default)


def _sum_metric(records: list[dict[str, Any]], key: str) -> float:
    return float(sum(_metric_values(records, key)))


def _all_true(records: list[dict[str, Any]], key: str) -> bool:
    return bool(records) and all(record.get(key) is True for record in records)


def _all_false(records: list[dict[str, Any]], key: str) -> bool:
    return bool(records) and all(record.get(key) is False for record in records)


def _entropy_counts(counts: dict[str, Any]) -> float:
    total = sum(int(value or 0) for value in counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        count = int(value or 0)
        if count <= 0:
            continue
        probability = count / float(total)
        entropy -= probability * math.log2(probability)
    return float(entropy)


def _dominant_fraction_counts(counts: dict[str, Any]) -> float:
    total = sum(int(value or 0) for value in counts.values())
    if total <= 0:
        return 0.0
    return float(max((int(value or 0) for value in counts.values()), default=0) / float(total))


def _json_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _write_text(path: Path, text: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8", newline="\n")


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    runtime = summary["runtime_replay"]
    future = summary["future_rollout"]
    lines = [
        "# OpenLORIS Route-Heldout Milestone",
        "",
        f"- Routes/scenes: `{summary['route_count']}` / `{summary['scenes']}`",
        f"- Heldout sequence: `{summary['heldout_sequence']}`",
        f"- Spatial checkpoint: `{summary['spatial_primary_checkpoint']}`",
        f"- Future rollout checkpoint: `{future['checkpoint']}`",
        f"- Runtime report: `{runtime['runtime_report']}`",
        f"- Decisions: `{runtime['decision_count']}`",
        f"- Entropy/dominant/unsafe: `{runtime['selected_candidate_entropy']}` / `{runtime['selected_candidate_dominant_fraction']}` / `{runtime['unsafe_selected_rate']}`",
        f"- Latency p50/p95 ms: `{runtime['latency_end_to_end_p50_ms']}` / `{runtime['latency_end_to_end_p95_ms']}`",
        f"- Future free/occupied/unknown IoU proxy: `{future['future_free_iou_or_proxy']}` / `{future['future_occupied_iou_or_proxy']}` / `{future['future_unknown_iou_or_proxy']}`",
        f"- Contact sheet: `{runtime['contact_sheet']}`",
        "- Safety: replay-only, not executed, `control_safe=false`, bounded cmd_vel proposals only, raw PWM false.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _parse_sequences(value: str) -> list[str]:
    sequences = [item.strip() for item in value.split(",") if item.strip()]
    if not sequences:
        raise argparse.ArgumentTypeError("at least one sequence is required")
    return sequences


def _dedupe_sequences(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real-data OpenLORIS route-heldout replay-as-live milestone "
            "without fake, mock, synthetic, or generated metric data."
        )
    )
    parser.add_argument("--openloris-root", default="data/public/openloris_scene")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sequences", type=_parse_sequences, default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--heldout-sequence", default=None)
    parser.add_argument(
        "--runtime-heldout-sequences",
        type=_parse_sequences,
        default=None,
        help="Comma-separated heldout routes to replay through the runtime; Goal28 requires at least two.",
    )
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--dino-device", default=None)
    parser.add_argument("--dino-image-size", type=int, default=224)
    parser.add_argument("--spatial-steps", type=int, default=80)
    parser.add_argument("--spatial-window-length", type=int, default=4)
    parser.add_argument("--spatial-batch-size", type=int, default=4)
    parser.add_argument("--max-spatial-folds", type=int, default=None)
    parser.add_argument("--future-steps", type=int, default=80)
    parser.add_argument("--future-batch-size", type=int, default=4)
    parser.add_argument("--future-max-examples", type=int, default=None)
    parser.add_argument("--trajectory-steps", type=int, default=160)
    parser.add_argument("--trajectory-batch-size", type=int, default=32)
    parser.add_argument(
        "--action-label-horizon-s",
        type=float,
        default=None,
        help="Future-motion horizon for behavior-cloning action labels; defaults to candidate max duration.",
    )
    parser.add_argument("--runtime-max-frames", type=int, default=None)
    parser.add_argument("--runtime-feature-source", choices=RUNTIME_FEATURE_SOURCES, default="dino")
    parser.add_argument("--student-feature-source", choices=("dino", "direct_rgbd"), default="dino")
    parser.add_argument(
        "--future-rollout-selection-mode",
        choices=("argmin", "safe_argmin", "guided_transparent"),
        default="guided_transparent",
    )
    parser.add_argument(
        "--runtime-policy-source",
        choices=("trajectory_scorer", "future_rollout", "auto"),
        default="trajectory_scorer",
    )
    parser.add_argument("--skip-direct-rgbd-runtime-slice", action="store_true")
    parser.add_argument("--skip-dino-features", action="store_true")
    parser.add_argument("--pose-warp-source", choices=V1_POSE_WARP_SOURCES, default="odom")
    parser.add_argument("--v1-policy-bev-source", choices=V1_POLICY_BEV_SOURCES, default="scene")
    args = parser.parse_args(argv)
    command = "python -m homebrain.tools.run_openloris_route_heldout_milestone " + " ".join(sys.argv[1:])
    summary = run_openloris_route_heldout_milestone(
        openloris_root=args.openloris_root,
        out_dir=args.out,
        sequences=args.sequences,
        heldout_sequence=args.heldout_sequence,
        runtime_heldout_sequences=args.runtime_heldout_sequences,
        max_frames=args.max_frames,
        download_missing=args.download_missing,
        dino_device=args.dino_device,
        dino_image_size=args.dino_image_size,
        spatial_steps=args.spatial_steps,
        spatial_window_length=args.spatial_window_length,
        spatial_batch_size=args.spatial_batch_size,
        max_spatial_folds=args.max_spatial_folds,
        future_steps=args.future_steps,
        future_batch_size=args.future_batch_size,
        future_max_examples=args.future_max_examples,
        trajectory_steps=args.trajectory_steps,
        trajectory_batch_size=args.trajectory_batch_size,
        action_label_horizon_s=args.action_label_horizon_s,
        runtime_max_frames=args.runtime_max_frames,
        runtime_feature_source=args.runtime_feature_source,
        student_feature_source=args.student_feature_source,
        future_rollout_selection_mode=args.future_rollout_selection_mode,
        runtime_policy_source=args.runtime_policy_source,
        run_direct_rgbd_runtime_slice=not args.skip_direct_rgbd_runtime_slice,
        build_dino_features=not args.skip_dino_features,
        pose_warp_source=args.pose_warp_source,
        policy_bev_source=args.v1_policy_bev_source,
        command=command,
    )
    print(
        json.dumps(
            {
                "milestone_report": (Path(args.out) / "milestone_report.json").as_posix(),
                "spatial_checkpoint": summary["spatial_primary_checkpoint"],
                "future_checkpoint": summary["future_rollout"]["checkpoint"],
                "runtime_report": summary["runtime_replay"]["runtime_report"],
                "decision_count": summary["runtime_replay"]["decision_count"],
                "control_safe": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
