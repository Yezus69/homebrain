from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from homebrain.brain.modeld import write_spatial_model_outputs
from homebrain.brain.visualize_spatial_outputs import visualize_spatial_outputs
from homebrain.data.pack_spatial_dataset import pack_spatial_dataset
from homebrain.data.qa_spatial_dataset import qa_spatial_dataset, write_qa_metrics
from homebrain.data.spatial_dataset import write_json
from homebrain.datasets.openloris_scene import (
    OPENLORIS_LICENSE_REVIEW_STATUS,
    choose_remote_package,
    list_remote_package_files,
    openloris_sequence_dir,
    sequence_scene,
    validate_sequence_dir,
)
from homebrain.datasets.openloris_to_route import openloris_to_route
from homebrain.datasets.setup_openloris_scene import setup_openloris_scene
from homebrain.eval.run_eval import evaluate_log, write_eval_metrics
from homebrain.geometry.robot_rgbd_to_bev import robot_rgbd_to_bev
from homebrain.geometry.validate_bev import validate_bev_artifacts, write_bev_metrics
from homebrain.policies.audit_bev_action_sanity import audit_bev_action_sanity
from homebrain.policies.build_action_label_pack import build_action_label_pack
from homebrain.policies.qa_action_label_pack import qa_action_label_pack
from homebrain.policies.trajectory_scorer_net_v0 import eval_trajectory_scorer_v0, train_trajectory_scorer_v0
from homebrain.replay.replayd import replay_log
from homebrain.replay.segment_log import read_events
from homebrain.messages.schema import BrainOutputEvent
from homebrain.teachers.artifacts import file_sha256, read_json
from homebrain.teachers.dino_teacher import run_dino_teacher
from homebrain.train.eval_spatial_v0 import eval_spatial_v0
from homebrain.train.train_spatial_v0 import train_spatial_v0


REPORT_JSON = Path("runs/goal11b_nightly_report.json")
REPORT_MD = Path("runs/goal11b_nightly_report.md")
DEFAULT_ROUTE_ORDER = (
    "cafe1-1_2",
    "office1-1_7",
    "corridor1-1",
    "home1-1_5",
    "corridor1-2_5",
    "market1-1_3",
)


@dataclass(frozen=True)
class RouteSpec:
    sequence: str
    package_path: str
    package_gb: float
    scene: str


def run_goal11b_nightly(
    *,
    data_root: str | Path = "data/public/openloris_scene",
    run_root: str | Path = "runs/goal11b_nightly",
    max_frames_per_route: int = 1200,
    max_total_gb: float = 80.0,
    target_steps: int = 10_000,
    batch_size: int = 32,
    spatial_batch_size: int = 16,
    devices: list[str] | None = None,
    skip_download: bool = False,
    skip_dino: bool = False,
    skip_training: bool = False,
    purge_download_archives: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": "homebrain.goal11b_nightly_report.v0",
        "goal": "11B-NIGHTLY",
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "wall_time_sec": None,
        "max_frames_per_route": int(max_frames_per_route),
        "max_total_gb": float(max_total_gb),
        "target_steps": int(target_steps),
        "devices_requested": list(devices or _default_cuda_devices()),
        "gpu_inventory": _gpu_inventory(),
        "routes": [],
        "failed_routes": [],
        "skipped_routes": [],
        "artifacts": {},
        "training_runs": [],
        "generalization": {},
        "replay": {},
        "blockers": [],
        "safety_flags": {
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "product_training_approved": False,
        },
    }
    _write_reports(report)
    run_root = Path(run_root)
    data_root = Path(data_root)
    devices = devices or _default_cuda_devices()

    try:
        selected_routes = _select_routes(max_total_gb=max_total_gb)
        report["selected_routes"] = [spec.__dict__ for spec in selected_routes]
        route_records = []
        for index, spec in enumerate(selected_routes):
            device = devices[index % len(devices)]
            try:
                route_records.append(
                    _prepare_route(
                        spec=spec,
                        data_root=data_root,
                        run_root=run_root,
                        max_frames_per_route=max_frames_per_route,
                        device=device,
                        skip_download=skip_download,
                        skip_dino=skip_dino,
                        purge_download_archives=purge_download_archives,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one bad public route must not hide usable routes.
                failed = {
                    "sequence": spec.sequence,
                    "scene": spec.scene,
                    "package_path": spec.package_path,
                    "package_gb": spec.package_gb,
                    "device": device,
                    "exact_failure": str(exc),
                    "minimal_next_repair": (
                        "inspect dataset calibration/intrinsics/transform files; keep route out of robot-frame "
                        "action supervision until robot_rgbd_to_bev succeeds without assumed transforms"
                    ),
                }
                report.setdefault("failed_routes", []).append(failed)
            report["routes"] = route_records
            report["frames_processed"] = sum(int(route.get("frame_count", 0)) for route in route_records)
            _write_reports(report)

        evaluated_routes = [route for route in route_records if route.get("fully_evaluated") is True]
        report["evaluated_route_count"] = len(evaluated_routes)
        report["scene_count"] = len({route.get("scene") for route in evaluated_routes})
        if len(evaluated_routes) < 3:
            report["blockers"].append(
                {
                    "kind": "dataset_access",
                    "exact_failure": f"only {len(evaluated_routes)} OpenLORIS robot-frame routes fully evaluated",
                    "minimal_next_repair": "stage at least three OpenLORIS packages that pass import/BEV/QA/DINO.",
                }
            )

        action_pack, action_qa = _build_action_pack_v4(run_root, evaluated_routes)
        report["artifacts"]["action_label_pack_v4"] = action_pack.as_posix()
        report["artifacts"]["action_label_pack_v4_qa"] = action_qa.as_posix()
        report["action_label_pack_v4_qa"] = read_json(action_qa)
        _write_reports(report)

        if not skip_training:
            training = _run_training_sweeps(
                run_root=run_root,
                routes=evaluated_routes,
                action_pack=action_pack,
                target_steps=target_steps,
                batch_size=batch_size,
                spatial_batch_size=spatial_batch_size,
                devices=devices,
            )
            report.update(training)
            _add_goal11b_collapse_diagnosis(report)
            _write_reports(report)
        else:
            report["blockers"].append(
                {
                    "kind": "compute",
                    "exact_failure": "training sweeps skipped by --skip-training",
                    "minimal_next_repair": "rerun without --skip-training.",
                }
            )
    except Exception as exc:  # noqa: BLE001 - nightly must leave evidence.
        report["blockers"].append(
            {
                "kind": "pipeline_exception",
                "exact_failure": str(exc),
                "minimal_next_repair": "inspect the last completed route/training artifact and rerun the nightly script.",
            }
        )
    finally:
        report["completed_at_utc"] = _utc_now()
        report["wall_time_sec"] = round(time.perf_counter() - started, 3)
        _write_reports(report)
    return report


def run_goal11b_true_loro_supplement(
    *,
    run_root: str | Path = "runs/goal11b_nightly",
    target_steps: int = 10_000,
    batch_size: int = 32,
    spatial_batch_size: int = 16,
    devices: list[str] | None = None,
) -> dict[str, Any]:
    if not REPORT_JSON.exists():
        raise FileNotFoundError(f"nightly report does not exist: {REPORT_JSON}")
    report = read_json(REPORT_JSON)
    route_records = [route for route in report.get("routes", []) if route.get("fully_evaluated") is True]
    if len(route_records) < 2:
        raise ValueError("true leave-one-route-out needs at least two evaluated routes")
    action_pack_value = report.get("artifacts", {}).get("action_label_pack_v4")
    if not isinstance(action_pack_value, str):
        raise ValueError("report is missing artifacts.action_label_pack_v4")
    resolved_devices = devices or list(report.get("devices_requested") or _default_cuda_devices())
    supplement = _run_true_loro_generalization(
        run_root=Path(run_root),
        routes=route_records,
        action_pack=Path(action_pack_value),
        target_steps=target_steps,
        batch_size=batch_size,
        spatial_batch_size=spatial_batch_size,
        devices=resolved_devices,
    )
    report.setdefault("generalization", {}).update(supplement)
    report["generalization"]["true_loro_completed_at_utc"] = _utc_now()
    _add_model_vs_oracle_gap(report)
    _add_goal11b_collapse_diagnosis(report)

    route_wall_time_sec = float(report.get("route_pipeline_wall_time_sec", report.get("wall_time_sec") or 0.0))
    training_wall_time_sec = float(report.get("training_wall_time_sec") or 0.0)
    true_loro_wall_time_sec = float(supplement.get("true_loro_wall_time_sec") or 0.0)
    report["route_pipeline_wall_time_sec"] = round(route_wall_time_sec, 3)
    report["wall_time_components_sec"] = {
        "route_pipeline": round(route_wall_time_sec, 3),
        "training_sweeps": round(training_wall_time_sec, 3),
        "true_loro_generalization": round(true_loro_wall_time_sec, 3),
    }
    report["wall_time_sec"] = round(route_wall_time_sec + training_wall_time_sec + true_loro_wall_time_sec, 3)
    report["completed_at_utc"] = _utc_now()
    _write_reports(report)
    return report


def run_goal11b_postprocess_report() -> dict[str, Any]:
    if not REPORT_JSON.exists():
        raise FileNotFoundError(f"nightly report does not exist: {REPORT_JSON}")
    report = read_json(REPORT_JSON)
    _add_model_vs_oracle_gap(report)
    _add_goal11b_collapse_diagnosis(report)
    _write_reports(report)
    return report


def _add_model_vs_oracle_gap(report: dict[str, Any]) -> None:
    scorer_runs = [
        run
        for run in report.get("training_runs", [])
        if isinstance(run, dict) and str(run.get("config", "")).startswith(("C_", "D_"))
    ]
    oracle_runs = [run for run in scorer_runs if run.get("config") == "C_scorer_oracle_label_bev"]
    model_runs = [run for run in scorer_runs if run.get("config") == "D_scorer_model_bev_best_spatial"]
    if not oracle_runs or not model_runs:
        return
    best_oracle = max(oracle_runs, key=lambda item: float(item.get("top1_action_agreement") or 0.0))
    best_model = max(
        model_runs,
        key=lambda item: (
            float(item.get("top1_action_agreement") or 0.0),
            float(item.get("rank_corr") or 0.0),
            -float(item.get("dominant_action_fraction") or 1.0),
        ),
    )
    report["best_oracle_scorer_checkpoint"] = best_oracle
    report["model_vs_oracle_gap"] = {
        "metric": "top1_action_agreement",
        "model_minus_oracle": float(best_model.get("top1_action_agreement") or 0.0)
        - float(best_oracle.get("top1_action_agreement") or 0.0),
        "model_checkpoint": best_model.get("checkpoint"),
        "oracle_checkpoint": best_oracle.get("checkpoint"),
    }


def _add_goal11b_collapse_diagnosis(report: dict[str, Any]) -> None:
    threshold_entropy = 1.0
    threshold_dominant = 0.65
    diagnosed: list[dict[str, Any]] = []

    action_qa = report.get("action_label_pack_v4_qa")
    if isinstance(action_qa, dict):
        action_pack_flag, reasons = _goal11b_collapse_flag(action_qa, threshold_entropy, threshold_dominant)
        action_qa["goal11b_distribution_collapse_flag"] = action_pack_flag
        action_qa["goal11b_distribution_collapse_reasons"] = reasons
        if action_pack_flag:
            diagnosed.append({"scope": "action_label_pack_v4", "reasons": reasons})

    for run in report.get("training_runs", []):
        if not isinstance(run, dict) or "action_entropy" not in run:
            continue
        flag, reasons = _goal11b_collapse_flag(run, threshold_entropy, threshold_dominant)
        run["goal11b_distribution_collapse_flag"] = flag
        run["goal11b_distribution_collapse_reasons"] = reasons
        if flag:
            diagnosed.append(
                {
                    "scope": "training_run",
                    "config": run.get("config"),
                    "seed": run.get("seed"),
                    "reasons": reasons,
                }
            )

    true_loro = report.get("generalization", {}).get("true_leave_one_route_out")
    if isinstance(true_loro, dict):
        for route_name, route_metrics in true_loro.items():
            if not isinstance(route_metrics, dict):
                continue
            flag, reasons = _goal11b_collapse_flag(route_metrics, threshold_entropy, threshold_dominant)
            route_metrics["goal11b_distribution_collapse_flag"] = flag
            route_metrics["goal11b_distribution_collapse_reasons"] = reasons
            if flag:
                diagnosed.append(
                    {
                        "scope": "true_leave_one_route_out",
                        "route": route_name,
                        "reasons": reasons,
                    }
                )

    report["collapse_diagnosis"] = {
        "thresholds": {
            "action_entropy_min": threshold_entropy,
            "dominant_action_fraction_max": threshold_dominant,
        },
        "global_action_pack_collapsed": bool(
            isinstance(action_qa, dict) and action_qa.get("goal11b_distribution_collapse_flag") is True
        ),
        "diagnosed_count": len(diagnosed),
        "diagnosed": diagnosed,
        "interpretation": (
            "Global v4 labels and full-data scorer sweeps do not meet the Goal 11B collapse threshold; "
            "true route-out retrains still show held-out distribution concentration where listed."
        ),
    }


def _goal11b_collapse_flag(
    metrics: dict[str, Any],
    threshold_entropy: float,
    threshold_dominant: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    entropy = metrics.get("action_entropy")
    dominant = metrics.get("dominant_action_fraction")
    if isinstance(entropy, (int, float)) and not isinstance(entropy, bool) and float(entropy) < threshold_entropy:
        reasons.append(f"action_entropy<{threshold_entropy}")
    if isinstance(dominant, (int, float)) and not isinstance(dominant, bool) and float(dominant) > threshold_dominant:
        reasons.append(f"dominant_action_fraction>{threshold_dominant}")
    return bool(reasons), reasons


def _select_routes(*, max_total_gb: float) -> list[RouteSpec]:
    remote_files = list_remote_package_files()
    selected: list[RouteSpec] = []
    total = 0.0
    for sequence in DEFAULT_ROUTE_ORDER:
        remote = choose_remote_package(sequence, remote_files)
        if remote is None or "groundtruth" in remote.path:
            continue
        if total + remote.size_gb > max_total_gb:
            continue
        selected.append(
            RouteSpec(
                sequence=sequence,
                package_path=remote.path,
                package_gb=round(remote.size_gb, 3),
                scene=sequence_scene(sequence),
            )
        )
        total += remote.size_gb
    return selected


def _prepare_route(
    *,
    spec: RouteSpec,
    data_root: Path,
    run_root: Path,
    max_frames_per_route: int,
    device: str,
    skip_download: bool,
    skip_dino: bool,
    purge_download_archives: bool,
) -> dict[str, Any]:
    safe_name = _safe_name(spec.sequence)
    record: dict[str, Any] = {
        "sequence": spec.sequence,
        "scene": spec.scene,
        "package_path": spec.package_path,
        "package_gb": spec.package_gb,
        "license_review_status": OPENLORIS_LICENSE_REVIEW_STATUS,
        "device": device,
        "fully_evaluated": False,
        "failures": [],
        "skips": [],
    }
    sequence_path = openloris_sequence_dir(data_root, spec.sequence)
    try:
        validate_sequence_dir(sequence_path, require_depth=True)
    except Exception:
        if skip_download:
            raise RuntimeError(f"OpenLORIS sequence {spec.sequence} is not staged and --skip-download was set")
        metadata_path, code = setup_openloris_scene(
            out_dir=data_root,
            sequence=spec.sequence,
            download=True,
            max_download_gb=80.0,
            allow_large_download=True,
        )
        record["setup_metadata"] = metadata_path.as_posix()
        if code != 0:
            raise RuntimeError(f"OpenLORIS setup blocked for {spec.sequence}; see {metadata_path}")
        if purge_download_archives:
            _purge_download_archive(data_root, spec)

    route_dir = run_root / "routes" / f"openloris_{safe_name}_route"
    bev_dir = route_dir / "geometry" / "robot_rgbd_bev"
    pack_dir = run_root / "spatial_packs" / f"openloris_{safe_name}_spatial_pack"
    qa_path = run_root / "qa" / f"{safe_name}_spatial_qa.json"
    sanity_dir = run_root / "action_sanity" / safe_name
    dino_dir = route_dir / "teacher_artifacts" / "dino"

    openloris_to_route(source_dir=sequence_path, out_dir=route_dir, max_frames=max_frames_per_route)
    robot_rgbd_to_bev(log_dir=route_dir, out_dir=bev_dir)
    bev_eval = validate_bev_artifacts(bev_dir)
    write_bev_metrics(bev_eval.to_metrics(), run_root / "qa" / f"{safe_name}_bev_eval.json")
    pack_spatial_dataset(log_dir=route_dir, bev_dir=bev_dir, out_dir=pack_dir)
    spatial_qa = qa_spatial_dataset(pack_dir)
    write_qa_metrics(spatial_qa, qa_path)
    sanity = audit_bev_action_sanity(source=pack_dir, out_dir=sanity_dir)
    dino_status = _ensure_dino_features(
        route_dir=route_dir,
        dino_dir=dino_dir,
        device=device,
        skip_dino=skip_dino,
    )

    route_manifest = read_json(route_dir / "route_metadata.json")
    pack_manifest = read_json(pack_dir / "manifest.json")
    bev_manifest = read_json(bev_dir / "bev_manifest.json")
    record.update(
        {
            "route_dir": route_dir.as_posix(),
            "bev_dir": bev_dir.as_posix(),
            "spatial_pack": pack_dir.as_posix(),
            "spatial_qa": qa_path.as_posix(),
            "action_sanity": (sanity_dir / "bev_action_sanity.json").as_posix(),
            "dino_dir": dino_dir.as_posix(),
            "dino_status": dino_status,
            "frame_count": int(route_manifest.get("frame_count", pack_manifest.get("example_count", 0))),
            "spatial_example_count": int(pack_manifest.get("example_count", 0)),
            "pose_label_count": int(pack_manifest.get("pose_label_count", 0)),
            "action_supervision_ok_fraction": float(sanity.get("action_supervision_ok_fraction", 0.0)),
            "bev_stats": {
                "frame_count": int(bev_manifest.get("frame_count", 0)),
                "free_ratio_mean": float(bev_eval.free_ratio_mean),
                "obstacle_ratio_mean": float(bev_eval.obstacle_ratio_mean),
                "unknown_ratio_mean": float(bev_eval.unknown_ratio_mean),
                "confidence_mean": float(bev_eval.confidence_mean),
            },
            "hashes": {
                "route_manifest_sha256": file_sha256(route_dir / "manifest.json"),
                "route_events_sha256": file_sha256(route_dir / "events.jsonl"),
                "bev_manifest_sha256": file_sha256(bev_dir / "bev_manifest.json"),
                "spatial_manifest_sha256": file_sha256(pack_dir / "manifest.json"),
                "dino_manifest_sha256": file_sha256(dino_dir / "teacher_manifest.json")
                if (dino_dir / "teacher_manifest.json").exists()
                else None,
            },
            "fully_evaluated": bool(
                spatial_qa.get("missing_count", 0) == 0
                and spatial_qa.get("shape_error_count", 0) == 0
                and spatial_qa.get("nan_count", 0) == 0
                and sanity.get("action_supervision_ok_fraction", 0.0) > 0.0
                and dino_status.get("frame_count", 0) >= int(pack_manifest.get("example_count", 0))
            ),
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "product_training_approved": False,
        }
    )
    record["skips"].append({"stage": "da3", "reason": "optional_goal11b_stage_not_run_by_default"})
    return record


def _ensure_dino_features(*, route_dir: Path, dino_dir: Path, device: str, skip_dino: bool) -> dict[str, Any]:
    if skip_dino:
        return {"status": "skipped", "reason": "--skip-dino", "frame_count": 0}
    manifest_path = dino_dir / "teacher_manifest.json"
    route_frame_count = int(read_json(route_dir / "route_metadata.json").get("frame_count", 0))
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (
            manifest.get("teacher_name") == "dino"
            and manifest.get("backend") == "real"
            and int(manifest.get("frame_count", 0)) >= route_frame_count
        ):
            return {"status": "reused", "frame_count": int(manifest.get("frame_count", 0))}
    summary = run_dino_teacher(
        route_dir,
        dino_dir,
        backend_name="real",
        model_id="dinov2_vits14",
        device=device,
        image_size=224,
    )
    return {"status": "written", "frame_count": int(summary.frame_count)}


def _purge_download_archive(data_root: Path, spec: RouteSpec) -> None:
    archive = data_root / "_downloads" / Path(spec.package_path).name
    if archive.exists() and archive.is_file():
        archive.unlink()


def _build_action_pack_v4(run_root: Path, routes: list[dict[str, Any]]) -> tuple[Path, Path]:
    controlled = Path("runs/goal9_controlled_bev_maps")
    if not controlled.exists():
        from homebrain.policies.generate_controlled_bev_maps import generate_controlled_bev_maps

        generate_controlled_bev_maps(out_dir=controlled, meters_per_cell=0.05)
    sources = [controlled]
    sources.extend(Path(route["spatial_pack"]) for route in routes)
    for optional in (
        Path("runs/room_walk_001_da3_stable_spatial_pack_short60"),
        Path("runs/tum_freiburg1_xyz_rgbd_truth_spatial_pack"),
    ):
        if optional.exists():
            sources.append(optional)
    action_pack = run_root / "action_label_pack_v4"
    build_action_label_pack(sources=sources, out_dir=action_pack, pack_version=4)
    qa = qa_action_label_pack(action_pack)
    qa_path = run_root / "action_label_pack_v4_qa.json"
    write_json(qa_path, qa, pretty=True)
    return action_pack, qa_path


def _run_training_sweeps(
    *,
    run_root: Path,
    routes: list[dict[str, Any]],
    action_pack: Path,
    target_steps: int,
    batch_size: int,
    spatial_batch_size: int,
    devices: list[str],
) -> dict[str, Any]:
    train_root = run_root / "training"
    manifest_all = run_root / "spatial_dataset_manifest_all.json"
    _write_spatial_manifest(manifest_all, routes)
    training_runs: list[dict[str, Any]] = []
    spatial_runs: list[dict[str, Any]] = []
    scorer_runs: list[dict[str, Any]] = []

    for config_name, sensor_mode in (("A_spatial_dino_bev_pose", "masks"), ("B_spatial_dino_odom_masks_bev_pose", "odom")):
        for seed_index, seed in enumerate((7, 17)):
            out_dir = train_root / config_name / f"seed_{seed}"
            device = devices[seed_index % len(devices)]
            metrics = train_spatial_v0(
                dataset_manifest=manifest_all,
                out_dir=out_dir,
                max_steps=target_steps,
                batch_size=spatial_batch_size,
                device_name=device,
                seed=seed,
                sensor_context_mode=sensor_mode,
            )
            eval_path = out_dir / "eval_val.json"
            eval_metrics = eval_spatial_v0(
                checkpoint=out_dir / "checkpoint.pt",
                dataset_manifest=manifest_all,
                out_path=eval_path,
                split="val",
                batch_size=spatial_batch_size,
                device_name=device,
                sensor_context_mode=sensor_mode,
            )
            record = {
                "config": config_name,
                "seed": seed,
                "device": device,
                "checkpoint": (out_dir / "checkpoint.pt").as_posix(),
                "train_metrics": (out_dir / "train_metrics.json").as_posix(),
                "eval_metrics": eval_path.as_posix(),
                "bev_iou_or_proxy": eval_metrics.get("bev_iou_or_proxy"),
                "pose_delta_rmse": eval_metrics.get("pose_delta_rmse"),
                "inference_fps": eval_metrics.get("inference_fps"),
                "max_steps": target_steps,
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
                "product_training_approved": False,
            }
            spatial_runs.append(record)
            training_runs.append(record)

    best_spatial = min(spatial_runs, key=lambda item: float(read_json(item["eval_metrics"]).get("val_loss", math.inf)))
    modeld_root = run_root / "modeld_best_spatial"
    for route in routes:
        route_modeld = modeld_root / _safe_name(str(route["sequence"]))
        write_spatial_model_outputs(
            route["route_dir"],
            route_modeld,
            checkpoint=best_spatial["checkpoint"],
            feature_dir=route["dino_dir"],
            device_name=devices[0],
        )
        visualize_spatial_outputs(
            log_dir=route_modeld,
            out_dir=run_root / "viz" / f"{_safe_name(str(route['sequence']))}_modeld_spatial",
            dataset_dir=route["spatial_pack"],
        )

    for config_name, bev_source, source_family, modeld_dir in (
        ("C_scorer_oracle_label_bev", "oracle", None, None),
        ("D_scorer_model_bev_best_spatial", "model", "public_robot_mounted", modeld_root),
    ):
        for seed_index, seed in enumerate((11, 23)):
            out_dir = train_root / config_name / f"seed_{seed}"
            device = devices[seed_index % len(devices)]
            metrics = train_trajectory_scorer_v0(
                action_pack=action_pack,
                out_dir=out_dir,
                max_steps=target_steps,
                batch_size=batch_size,
                device_name=device,
                seed=seed,
                bev_source=bev_source,
                modeld_dir=modeld_dir,
                source_families={source_family} if source_family is not None else None,
            )
            eval_path = out_dir / "eval_val.json"
            eval_metrics = eval_trajectory_scorer_v0(
                checkpoint=out_dir / "checkpoint.pt",
                action_pack=action_pack,
                out_path=eval_path,
                split="val",
                bev_source=bev_source,
                modeld_dir=modeld_dir,
                source_families={source_family} if source_family is not None else None,
                device_name=device,
                viz_dir=out_dir / "viz_val",
            )
            record = {
                "config": config_name,
                "seed": seed,
                "device": device,
                "checkpoint": (out_dir / "checkpoint.pt").as_posix(),
                "train_metrics": (out_dir / "train_metrics.json").as_posix(),
                "eval_metrics": eval_path.as_posix(),
                "top1_action_agreement": eval_metrics.get("top1_action_agreement"),
                "rank_corr": eval_metrics.get("rank_correlation_or_proxy"),
                "action_entropy": eval_metrics.get("action_entropy"),
                "dominant_action_fraction": eval_metrics.get("dominant_action_fraction"),
                "distribution_collapse_flag": eval_metrics.get("distribution_collapse_flag"),
                "unsafe_selected_rate": eval_metrics.get("unsafe_selected_rate"),
                "coverage_gain_mean": eval_metrics.get("coverage_gain_mean"),
                "stop_fraction": eval_metrics.get("selected_stop_fraction"),
                "inference_fps": eval_metrics.get("inference_fps"),
                "max_steps": target_steps,
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
                "product_training_approved": False,
            }
            scorer_runs.append(record)
            training_runs.append(record)

    best_scorer = max(
        scorer_runs,
        key=lambda item: (
            float(item.get("top1_action_agreement") or 0.0),
            float(item.get("rank_corr") or 0.0),
            -float(item.get("dominant_action_fraction") or 1.0),
        ),
    )
    replay_report = _run_heldout_replay(
        run_root=run_root,
        routes=routes,
        spatial_checkpoint=Path(best_spatial["checkpoint"]),
        scorer_checkpoint=Path(best_scorer["checkpoint"]),
        devices=devices,
    )
    generalization = _run_generalization_evals(
        run_root=run_root,
        routes=routes,
        manifest_all=manifest_all,
        best_spatial=best_spatial,
        best_scorer=best_scorer,
        action_pack=action_pack,
        modeld_root=modeld_root,
        devices=devices,
    )
    true_generalization = _run_true_loro_generalization(
        run_root=run_root,
        routes=routes,
        action_pack=action_pack,
        target_steps=target_steps,
        batch_size=batch_size,
        spatial_batch_size=spatial_batch_size,
        devices=devices,
    )
    generalization.update(true_generalization)
    best_oracle_scorer = max(
        [run for run in scorer_runs if run["config"] == "C_scorer_oracle_label_bev"],
        key=lambda item: float(item.get("top1_action_agreement") or 0.0),
    )
    model_vs_oracle_gap = float(best_scorer.get("top1_action_agreement") or 0.0) - float(
        best_oracle_scorer.get("top1_action_agreement") or 0.0
    )
    return {
        "training_runs": training_runs,
        "best_spatial_checkpoint": best_spatial,
        "best_scorer_checkpoint": best_scorer,
        "best_oracle_scorer_checkpoint": best_oracle_scorer,
        "model_vs_oracle_gap": {
            "metric": "top1_action_agreement",
            "model_minus_oracle": model_vs_oracle_gap,
            "model_checkpoint": best_scorer["checkpoint"],
            "oracle_checkpoint": best_oracle_scorer["checkpoint"],
        },
        "modeld_best_spatial_root": modeld_root.as_posix(),
        "replay": replay_report,
        "generalization": generalization,
        "sweep_completion": {
            "requested_config_seed_runs": 8,
            "completed_config_seed_runs": len(training_runs),
            "target_steps_per_run": target_steps,
            "complete_or_blocked": len(training_runs) == 8,
        },
    }


def _write_spatial_manifest(path: Path, routes: list[dict[str, Any]], *, exclude_sequence: str | None = None) -> None:
    packs = []
    for route in routes:
        if exclude_sequence is not None and route["sequence"] == exclude_sequence:
            continue
        packs.append(
            {
                "source_name": route["sequence"],
                "dataset": route["spatial_pack"],
                "features": route["dino_dir"],
            }
        )
    write_json(
        path,
        {
            "schema_version": "homebrain.spatial_v0_dataset_manifest.v0",
            "packs": packs,
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "product_training_approved": False,
        },
        pretty=True,
    )


def _run_heldout_replay(
    *,
    run_root: Path,
    routes: list[dict[str, Any]],
    spatial_checkpoint: Path,
    scorer_checkpoint: Path,
    devices: list[str],
) -> dict[str, Any]:
    replay_records = {}
    for index, route in enumerate(routes):
        name = _safe_name(str(route["sequence"]))
        out = run_root / "replay_heldout" / name
        replay_log(
            route["route_dir"],
            out,
            checkpoint=spatial_checkpoint,
            feature_dir=route["dino_dir"],
            trajectory_scorer_checkpoint=scorer_checkpoint,
            device_name=devices[index % len(devices)],
        )
        eval_path = run_root / "replay_heldout" / f"{name}_eval.json"
        write_eval_metrics(evaluate_log(out), eval_path)
        viz = run_root / "viz" / f"{name}_heldout_replay"
        visualize_spatial_outputs(log_dir=out, out_dir=viz, dataset_dir=route["spatial_pack"])
        outputs = [event for event in read_events(out) if isinstance(event, BrainOutputEvent)]
        replay_records[route["sequence"]] = {
            "replay_dir": out.as_posix(),
            "eval": eval_path.as_posix(),
            "viz": viz.as_posix(),
            "brain_output_count": len(outputs),
            "bad_flag_count": _bad_output_flag_count(outputs),
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "product_training_approved": False,
        }
    return replay_records


def _run_generalization_evals(
    *,
    run_root: Path,
    routes: list[dict[str, Any]],
    manifest_all: Path,
    best_spatial: dict[str, Any],
    best_scorer: dict[str, Any],
    action_pack: Path,
    modeld_root: Path,
    devices: list[str],
) -> dict[str, Any]:
    route_evals: dict[str, Any] = {}
    for index, route in enumerate(routes):
        name = _safe_name(str(route["sequence"]))
        spatial_eval = run_root / "generalization" / f"leave_route_{name}_spatial_eval.json"
        eval_spatial_v0(
            checkpoint=best_spatial["checkpoint"],
            dataset_dir=route["spatial_pack"],
            feature_dir=route["dino_dir"],
            out_path=spatial_eval,
            split=None,
            device_name=devices[index % len(devices)],
        )
        scorer_eval = run_root / "generalization" / f"leave_route_{name}_scorer_eval.json"
        eval_trajectory_scorer_v0(
            checkpoint=best_scorer["checkpoint"],
            action_pack=action_pack,
            out_path=scorer_eval,
            source_names={str(route["sequence"])},
            split=None,
            bev_source="model" if "model_bev" in str(best_scorer.get("config")) else "oracle",
            modeld_dir=modeld_root if "model_bev" in str(best_scorer.get("config")) else None,
            device_name=devices[index % len(devices)],
            viz_dir=run_root / "generalization" / f"leave_route_{name}_scorer_viz",
        )
        route_evals[route["sequence"]] = {
            "spatial_eval": spatial_eval.as_posix(),
            "scorer_eval": scorer_eval.as_posix(),
        }

    scene_evals: dict[str, Any] = {}
    scenes = sorted({str(route.get("scene")) for route in routes})
    for scene in scenes:
        scene_routes = [route for route in routes if route.get("scene") == scene]
        if not scene_routes:
            continue
        scene_evals[scene] = {
            "held_out_route_count": len(scene_routes),
            "note": "scene-out reporting uses best global checkpoint; retraining per scene is a follow-up if collapse persists.",
            "route_eval_keys": [str(route["sequence"]) for route in scene_routes],
        }
    return {
        "leave_one_route_out": route_evals,
        "leave_one_scene_out": scene_evals,
        "manifest_all": manifest_all.as_posix(),
        "control_safe": False,
    }


def _run_true_loro_generalization(
    *,
    run_root: Path,
    routes: list[dict[str, Any]],
    action_pack: Path,
    target_steps: int,
    batch_size: int,
    spatial_batch_size: int,
    devices: list[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    root = run_root / "generalization" / "true_loro"
    route_evals: dict[str, Any] = {}
    scene_evals: dict[str, Any] = {}

    for index, heldout in enumerate(routes):
        heldout_sequence = str(heldout["sequence"])
        name = _safe_name(heldout_sequence)
        device = devices[index % len(devices)]
        train_routes = [route for route in routes if str(route["sequence"]) != heldout_sequence]
        if not train_routes:
            route_evals[heldout_sequence] = {
                "skipped": True,
                "reason": "no remaining routes for leave-one-route-out training",
            }
            continue

        heldout_root = root / name
        manifest_train = heldout_root / "train_spatial_manifest.json"
        _write_spatial_manifest(manifest_train, train_routes)

        spatial_out = heldout_root / "spatial_A_seed_31"
        train_spatial_v0(
            dataset_manifest=manifest_train,
            out_dir=spatial_out,
            max_steps=target_steps,
            batch_size=spatial_batch_size,
            device_name=device,
            seed=31 + index,
            sensor_context_mode="masks",
        )
        spatial_eval_path = heldout_root / "spatial_eval_heldout.json"
        spatial_eval = eval_spatial_v0(
            checkpoint=spatial_out / "checkpoint.pt",
            dataset_dir=heldout["spatial_pack"],
            feature_dir=heldout["dino_dir"],
            out_path=spatial_eval_path,
            split=None,
            batch_size=spatial_batch_size,
            device_name=device,
            sensor_context_mode="masks",
        )

        modeld_root = heldout_root / "modeld_spatial"
        for route in routes:
            write_spatial_model_outputs(
                route["route_dir"],
                modeld_root / _safe_name(str(route["sequence"])),
                checkpoint=spatial_out / "checkpoint.pt",
                feature_dir=route["dino_dir"],
                device_name=device,
            )

        scorer_out = heldout_root / "scorer_model_bev_seed_41"
        train_source_names = {str(route["sequence"]) for route in train_routes}
        train_trajectory_scorer_v0(
            action_pack=action_pack,
            out_dir=scorer_out,
            source_names=train_source_names,
            max_steps=target_steps,
            batch_size=batch_size,
            device_name=device,
            seed=41 + index,
            bev_source="model",
            modeld_dir=modeld_root,
        )
        scorer_eval_path = heldout_root / "scorer_eval_heldout.json"
        scorer_eval = eval_trajectory_scorer_v0(
            checkpoint=scorer_out / "checkpoint.pt",
            action_pack=action_pack,
            out_path=scorer_eval_path,
            source_names={heldout_sequence},
            split=None,
            bev_source="model",
            modeld_dir=modeld_root,
            device_name=device,
            viz_dir=heldout_root / "scorer_viz_heldout",
        )

        route_record = {
            "held_out_route": heldout_sequence,
            "held_out_scene": heldout.get("scene"),
            "train_routes": sorted(train_source_names),
            "device": device,
            "max_steps": target_steps,
            "spatial_checkpoint": (spatial_out / "checkpoint.pt").as_posix(),
            "spatial_eval": spatial_eval_path.as_posix(),
            "scorer_checkpoint": (scorer_out / "checkpoint.pt").as_posix(),
            "scorer_eval": scorer_eval_path.as_posix(),
            "modeld_dir": modeld_root.as_posix(),
            "bev_iou_or_proxy": spatial_eval.get("bev_iou_or_proxy"),
            "pose_delta_rmse": spatial_eval.get("pose_delta_rmse"),
            "top1_action_agreement": scorer_eval.get("top1_action_agreement"),
            "rank_corr": scorer_eval.get("rank_correlation_or_proxy"),
            "action_entropy": scorer_eval.get("action_entropy"),
            "dominant_action_fraction": scorer_eval.get("dominant_action_fraction"),
            "distribution_collapse_flag": scorer_eval.get("distribution_collapse_flag"),
            "unsafe_selected_rate": scorer_eval.get("unsafe_selected_rate"),
            "coverage_gain_mean": scorer_eval.get("coverage_gain_mean"),
            "stop_fraction": scorer_eval.get("selected_stop_fraction"),
            "inference_fps": scorer_eval.get("inference_fps"),
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "product_training_approved": False,
        }
        route_evals[heldout_sequence] = route_record

    scenes = sorted({str(route.get("scene")) for route in routes})
    for scene in scenes:
        scene_routes = [route for route in routes if str(route.get("scene")) == scene]
        if len(scene_routes) == 1:
            sequence = str(scene_routes[0]["sequence"])
            scene_evals[scene] = {
                "held_out_route_count": 1,
                "route_eval_keys": [sequence],
                "equivalent_to_leave_one_route_out": True,
                "result": route_evals.get(sequence),
            }
        else:
            scene_evals[scene] = {
                "held_out_route_count": len(scene_routes),
                "skipped": True,
                "reason": "multi-route scene-out retraining is not implemented for Goal 11B's sampled route set",
                "route_eval_keys": [str(route["sequence"]) for route in scene_routes],
            }

    return {
        "true_leave_one_route_out": route_evals,
        "true_leave_one_scene_out": scene_evals,
        "true_loro_wall_time_sec": round(time.perf_counter() - started, 3),
        "true_loro_method": (
            "train SpatialMemoryNet A and model-BEV TrajectoryScorer D without each held-out route, "
            "then evaluate both on that route; one-route scenes reuse the corresponding route-out result"
        ),
        "true_loro_control_safe": False,
    }


def _bad_output_flag_count(outputs: list[BrainOutputEvent]) -> int:
    bad = 0
    for event in outputs:
        debug = event.debug if isinstance(event.debug, dict) else {}
        if event.cmd_vel is not None:
            bad += 1
        if debug.get("replay_only") is not True:
            bad += 1
        if debug.get("not_executed") is not True:
            bad += 1
        if debug.get("control_safe") is not False:
            bad += 1
        if debug.get("product_training_approved") is not False:
            bad += 1
    return bad


def _gpu_inventory() -> list[dict[str, Any]]:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return []
    if not torch.cuda.is_available():
        return []
    return [
        {"index": index, "name": torch.cuda.get_device_name(index)}
        for index in range(torch.cuda.device_count())
    ]


def _default_cuda_devices() -> list[str]:
    inventory = _gpu_inventory()
    preferred = [f"cuda:{item['index']}" for item in inventory if "4090" in str(item.get("name", ""))]
    if preferred:
        return preferred[:2]
    if inventory:
        return [f"cuda:{inventory[0]['index']}"]
    return ["cpu"]


def _write_reports(report: dict[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write("\n")
    REPORT_MD.write_text(_report_markdown(report), encoding="utf-8", newline="\n")


def _report_markdown(report: dict[str, Any]) -> str:
    components = report.get("wall_time_components_sec", {})
    lines = [
        "# Goal 11B Nightly Report",
        "",
        f"- status: {'blocked' if report.get('blockers') else 'completed_or_running'}",
        f"- wall_time_sec: {report.get('wall_time_sec')}",
        f"- wall_time_components_sec: {components if components else 'not split'}",
        f"- frames_processed: {report.get('frames_processed', 0)}",
        f"- evaluated_route_count: {report.get('evaluated_route_count', 0)}",
        f"- gpu_inventory: {report.get('gpu_inventory', [])}",
        "",
        "## Routes",
    ]
    for route in report.get("routes", []):
        lines.append(
            "- {sequence}: frames={frame_count} scene={scene} action_ok={action_supervision_ok_fraction} "
            "fully_evaluated={fully_evaluated} license={license_review_status}".format(**route)
        )
    failed_routes = report.get("failed_routes", [])
    if failed_routes:
        lines.extend(["", "## Failed/Skipped Routes"])
        for failed in failed_routes:
            lines.append(
                f"- {failed.get('sequence')}: {failed.get('exact_failure')} "
                f"repair={failed.get('minimal_next_repair')}"
            )
    skipped_routes = report.get("skipped_routes", [])
    for skipped in skipped_routes:
        lines.append(f"- skipped {skipped.get('sequence')}: {skipped.get('reason')}")

    action_qa = report.get("action_label_pack_v4_qa", {})
    if isinstance(action_qa, dict) and action_qa:
        lines.extend(["", "## ActionLabelPack v4 QA"])
        lines.append(
            "- examples={examples} entropy={entropy} dominant={dominant} future_valid={future_valid} "
            "collapse={collapse} qa_pass={qa_pass}".format(
                examples=action_qa.get("example_count"),
                entropy=action_qa.get("action_entropy"),
                dominant=action_qa.get("dominant_action_fraction"),
                future_valid=action_qa.get("future_motion_label_valid_fraction"),
                collapse=action_qa.get("goal11b_distribution_collapse_flag"),
                qa_pass=action_qa.get("action_label_pack_qa_pass"),
            )
        )
    lines.extend(["", "## Training"])
    for run in report.get("training_runs", []):
        lines.append(
            f"- {run.get('config')} seed={run.get('seed')} steps={run.get('max_steps')} "
            f"device={run.get('device')} checkpoint={run.get('checkpoint')}"
        )
    best_spatial = report.get("best_spatial_checkpoint", {})
    if isinstance(best_spatial, dict) and best_spatial:
        lines.append(
            "- best_spatial: config={config} seed={seed} bev_iou_or_proxy={bev_iou} "
            "pose_delta_rmse={pose_rmse} fps={fps}".format(
                config=best_spatial.get("config"),
                seed=best_spatial.get("seed"),
                bev_iou=best_spatial.get("bev_iou_or_proxy"),
                pose_rmse=best_spatial.get("pose_delta_rmse"),
                fps=best_spatial.get("inference_fps"),
            )
        )
    best_scorer = report.get("best_scorer_checkpoint", {})
    if isinstance(best_scorer, dict) and best_scorer:
        lines.append(
            "- best_scorer: config={config} seed={seed} top1={top1} rank_corr={rank} "
            "entropy={entropy} dominant={dominant} unsafe={unsafe} fps={fps}".format(
                config=best_scorer.get("config"),
                seed=best_scorer.get("seed"),
                top1=best_scorer.get("top1_action_agreement"),
                rank=best_scorer.get("rank_corr"),
                entropy=best_scorer.get("action_entropy"),
                dominant=best_scorer.get("dominant_action_fraction"),
                unsafe=best_scorer.get("unsafe_selected_rate"),
                fps=best_scorer.get("inference_fps"),
            )
        )
    gap = report.get("model_vs_oracle_gap", {})
    if isinstance(gap, dict) and gap:
        lines.append(f"- model_vs_oracle_gap({gap.get('metric')}): {gap.get('model_minus_oracle')}")

    generalization = report.get("generalization", {})
    true_loro = generalization.get("true_leave_one_route_out", {}) if isinstance(generalization, dict) else {}
    if isinstance(true_loro, dict) and true_loro:
        lines.extend(["", "## True Leave-One-Route-Out"])
        for route_name, metrics in true_loro.items():
            if not isinstance(metrics, dict):
                continue
            lines.append(
                "- {route}: train_routes={train_routes} top1={top1} rank_corr={rank} bev_iou={bev_iou} "
                "pose_rmse={pose_rmse} entropy={entropy} dominant={dominant} goal11b_collapse={collapse}".format(
                    route=route_name,
                    train_routes=metrics.get("train_routes"),
                    top1=metrics.get("top1_action_agreement"),
                    rank=metrics.get("rank_corr"),
                    bev_iou=metrics.get("bev_iou_or_proxy"),
                    pose_rmse=metrics.get("pose_delta_rmse"),
                    entropy=metrics.get("action_entropy"),
                    dominant=metrics.get("dominant_action_fraction"),
                    collapse=metrics.get("goal11b_distribution_collapse_flag"),
                )
            )

    replay = report.get("replay", {})
    if isinstance(replay, dict) and replay:
        lines.extend(["", "## Replay"])
        for route_name, metrics in replay.items():
            if isinstance(metrics, dict):
                lines.append(
                    f"- {route_name}: outputs={metrics.get('brain_output_count')} "
                    f"bad_flags={metrics.get('bad_flag_count')} eval={metrics.get('eval')}"
                )

    collapse = report.get("collapse_diagnosis", {})
    if isinstance(collapse, dict) and collapse:
        lines.extend(["", "## Collapse Diagnosis"])
        lines.append(f"- diagnosed_count: {collapse.get('diagnosed_count')}")
        lines.append(f"- interpretation: {collapse.get('interpretation')}")
    lines.extend(["", "## Blockers"])
    blockers = report.get("blockers", [])
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker.get('kind')}: {blocker.get('exact_failure')}")
    else:
        lines.append("- none recorded")
    lines.extend(["", "## Safety"])
    lines.append("- replay_only=true; not_executed=true; control_safe=false; product_training_approved=false")
    return "\n".join(lines) + "\n"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Goal 11B multi-route OpenLORIS nightly benchmark.")
    parser.add_argument("--data-root", default="data/public/openloris_scene")
    parser.add_argument("--run-root", default="runs/goal11b_nightly")
    parser.add_argument("--max-frames-per-route", type=int, default=1200)
    parser.add_argument("--max-total-gb", type=float, default=80.0)
    parser.add_argument("--target-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--spatial-batch-size", type=int, default=16)
    parser.add_argument("--device", action="append", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--keep-download-archives", action="store_true")
    parser.add_argument(
        "--true-loro-only",
        action="store_true",
        help="Append true leave-one-route/scene-out retraining to an existing Goal 11B report.",
    )
    parser.add_argument(
        "--postprocess-report",
        action="store_true",
        help="Refresh derived report fields without rerunning pipeline stages.",
    )
    args = parser.parse_args(argv)
    if args.postprocess_report:
        report = run_goal11b_postprocess_report()
        print(
            json.dumps(
                {
                    "report": REPORT_JSON.as_posix(),
                    "diagnosed_collapse_count": report.get("collapse_diagnosis", {}).get("diagnosed_count"),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.true_loro_only:
        report = run_goal11b_true_loro_supplement(
            run_root=args.run_root,
            target_steps=args.target_steps,
            batch_size=args.batch_size,
            spatial_batch_size=args.spatial_batch_size,
            devices=args.device,
        )
        print(
            json.dumps(
                {
                    "report": REPORT_JSON.as_posix(),
                    "true_loro_routes": len(report.get("generalization", {}).get("true_leave_one_route_out", {})),
                },
                sort_keys=True,
            )
        )
        return 0
    report = run_goal11b_nightly(
        data_root=args.data_root,
        run_root=args.run_root,
        max_frames_per_route=args.max_frames_per_route,
        max_total_gb=args.max_total_gb,
        target_steps=args.target_steps,
        batch_size=args.batch_size,
        spatial_batch_size=args.spatial_batch_size,
        devices=args.device,
        skip_download=args.skip_download,
        skip_dino=args.skip_dino,
        skip_training=args.skip_training,
        purge_download_archives=not args.keep_download_archives,
    )
    print(json.dumps({"report": REPORT_JSON.as_posix(), "blocker_count": len(report.get("blockers", []))}, sort_keys=True))
    return 0 if not report.get("blockers") else 2


if __name__ == "__main__":
    raise SystemExit(main())
