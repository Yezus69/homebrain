from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

from homebrain.brain.spatial_memory_v1 import load_checkpoint
from homebrain.data.spatial_dataset import read_json, write_json
from homebrain.datasets.usage_policy import openloris_usage_policy
from homebrain.train.eval_spatial_v1 import eval_spatial_v1
from homebrain.train.spatial_dataset import (
    SpatialPackSpec,
    dataset_manifest_hashes,
    load_spatial_dataset_manifest,
)
from homebrain.train.spatial_temporal_dataset import SpatialTemporalMultiPackDataset, SpatialTemporalTrainDataset
from homebrain.train.spatial_v1_hard_eval import evaluate_deployment_memory, evaluate_spatial_v1_hard
from homebrain.train.train_spatial_v1 import train_spatial_v1


GOAL12C_REPORT_SCHEMA_VERSION = "homebrain.goal12c_spatial_memory_v1_hard_validation_report.v0"
DEFAULT_GOAL12B_MANIFEST = "runs/goal12b_spatial_memory_v1/spatial_dataset_manifest_goal12b.json"
DEFAULT_V1_WINDOW1 = "runs/goal12b_spatial_memory_v1/v1_window1_memory_disabled_warm_start/checkpoint.pt"
DEFAULT_V1_WINDOW4 = "runs/goal12b_spatial_memory_v1/v1_window4_route_pose_warm_start/checkpoint.pt"
DEFAULT_V0_BASELINE = "runs/goal11b_nightly/training/A_spatial_dino_bev_pose/seed_17/checkpoint.pt"
DEFAULT_TRUE_LORO_ROOT = "runs/goal11b_nightly/generalization/true_loro"


def build_goal12c_report(
    *,
    hard_metrics: dict[str, Any],
    true_route_out: dict[str, Any],
    dataset_manifest: str | Path,
    v1_window1_checkpoint: str | Path,
    v1_window4_checkpoint: str | Path,
    v0_checkpoint: str | Path,
    commands_run: list[str] | None = None,
    artifacts_created: dict[str, str] | None = None,
) -> dict[str, Any]:
    policy = openloris_usage_policy()
    deployment = hard_metrics["deployment_style"]
    future = hard_metrics["future_observation"]
    occlusion = hard_metrics["occlusion_stress"]
    pose = hard_metrics["pose_ablation"]
    true_route_out_pass = bool(true_route_out.get("available")) and not bool(true_route_out.get("blocked"))
    pass_fail_gates = {
        "openloris_poc_policy_pass": bool(
            policy["poc_training_eval_allowed"]
            and not policy["product_training_approved"]
            and not policy["runtime_dependency"]
            and not policy["derived_dataset_redistribution_allowed"]
            and policy["attribution_required"]
        ),
        "deployment_memory_benefit_pass": bool(deployment["deployment_memory_benefit_pass"]),
        "hidden_memory_benefit_pass": bool(future["hidden_memory_benefit_pass"]),
        "occlusion_recovery_pass": bool(occlusion["occlusion_recovery_pass"]),
        "pose_ablation_pass": bool(pose["pose_ablation_pass"]),
        "true_leave_one_route_out_pass": true_route_out_pass,
    }
    pass_fail_gates["goal12c_hard_validation_pass"] = bool(all(pass_fail_gates.values()))
    blockers = _blockers(pass_fail_gates, true_route_out=true_route_out)
    report = {
        "schema_version": GOAL12C_REPORT_SCHEMA_VERSION,
        "goal": "12C OpenLORIS PoC policy + SpatialMemoryV1 hard validation",
        "dataset_usage_policy_status": {
            "OpenLORIS-Scene": policy,
            "generated_openloris_derived_artifacts": {
                "write_under_runs_or_data_public": True,
                "committed_or_distributed": False,
                "derived_dataset_redistribution_allowed": False,
            },
        },
        "openloris_poc_allowed_status": {
            "poc_training_eval_allowed": policy["poc_training_eval_allowed"],
            "product_training_approved": policy["product_training_approved"],
            "runtime_dependency": policy["runtime_dependency"],
            "derived_dataset_redistribution_allowed": policy["derived_dataset_redistribution_allowed"],
            "attribution_required": policy["attribution_required"],
            "license_name": policy["license_name"],
            "license_review_status": policy["license_review_status"],
        },
        "checkpoints": {
            "v0_baseline": Path(v0_checkpoint).as_posix(),
            "v1_window1": Path(v1_window1_checkpoint).as_posix(),
            "v1_window4": Path(v1_window4_checkpoint).as_posix(),
        },
        "dataset_manifest": Path(dataset_manifest).as_posix(),
        "deployment_style_metrics": deployment,
        "future_observation_metrics": future,
        "occlusion_stress_metrics": occlusion,
        "pose_ablation": pose,
        "true_route_out_metrics": true_route_out,
        "pass_fail_gates": pass_fail_gates,
        "blockers": blockers,
        "commands_run": commands_run or [],
        "artifacts_created": artifacts_created or {},
        "safety_flags": {
            "control_safe": False,
            "replay_only": True,
            "not_executed": True,
            "product_training_approved": False,
            "runtime_dependency": False,
            "cmd_vel_emitted": False,
            "uses_label_update_masks": False,
            "labels_used_only_after_prediction": True,
        },
        "next_goal": _next_goal(pass_fail_gates),
    }
    return report


def run_goal12c_report(
    *,
    dataset_manifest: str | Path,
    v1_window1_checkpoint: str | Path,
    v1_window4_checkpoint: str | Path,
    v0_checkpoint: str | Path,
    out_json: str | Path,
    out_md: str | Path,
    device_name: str | None = None,
    batch_size: int = 32,
    future_horizon: int = 3,
    run_true_route_out: bool = False,
    true_route_out_root: str | Path = "runs/goal12c_true_route_out",
    true_route_out_steps: int = 25,
    true_loro_v0_root: str | Path = DEFAULT_TRUE_LORO_ROOT,
    command: str | None = None,
) -> dict[str, Any]:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    hard_metrics = _hard_eval_checkpoint(
        checkpoint=v1_window4_checkpoint,
        dataset_manifest=dataset_manifest,
        window_length=4,
        split="val",
        batch_size=batch_size,
        device=device,
        future_horizon=future_horizon,
    )
    true_route_out = (
        run_true_leave_one_route_out(
            dataset_manifest=dataset_manifest,
            out_root=true_route_out_root,
            v0_fallback_checkpoint=v0_checkpoint,
            true_loro_v0_root=true_loro_v0_root,
            device_name=device_name,
            batch_size=batch_size,
            window4_steps=true_route_out_steps,
        )
        if run_true_route_out
        else {
            "available": False,
            "blocked": True,
            "reason": "run_goal12c_report called without run_true_route_out=True",
            "true_leave_one_route_out": False,
        }
    )
    report = build_goal12c_report(
        hard_metrics=hard_metrics,
        true_route_out=true_route_out,
        dataset_manifest=dataset_manifest,
        v1_window1_checkpoint=v1_window1_checkpoint,
        v1_window4_checkpoint=v1_window4_checkpoint,
        v0_checkpoint=v0_checkpoint,
        commands_run=[command] if command else [],
        artifacts_created={
            "json_report": Path(out_json).as_posix(),
            "markdown_report": Path(out_md).as_posix(),
        },
    )
    write_json(out_json, report, pretty=True)
    _write_markdown(out_md, report)
    return report


def run_true_leave_one_route_out(
    *,
    dataset_manifest: str | Path,
    out_root: str | Path,
    v0_fallback_checkpoint: str | Path,
    true_loro_v0_root: str | Path,
    device_name: str | None,
    batch_size: int,
    window4_steps: int,
) -> dict[str, Any]:
    specs = load_spatial_dataset_manifest(dataset_manifest)
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    folds: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for heldout in specs:
        train_specs = [spec for spec in specs if spec.source_name != heldout.source_name]
        if not train_specs:
            errors.append({"heldout_route": heldout.source_name, "error": "no training routes remain"})
            continue
        fold_dir = root / _safe_name(heldout.source_name)
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_manifest = fold_dir / "train_manifest.json"
        eval_manifest = fold_dir / "heldout_manifest.json"
        _write_pack_manifest(train_manifest, train_specs)
        _write_pack_manifest(eval_manifest, [heldout])
        fold_v0 = _loro_v0_checkpoint(heldout.source_name, true_loro_v0_root) or Path(v0_fallback_checkpoint)
        v0_source = "goal11b_true_loro" if _loro_v0_checkpoint(heldout.source_name, true_loro_v0_root) else "fallback_full_data_v0"
        try:
            window1_dir = fold_dir / "v1_window1"
            window4_dir = fold_dir / "v1_window4"
            train_spatial_v1(
                dataset_manifest=train_manifest,
                out_dir=window1_dir,
                max_steps=0,
                window_length=1,
                batch_size=batch_size,
                device_name=device_name,
                warm_start_v0_checkpoint=fold_v0,
                freeze_current_bev=True,
                missing_pose_behavior="masked_update",
                seed=31,
                command="goal12c true route-out window1 warm-start",
            )
            train_spatial_v1(
                dataset_manifest=train_manifest,
                out_dir=window4_dir,
                max_steps=window4_steps,
                window_length=4,
                batch_size=batch_size,
                device_name=device_name,
                learning_rate=5e-4,
                warm_start_v0_checkpoint=fold_v0,
                freeze_current_bev=True,
                missing_pose_behavior="masked_update",
                seed=43,
                command="goal12c true route-out window4 train",
            )
            window1_eval_path = fold_dir / "v1_window1_eval.json"
            window4_eval_path = fold_dir / "v1_window4_eval.json"
            window1_eval = eval_spatial_v1(
                checkpoint=window1_dir / "checkpoint.pt",
                dataset_manifest=eval_manifest,
                out_path=window1_eval_path,
                split="val",
                batch_size=batch_size,
                window_length=1,
                device_name=device_name,
                baseline_checkpoint_v0=fold_v0,
                command="goal12c true route-out window1 eval",
            )
            window4_eval = eval_spatial_v1(
                checkpoint=window4_dir / "checkpoint.pt",
                dataset_manifest=eval_manifest,
                out_path=window4_eval_path,
                split="val",
                batch_size=batch_size,
                window_length=4,
                device_name=device_name,
                baseline_checkpoint_v0=fold_v0,
                command="goal12c true route-out window4 eval",
            )
            device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
            window1_deployment = _deployment_eval_checkpoint(
                checkpoint=window1_dir / "checkpoint.pt",
                dataset_manifest=eval_manifest,
                window_length=1,
                split="val",
                batch_size=batch_size,
                device=device,
            )
            window4_deployment = _deployment_eval_checkpoint(
                checkpoint=window4_dir / "checkpoint.pt",
                dataset_manifest=eval_manifest,
                window_length=4,
                split="val",
                batch_size=batch_size,
                device=device,
            )
            folds.append(
                {
                    "heldout_route": heldout.source_name,
                    "true_leave_one_route_out": True,
                    "train_routes": [spec.source_name for spec in train_specs],
                    "heldout_manifest": eval_manifest.as_posix(),
                    "train_manifest": train_manifest.as_posix(),
                    "v0_checkpoint": fold_v0.as_posix(),
                    "v0_checkpoint_source": v0_source,
                    "v0_current_iou_window1_eval": window1_eval["baseline_comparison"]["baseline_metrics"].get(
                        "current_bev_iou_or_proxy"
                    ),
                    "v0_current_iou_window4_eval": window4_eval["baseline_comparison"]["baseline_metrics"].get(
                        "current_bev_iou_or_proxy"
                    ),
                    "v1_window1_eval": _route_out_metric_subset(window1_eval),
                    "v1_window4_eval": _route_out_metric_subset(window4_eval),
                    "v1_window1_deployment": window1_deployment,
                    "v1_window4_deployment": window4_deployment,
                    "deployment_memory_delta_window4_vs_window1": float(
                        window4_deployment["deployment_memory_iou"] - window1_deployment["deployment_memory_iou"]
                    ),
                    "artifacts": {
                        "v1_window1_checkpoint": (window1_dir / "checkpoint.pt").as_posix(),
                        "v1_window4_checkpoint": (window4_dir / "checkpoint.pt").as_posix(),
                        "v1_window1_eval": window1_eval_path.as_posix(),
                        "v1_window4_eval": window4_eval_path.as_posix(),
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001 - route-out report must keep partial evidence.
            errors.append({"heldout_route": heldout.source_name, "error": str(exc)})
    expected = {"cafe", "office", "corridor"}
    scenes = {_scene_from_source(fold["heldout_route"]) for fold in folds}
    return {
        "schema_version": "homebrain.goal12c_true_leave_one_route_out.v0",
        "available": bool(folds),
        "blocked": bool(errors) or not expected.issubset(scenes),
        "true_leave_one_route_out": bool(folds),
        "method": "train_on_all_but_one_route_eval_heldout_route",
        "fold_count": len(folds),
        "expected_scene_families": sorted(expected),
        "observed_scene_families": sorted(scenes),
        "folds": folds,
        "errors": errors,
        "product_training_approved": False,
        "control_safe": False,
        "replay_only": True,
    }


def _hard_eval_checkpoint(
    *,
    checkpoint: str | Path,
    dataset_manifest: str | Path,
    window_length: int,
    split: str,
    batch_size: int,
    device: torch.device,
    future_horizon: int,
) -> dict[str, Any]:
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model.to(device)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    dataset = _build_temporal_dataset(
        dataset_manifest=dataset_manifest,
        split=split,
        window_length=window_length,
        sensor_context_mode=str(metadata.get("sensor_context_mode", "masks")),
        missing_pose_behavior=str(metadata.get("missing_pose_behavior", model.config.missing_pose_behavior)),
    )
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    metrics = evaluate_spatial_v1_hard(model, loader, device=device, future_horizon=future_horizon)
    metrics["checkpoint"] = Path(checkpoint).as_posix()
    metrics["dataset_manifest"] = Path(dataset_manifest).as_posix()
    metrics["window_length"] = window_length
    metrics["split"] = split
    metrics["data_manifest_hashes"] = dataset_manifest_hashes(
        dataset_manifest=dataset_manifest,
        pack_specs=load_spatial_dataset_manifest(dataset_manifest),
    )
    return metrics


def _deployment_eval_checkpoint(
    *,
    checkpoint: str | Path,
    dataset_manifest: str | Path,
    window_length: int,
    split: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model.to(device)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    dataset = _build_temporal_dataset(
        dataset_manifest=dataset_manifest,
        split=split,
        window_length=window_length,
        sensor_context_mode=str(metadata.get("sensor_context_mode", "masks")),
        missing_pose_behavior=str(metadata.get("missing_pose_behavior", model.config.missing_pose_behavior)),
    )
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    return evaluate_deployment_memory(model, loader, device=device, pose_mode="route_pose")


def _build_temporal_dataset(
    *,
    dataset_manifest: str | Path,
    split: str,
    window_length: int,
    sensor_context_mode: str,
    missing_pose_behavior: str,
) -> SpatialTemporalMultiPackDataset | SpatialTemporalTrainDataset:
    specs = load_spatial_dataset_manifest(dataset_manifest)
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


def _write_pack_manifest(path: Path, specs: list[SpatialPackSpec]) -> None:
    write_json(
        path,
        {
            "schema_version": "homebrain.spatial_v0_dataset_manifest.v0",
            "packs": [
                {
                    "source_name": spec.source_name,
                    "dataset": spec.dataset_dir.as_posix(),
                    "features": spec.feature_dir.as_posix(),
                }
                for spec in specs
            ],
            "control_safe": False,
            "replay_only": True,
            "not_executed": True,
            "product_training_approved": False,
        },
        pretty=True,
    )


def _loro_v0_checkpoint(source_name: str, root: str | Path) -> Path | None:
    candidate = Path(root) / _safe_name(source_name) / "spatial_A_seed_31" / "checkpoint.pt"
    return candidate if candidate.exists() else None


def _route_out_metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "checkpoint",
        "window_length",
        "window_count",
        "current_bev_iou_or_proxy",
        "fused_memory_bev_iou_or_proxy",
        "unknown_reduction_vs_current",
        "temporal_reprojection_consistency_iou",
        "update_mask_coverage_mean",
        "memory_overwrite_fraction",
        "valid_warp_fraction",
        "memory_benefit_pass",
        "failure_flags",
        "control_safe",
        "product_training_approved",
    ]
    subset = {key: metrics.get(key) for key in keys if key in metrics}
    subset["baseline_comparison"] = metrics.get("baseline_comparison")
    return subset


def _blockers(pass_fail_gates: dict[str, bool], *, true_route_out: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    labels = {
        "openloris_poc_policy_pass": "OpenLORIS usage policy flags are not PoC-only/product-blocking as required",
        "deployment_memory_benefit_pass": "deployment-style memory did not beat current-frame prediction",
        "hidden_memory_benefit_pass": "future-hidden-cell memory did not beat current-frame prediction",
        "occlusion_recovery_pass": "occlusion stress did not show memory recovery across all deterministic masks",
        "pose_ablation_pass": "route-pose memory was not best among pose ablations",
        "true_leave_one_route_out_pass": "true leave-one-route-out validation did not complete for cafe/office/corridor",
    }
    for key, message in labels.items():
        if not pass_fail_gates.get(key, False):
            blockers.append(message)
    for error in true_route_out.get("errors", []):
        if isinstance(error, dict):
            blockers.append(f"route-out {error.get('heldout_route')}: {error.get('error')}")
    return blockers or ["none"]


def _next_goal(pass_fail_gates: dict[str, bool]) -> str:
    if pass_fail_gates.get("goal12c_hard_validation_pass"):
        return "Extend SpatialMemoryV1 hard validation to more robot-frame routes and owned logs before any product-training claim."
    return "Repair the failing hard SpatialMemoryV1 gate before tuning memory fusion or making control-facing claims."


def _safe_name(value: str) -> str:
    return value.replace("-", "_").replace("/", "_").replace("\\", "_")


def _scene_from_source(value: str) -> str:
    lower = value.lower()
    for scene in ("cafe", "office", "corridor"):
        if lower.startswith(scene):
            return scene
    return lower.split("1", 1)[0].split("-", 1)[0]


def _write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    gates = report["pass_fail_gates"]
    deployment = report["deployment_style_metrics"]
    future = report["future_observation_metrics"]
    occlusion = report["occlusion_stress_metrics"]
    route_out = report["true_route_out_metrics"]
    lines = [
        "# Goal 12C SpatialMemoryV1 Hard Validation",
        "",
        "## Dataset Policy",
        f"- OpenLORIS PoC training/eval allowed: `{str(report['openloris_poc_allowed_status']['poc_training_eval_allowed']).lower()}`",
        f"- product_training_approved: `{str(report['openloris_poc_allowed_status']['product_training_approved']).lower()}`",
        f"- runtime_dependency: `{str(report['openloris_poc_allowed_status']['runtime_dependency']).lower()}`",
        f"- derived_dataset_redistribution_allowed: `{str(report['openloris_poc_allowed_status']['derived_dataset_redistribution_allowed']).lower()}`",
        f"- license_name: `{report['openloris_poc_allowed_status']['license_name']}`",
        "",
        "## Deployment Metrics",
        f"- deployment_current_iou: `{deployment['deployment_current_iou']}`",
        f"- deployment_memory_iou: `{deployment['deployment_memory_iou']}`",
        f"- deployment_memory_delta: `{deployment['deployment_memory_delta']}`",
        f"- deployment_update_mask_coverage: `{deployment['deployment_update_mask_coverage']}`",
        f"- deployment_overwrite_fraction: `{deployment['deployment_overwrite_fraction']}`",
        "",
        "## Future Hidden Cells",
        f"- hidden_cell_iou: `{future['hidden_cell_iou']}`",
        f"- hidden_free_iou: `{future['hidden_free_iou']}`",
        f"- hidden_obstacle_iou: `{future['hidden_obstacle_iou']}`",
        f"- future_reveal_count: `{future['future_reveal_count']}`",
        f"- hidden_memory_benefit_pass: `{str(future['hidden_memory_benefit_pass']).lower()}`",
        "",
        "## Occlusion Stress",
        f"- occluded_current_iou: `{occlusion['occluded_current_iou']}`",
        f"- occluded_memory_iou: `{occlusion['occluded_memory_iou']}`",
        f"- occlusion_recovery_delta: `{occlusion['occlusion_recovery_delta']}`",
        "",
        "## True Route Out",
        f"- available: `{str(route_out.get('available')).lower()}`",
        f"- true_leave_one_route_out: `{str(route_out.get('true_leave_one_route_out')).lower()}`",
        f"- fold_count: `{route_out.get('fold_count', 0)}`",
        "",
        "## Pose Ablation",
    ]
    for row in report["pose_ablation"]["table"]:
        lines.append(
            f"- `{row['pose_mode']}`: memory `{row['deployment_memory_iou']}`, "
            f"current `{row['deployment_current_iou']}`, delta `{row['deployment_memory_delta']}`"
        )
    lines.extend(
        [
            "",
            "## Gates",
        ]
    )
    for key, value in gates.items():
        lines.append(f"- {key}: `{str(value).lower()}`")
    lines.extend(
        [
            f"- blockers: `{', '.join(report['blockers'])}`",
            "",
            "## Safety",
            "- Replay/eval only. `control_safe=false`, `product_training_approved=false`, `runtime_dependency=false`, and no `cmd_vel` is emitted.",
            f"- next_goal: {report['next_goal']}",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Goal 12C SpatialMemoryV1 hard-validation report.")
    parser.add_argument("--dataset-manifest", default=DEFAULT_GOAL12B_MANIFEST)
    parser.add_argument("--v1-window1-checkpoint", default=DEFAULT_V1_WINDOW1)
    parser.add_argument("--v1-window4-checkpoint", default=DEFAULT_V1_WINDOW4)
    parser.add_argument("--v0-checkpoint", default=DEFAULT_V0_BASELINE)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--future-horizon", type=int, default=3)
    parser.add_argument("--run-true-route-out", action="store_true")
    parser.add_argument("--true-route-out-root", default="runs/goal12c_true_route_out")
    parser.add_argument("--true-route-out-steps", type=int, default=25)
    parser.add_argument("--true-loro-v0-root", default=DEFAULT_TRUE_LORO_ROOT)
    args = parser.parse_args(argv)
    command = "python -m homebrain.tools.goal12c_hard_validation " + " ".join(sys.argv[1:])
    report = run_goal12c_report(
        dataset_manifest=args.dataset_manifest,
        v1_window1_checkpoint=args.v1_window1_checkpoint,
        v1_window4_checkpoint=args.v1_window4_checkpoint,
        v0_checkpoint=args.v0_checkpoint,
        out_json=args.out_json,
        out_md=args.out_md,
        device_name=args.device,
        batch_size=args.batch_size,
        future_horizon=args.future_horizon,
        run_true_route_out=args.run_true_route_out,
        true_route_out_root=args.true_route_out_root,
        true_route_out_steps=args.true_route_out_steps,
        true_loro_v0_root=args.true_loro_v0_root,
        command=command,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
