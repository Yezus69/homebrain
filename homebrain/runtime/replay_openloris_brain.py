from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from homebrain.brain.modeld import write_spatial_model_outputs
from homebrain.data.spatial_dataset import read_json
from homebrain.eval.closed_loop_replay_report import build_closed_loop_replay_report, write_report


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
    v1_pose_warp_source: str = "route_pose",
    v1_policy_bev_source: str = "memory",
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
    write_report(
        report,
        out_json=output / "runtime_report.json",
        out_md=output / "runtime_report.md",
    )
    return report


def _route_metadata(route: Path) -> dict[str, Any]:
    path = route / "route_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"missing route metadata: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"route metadata must be a JSON object: {path}")
    return data


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
    parser.add_argument("--v1-pose-warp-source", choices=("route_pose", "predicted_pose", "none"), default="route_pose")
    parser.add_argument("--v1-policy-bev-source", choices=("current", "memory"), default="memory")
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
