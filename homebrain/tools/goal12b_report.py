from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from homebrain.data.spatial_dataset import read_json, write_json


def build_goal12b_report(
    *,
    memory_disabled_metrics: dict[str, Any],
    memory_metrics: dict[str, Any],
    memory_window8_metrics: dict[str, Any] | None = None,
    v0_eval_metrics: dict[str, Any] | None = None,
    commands_run: list[str] | None = None,
    artifacts_created: dict[str, str] | None = None,
) -> dict[str, Any]:
    v0_baseline = _v0_baseline(memory_disabled_metrics, v0_eval_metrics)
    v1_disabled = _metric_subset(memory_disabled_metrics)
    v1_memory = _metric_subset(memory_metrics)
    current_parity = _current_parity(v1_disabled, v0_baseline)
    memory_delta = _memory_delta(v1_disabled, v1_memory)
    memory_benefit_pass = (
        bool(current_parity["current_bev_parity_pass"])
        and bool(memory_delta["memory_benefit_condition_pass"])
        and float(memory_delta["current_bev_iou_delta_vs_memory_disabled"]) >= -0.02
    )
    failure_flags = sorted(
        set(
            list(memory_disabled_metrics.get("failure_flags", []))
            + list(memory_metrics.get("failure_flags", []))
            + ([] if current_parity["current_bev_parity_pass"] else ["current_bev_parity_failed"])
            + ([] if memory_benefit_pass else ["memory_benefit_gate_failed"])
        )
    )
    blockers: list[str] = []
    if not current_parity["current_bev_parity_pass"]:
        blockers.append("current BEV parity failed; v1 cannot be promoted")
    if current_parity["current_bev_parity_pass"] and not memory_benefit_pass:
        blockers.append("memory benefit gate failed; keep v1 replay/eval only")
    if not blockers:
        blockers.append("none")

    report: dict[str, Any] = {
        "schema_version": "homebrain.goal12b_spatial_memory_v1_parity_report.v0",
        "goal": "12B SpatialMemoryNetV1 parity + memory-sanity repair",
        "v0_baseline_metrics": v0_baseline,
        "v1_memory_disabled_metrics": v1_disabled,
        "v1_memory_metrics": v1_memory,
        "v1_memory_window8_metrics": _metric_subset(memory_window8_metrics) if memory_window8_metrics is not None else None,
        **current_parity,
        "memory_benefit_pass": bool(memory_benefit_pass),
        "fused_memory_iou_delta": memory_delta["fused_memory_iou_delta"],
        "temporal_consistency_delta": memory_delta["temporal_consistency_delta"],
        "current_bev_iou_delta_vs_memory_disabled": memory_delta["current_bev_iou_delta_vs_memory_disabled"],
        "unknown_reduction_vs_current": v1_memory["unknown_reduction_vs_current"],
        "update_mask_coverage_mean": v1_memory["update_mask_coverage_mean"],
        "memory_overwrite_fraction": v1_memory["memory_overwrite_fraction"],
        "pose_warp_source": v1_memory["pose_warp_source"],
        "valid_warp_fraction": v1_memory["valid_warp_fraction"],
        "predicted_pose_warp_ablation": bool(v1_memory["predicted_pose_warp_ablation"]),
        "route_out_metrics": memory_metrics.get("route_out_metrics"),
        "per_source_metrics": memory_metrics.get("sources", []),
        "failure_flags": failure_flags,
        "blockers": blockers,
        "commands_run": commands_run or [],
        "artifacts_created": artifacts_created or {},
        "safety_flags": {
            "control_safe": False,
            "replay_only": True,
            "not_executed": True,
            "product_training_approved": False,
            "cmd_vel_emitted": False,
        },
        "next_goal": _next_goal(memory_benefit_pass),
    }
    return report


def write_goal12b_report(
    *,
    memory_disabled_eval: str | Path,
    memory_eval: str | Path,
    out_json: str | Path,
    out_md: str | Path,
    memory_window8_eval: str | Path | None = None,
    v0_eval: str | Path | None = None,
    commands_run: list[str] | None = None,
    artifacts_created: dict[str, str] | None = None,
) -> dict[str, Any]:
    memory_disabled_metrics = read_json(memory_disabled_eval)
    memory_metrics = read_json(memory_eval)
    memory_window8_metrics = read_json(memory_window8_eval) if memory_window8_eval is not None else None
    v0_eval_metrics = read_json(v0_eval) if v0_eval is not None else None
    report = build_goal12b_report(
        memory_disabled_metrics=memory_disabled_metrics,
        memory_metrics=memory_metrics,
        memory_window8_metrics=memory_window8_metrics,
        v0_eval_metrics=v0_eval_metrics,
        commands_run=commands_run,
        artifacts_created=artifacts_created,
    )
    write_json(out_json, report, pretty=True)
    _write_markdown(out_md, report)
    return report


def _v0_baseline(metrics: dict[str, Any], v0_eval_metrics: dict[str, Any] | None) -> dict[str, Any]:
    baseline = metrics.get("baseline_comparison", {}).get("baseline_metrics")
    if isinstance(baseline, dict):
        return dict(baseline)
    if v0_eval_metrics is not None:
        return {
            "baseline_name": "SpatialMemoryNetV0_current_frame_only",
            "checkpoint": v0_eval_metrics.get("checkpoint"),
            "current_bev_iou_or_proxy": v0_eval_metrics.get("bev_iou_or_proxy", 0.0),
            "fused_memory_bev_iou_or_proxy": v0_eval_metrics.get("bev_iou_or_proxy", 0.0),
            "temporal_reprojection_consistency_iou": 0.0,
            "unknown_reduction_vs_current": 0.0,
            "frame_count": v0_eval_metrics.get("example_count"),
        }
    return {
        "baseline_name": "missing_v0_baseline",
        "current_bev_iou_or_proxy": 0.0,
        "fused_memory_bev_iou_or_proxy": 0.0,
        "temporal_reprojection_consistency_iou": 0.0,
        "unknown_reduction_vs_current": 0.0,
    }


def _metric_subset(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if metrics is None:
        return {}
    keys = [
        "checkpoint",
        "dataset_manifest",
        "window_length",
        "device",
        "window_count",
        "current_bev_iou_or_proxy",
        "fused_memory_bev_iou_or_proxy",
        "temporal_reprojection_consistency_iou",
        "unknown_reduction_vs_current",
        "update_mask_coverage_mean",
        "memory_overwrite_fraction",
        "pose_warp_source",
        "valid_warp_fraction",
        "predicted_pose_warp_ablation",
        "memory_warp_valid_fraction",
        "memory_reset_fraction",
        "failure_flags",
        "control_safe",
        "replay_only",
        "not_executed",
        "product_training_approved",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def _current_parity(v1_disabled: dict[str, Any], v0_baseline: dict[str, Any]) -> dict[str, Any]:
    delta = float(v1_disabled.get("current_bev_iou_or_proxy", 0.0)) - float(
        v0_baseline.get("current_bev_iou_or_proxy", 0.0)
    )
    return {
        "current_bev_parity_pass": bool(delta >= -0.02),
        "current_bev_iou_delta": delta,
        "current_bev_parity_tolerance": 0.02,
    }


def _memory_delta(v1_disabled: dict[str, Any], v1_memory: dict[str, Any]) -> dict[str, Any]:
    fused_delta = float(v1_memory.get("fused_memory_bev_iou_or_proxy", 0.0)) - float(
        v1_disabled.get("fused_memory_bev_iou_or_proxy", 0.0)
    )
    temporal_delta = float(v1_memory.get("temporal_reprojection_consistency_iou", 0.0)) - float(
        v1_disabled.get("temporal_reprojection_consistency_iou", 0.0)
    )
    current_delta = float(v1_memory.get("current_bev_iou_or_proxy", 0.0)) - float(
        v1_disabled.get("current_bev_iou_or_proxy", 0.0)
    )
    return {
        "fused_memory_iou_delta": fused_delta,
        "temporal_consistency_delta": temporal_delta,
        "current_bev_iou_delta_vs_memory_disabled": current_delta,
        "memory_benefit_condition_pass": bool(fused_delta > 0.0 or temporal_delta > 0.0),
    }


def _next_goal(memory_benefit_pass: bool) -> str:
    if memory_benefit_pass:
        return "Run longer route-out v1 sweeps and tune fusion only after keeping current-BEV parity stable."
    return "Repair fused-memory training/eval before adding new v1 features; keep v1 replay/eval only."


def _write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Goal 12B SpatialMemoryNetV1 Parity Report",
        "",
        f"- current_bev_parity_pass: `{str(report['current_bev_parity_pass']).lower()}`",
        f"- memory_benefit_pass: `{str(report['memory_benefit_pass']).lower()}`",
        f"- current_bev_iou_delta: `{report['current_bev_iou_delta']}`",
        f"- fused_memory_iou_delta: `{report['fused_memory_iou_delta']}`",
        f"- temporal_consistency_delta: `{report['temporal_consistency_delta']}`",
        f"- unknown_reduction_vs_current: `{report['unknown_reduction_vs_current']}`",
        f"- update_mask_coverage_mean: `{report['update_mask_coverage_mean']}`",
        f"- memory_overwrite_fraction: `{report['memory_overwrite_fraction']}`",
        f"- pose_warp_source: `{report['pose_warp_source']}`",
        f"- valid_warp_fraction: `{report['valid_warp_fraction']}`",
        "",
        "## Metrics",
        f"- v0 current IoU: `{report['v0_baseline_metrics'].get('current_bev_iou_or_proxy')}`",
        f"- v1 memory-disabled current IoU: `{report['v1_memory_disabled_metrics'].get('current_bev_iou_or_proxy')}`",
        f"- v1 memory current IoU: `{report['v1_memory_metrics'].get('current_bev_iou_or_proxy')}`",
        f"- v1 memory fused IoU: `{report['v1_memory_metrics'].get('fused_memory_bev_iou_or_proxy')}`",
        "",
        "## Per Source",
    ]
    for item in report.get("per_source_metrics", []):
        lines.append(
            f"- `{item.get('source_name')}`: current `{item.get('current_bev_iou_or_proxy')}`, "
            f"fused `{item.get('fused_memory_bev_iou_or_proxy')}`, warp `{item.get('valid_warp_fraction')}`"
        )
    lines.extend(
        [
            "",
            "## Gates",
            f"- failure_flags: `{', '.join(report['failure_flags'])}`",
            f"- blockers: `{', '.join(report['blockers'])}`",
            "",
            "## Safety",
            "- Replay/eval only. `control_safe=false`, `product_training_approved=false`, and no `cmd_vel` is emitted.",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the Goal 12B SpatialMemoryNet v1 parity report.")
    parser.add_argument("--memory-disabled-eval", "--memory-disabled", dest="memory_disabled_eval", required=True)
    parser.add_argument("--memory-eval", "--memory", dest="memory_eval", required=True)
    parser.add_argument("--memory-window8-eval", "--memory-window8", dest="memory_window8_eval", default=None)
    parser.add_argument("--v0-eval", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args(argv)
    report = write_goal12b_report(
        memory_disabled_eval=args.memory_disabled_eval,
        memory_eval=args.memory_eval,
        memory_window8_eval=args.memory_window8_eval,
        v0_eval=args.v0_eval,
        out_json=args.out_json,
        out_md=args.out_md,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
