from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from homebrain.data.spatial_dataset import read_json, write_json
from homebrain.viz.checkpoint_3d_scene_types_v0 import safety_flags


def eval_checkpoint_3d_visualizer_v0(*, viz_dir: str | Path, out: str | Path) -> Path:
    root = Path(viz_dir)
    summary = _read_optional(root / "summary.json")
    frames = _read_jsonl(root / "scene_timeline.jsonl")
    accumulated = _read_optional(root / "accumulated_scene.json")
    final_map = accumulated.get("final_map") if isinstance(accumulated.get("final_map"), dict) else {}
    artifact_paths = [path for path in root.rglob("*") if path.is_file()]
    html_exists = (root / "viewer.html").exists()
    scene_timeline_exists = (root / "scene_timeline.jsonl").exists()
    accumulated_map_exists = (root / "accumulated_scene.json").exists()
    ply_exists = (root / "accumulated_map.ply").exists() or (root / "camera_trajectory.ply").exists()
    frame_count = len(frames) if frames else int(summary.get("frame_count", 0) or 0)
    model_present, model_missing = _model_channels(frames, summary)
    warnings = sorted(set(_string_list(summary.get("warnings")) + [warning for frame in frames for warning in _string_list(frame.get("warnings"))]))
    hard_failures = sorted(set(_string_list(summary.get("hard_failures"))))
    safety = summary.get("safety") if isinstance(summary.get("safety"), dict) else safety_flags()
    report = {
        "schema_version": "homebrain.eval_checkpoint_3d_visualizer_v0.v0",
        "accepted_checkpoint_3d_visualizer_v0": False,
        "gates_improved": [],
        "artifact_count": len(artifact_paths),
        "html_exists": html_exists,
        "scene_timeline_exists": scene_timeline_exists,
        "accumulated_map_exists": accumulated_map_exists,
        "ply_exists": ply_exists,
        "frame_count": frame_count,
        "map_accumulates_over_time": bool(summary.get("map_accumulates_over_time") is True),
        "incremental_no_future_leakage_passed": bool(summary.get("incremental_no_future_leakage_passed") is True),
        "camera_pose_rendered": bool(summary.get("camera_pose_rendered") is True),
        "robot_pose_rendered": bool(summary.get("robot_pose_rendered") is True),
        "trajectory_rendered": bool(summary.get("trajectory_rendered") is True),
        "local_bev_rendered": any(isinstance(frame.get("local_bev"), dict) for frame in frames),
        "accumulated_map_rendered": _cell_total(final_map) > 0,
        "candidate_trajectories_rendered": bool(summary.get("candidate_trajectories_rendered") is True),
        "safety_debug_rendered": bool(summary.get("safety_debug_rendered") is True),
        "geometry_source": summary.get("geometry_source"),
        "pose_source": summary.get("pose_source"),
        "pose_source_is_oracle": bool(summary.get("pose_source_is_oracle") is True),
        "dense_3d_claimed": bool(summary.get("dense_3d_claimed") is True),
        "model_channels_present": model_present,
        "model_channels_missing": model_missing,
        "warnings": warnings,
        "hard_failures": hard_failures,
        "safety": safety,
    }
    checks = _acceptance_checks(report)
    accepted = all(checks.values())
    report["accepted_checkpoint_3d_visualizer_v0"] = bool(accepted)
    report["gates_improved"] = ["Gate A", "Gate D"] if accepted else []
    report["acceptance_checks"] = checks
    report["failed_checks"] = [key for key, value in checks.items() if not value]
    write_json(out, report, pretty=True)
    return Path(out)


def _acceptance_checks(report: dict[str, Any]) -> dict[str, bool]:
    camera_ok = bool(report["camera_pose_rendered"]) or "missing_camera_to_base_extrinsics" in report["warnings"]
    safety = report["safety"]
    return {
        "viewer_html_exists": bool(report["html_exists"]),
        "scene_timeline_exists": bool(report["scene_timeline_exists"]),
        "accumulated_scene_exists": bool(report["accumulated_map_exists"]),
        "ply_exists": bool(report["ply_exists"]),
        "frame_count_gt_1": int(report["frame_count"]) > 1,
        "map_accumulates_over_time": bool(report["map_accumulates_over_time"]),
        "incremental_no_future_leakage_passed": bool(report["incremental_no_future_leakage_passed"]),
        "robot_pose_rendered": bool(report["robot_pose_rendered"]),
        "camera_pose_or_missing_warning_rendered": camera_ok,
        "trajectory_rendered": bool(report["trajectory_rendered"]),
        "local_bev_or_depth_geometry_rendered": bool(report["local_bev_rendered"]) or report["geometry_source"] in {"depth_pointcloud", "rgbd_pointcloud"},
        "accumulated_map_rendered": bool(report["accumulated_map_rendered"]),
        "geometry_source_explicit": report["geometry_source"] in {"depth_pointcloud", "rgbd_pointcloud", "bev_extruded_2p5d", "fixture_synthetic"},
        "dense_3d_honest": report["dense_3d_claimed"] is False or report["geometry_source"] in {"depth_pointcloud", "rgbd_pointcloud"},
        "safety_flags_conservative": safety.get("replay_only") is True
        and safety.get("not_executed") is True
        and safety.get("control_safe") is False
        and safety.get("raw_pwm_emitted") is False
        and safety.get("hardware_validated") is False,
        "no_hard_failures": not report["hard_failures"],
    }


def _model_channels(frames: list[dict[str, Any]], summary: dict[str, Any]) -> tuple[list[str], list[str]]:
    present = set(str(item) for item in summary.get("model_channels_rendered", []) if isinstance(item, str))
    missing = set()
    for frame in frames:
        for item in frame.get("model_channel_summaries", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            if item.get("rendered") is True:
                present.add(name)
            elif name:
                missing.add(name)
    return sorted(present), sorted(missing - present)


def _read_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data = json.loads(line)
                if isinstance(data, dict):
                    out.append(data)
    return out


def _cell_total(final_map: dict[str, Any]) -> int:
    return sum(len(final_map.get(key, [])) for key in ("free_cells", "obstacle_cells", "unknown_cells", "risky_cells", "hazard_cells", "stale_cells", "point_cloud"))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint 3D visualizer artifacts.")
    parser.add_argument("--viz-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    eval_checkpoint_3d_visualizer_v0(viz_dir=args.viz_dir, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
