from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import shutil
from typing import Any

from homebrain.data.spatial_dataset import write_json
from homebrain.viz.accumulated_scene3d_v0 import AccumulatedScene3DV0
from homebrain.viz.checkpoint_3d_scene_types_v0 import (
    AccumulatedMapStateV0,
    RobotPoseVizV0,
    SceneTimelineFrameV0,
    safety_flags,
)
from homebrain.viz.checkpoint_3d_visualizer_fixtures_v0 import build_fixture_sequence
from homebrain.viz.checkpoint_output_adapter_v0 import Checkpoint3DOutputAdapterV0
from homebrain.viz.export_checkpoint_3d_artifacts_v0 import (
    camera_frustum_lines_from_frames,
    trajectory_lines,
    write_accumulated_scene_json,
    write_artifact_manifest,
    write_ply_lines,
    write_ply_point_cloud,
    write_scene_timeline_jsonl,
    write_viewer_html,
)
from homebrain.viz.checkpoint_3d_visualizer_summary_v0 import (
    empty_accumulated_map,
    map_accumulates,
    no_future_leakage,
    visualizer_acceptance_report,
)
from homebrain.viz.pose_camera_geometry_v0 import base_axes_lines, camera_frustum_from_intrinsics, compose_transforms, robot_footprint_polygon


def visualize_checkpoint_3d_v0(
    *,
    out: str | Path,
    fixture: str | None = None,
    checkpoint: str | Path | None = None,
    pack: str | Path | None = None,
    route_id: str | None = None,
    device: str | None = "cpu",
    max_frames: int | None = 300,
    stride: int = 1,
    pose_source: str = "online",
    write_html: bool = False,
    write_ply: bool = False,
    write_jsonl: bool = False,
    max_points: int = 50_000,
    max_cells: int = 20_000,
    review_assumed_extrinsics: bool = False,
    debug_oracle_pose: bool = False,
    debug_teacher_overlay: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _prepare_output_dirs(out_dir)
    if fixture:
        frames, settings = _fixture_frames(fixture)
    else:
        frames, settings = _real_frames(
            checkpoint=checkpoint,
            pack=pack,
            route_id=route_id,
            device=device,
            max_frames=max_frames,
            stride=stride,
            pose_source=pose_source,
            review_assumed_extrinsics=review_assumed_extrinsics,
        )
    if debug_teacher_overlay is not None:
        settings["warnings"].append("teacher_overlay_enabled_debug_only")
        settings["teacher_overlay_enabled"] = True
        settings["overlay_is_runtime_truth"] = False
        settings["force_unaccepted"] = True

    timeline, timeline_maps = _accumulate_frames(frames, settings=settings, max_points=max_points, max_cells=max_cells)
    final_map = timeline_maps[-1] if timeline_maps else empty_accumulated_map(settings)
    summary = _summary(
        timeline,
        timeline_maps,
        final_map,
        settings=settings,
        checkpoint=checkpoint,
        route_id=route_id or settings["route_id"],
        debug_oracle_pose=debug_oracle_pose,
    )
    accumulated_payload = {
        "schema_version": "homebrain.checkpoint_3d_accumulated_scene.v0",
        "summary": summary,
        "final_map": final_map.to_dict(),
        "timeline_maps": [state.to_dict() for state in timeline_maps],
    }
    artifacts = _write_artifacts(
        out_dir,
        frames=timeline,
        timeline_maps=timeline_maps,
        final_map=final_map,
        accumulated_payload=accumulated_payload,
        summary=summary,
        write_html=write_html,
        write_ply=write_ply,
        write_jsonl=write_jsonl,
    )
    artifacts = sorted(set([*artifacts, "summary.json", "visualizer_acceptance.json", "manifest.json", "frame_debug/"]))
    summary["artifacts"] = artifacts
    write_accumulated_scene_json(out_dir / "accumulated_scene.json", final_map=final_map, timeline_maps=timeline_maps, summary=summary)
    write_json(out_dir / "summary.json", summary, pretty=True)
    acceptance = visualizer_acceptance_report(summary)
    write_json(out_dir / "visualizer_acceptance.json", acceptance, pretty=True)
    write_artifact_manifest(out_dir / "manifest.json", root=out_dir, artifacts=[item for item in artifacts if item != "manifest.json" and not item.endswith("/")], summary=summary)
    return summary


def _prepare_output_dirs(out_dir: Path) -> None:
    for name in ("frame_debug", "rgb_frames"):
        path = out_dir / name
        if path.exists():
            shutil.rmtree(path)
    (out_dir / "frame_debug").mkdir(parents=True, exist_ok=True)


def _fixture_frames(name: str) -> tuple[list[SceneTimelineFrameV0], dict[str, Any]]:
    fixture = build_fixture_sequence(name)
    return fixture.frames, {
        "route_id": fixture.route_id,
        "resolution_m": fixture.resolution_m,
        "decay_sec": fixture.decay_sec,
        "geometry_source": fixture.geometry_source,
        "pose_source": fixture.pose_source,
        "pose_source_is_oracle": fixture.pose_source_is_oracle,
        "dense_3d_claimed": fixture.dense_3d_claimed,
        "force_unaccepted": fixture.force_unaccepted,
        "expected": fixture.expected,
        "warnings": [],
        "hard_failures": [],
        "teacher_overlay_enabled": False,
        "overlay_is_runtime_truth": False,
    }


def _real_frames(**kwargs: Any) -> tuple[list[SceneTimelineFrameV0], dict[str, Any]]:
    checkpoint = kwargs["checkpoint"]
    pack = kwargs["pack"]
    route_id = kwargs["route_id"]
    if checkpoint is None or pack is None or route_id is None:
        raise ValueError("real mode requires --checkpoint, --pack, and --route-id")
    adapter = Checkpoint3DOutputAdapterV0(device=kwargs["device"], review_assumed_extrinsics=kwargs["review_assumed_extrinsics"])
    frames = []
    warnings: list[str] = []
    depth_point_clouds: list[list[list[float]] | None] = []
    for output in adapter.iter_route_outputs(
        checkpoint=checkpoint,
        pack=pack,
        route_id=route_id,
        max_frames=kwargs["max_frames"],
        stride=kwargs["stride"],
        pose_source=kwargs["pose_source"],
    ):
        camera = None
        T_map_camera = None
        if output.T_base_camera is not None:
            T_map_camera = compose_transforms(output.T_map_base, output.T_base_camera)
            camera = camera_frustum_from_intrinsics(T_map_camera=T_map_camera, intrinsics=output.intrinsics, timestamp_ns=output.timestamp_ns)
        warnings.extend(output.warnings)
        depth_point_clouds.append(output.depth_points_camera)
        robot = RobotPoseVizV0(
            timestamp_ns=output.timestamp_ns,
            T_map_base=output.T_map_base,
            T_map_camera=T_map_camera,
            base_axes_lines_map=base_axes_lines(output.T_map_base),
            footprint_polygon_map=robot_footprint_polygon(output.T_map_base),
            pose_confidence=None,
            pose_source=output.T_map_base.source,
            pose_source_is_oracle=output.T_map_base.is_oracle,
        )
        frames.append(
            SceneTimelineFrameV0(
                route_id=str(route_id),
                frame_id=output.frame_id,
                timestamp_ns=output.timestamp_ns,
                robot_pose=robot,
                camera_frustum=camera,
                local_bev=output.local_bev,
                rgb_frame=output.rgb_frame,
                depth_point_cloud_camera=output.depth_points_camera or [],
                candidate_trajectories=output.candidate_trajectories,
                safety_debug=output.safety_debug,
                model_channel_summaries=output.model_channel_summaries,
                warnings=output.warnings,
            )
        )
    return frames, {
        "route_id": str(route_id),
        "resolution_m": 0.05,
        "decay_sec": 5.0,
        "geometry_source": "depth_pointcloud" if any(depth_point_clouds) else "bev_extruded_2p5d",
        "pose_source": "odom" if kwargs["pose_source"] == "online" else str(kwargs["pose_source"]),
        "pose_source_is_oracle": False,
        "dense_3d_claimed": bool(any(depth_point_clouds)),
        "depth_point_clouds": depth_point_clouds,
        "force_unaccepted": False,
        "expected": {},
        "warnings": sorted(set(warnings)),
        "hard_failures": [],
        "teacher_overlay_enabled": False,
        "overlay_is_runtime_truth": False,
    }


def _accumulate_frames(
    frames: list[SceneTimelineFrameV0],
    *,
    settings: dict[str, Any],
    max_points: int,
    max_cells: int,
) -> tuple[list[SceneTimelineFrameV0], list[AccumulatedMapStateV0]]:
    accumulator = AccumulatedScene3DV0(
        resolution_m=float(settings["resolution_m"]),
        decay_sec=float(settings["decay_sec"]),
        max_points=max_points,
        geometry_source=str(settings["geometry_source"]),
        pose_source=str(settings["pose_source"]),
        dense_3d_claimed=bool(settings["dense_3d_claimed"]),
    )
    timeline: list[SceneTimelineFrameV0] = []
    maps: list[AccumulatedMapStateV0] = []
    depth_point_clouds = settings.get("depth_point_clouds", [])
    use_depth = settings.get("geometry_source") in {"depth_pointcloud", "rgbd_pointcloud"}
    for index, frame in enumerate(frames):
        depth_points = depth_point_clouds[index] if isinstance(depth_point_clouds, list) and index < len(depth_point_clouds) else None
        if use_depth and frame.robot_pose is not None:
            accumulator.record_frame_pose(frame.robot_pose.T_map_base, camera_frustum=frame.camera_frustum)
        if use_depth and depth_points and frame.robot_pose is not None and frame.robot_pose.T_map_camera is not None:
            accumulator.update_from_depth_pointcloud(depth_points, T_map_camera=frame.robot_pose.T_map_camera, frame_id=frame.frame_id, timestamp_ns=frame.timestamp_ns)
        elif frame.local_bev is not None:
            accumulator.update_from_local_bev(frame.local_bev, camera_frustum=frame.camera_frustum)
        snapshot = accumulator.snapshot(frame.frame_id, frame.timestamp_ns)
        snapshot = _limited_snapshot(snapshot, max_cells=max_cells, max_points=max_points)
        maps.append(snapshot)
        timeline.append(replace(frame, accumulated_map_summary=snapshot.to_dict()))
    return timeline, maps


def _limited_snapshot(snapshot: AccumulatedMapStateV0, *, max_cells: int, max_points: int) -> AccumulatedMapStateV0:
    limit = max(0, int(max_cells))
    point_limit = max(0, int(max_points))
    return replace(
        snapshot,
        free_cells=snapshot.free_cells[:limit],
        obstacle_cells=snapshot.obstacle_cells[:limit],
        unknown_cells=snapshot.unknown_cells[:limit],
        risky_cells=snapshot.risky_cells[:limit],
        hazard_cells=snapshot.hazard_cells[:limit],
        stale_cells=snapshot.stale_cells[:limit],
        point_cloud=snapshot.point_cloud[:point_limit],
    )


def _summary(
    frames: list[SceneTimelineFrameV0],
    maps: list[AccumulatedMapStateV0],
    final_map: AccumulatedMapStateV0,
    *,
    settings: dict[str, Any],
    checkpoint: str | Path | None,
    route_id: str,
    debug_oracle_pose: bool,
) -> dict[str, Any]:
    warnings = sorted(set([*settings["warnings"], *(warning for frame in frames for warning in frame.warnings)]))
    hard_failures = list(settings["hard_failures"])
    if settings["pose_source_is_oracle"] and not debug_oracle_pose:
        hard_failures.append("oracle_pose_requires_debug_oracle_pose")
    if settings["geometry_source"] != "fixture_synthetic" and "missing_camera_to_base_extrinsics" in warnings:
        hard_failures.append("missing_extrinsics_real_mode_unaccepted")
    if settings["geometry_source"] != "fixture_synthetic" and "assumed_camera_to_base_extrinsics_review_required" in warnings:
        hard_failures.append("assumed_extrinsics_real_mode_unaccepted")
    if settings["dense_3d_claimed"] and settings["geometry_source"] not in {"depth_pointcloud", "rgbd_pointcloud"}:
        hard_failures.append("dense_3d_claim_without_depth_or_rgbd")
    channel_names = sorted({summary.name for frame in frames for summary in frame.model_channel_summaries if summary.rendered})
    missing_channel_names = sorted({summary.name for frame in frames for summary in frame.model_channel_summaries if not summary.rendered})
    accepted = _summary_accepts(frames, maps, final_map, settings=settings, hard_failures=hard_failures)
    return {
        "accepted_checkpoint_3d_visualizer_v0": bool(accepted),
        "checkpoint_path": str(checkpoint) if checkpoint is not None else None,
        "route_id": route_id,
        "frame_count": len(frames),
        "geometry_source": settings["geometry_source"],
        "pose_source": settings["pose_source"],
        "pose_source_is_oracle": bool(settings["pose_source_is_oracle"]),
        "dense_3d_claimed": bool(settings["dense_3d_claimed"]),
        "map_accumulates_over_time": map_accumulates(maps),
        "incremental_no_future_leakage_passed": no_future_leakage(maps, settings.get("expected", {})),
        "camera_pose_rendered": any(frame.camera_frustum is not None and frame.camera_frustum.valid for frame in frames),
        "camera_extrinsics_assumed": any(
            frame.robot_pose is not None and frame.robot_pose.T_map_camera is not None and frame.robot_pose.T_map_camera.is_assumed for frame in frames
        ),
        "robot_pose_rendered": all(frame.robot_pose is not None for frame in frames) and bool(frames),
        "rgb_frame_rendered": any(frame.rgb_frame is not None for frame in frames),
        "camera_frame_point_cloud_rendered": any(frame.depth_point_cloud_camera for frame in frames),
        "trajectory_rendered": len(final_map.trajectory_points) > 1,
        "model_channels_rendered": channel_names,
        "model_channels_missing": missing_channel_names,
        "candidate_trajectories_rendered": any(frame.candidate_trajectories for frame in frames),
        "safety_debug_rendered": any(frame.safety_debug is not None for frame in frames),
        "teacher_overlay_enabled": bool(settings.get("teacher_overlay_enabled", False)),
        "overlay_is_runtime_truth": bool(settings.get("overlay_is_runtime_truth", False)),
        "artifacts": [],
        "warnings": warnings,
        "hard_failures": sorted(set(hard_failures)),
        "safety": safety_flags(),
    }


def _summary_accepts(
    frames: list[SceneTimelineFrameV0],
    maps: list[AccumulatedMapStateV0],
    final_map: AccumulatedMapStateV0,
    *,
    settings: dict[str, Any],
    hard_failures: list[str],
) -> bool:
    return bool(
        not settings.get("force_unaccepted", False)
        and not hard_failures
        and len(frames) > 1
        and map_accumulates(maps)
        and no_future_leakage(maps, settings.get("expected", {}))
        and all(frame.robot_pose is not None for frame in frames)
        and len(final_map.trajectory_points) > 1
        and (final_map.free_cells or final_map.obstacle_cells or final_map.unknown_cells or settings["geometry_source"] in {"depth_pointcloud", "rgbd_pointcloud"})
        and (settings["dense_3d_claimed"] is False or settings["geometry_source"] in {"depth_pointcloud", "rgbd_pointcloud"})
    )


def _write_artifacts(
    out_dir: Path,
    *,
    frames: list[SceneTimelineFrameV0],
    timeline_maps: list[AccumulatedMapStateV0],
    final_map: AccumulatedMapStateV0,
    accumulated_payload: dict[str, Any],
    summary: dict[str, Any],
    write_html: bool,
    write_ply: bool,
    write_jsonl: bool,
) -> list[str]:
    artifacts: list[str] = []
    frames, rgb_artifacts = _materialize_rgb_frames(out_dir, frames)
    artifacts.extend(rgb_artifacts)
    if write_jsonl:
        write_scene_timeline_jsonl(out_dir / "scene_timeline.jsonl", frames)
        artifacts.append("scene_timeline.jsonl")
    write_accumulated_scene_json(out_dir / "accumulated_scene.json", final_map=final_map, timeline_maps=timeline_maps, summary=summary)
    artifacts.append("accumulated_scene.json")
    if write_ply:
        write_ply_point_cloud(out_dir / "accumulated_map.ply", final_map, metadata=summary)
        write_ply_lines(
            out_dir / "camera_trajectory.ply",
            [*trajectory_lines(final_map.trajectory_points), *camera_frustum_lines_from_frames(frames)],
            metadata=summary,
        )
        artifacts.extend(["accumulated_map.ply", "camera_trajectory.ply"])
    if write_html:
        write_viewer_html(out_dir / "viewer.html", frames=frames, accumulated_scene=accumulated_payload, summary=summary)
        artifacts.append("viewer.html")
    for frame in frames:
        write_json(
            out_dir / "frame_debug" / f"frame_{int(frame.frame_id):06d}.json",
            {
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "warnings": frame.warnings,
                "candidate_count": len(frame.candidate_trajectories),
                "rgb_frame": _debug_rgb_frame(frame.rgb_frame),
                "stop_reasons": frame.safety_debug.stop_reasons if frame.safety_debug is not None else [],
            },
            pretty=True,
        )
    return sorted(artifacts)


def _materialize_rgb_frames(out_dir: Path, frames: list[SceneTimelineFrameV0]) -> tuple[list[SceneTimelineFrameV0], list[str]]:
    updated: list[SceneTimelineFrameV0] = []
    artifacts: list[str] = []
    rgb_dir = out_dir / "rgb_frames"
    for index, frame in enumerate(frames):
        rgb = dict(frame.rgb_frame or {})
        source = _rgb_source_path(rgb)
        if source is not None and source.exists():
            rgb_dir.mkdir(parents=True, exist_ok=True)
            suffix = source.suffix if source.suffix else ".img"
            target = rgb_dir / f"frame_{index:06d}{suffix}"
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
            rgb["relative_path"] = target.relative_to(out_dir).as_posix()
            if "src" in rgb:
                rgb["source_uri"] = rgb.pop("src")
            rgb["src"] = rgb["relative_path"]
            rgb["available"] = True
            rgb["materialized"] = True
            artifacts.append("rgb_frames/")
        updated.append(replace(frame, rgb_frame=rgb if rgb else None))
    return updated, sorted(set(artifacts))


def _rgb_source_path(rgb_frame: dict[str, Any]) -> Path | None:
    source = rgb_frame.get("source_path")
    if isinstance(source, str) and source:
        path = Path(source)
        if path.exists():
            return path
        resolved = path.expanduser().resolve()
        if resolved.exists():
            return resolved
    src = rgb_frame.get("src")
    if isinstance(src, str) and src.startswith("file://"):
        return Path(src[7:])
    return None


def _debug_rgb_frame(rgb_frame: dict[str, Any] | None) -> dict[str, Any] | None:
    if rgb_frame is None:
        return None
    return {
        key: rgb_frame.get(key)
        for key in ("kind", "frame_id", "timestamp_ns", "source_path", "relative_path", "available", "materialized")
        if key in rgb_frame
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a replay-only checkpoint 3D/2.5D timeline visualization.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--pack")
    parser.add_argument("--route-id")
    parser.add_argument("--fixture")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--pose-source", default="online")
    parser.add_argument("--write-html", action="store_true")
    parser.add_argument("--write-ply", action="store_true")
    parser.add_argument("--write-jsonl", action="store_true")
    parser.add_argument("--max-points", type=int, default=50_000)
    parser.add_argument("--max-cells", type=int, default=20_000)
    parser.add_argument("--review-assumed-extrinsics", action="store_true")
    parser.add_argument("--debug-oracle-pose", action="store_true")
    parser.add_argument("--debug-teacher-overlay")
    args = parser.parse_args(argv)
    visualize_checkpoint_3d_v0(
        out=args.out,
        fixture=args.fixture,
        checkpoint=args.checkpoint,
        pack=args.pack,
        route_id=args.route_id,
        device=args.device,
        max_frames=args.max_frames,
        stride=args.stride,
        pose_source=args.pose_source,
        write_html=args.write_html,
        write_ply=args.write_ply,
        write_jsonl=args.write_jsonl,
        max_points=args.max_points,
        max_cells=args.max_cells,
        review_assumed_extrinsics=args.review_assumed_extrinsics,
        debug_oracle_pose=args.debug_oracle_pose,
        debug_teacher_overlay=args.debug_teacher_overlay,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
