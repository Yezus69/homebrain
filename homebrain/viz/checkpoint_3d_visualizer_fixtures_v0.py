from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import quote

import numpy as np

from homebrain.viz.checkpoint_3d_scene_types_v0 import (
    CandidateTrajectoryVizV0,
    LocalBEVFrameV0,
    ModelChannelSummaryV0,
    RobotPoseVizV0,
    SafetyDebugVizV0,
    SceneTimelineFrameV0,
    TransformV0,
)
from homebrain.viz.pose_camera_geometry_v0 import (
    base_axes_lines,
    build_T_map_base_from_pose_sequence,
    camera_frustum_from_intrinsics,
    compose_transforms,
    robot_footprint_polygon,
    transform_from_matrix,
    transform_points,
)


@dataclass(frozen=True)
class FixtureSequenceV0:
    name: str
    route_id: str
    frames: list[SceneTimelineFrameV0]
    resolution_m: float
    decay_sec: float
    geometry_source: str
    pose_source: str
    pose_source_is_oracle: bool
    dense_3d_claimed: bool
    expected: dict[str, Any]
    force_unaccepted: bool = False


def available_fixtures() -> list[str]:
    return [
        "stitched_room_tiny",
        "turning_camera_frustum",
        "hazard_free_but_unsafe",
        "unknown_not_free",
        "stale_memory",
        "missing_extrinsics_real_mode",
        "oracle_pose_debug",
        "no_future_leakage",
        "candidate_trajectory_overlay",
        "safety_stop_overlay",
    ]


def build_fixture_sequence(name: str) -> FixtureSequenceV0:
    if name not in available_fixtures():
        raise ValueError(f"unknown checkpoint 3d visualizer fixture {name!r}")
    resolution = 0.25
    frames: list[SceneTimelineFrameV0] = []
    force_unaccepted = name in {"missing_extrinsics_real_mode", "oracle_pose_debug"}
    pose_source = "fixture_pose" if name != "missing_extrinsics_real_mode" else "route_pose"
    pose_is_oracle = name == "oracle_pose_debug"
    geometry_source = "fixture_synthetic"
    for index in range(5):
        timestamp_ns = index * 100_000_000
        yaw = 0.0
        x_m = index * 0.25
        y_m = 0.0
        if name == "turning_camera_frustum":
            x_m = 0.0
            yaw = index * 0.35
        T_map_base = build_T_map_base_from_pose_sequence(
            (x_m, y_m, yaw),
            timestamp_ns=timestamp_ns,
            source=pose_source,
            is_oracle=pose_is_oracle,
        )
        camera = None if name == "missing_extrinsics_real_mode" else _camera_frustum(T_map_base, timestamp_ns)
        robot = _robot_pose(T_map_base, camera.T_map_camera if camera is not None else None, pose_source, pose_is_oracle)
        local = _local_bev_frame(name, index, timestamp_ns, T_map_base, resolution)
        candidates = _candidates(T_map_base, timestamp_ns, include=name in {"stitched_room_tiny", "candidate_trajectory_overlay", "safety_stop_overlay"})
        safety = _safety(timestamp_ns, stopped=name == "safety_stop_overlay" and index >= 2)
        warnings = []
        if camera is None:
            warnings.append("missing_camera_to_base_extrinsics")
        if pose_is_oracle:
            warnings.append("pose_source_is_oracle_debug_only")
        frames.append(
            SceneTimelineFrameV0(
                route_id=f"fixture_{name}",
                frame_id=index,
                timestamp_ns=timestamp_ns,
                robot_pose=robot,
                camera_frustum=camera,
                local_bev=local,
                rgb_frame=_fixture_rgb_frame(name, index),
                candidate_trajectories=candidates,
                safety_debug=safety,
                model_channel_summaries=local.channel_summaries,
                warnings=warnings,
            )
        )
    if name == "stale_memory":
        frames = [replace(frame, timestamp_ns=index * 500_000_000) for index, frame in enumerate(frames)]
        frames = [_retime_frame(frame, index * 500_000_000) for index, frame in enumerate(frames)]
    return FixtureSequenceV0(
        name=name,
        route_id=f"fixture_{name}",
        frames=frames,
        resolution_m=resolution,
        decay_sec=0.3 if name == "stale_memory" else 10.0,
        geometry_source=geometry_source,
        pose_source=pose_source,
        pose_source_is_oracle=pose_is_oracle,
        dense_3d_claimed=False,
        expected=_expected(name),
        force_unaccepted=force_unaccepted,
    )


def _retime_frame(frame: SceneTimelineFrameV0, timestamp_ns: int) -> SceneTimelineFrameV0:
    local = frame.local_bev
    robot = frame.robot_pose
    camera = frame.camera_frustum
    if local is not None:
        local = replace(local, timestamp_ns=timestamp_ns)
    if robot is not None:
        robot = replace(robot, timestamp_ns=timestamp_ns)
    if camera is not None:
        camera = replace(camera, timestamp_ns=timestamp_ns)
    return replace(frame, timestamp_ns=timestamp_ns, local_bev=local, robot_pose=robot, camera_frustum=camera)


def _camera_frustum(T_map_base: TransformV0, timestamp_ns: int) -> Any:
    T_base_camera = transform_from_matrix(
        frame_from="base_link",
        frame_to="camera",
        matrix_4x4=[
            [0.0, 0.0, 1.0, 0.12],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.24],
            [0.0, 0.0, 0.0, 1.0],
        ],
        source="fixture_camera_extrinsics",
        timestamp_ns=timestamp_ns,
    )
    return camera_frustum_from_intrinsics(
        T_map_camera=compose_transforms(T_map_base, T_base_camera),
        intrinsics={"fx": 120.0, "fy": 120.0, "cx": 80.0, "cy": 60.0, "width": 160, "height": 120},
        timestamp_ns=timestamp_ns,
        far_m=1.0,
    )


def _robot_pose(T_map_base: TransformV0, T_map_camera: TransformV0 | None, pose_source: str, pose_is_oracle: bool) -> RobotPoseVizV0:
    return RobotPoseVizV0(
        timestamp_ns=T_map_base.timestamp_ns,
        T_map_base=T_map_base,
        T_map_camera=T_map_camera,
        base_axes_lines_map=base_axes_lines(T_map_base),
        footprint_polygon_map=robot_footprint_polygon(T_map_base),
        pose_confidence=1.0,
        pose_source=pose_source,
        pose_source_is_oracle=pose_is_oracle,
    )


def _local_bev_frame(name: str, index: int, timestamp_ns: int, T_map_base: TransformV0, resolution_m: float) -> LocalBEVFrameV0:
    shape = (7, 7)
    free = np.zeros(shape, dtype=np.float32)
    obstacle = np.zeros(shape, dtype=np.float32)
    unknown = np.zeros(shape, dtype=np.float32)
    risky = np.zeros(shape, dtype=np.float32)
    hazard = np.zeros(shape, dtype=np.float32)
    memory_age = np.zeros(shape, dtype=np.float32)
    free[3:, :] = 1.0
    if name == "unknown_not_free":
        free[:] = 0.0
        unknown[1:4, 1:4] = 1.0
    elif name == "hazard_free_but_unsafe":
        free[3, 2] = 1.0
        hazard[3, 2] = 1.0
    elif name == "no_future_leakage":
        if index >= 4:
            obstacle[1, 2] = 1.0
    elif name == "stale_memory":
        if index == 0:
            obstacle[2, 3] = 1.0
        else:
            free[3:, :] = 1.0
            memory_age[2, 3] = float(index) * 0.5
    else:
        obstacle[1, 2] = 1.0
        obstacle[1, 4] = 1.0
        obstacle[2, 5] = 1.0
        unknown[0, 0] = 1.0
        unknown[0, 6] = 1.0
        risky[4, 1] = 1.0
        hazard[5, 4] = 1.0
        free[5, 4] = 1.0
        if name == "safety_stop_overlay" and index >= 2:
            risky[3, 3] = 1.0
            hazard[3, 3] = 1.0
    channels = {
        "free": free.tolist(),
        "obstacle": obstacle.tolist(),
        "unknown": unknown.tolist(),
        "risky": risky.tolist(),
        "hazard": hazard.tolist(),
        "memory_age": memory_age.tolist(),
    }
    summaries = [_summary(key, np.asarray(value, dtype=np.float32)) for key, value in channels.items()]
    return LocalBEVFrameV0(
        timestamp_ns=timestamp_ns,
        frame_id=index,
        T_map_base=T_map_base,
        resolution_m=resolution_m,
        origin_convention="robot_bottom_center_forward_rows_decrease_left_cols_increase",
        channels=channels,
        channel_summaries=summaries,
        rendered_cells_map=[],
    )


def _fixture_rgb_frame(name: str, index: int) -> dict[str, Any]:
    shift = index * 10
    obstacle_x = 156 + shift
    hazard_x = 214 - index * 4
    label = f"{name} frame {index}"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="360" height="220" viewBox="0 0 360 220">
<rect width="360" height="220" fill="#d9e7f2"/>
<rect y="0" width="360" height="88" fill="#edf4f8"/>
<polygon points="18,220 136,92 224,92 342,220" fill="#8f9b92"/>
<polygon points="72,220 154,112 205,112 286,220" fill="#b6c3ad"/>
<line x1="180" y1="92" x2="180" y2="220" stroke="#f2f2ea" stroke-width="3"/>
<rect x="{obstacle_x}" y="118" width="42" height="54" fill="#c84d46" stroke="#6f2422" stroke-width="3"/>
<rect x="76" y="138" width="34" height="35" fill="#9aa3ad" stroke="#626b73" stroke-width="3"/>
<polygon points="{hazard_x},182 {hazard_x + 62},190 {hazard_x + 52},211 {hazard_x - 12},203" fill="#c84897" opacity="0.78"/>
<polygon points="108,188 151,182 163,205 119,211" fill="#e3ad3f" opacity="0.8"/>
<polyline points="180,214 178,176 183,150" fill="none" stroke="#111827" stroke-width="5"/>
<text x="14" y="24" font-family="Arial,sans-serif" font-size="15" fill="#18202a">fixture RGB reference</text>
<text x="14" y="44" font-family="Arial,sans-serif" font-size="12" fill="#394554">{label}</text>
</svg>"""
    return {
        "kind": "fixture_synthetic_rgb_reference",
        "frame_id": index,
        "width": 360,
        "height": 220,
        "data_uri": "data:image/svg+xml;charset=utf-8," + quote(svg, safe=""),
        "notes": ["fixture_reference_not_real_sensor_rgb"],
    }


def _summary(name: str, array: np.ndarray) -> ModelChannelSummaryV0:
    valid = np.isfinite(array)
    values = array[valid]
    return ModelChannelSummaryV0(
        name=f"bev_{name}" if name not in {"memory_age"} else name,
        shape=[int(array.shape[0]), int(array.shape[1])],
        min=float(values.min()) if values.size else None,
        max=float(values.max()) if values.size else None,
        mean=float(values.mean()) if values.size else None,
        valid_fraction=float(np.count_nonzero(valid) / max(valid.size, 1)),
        source="fixture_model_output",
        rendered=True,
    )


def _candidates(T_map_base: TransformV0, timestamp_ns: int, *, include: bool) -> list[CandidateTrajectoryVizV0]:
    if not include:
        return []
    specs = [
        ("stop", [[0.0, 0.0, 0.02], [0.0, 0.0, 0.02]], 0.2, True, False, []),
        ("straight", [[0.0, 0.0, 0.02], [0.35, 0.0, 0.02], [0.7, 0.0, 0.02]], 0.1, False, False, []),
        ("left_vetoed", [[0.0, 0.0, 0.02], [0.25, 0.2, 0.02], [0.45, 0.4, 0.02]], 0.8, False, True, ["hazard"]),
    ]
    out = []
    for candidate_id, points, score, selected, vetoed, reasons in specs:
        mapped = transform_points(T_map_base, points).tolist()
        out.append(
            CandidateTrajectoryVizV0(
                timestamp_ns=timestamp_ns,
                candidate_id=candidate_id,
                points_map=[[round(float(v), 6) for v in point] for point in mapped],
                score=score,
                risk=score,
                hazard_exposure=0.7 if vetoed else 0.0,
                unknown_exposure=0.0,
                selected=selected,
                vetoed=vetoed,
                stop_reasons=reasons,
            )
        )
    return out


def _safety(timestamp_ns: int, *, stopped: bool) -> SafetyDebugVizV0:
    return SafetyDebugVizV0(
        timestamp_ns=timestamp_ns,
        proposed_cmd_vel={"linear_mps": 0.2, "angular_radps": 0.0},
        safe_cmd_vel={"linear_mps": 0.0 if stopped else 0.2, "angular_radps": 0.0},
        accepted=not stopped,
        stop_reasons=["high_risk_near_footprint"] if stopped else [],
        high_risk=stopped,
        obstacle_too_close=stopped,
    )


def _expected(name: str) -> dict[str, Any]:
    return {
        "fixture": name,
        "frame_count": 5,
        "expected_accumulated_obstacle_min": 1 if name != "unknown_not_free" else 0,
        "expected_no_future_obstacle_before_frame": 4 if name == "no_future_leakage" else None,
        "expected_selected_candidate_id": "stop" if name in {"stitched_room_tiny", "candidate_trajectory_overlay", "safety_stop_overlay"} else None,
    }
