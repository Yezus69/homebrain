from __future__ import annotations

from typing import Any

from homebrain.viz.checkpoint_3d_scene_types_v0 import AccumulatedMapStateV0, safety_flags


def visualizer_acceptance_report(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "summary_accepted": summary.get("accepted_checkpoint_3d_visualizer_v0") is True,
        "safety_replay_only": summary.get("safety", {}).get("replay_only") is True and summary.get("safety", {}).get("not_executed") is True,
        "safety_not_control": summary.get("safety", {}).get("control_safe") is False
        and summary.get("safety", {}).get("raw_pwm_emitted") is False
        and summary.get("safety", {}).get("hardware_validated") is False,
        "dense_3d_honest": summary.get("dense_3d_claimed") is False or summary.get("geometry_source") in {"depth_pointcloud", "rgbd_pointcloud"},
    }
    accepted = all(checks.values())
    return {
        "schema_version": "homebrain.checkpoint_3d_visualizer_acceptance.v0",
        "accepted_checkpoint_3d_visualizer_v0": accepted,
        "gates_improved": ["Gate A", "Gate D"] if accepted else [],
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "safety": summary.get("safety", safety_flags()),
    }


def map_accumulates(maps: list[AccumulatedMapStateV0]) -> bool:
    if len(maps) <= 1:
        return False
    counts = [_cell_total(state) for state in maps]
    return counts[-1] > counts[0] and all(after >= before for before, after in zip(counts, counts[1:]))


def no_future_leakage(maps: list[AccumulatedMapStateV0], expected: dict[str, Any]) -> bool:
    marker = expected.get("expected_no_future_obstacle_before_frame")
    if marker is not None:
        for state in maps[: int(marker)]:
            if state.obstacle_cells:
                return False
    return all(state.observation_count <= index + 1 for index, state in enumerate(maps))


def empty_accumulated_map(settings: dict[str, Any]) -> AccumulatedMapStateV0:
    return AccumulatedMapStateV0(
        timestamp_ns=None,
        frame_id=0,
        map_resolution_m=float(settings.get("resolution_m", 0.05)),
        global_bounds_m={"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0},
        free_cells=[],
        obstacle_cells=[],
        unknown_cells=[],
        risky_cells=[],
        hazard_cells=[],
        stale_cells=[],
        trajectory_points=[],
        camera_frustums=[],
        point_cloud=[],
        conflict_count=0,
        stale_update_count=0,
        observation_count=0,
        geometry_source=str(settings.get("geometry_source", "bev_extruded_2p5d")),
        pose_source=str(settings.get("pose_source", "assumed_stationary")),
        dense_3d_claimed=False,
    )


def _cell_total(state: AccumulatedMapStateV0) -> int:
    return len(state.free_cells) + len(state.obstacle_cells) + len(state.unknown_cells) + len(state.risky_cells) + len(state.hazard_cells) + len(state.point_cloud)
