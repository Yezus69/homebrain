from __future__ import annotations

from homebrain.viz.accumulated_scene3d_v0 import AccumulatedScene3DV0
from homebrain.viz.pose_camera_geometry_v0 import build_T_map_base_from_pose_sequence
from tests.checkpoint_3d_visualizer_fixtures import build_fixture_sequence


def _snapshots(name: str):
    fixture = build_fixture_sequence(name)
    scene = AccumulatedScene3DV0(
        resolution_m=fixture.resolution_m,
        decay_sec=fixture.decay_sec,
        max_points=1000,
        geometry_source=fixture.geometry_source,
        pose_source=fixture.pose_source,
    )
    snapshots = []
    for frame in fixture.frames:
        assert frame.local_bev is not None
        scene.update_from_local_bev(frame.local_bev, camera_frustum=frame.camera_frustum)
        snapshots.append(scene.snapshot(frame.frame_id, frame.timestamp_ns))
    return snapshots


def test_accumulated_map_grows_and_stitches_obstacle() -> None:
    snapshots = _snapshots("stitched_room_tiny")
    counts = [len(state.free_cells) + len(state.obstacle_cells) for state in snapshots]

    assert counts[-1] > counts[0]
    assert all(after >= before for before, after in zip(counts, counts[1:]))
    assert len(snapshots[-1].obstacle_cells) >= 1
    assert len(snapshots[-1].trajectory_points) == 5


def test_timeline_snapshot_uses_only_past_frames() -> None:
    snapshots = _snapshots("no_future_leakage")

    assert all(not state.obstacle_cells for state in snapshots[:4])
    assert snapshots[4].obstacle_cells


def test_hazard_remains_free_but_not_obstacle() -> None:
    final = _snapshots("hazard_free_but_unsafe")[-1]

    assert final.hazard_cells
    assert final.free_cells
    assert not final.obstacle_cells


def test_unknown_cells_do_not_become_free() -> None:
    final = _snapshots("unknown_not_free")[-1]

    assert final.unknown_cells
    assert not final.free_cells


def test_stale_memory_is_tracked() -> None:
    final = _snapshots("stale_memory")[-1]

    assert final.stale_cells
    assert final.stale_update_count > 0


def test_depth_pointcloud_accumulates_real_geometry_points() -> None:
    scene = AccumulatedScene3DV0(resolution_m=0.1, decay_sec=5.0, max_points=10, geometry_source="depth_pointcloud")
    T_map_camera = build_T_map_base_from_pose_sequence((0.0, 0.0, 0.0), source="fixture_pose")

    scene.record_frame_pose(T_map_camera)
    scene.update_from_depth_pointcloud([[0.0, 0.0, 1.0, 10, 20, 30], [0.1, 0.0, 1.2, 40, 50, 60]], T_map_camera=T_map_camera, frame_id=0, timestamp_ns=0)
    snapshot = scene.snapshot(0, 0)

    assert snapshot.point_cloud == [[0.0, 0.0, 1.0, 10, 20, 30], [0.1, 0.0, 1.2, 40, 50, 60]]
    assert snapshot.obstacle_cells
    assert snapshot.geometry_source == "depth_pointcloud"
