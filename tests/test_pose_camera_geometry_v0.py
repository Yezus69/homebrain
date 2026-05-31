from __future__ import annotations

import numpy as np

from homebrain.viz.pose_camera_geometry_v0 import (
    bev_cell_centers_to_robot_frame,
    build_T_map_base_from_pose_sequence,
    camera_frustum_from_intrinsics,
    compose_transforms,
    invert_transform,
    robot_footprint_polygon,
    robot_frame_to_map_frame,
    transform_from_matrix,
    transform_points,
)


def test_transform_composition_and_inversion_are_sane() -> None:
    T_map_base = build_T_map_base_from_pose_sequence((1.0, 2.0, 0.0), source="odom")
    T_base_camera = transform_from_matrix(
        frame_from="base_link",
        frame_to="camera",
        matrix_4x4=[[1.0, 0.0, 0.0, 0.3], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.2], [0.0, 0.0, 0.0, 1.0]],
        source="fixture",
    )
    T_map_camera = compose_transforms(T_map_base, T_base_camera)
    T_camera_map = invert_transform(T_map_camera)
    point = np.asarray([[0.1, 0.0, 0.0]])
    round_trip = transform_points(T_camera_map, transform_points(T_map_camera, point))

    assert T_map_camera.frame_from == "map"
    assert T_map_camera.frame_to == "camera"
    assert np.allclose(round_trip, point)


def test_camera_frustum_and_robot_footprint_render_in_map_frame() -> None:
    T_map_base = build_T_map_base_from_pose_sequence((1.0, 0.5, 0.0), source="fixture")
    T_base_camera = transform_from_matrix(
        frame_from="base_link",
        frame_to="camera",
        matrix_4x4=[[0.0, 0.0, 1.0, 0.1], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0]],
        source="fixture",
    )
    frustum = camera_frustum_from_intrinsics(
        T_map_camera=compose_transforms(T_map_base, T_base_camera),
        intrinsics={"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0, "width": 100, "height": 80},
    )
    footprint = robot_footprint_polygon(T_map_base)

    assert frustum.valid is True
    assert len(frustum.frustum_lines_map) == 8
    assert len(footprint) == 5
    assert footprint[0][0] > 1.0


def test_bev_cells_transform_from_robot_to_map_frame() -> None:
    T_map_base = build_T_map_base_from_pose_sequence((1.0, 0.0, 0.0), source="fixture")
    robot_points = bev_cell_centers_to_robot_frame((4, 4), 0.25, [(3, 2), (2, 2)])
    map_points = robot_frame_to_map_frame(T_map_base, robot_points)

    assert np.allclose(robot_points[0], [0.125, 0.125, 0.0])
    assert map_points[1, 0] > map_points[0, 0]

