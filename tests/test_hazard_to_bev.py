import json
from pathlib import Path

import numpy as np
import pytest

from homebrain.geometry.camera_config import CameraConfig, write_camera_config
from homebrain.geometry.hazard_to_bev import convert_hazard_to_bev, project_hazard_masks_to_bev
from homebrain.replay.generate_dummy_log import generate_dummy_log
from homebrain.teachers.hazard_teacher import run_hazard_teacher


def test_hazard_to_bev_projects_floor_mask_to_robot_frame_cell(tmp_path: Path) -> None:
    config = CameraConfig.from_dict(
        {
            "camera_height_m": 0.5,
            "pitch_deg": 45.0,
            "roll_deg": 0.0,
            "yaw_deg": 0.0,
            "grid_size_m": 1.0,
            "meters_per_cell": 0.25,
            "max_depth_m": 3.0,
            "floor_height_tol_m": 0.04,
            "obstacle_height_min_m": 0.12,
            "robot_radius_m": 0.0,
        }
    )
    masks = np.zeros((1, 3, 3), dtype=np.float32)
    masks[0, 1, 1] = 1.0

    hazard, valid, stats = project_hazard_masks_to_bev(
        hazard_masks=masks,
        frame_width=3,
        frame_height=3,
        intrinsics={"fx": 2.0, "fy": 2.0, "cx": 1.0, "cy": 1.0},
        config=config,
    )

    assert hazard.shape == (4, 4)
    assert valid.shape == (4, 4)
    assert hazard[2, 2] == 1.0
    assert stats["hazard_positive_cell_count"] == 1


def test_hazard_projection_requires_extrinsics_unless_review_assumed(tmp_path: Path) -> None:
    route = tmp_path / "route"
    teacher = tmp_path / "hazard"
    generate_dummy_log(route)
    run_hazard_teacher(route, teacher, backend_name="fake")

    with pytest.raises(ValueError, match="camera_config_path is required"):
        convert_hazard_to_bev(log_dir=route, hazard_artifacts_dir=teacher, out_dir=tmp_path / "bev")

    manifest_path = convert_hazard_to_bev(
        log_dir=route,
        hazard_artifacts_dir=teacher,
        out_dir=tmp_path / "bev_review",
        review_assumed_extrinsics=True,
    )
    assert manifest_path.exists()


def test_hazard_channel_does_not_modify_geometry_channels(tmp_path: Path) -> None:
    free = np.ones((4, 4), dtype=np.uint8)
    obstacle = np.zeros((4, 4), dtype=np.uint8)
    unknown = np.zeros((4, 4), dtype=np.uint8)
    hazard = np.zeros((4, 4), dtype=np.float32)
    hazard[1, 2] = 1.0

    free_before = free.copy()
    obstacle_before = obstacle.copy()
    unknown_before = unknown.copy()
    _ = hazard

    assert np.array_equal(free, free_before)
    assert np.array_equal(obstacle, obstacle_before)
    assert np.array_equal(unknown, unknown_before)
    assert free[1, 2] == 1
    assert obstacle[1, 2] == 0
    assert unknown[1, 2] == 0


def test_hazard_to_bev_conversion_writes_hazard_artifacts(tmp_path: Path) -> None:
    route = tmp_path / "route"
    teacher = tmp_path / "hazard"
    camera = tmp_path / "camera.json"
    bev = tmp_path / "bev"
    generate_dummy_log(route)
    run_hazard_teacher(route, teacher, backend_name="fake")
    write_camera_config(
        camera,
        CameraConfig.from_dict(
            {
                "camera_height_m": 0.5,
                "pitch_deg": 45.0,
                "roll_deg": 0.0,
                "yaw_deg": 0.0,
                "grid_size_m": 2.0,
                "meters_per_cell": 0.5,
                "max_depth_m": 3.0,
                "floor_height_tol_m": 0.04,
                "obstacle_height_min_m": 0.12,
                "robot_radius_m": 0.0,
            }
        ),
    )

    manifest_path = convert_hazard_to_bev(
        log_dir=route,
        hazard_artifacts_dir=teacher,
        camera_config_path=camera,
        out_dir=bev,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = manifest["frames"][0]
    assert manifest["weak_label"] is True
    assert manifest["control_safe"] is False
    assert "bev_hazard" in first["artifacts"]
    assert "hazard_valid_mask" in first["artifacts"]


def test_hazard_to_bev_carries_hand_verified_positive_frames(tmp_path: Path) -> None:
    route = tmp_path / "route"
    teacher = tmp_path / "hazard"
    camera = tmp_path / "camera.json"
    verification = tmp_path / "hand_verified_hazards.json"
    bev = tmp_path / "bev"
    generate_dummy_log(route)
    run_hazard_teacher(route, teacher, backend_name="fake")
    write_camera_config(
        camera,
        CameraConfig.from_dict(
            {
                "camera_height_m": 0.5,
                "pitch_deg": 45.0,
                "roll_deg": 0.0,
                "yaw_deg": 0.0,
                "grid_size_m": 2.0,
                "meters_per_cell": 0.5,
                "max_depth_m": 3.0,
                "floor_height_tol_m": 0.04,
                "obstacle_height_min_m": 0.12,
                "robot_radius_m": 0.0,
            }
        ),
    )
    verification.write_text(
        json.dumps(
            {
                "schema_version": "homebrain.hazard_hand_verification.v0",
                "hand_verified": True,
                "reviewer": "unit_test",
                "frames": [
                    {"frame_id": 0, "hazard_present": True, "classes": ["cable"]},
                    {"frame_id": 999, "hazard_present": True, "classes": ["sock"]},
                    {"frame_id": 1, "hazard_present": False},
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = convert_hazard_to_bev(
        log_dir=route,
        hazard_artifacts_dir=teacher,
        camera_config_path=camera,
        out_dir=bev,
        hand_verification_path=verification,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["hand_verified_hazard_positive_frame_count"] == 1
    assert manifest["hand_verified_hazard_positive_frame_ids"] == [0]
    assert manifest["hand_verified_hazard_missing_frame_ids"] == [999]
    assert manifest["frames"][0]["hand_verified_hazard_positive"] is True
    assert manifest["frames"][1]["hand_verified_hazard_positive"] is False
