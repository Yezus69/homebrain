from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from homebrain.geometry.bev_projector import robot_points_to_bev_arrays
from homebrain.teachers.artifacts import file_sha256, relative_to_root, save_array
from homebrain.teachers.scene_teacher import SCENE_TEACHER_PACK_SCHEMA_VERSION, write_scene_teacher_json
from homebrain.tools.validate_moge_robot_bev_projection import (
    NEXT_LOCAL_REVIEW,
    NEXT_REVIEW_ONLY,
    convention_candidates,
    validate_moge_robot_bev_projection,
    write_aggregate_report,
)

CAMERA_TO_BASE = [
    [0.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.5],
    [0.0, 0.0, 0.0, 1.0],
]
KNOWN_CONVENTION = "cam_x=+moge_y__cam_y=+moge_z__cam_z=+moge_x"
GRID_CONFIG = {
    "grid_cell_count": 8,
    "grid_size_m": 2.0,
    "meters_per_cell": 0.25,
    "floor_height_tol_m": 0.05,
    "obstacle_height_min_m": 0.10,
    "obstacle_dilation_cells": 0,
    "robot_radius_m": 0.0,
}


def test_goal18a_projection_finds_known_convention_and_keeps_safety_flags(tmp_path: Path) -> None:
    route, scene, spatial = _projection_fixture(tmp_path, mock=False)

    report = validate_moge_robot_bev_projection(
        route_dir=route,
        scene_teacher_dir=scene,
        spatial_pack_dir=spatial,
        out_json=tmp_path / "projection.json",
        out_md=tmp_path / "projection.md",
        out_viz=tmp_path / "projection.ppm",
    )

    assert report["matched_frame_count"] == 4
    assert report["candidate_convention_count"] == len(convention_candidates())
    assert report["best_convention_name"] == KNOWN_CONVENTION
    assert report["selected_convention"]["calibration_split_metrics"]["obstacle_iou"] == 1.0
    assert report["heldout_split_metrics"]["free_iou"] == 1.0
    assert report["heldout_split_metrics"]["obstacle_iou"] == 1.0
    assert report["heldout_split_metrics"]["false_free_over_obstacle_rate"] == 0.0
    assert report["next_allowed_use"] == NEXT_LOCAL_REVIEW
    _assert_safety_flags(report)
    assert (tmp_path / "projection.json").exists()
    assert (tmp_path / "projection.md").exists()
    assert (tmp_path / "projection.ppm").exists()


def test_goal18a_fake_scene_teacher_cannot_be_promoted_beyond_review_only(tmp_path: Path) -> None:
    route, scene, spatial = _projection_fixture(tmp_path, mock=True)

    report = validate_moge_robot_bev_projection(
        route_dir=route,
        scene_teacher_dir=scene,
        spatial_pack_dir=spatial,
        out_json=tmp_path / "projection_fake.json",
        out_md=tmp_path / "projection_fake.md",
        out_viz=tmp_path / "projection_fake.ppm",
    )

    assert report["matched_frame_count"] == 4
    assert report["best_convention_name"] == KNOWN_CONVENTION
    assert report["heldout_split_metrics"]["obstacle_iou"] == 1.0
    assert report["scene_teacher"]["mock"] is True
    assert report["scene_teacher"]["synthetic"] is True
    assert report["next_allowed_use"] == NEXT_REVIEW_ONLY
    _assert_safety_flags(report)


def test_goal18a_aggregate_keeps_product_and_control_flags_false(tmp_path: Path) -> None:
    route, scene, spatial = _projection_fixture(tmp_path, mock=False)
    report = validate_moge_robot_bev_projection(
        route_dir=route,
        scene_teacher_dir=scene,
        spatial_pack_dir=spatial,
        out_json=tmp_path / "projection.json",
        out_md=tmp_path / "projection.md",
        out_viz=tmp_path / "projection.ppm",
    )

    aggregate = write_aggregate_report(
        reports=[report],
        out_json=tmp_path / "aggregate.json",
        out_md=tmp_path / "aggregate.md",
    )

    assert aggregate["route_count"] == 1
    assert aggregate["best_convention_distribution"][KNOWN_CONVENTION] == 1
    assert aggregate["next_allowed_use"] == NEXT_LOCAL_REVIEW
    _assert_safety_flags(aggregate)


def _projection_fixture(root: Path, *, mock: bool) -> tuple[Path, Path, Path]:
    route = root / "route"
    scene = root / "scene_teacher"
    spatial = root / "spatial_pack"
    route.mkdir()
    (spatial / "examples").mkdir(parents=True)
    _write_route(route, frame_count=4)
    _write_scene_teacher(scene, frame_count=4, mock=mock)
    _write_spatial_pack(spatial, frame_count=4)
    return route, scene, spatial


def _write_route(route: Path, *, frame_count: int) -> None:
    write_scene_teacher_json(
        route / "route_metadata.json",
        {
            "schema_version": "homebrain.route_source.v0",
            "source_type": "openloris_scene",
            "sequence": "unit_route",
            "scene": "unit",
            "frame_count": frame_count,
            "imported_frame_count": frame_count,
            "has_rgb": True,
            "has_depth": True,
            "has_intrinsics": True,
            "has_camera_to_base_transform": True,
            "has_groundtruth_pose": True,
            "has_odometry": True,
            "has_robot_base_pose": False,
            "has_wheel_odometry": False,
            "has_commands": False,
            "extrinsics_source": "unit_measured_static_tf",
            "robot_frame_truth": True,
            "robot_frame_truth_candidate": True,
            "license_name": "unit_fixture",
            "license_review_status": "unit_test",
            "control_safe": False,
            "product_training_approved": False,
            "associations_file": "openloris_scene_associations.json",
        },
    )
    frames = []
    for frame_id in range(frame_count):
        frames.append(
            {
                "frame_id": frame_id,
                "timestamp_ns": 1_700_000_000_000_000_000 + frame_id,
                "camera_to_base": CAMERA_TO_BASE,
                "base_pose": {"tx": float(frame_id), "ty": 0.0, "tz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
            }
        )
    write_scene_teacher_json(
        route / "openloris_scene_associations.json",
        {
            "schema_version": "homebrain.openloris_scene_associations.v0",
            "source_type": "openloris_scene_associations",
            "source_sequence": "unit_route",
            "camera_to_base": CAMERA_TO_BASE,
            "camera_to_base_source": "unit_measured_static_tf",
            "intrinsics": {"available": True, "fx": 4.0, "fy": 4.0, "cx": 1.5, "cy": 1.5, "depth_scale": 1000.0},
            "robot_frame_truth": True,
            "robot_frame_truth_candidate": True,
            "control_safe": False,
            "frames": frames,
        },
    )


def _write_scene_teacher(scene: Path, *, frame_count: int, mock: bool) -> None:
    frame_records = []
    for frame_id in range(frame_count):
        frame_dir = scene / "frames" / f"d400_color_{frame_id:06d}"
        camera_points, depth = _camera_points_for_frame(frame_id)
        point_map = np.full((4, 4, 3), np.nan, dtype=np.float32)
        depth_map = np.zeros((4, 4), dtype=np.float32)
        confidence = np.zeros((4, 4), dtype=np.float32)
        validity = np.zeros((4, 4), dtype=np.uint8)
        for index, point in enumerate(camera_points):
            row = index // 4
            col = index % 4
            point_map[row, col, :] = point[[2, 0, 1]]
            depth_map[row, col] = depth[index]
            confidence[row, col] = 1.0
            validity[row, col] = 1
        artifacts = {
            "point_map": _artifact(scene, frame_dir / "point_map.npy", point_map),
            "depth": _artifact(scene, frame_dir / "depth.npy", depth_map),
            "confidence": _artifact(scene, frame_dir / "confidence.npy", confidence),
            "validity_mask": _artifact(scene, frame_dir / "validity_mask.npy", validity),
            "intrinsics": _artifact(scene, frame_dir / "intrinsics.npy", np.eye(3, dtype=np.float32)),
        }
        metadata_path = frame_dir / "metadata.json"
        write_scene_teacher_json(
            metadata_path,
            {
                "schema_version": SCENE_TEACHER_PACK_SCHEMA_VERSION,
                "frame_id": frame_id,
                "mock": mock,
                "synthetic": mock,
                "robot_frame_truth": False,
                "moge_robot_frame_truth": False,
                "action_supervision_ok": False,
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
                "product_training_approved": False,
                "raw_pwm_emitted": False,
            },
        )
        frame_records.append(
            {
                "sequence_id": "unit_route",
                "camera_id": "d400_color",
                "frame_id": frame_id,
                "timestamp_ns": 1_700_000_000_000_000_000 + frame_id,
                "width": 4,
                "height": 4,
                "format": "unit",
                "source_data_ref": f"frames/frame_{frame_id:06d}.png",
                "scale_status": "synthetic_metric_test_scale" if mock else "metric",
                "mock": mock,
                "synthetic": mock,
                "real_perception": not mock,
                "robot_frame_truth": False,
                "not_robot_frame_truth": True,
                "action_supervision_ok": False,
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
                "product_training_approved": False,
                "raw_pwm_emitted": False,
                "metadata_path": relative_to_root(metadata_path, scene),
                "metadata_sha256": file_sha256(metadata_path),
                "artifacts": artifacts,
            }
        )
    write_scene_teacher_json(
        scene / "scene_teacher_manifest.json",
        {
            "schema_version": SCENE_TEACHER_PACK_SCHEMA_VERSION,
            "pack_name": "SceneTeacherPack",
            "pack_version": "v0",
            "teacher_name": "moge",
            "teacher_version": "unit",
            "backend": "fake" if mock else "real",
            "model_id": "unit",
            "model_source": "unit",
            "license_review_status": "unit_test",
            "mock": mock,
            "synthetic": mock,
            "real_perception": not mock,
            "deterministic": True,
            "scale_status": "synthetic_metric_test_scale" if mock else "metric",
            "frame_count": frame_count,
            "source_frame_count": frame_count,
            "artifact_kinds": ["point_map", "depth", "confidence", "validity_mask", "intrinsics"],
            "robot_frame_truth": False,
            "moge_robot_frame_truth": False,
            "not_robot_frame_truth": True,
            "action_supervision_ok": False,
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "product_training_approved": False,
            "raw_pwm_emitted": False,
            "frames": frame_records,
            "windows": [],
        },
    )


def _write_spatial_pack(spatial: Path, *, frame_count: int) -> None:
    examples = []
    for frame_id in range(frame_count):
        camera_points, _depth = _camera_points_for_frame(frame_id)
        base_points = _camera_to_base_points(camera_points)
        arrays, _stats, _raw = robot_points_to_bev_arrays(
            forward_m=base_points[:, 0],
            left_m=base_points[:, 1],
            height_m=base_points[:, 2],
            confidence=np.ones((base_points.shape[0],), dtype=np.float32),
            grid_cells=GRID_CONFIG["grid_cell_count"],
            meters_per_cell=GRID_CONFIG["meters_per_cell"],
            grid_size_m=GRID_CONFIG["grid_size_m"],
            floor_height_tol_m=GRID_CONFIG["floor_height_tol_m"],
            obstacle_height_min_m=GRID_CONFIG["obstacle_height_min_m"],
            obstacle_dilation_cells=GRID_CONFIG["obstacle_dilation_cells"],
        )
        example_path = spatial / "examples" / f"d400_color_{frame_id:06d}.npz"
        np.savez(
            example_path,
            frame_id=np.asarray(frame_id),
            bev_free=arrays["bev_free"].astype(np.float32),
            bev_obstacle=arrays["bev_obstacle"].astype(np.float32),
            bev_unknown=arrays["bev_unknown"].astype(np.float32),
            bev_confidence=arrays["bev_confidence"].astype(np.float32),
            robot_frame_truth=np.asarray(True),
            control_safe=np.asarray(False),
        )
        examples.append(
            {
                "sequence_id": "unit_route",
                "camera_id": "d400_color",
                "frame_id": frame_id,
                "timestamp_ns": 1_700_000_000_000_000_000 + frame_id,
                "example_path": relative_to_root(example_path, spatial),
                "robot_frame_truth": True,
                "robot_frame_truth_candidate": True,
                "control_safe": False,
                "weak_label": True,
            }
        )
    write_scene_teacher_json(
        spatial / "manifest.json",
        {
            "schema_version": "homebrain.spatial_dataset.v0",
            "package_type": "SpatialTrainPack",
            "example_count": frame_count,
            "source_family": "public_robot_mounted",
            "dataset_frame_type": "public_robot_mounted",
            "robot_frame_truth": True,
            "control_safe": False,
            "camera_config": dict(GRID_CONFIG),
            "grid_shape": [GRID_CONFIG["grid_cell_count"], GRID_CONFIG["grid_cell_count"]],
            "examples": examples,
        },
    )


def _camera_points_for_frame(frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    base_points = np.asarray(
        [
            [0.25, -0.50, 0.00],
            [0.50, 0.00, 0.00],
            [0.75, 0.50, 0.00],
            [1.00, -0.25, 0.00],
            [1.25, 0.00, 0.22],
            [1.50, 0.50, 0.24],
        ],
        dtype=np.float32,
    )
    base_points[:, 1] += np.float32(0.01 * frame_id)
    camera_points = np.zeros_like(base_points)
    camera_points[:, 2] = base_points[:, 0]
    camera_points[:, 0] = -base_points[:, 1]
    camera_points[:, 1] = np.float32(0.5) - base_points[:, 2]
    return camera_points, camera_points[:, 2].copy()


def _camera_to_base_points(camera_points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(CAMERA_TO_BASE, dtype=np.float32)
    return camera_points @ matrix[:3, :3].T + matrix[:3, 3][None, :]


def _artifact(root: Path, path: Path, array: np.ndarray) -> dict[str, object]:
    record = save_array(path, array)
    return {"kind": path.stem, "path": relative_to_root(path, root), **record}


def _assert_safety_flags(report: dict[str, object]) -> None:
    safety = report["safety_flags"]
    assert isinstance(safety, dict)
    assert safety["replay_only"] is True
    assert safety["not_executed"] is True
    assert safety["control_safe"] is False
    assert safety["product_training_approved"] is False
    assert safety["action_supervision_ok"] is False
    assert safety["raw_pwm_emitted"] is False
    assert safety["cmd_vel_emitted"] is False
    assert safety["moge_robot_frame_truth"] is False
    encoded = json.dumps(report)
    assert '"control_safe": true' not in encoded
    assert '"product_training_approved": true' not in encoded
    assert '"action_supervision_ok": true' not in encoded
    assert '"cmd_vel_emitted": true' not in encoded
    assert '"raw_pwm_emitted": true' not in encoded
