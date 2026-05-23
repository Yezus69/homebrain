import json
from pathlib import Path

import numpy as np
import pytest

from homebrain.geometry.camera_config import CameraConfig, write_camera_config
from homebrain.geometry.depth_to_points import project_depth_to_robot_points
from homebrain.geometry.run_depth_to_bev import convert_depth_artifacts_to_bev
from homebrain.geometry.validate_bev import load_bev_manifest, validate_bev_artifacts
from homebrain.geometry.visualize_bev import visualize_bev
from homebrain.messages.schema import FrameEvent, deterministic_json
from homebrain.replay.generate_dummy_log import generate_dummy_log
from homebrain.replay.segment_log import read_events
from homebrain.teachers.artifacts import (
    BEV_PREVIEW_SHAPE,
    DEPTH_PRO_ARTIFACT_KINDS,
    TEACHER_ARTIFACT_SCHEMA_VERSION,
    file_sha256,
    relative_to_root,
    save_array,
    write_json,
    write_teacher_manifest,
)


def _test_config() -> CameraConfig:
    return CameraConfig.from_dict(
        {
            "camera_height_m": 0.5,
            "pitch_deg": 90.0,
            "roll_deg": 0.0,
            "yaw_deg": 0.0,
            "grid_size_m": 2.0,
            "meters_per_cell": 0.5,
            "max_depth_m": 2.0,
            "floor_height_tol_m": 0.08,
            "obstacle_height_min_m": 0.14,
            "robot_radius_m": 0.0,
        }
    )


def _write_config(path: Path) -> None:
    write_camera_config(path, _test_config())


def _frames(route: Path) -> list[FrameEvent]:
    return [event for event in read_events(route) if isinstance(event, FrameEvent)]


def _write_fake_depth_artifacts(route: Path, out: Path) -> None:
    frames = _frames(route)
    frame_records = []
    for frame in frames:
        frame_dir = out / "frames" / f"{frame.camera_id}_{frame.frame_id:06d}"
        depth = np.full((frame.height, frame.width), 0.5 + frame.frame_id * 0.01, dtype=np.float32)
        confidence = np.full((frame.height, frame.width), 0.8, dtype=np.float32)
        focal = np.asarray([1.0], dtype=np.float32)
        bev_preview = np.zeros(BEV_PREVIEW_SHAPE, dtype=np.float32)
        arrays = {
            "depth_m": depth,
            "depth_confidence": confidence,
            "focallength_px": focal,
            "bev_preview": bev_preview,
        }
        artifact_records = {}
        for kind in DEPTH_PRO_ARTIFACT_KINDS:
            target = frame_dir / f"{kind}.npy"
            artifact_records[kind] = {
                "kind": kind,
                "path": relative_to_root(target, out),
                **save_array(target, arrays[kind]),
            }
        metadata_path = frame_dir / "metadata.json"
        write_json(
            metadata_path,
            {
                "schema_version": TEACHER_ARTIFACT_SCHEMA_VERSION,
                "teacher_name": "depth_pro",
                "mock": True,
                "synthetic": True,
                "real_perception": False,
                "frame_id": frame.frame_id,
            },
        )
        frame_records.append(
            {
                "sequence_id": frame.sequence_id,
                "camera_id": frame.camera_id,
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "width": frame.width,
                "height": frame.height,
                "format": frame.format,
                "source_data_ref": frame.data_ref,
                "metadata_path": relative_to_root(metadata_path, out),
                "metadata_sha256": file_sha256(metadata_path),
                "artifacts": artifact_records,
            }
        )

    write_teacher_manifest(
        out,
        {
            "schema_version": TEACHER_ARTIFACT_SCHEMA_VERSION,
            "teacher_name": "depth_pro",
            "teacher_version": "depth_pro.fake_for_geometry_tests",
            "backend": "fake",
            "mock": True,
            "synthetic": True,
            "real_perception": False,
            "deterministic": True,
            "source_log": route.as_posix(),
            "frame_count": len(frame_records),
            "artifact_kinds": list(DEPTH_PRO_ARTIFACT_KINDS),
            "control_safety": "not_control_safe_training_teacher_only",
            "frames": frame_records,
        },
    )


def _build_bev_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    route = tmp_path / "dummy_route"
    depth = tmp_path / "depth_pro"
    config = tmp_path / "camera.json"
    bev = tmp_path / "bev"
    generate_dummy_log(route)
    _write_fake_depth_artifacts(route, depth)
    _write_config(config)
    convert_depth_artifacts_to_bev(
        log_dir=route,
        depth_artifacts_dir=depth,
        camera_config_path=config,
        out_dir=bev,
        pixel_stride=1,
    )
    return route, depth, config, bev


def test_projection_math_sanity_uses_pinhole_and_center_assumption() -> None:
    config = _test_config()
    depth = np.full((3, 3), 2.0, dtype=np.float32)

    points = project_depth_to_robot_points(depth, 2.0, config, principal_point_px=None)

    assert "principal_point_assumed_image_center" in points.assumptions
    center_index = np.where((np.isclose(points.left_m, 0.0)) & (np.isclose(points.depth_m, 2.0)))[0]
    assert center_index.size >= 1
    assert np.isclose(points.principal_point_px[0], 1.0)
    assert np.isclose(points.principal_point_px[1], 1.0)
    assert points.valid_point_count == 9


def test_depth_to_bev_writes_artifacts_and_manifest_round_trip(tmp_path) -> None:
    _route, _depth, _config, bev = _build_bev_fixture(tmp_path)

    manifest = load_bev_manifest(bev)
    assert manifest["weak_label"] is True
    assert manifest["control_safe"] is False
    assert manifest["frame_count"] == 6
    assert manifest["grid_shape"] == [4, 4]

    first_frame = manifest["frames"][0]
    metadata = json.loads((bev / first_frame["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["weak_label"] is True
    assert metadata["control_safe"] is False

    for kind in ("bev_free", "bev_obstacle", "bev_unknown", "bev_floor_candidate"):
        array = np.load(bev / first_frame["artifacts"][kind]["path"])
        assert array.shape == (4, 4)
        assert array.dtype == np.uint8
    assert np.load(bev / first_frame["artifacts"]["bev_height"]["path"]).dtype == np.float32

    round_trip = tmp_path / "round_trip"
    round_trip.mkdir()
    (round_trip / "bev_manifest.json").write_text(deterministic_json(manifest) + "\n", encoding="utf-8")
    assert load_bev_manifest(round_trip) == manifest


def test_validate_bev_detects_missing_and_malformed_artifacts(tmp_path) -> None:
    _route, _depth, _config, bev = _build_bev_fixture(tmp_path)
    manifest = load_bev_manifest(bev)

    missing_file = bev / manifest["frames"][0]["artifacts"]["bev_free"]["path"]
    missing_file.unlink()
    missing = validate_bev_artifacts(bev)
    assert missing.bev_missing_count == 1
    assert missing.load_success is False

    _route, _depth, _config, malformed = _build_bev_fixture(tmp_path / "malformed")
    malformed_manifest = load_bev_manifest(malformed)
    bad_height = malformed / malformed_manifest["frames"][0]["artifacts"]["bev_height"]["path"]
    with bad_height.open("wb") as handle:
        np.save(handle, np.zeros((1, 1), dtype=np.float32), allow_pickle=False)
    validation = validate_bev_artifacts(malformed)
    assert validation.bev_shape_error_count == 1
    assert validation.load_success is False


def test_depth_to_bev_rejects_malformed_source_depth_artifacts(tmp_path) -> None:
    route = tmp_path / "dummy_route"
    depth = tmp_path / "depth_pro"
    config = tmp_path / "camera.json"
    bev = tmp_path / "bev"
    generate_dummy_log(route)
    _write_fake_depth_artifacts(route, depth)
    _write_config(config)

    manifest = json.loads((depth / "teacher_manifest.json").read_text(encoding="utf-8"))
    focal_path = depth / manifest["frames"][0]["artifacts"]["focallength_px"]["path"]
    with focal_path.open("wb") as handle:
        np.save(handle, np.asarray([1.0, 2.0], dtype=np.float32), allow_pickle=False)

    with pytest.raises(ValueError, match="focallength_px"):
        convert_depth_artifacts_to_bev(
            log_dir=route,
            depth_artifacts_dir=depth,
            camera_config_path=config,
            out_dir=bev,
            pixel_stride=1,
        )


def test_visualize_bev_writes_preview_files(tmp_path) -> None:
    _route, _depth, _config, bev = _build_bev_fixture(tmp_path)
    preview = tmp_path / "preview"

    manifest_path = visualize_bev(bev, preview)
    visualization = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert visualization["weak_label"] is True
    assert visualization["control_safe"] is False
    assert visualization["visualization_written"] is True
    first_frame = visualization["frames"][0]
    assert (preview / first_frame["previews"]["bev_labels"]).exists()
    assert (preview / first_frame["previews"]["bev_confidence"]).exists()
    assert (preview / first_frame["previews"]["bev_height"]).exists()


def test_bev_eval_metrics(tmp_path) -> None:
    _route, _depth, _config, bev = _build_bev_fixture(tmp_path)

    validation = validate_bev_artifacts(bev)
    metrics = validation.to_metrics()

    assert metrics["bev_frame_count"] == 6
    assert metrics["bev_missing_count"] == 0
    assert metrics["bev_shape_error_count"] == 0
    assert metrics["bev_nan_count"] == 0
    assert metrics["weak_label"] is True
    assert metrics["control_safe"] is False
    assert 0.0 <= metrics["free_ratio_mean"] <= 1.0
    assert 0.0 <= metrics["obstacle_ratio_mean"] <= 1.0
    assert 0.0 <= metrics["unknown_ratio_mean"] <= 1.0
    assert 0.0 <= metrics["confidence_mean"] <= 1.0
    assert 0.0 <= metrics["temporal_jitter_mean"] <= 1.0
