import json
from pathlib import Path

import numpy as np

from homebrain.messages.schema import FrameEvent
from homebrain.replay.generate_dummy_log import generate_dummy_log
from homebrain.replay.segment_log import read_events
from homebrain.teachers.artifacts import (
    EXPECTED_ARTIFACT_KINDS,
    load_array,
    load_teacher_manifest,
    validate_teacher_artifacts,
    write_teacher_manifest,
)
from homebrain.teachers.base import Teacher, TeacherRunConfig
from homebrain.teachers.mock_teacher import MockTeacher, run_mock_teacher
from homebrain.teachers.visualize_artifacts import visualize_artifacts


def _relative_files(root: Path) -> list[Path]:
    return sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())


def test_mock_teacher_interface_artifact_creation_and_manifest_round_trip(tmp_path) -> None:
    source = tmp_path / "dummy_route"
    artifacts = tmp_path / "teacher_artifacts" / "mock_teacher"
    round_trip = tmp_path / "round_trip"
    generate_dummy_log(source)

    teacher: Teacher = MockTeacher()
    summary = teacher.run(TeacherRunConfig(log_dir=source, out_dir=artifacts))

    assert summary.teacher_name == "mock_teacher"
    assert summary.mock is True
    assert summary.frame_count == 6
    assert summary.manifest_path == artifacts / "teacher_manifest.json"

    manifest = load_teacher_manifest(artifacts)
    write_teacher_manifest(round_trip, manifest)
    assert load_teacher_manifest(round_trip) == manifest
    assert manifest["mock"] is True
    assert manifest["synthetic"] is True
    assert manifest["real_perception"] is False
    assert manifest["artifact_kinds"] == list(EXPECTED_ARTIFACT_KINDS)

    first_frame = manifest["frames"][0]
    metadata = json.loads((artifacts / first_frame["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["mock"] is True
    assert metadata["real_perception"] is False
    assert "not real perception" in metadata["note"]

    depth = load_array(artifacts / first_frame["artifacts"]["depth"]["path"])
    confidence = load_array(artifacts / first_frame["artifacts"]["depth_confidence"]["path"])
    features = load_array(artifacts / first_frame["artifacts"]["dense_features"]["path"])
    dynamic_mask = load_array(artifacts / first_frame["artifacts"]["dynamic_mask"]["path"])
    bev_preview = load_array(artifacts / first_frame["artifacts"]["bev_preview"]["path"])

    assert depth.shape == (2, 2)
    assert confidence.shape == (2, 2)
    assert features.shape == (2, 2, 4)
    assert dynamic_mask.shape == (2, 2)
    assert bev_preview.shape == (16, 16)
    assert depth.dtype == np.float32
    assert dynamic_mask.dtype == np.uint8

    frames = [event for event in read_events(source) if isinstance(event, FrameEvent)]
    validation = validate_teacher_artifacts(artifacts, frames=frames)
    assert validation.teacher_artifact_count == 30
    assert validation.teacher_mock_used is True
    assert validation.frames_with_teacher_artifacts == 6
    assert validation.missing_artifact_count == 0
    assert validation.artifact_shape_error_count == 0
    assert validation.artifact_determinism_pass is True


def test_mock_teacher_is_deterministic_across_runs(tmp_path) -> None:
    source = tmp_path / "dummy_route"
    out_a = tmp_path / "teacher_a"
    out_b = tmp_path / "teacher_b"
    generate_dummy_log(source)

    run_mock_teacher(source, out_a)
    run_mock_teacher(source, out_b)

    assert _relative_files(out_a) == _relative_files(out_b)
    for relative in _relative_files(out_a):
        assert (out_a / relative).read_bytes() == (out_b / relative).read_bytes()


def test_visualize_artifacts_writes_preview_files(tmp_path) -> None:
    source = tmp_path / "dummy_route"
    artifacts = tmp_path / "teacher_artifacts"
    preview = tmp_path / "preview"
    generate_dummy_log(source)
    run_mock_teacher(source, artifacts)

    manifest_path = visualize_artifacts(artifacts, preview)
    visualization = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert visualization["visualization_written"] is True
    assert visualization["mock"] is True
    assert visualization["frame_count"] == 6
    first_previews = visualization["frames"][0]["previews"]
    assert (preview / first_previews["depth"]).exists()
    assert (preview / first_previews["dense_features"]).exists()


def test_validate_teacher_artifacts_detects_missing_and_malformed_files(tmp_path) -> None:
    source = tmp_path / "dummy_route"
    missing_artifacts = tmp_path / "missing_artifacts"
    malformed_artifacts = tmp_path / "malformed_artifacts"
    generate_dummy_log(source)
    run_mock_teacher(source, missing_artifacts)
    run_mock_teacher(source, malformed_artifacts)
    frames = [event for event in read_events(source) if isinstance(event, FrameEvent)]

    missing_manifest = load_teacher_manifest(missing_artifacts)
    missing_depth = missing_artifacts / missing_manifest["frames"][0]["artifacts"]["depth"]["path"]
    missing_depth.unlink()
    missing_validation = validate_teacher_artifacts(missing_artifacts, frames=frames)
    assert missing_validation.frames_with_teacher_artifacts == 5
    assert missing_validation.missing_artifact_count == 1
    assert missing_validation.artifact_determinism_pass is False

    malformed_manifest = load_teacher_manifest(malformed_artifacts)
    malformed_confidence = (
        malformed_artifacts
        / malformed_manifest["frames"][0]["artifacts"]["depth_confidence"]["path"]
    )
    with malformed_confidence.open("wb") as handle:
        np.save(handle, np.zeros((1, 1), dtype=np.float32), allow_pickle=False)
    malformed_validation = validate_teacher_artifacts(malformed_artifacts, frames=frames)
    assert malformed_validation.frames_with_teacher_artifacts == 5
    assert malformed_validation.artifact_shape_error_count == 1
    assert malformed_validation.artifact_determinism_pass is False
