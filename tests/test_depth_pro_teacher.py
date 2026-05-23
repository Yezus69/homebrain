import json
from pathlib import Path

import numpy as np

from homebrain.eval.run_eval import evaluate_log
from homebrain.messages.schema import FrameEvent
from homebrain.replay.generate_dummy_log import generate_dummy_log
from homebrain.replay.segment_log import read_events
from homebrain.teachers.artifacts import (
    DEPTH_PRO_ARTIFACT_KINDS,
    load_array,
    load_teacher_manifest,
    validate_teacher_artifacts,
)
from homebrain.teachers.depth_pro_teacher import (
    _resolve_depth_pro_checkpoint_uri,
    run_depth_pro_teacher,
)
from homebrain.teachers.registry import create_teacher
from homebrain.teachers.run_teacher import main as run_teacher_main
from homebrain.teachers.visualize_artifacts import visualize_artifacts


def _frames(route: Path) -> list[FrameEvent]:
    return [event for event in read_events(route) if isinstance(event, FrameEvent)]


def test_depth_pro_registry_and_fake_cli_backend_work(tmp_path) -> None:
    source = tmp_path / "dummy_route"
    artifacts = tmp_path / "depth_pro_artifacts"
    generate_dummy_log(source)

    teacher = create_teacher("depth_pro", backend_name="fake")
    assert teacher.name == "depth_pro"

    exit_code = run_teacher_main(
        [
            "--teacher",
            "depth_pro",
            "--backend",
            "fake",
            "--log",
            str(source),
            "--out",
            str(artifacts),
        ]
    )
    assert exit_code == 0
    assert (artifacts / "teacher_manifest.json").exists()


def test_depth_pro_checkpoint_resolution_prefers_env(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "custom_depth_pro.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setenv("DEPTH_PRO_CHECKPOINT", str(checkpoint))

    class DummyDepthPro:
        __file__ = str(tmp_path / "ml-depth-pro" / "src" / "depth_pro" / "__init__.py")

    assert _resolve_depth_pro_checkpoint_uri(DummyDepthPro) == str(checkpoint)


def test_depth_pro_checkpoint_resolution_finds_editable_repo_checkpoint(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "ml-depth-pro"
    package_dir = repo / "src" / "depth_pro"
    package_dir.mkdir(parents=True)
    checkpoint = repo / "checkpoints" / "depth_pro.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.delenv("DEPTH_PRO_CHECKPOINT", raising=False)

    class DummyDepthPro:
        __file__ = str(package_dir / "__init__.py")

    assert _resolve_depth_pro_checkpoint_uri(DummyDepthPro) == str(checkpoint)


def test_fake_depth_pro_backend_writes_artifacts_and_manifest(tmp_path) -> None:
    source = tmp_path / "dummy_route"
    artifacts = tmp_path / "teacher_artifacts" / "depth_pro"
    generate_dummy_log(source)

    summary = run_depth_pro_teacher(source, artifacts, backend_name="fake")

    assert summary.teacher_name == "depth_pro"
    assert summary.mock is True
    assert summary.frame_count == 6

    manifest = load_teacher_manifest(artifacts)
    assert manifest["teacher_name"] == "depth_pro"
    assert manifest["backend"] == "fake"
    assert manifest["mock"] is True
    assert manifest["synthetic"] is True
    assert manifest["real_perception"] is False
    assert manifest["license_review_status"] == "pending_human_review"
    assert manifest["dependency_status"]["backend"] == "fake"
    assert manifest["dependency_status"]["downloads_attempted_by_homebrain"] is False
    assert manifest["artifact_kinds"] == list(DEPTH_PRO_ARTIFACT_KINDS)

    first_frame = manifest["frames"][0]
    metadata = json.loads((artifacts / first_frame["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["teacher_name"] == "depth_pro"
    assert metadata["mock"] is True
    assert "not_control_safe" in metadata["control_safety"]
    assert metadata["depth_confidence_source"] == "homebrain_heuristic_from_depth_finiteness_and_edges"

    depth = load_array(artifacts / first_frame["artifacts"]["depth_m"]["path"])
    confidence = load_array(artifacts / first_frame["artifacts"]["depth_confidence"]["path"])
    focallength = load_array(artifacts / first_frame["artifacts"]["focallength_px"]["path"])
    bev_preview = load_array(artifacts / first_frame["artifacts"]["bev_preview"]["path"])

    assert depth.shape == (2, 2)
    assert confidence.shape == (2, 2)
    assert focallength.shape == (1,)
    assert bev_preview.shape == (16, 16)
    assert depth.dtype == np.float32
    assert float(depth.min()) > 0.0

    validation = validate_teacher_artifacts(artifacts, frames=_frames(source))
    assert validation.teacher_artifact_count == 24
    assert validation.teacher_mock_used is True
    assert validation.artifact_load_success is True
    assert validation.depth_frame_count == 6
    assert validation.depth_missing_count == 0
    assert validation.depth_nan_count == 0
    assert validation.depth_nonpositive_count == 0
    assert validation.depth_shape_error_count == 0


def test_eval_reports_depth_pro_depth_quality_counts(tmp_path) -> None:
    source = tmp_path / "dummy_route"
    missing_artifacts = tmp_path / "missing_depth"
    malformed_artifacts = tmp_path / "malformed_depth"
    invalid_artifacts = tmp_path / "invalid_depth_values"
    generate_dummy_log(source)

    run_depth_pro_teacher(source, missing_artifacts, backend_name="fake")
    run_depth_pro_teacher(source, malformed_artifacts, backend_name="fake")
    run_depth_pro_teacher(source, invalid_artifacts, backend_name="fake")

    missing_manifest = load_teacher_manifest(missing_artifacts)
    missing_depth = missing_artifacts / missing_manifest["frames"][0]["artifacts"]["depth_m"]["path"]
    missing_depth.unlink()
    missing_metrics = evaluate_log(source, teacher_artifacts=missing_artifacts, eval_runtime_sec=0.0)
    assert missing_metrics["depth_frame_count"] == 5
    assert missing_metrics["depth_missing_count"] == 1
    assert missing_metrics["artifact_load_success"] is False

    malformed_manifest = load_teacher_manifest(malformed_artifacts)
    malformed_depth = malformed_artifacts / malformed_manifest["frames"][0]["artifacts"]["depth_m"]["path"]
    with malformed_depth.open("wb") as handle:
        np.save(handle, np.zeros((1, 1), dtype=np.float32), allow_pickle=False)
    malformed_metrics = evaluate_log(source, teacher_artifacts=malformed_artifacts, eval_runtime_sec=0.0)
    assert malformed_metrics["depth_shape_error_count"] == 1
    assert malformed_metrics["artifact_load_success"] is False

    invalid_manifest = load_teacher_manifest(invalid_artifacts)
    invalid_depth = invalid_artifacts / invalid_manifest["frames"][0]["artifacts"]["depth_m"]["path"]
    bad_depth = np.ones((2, 2), dtype=np.float32)
    bad_depth[0, 0] = np.nan
    bad_depth[0, 1] = -1.0
    with invalid_depth.open("wb") as handle:
        np.save(handle, bad_depth, allow_pickle=False)
    invalid_metrics = evaluate_log(source, teacher_artifacts=invalid_artifacts, eval_runtime_sec=0.0)
    assert invalid_metrics["depth_nan_count"] == 1
    assert invalid_metrics["depth_nonpositive_count"] == 1
    assert invalid_metrics["artifact_determinism_pass"] is False


def test_visualize_depth_pro_artifacts_writes_previews(tmp_path) -> None:
    source = tmp_path / "dummy_route"
    artifacts = tmp_path / "teacher_artifacts" / "depth_pro"
    preview = tmp_path / "preview"
    generate_dummy_log(source)
    run_depth_pro_teacher(source, artifacts, backend_name="fake")

    manifest_path = visualize_artifacts(artifacts, preview)
    visualization = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert visualization["teacher_name"] == "depth_pro"
    assert visualization["visualization_written"] is True
    assert visualization["frame_count"] == 6
    first_frame = visualization["frames"][0]
    assert (preview / first_frame["previews"]["depth_m"]).exists()
    assert (preview / first_frame["previews"]["depth_confidence"]).exists()
    assert (preview / first_frame["previews"]["bev_preview"]).exists()
    assert "focallength_px" in first_frame["scalar_values"]
