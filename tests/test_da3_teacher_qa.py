import json

import numpy as np
import pytest

from homebrain.geometry.da3_to_weak_bev import convert_da3_to_weak_bev
from homebrain.geometry.qa_self_calibration import run_self_calibration_qa
from homebrain.geometry.validate_bev import validate_bev_artifacts
from homebrain.ingest.video import ingest_video
from homebrain.replay.generate_dummy_log import generate_dummy_log
from homebrain.teachers.artifacts import DA3_ARTIFACT_KINDS, load_array, load_teacher_manifest, validate_teacher_artifacts
from homebrain.teachers.run_teacher import main as run_teacher_main


def test_da3_fake_backend_writes_artifacts_and_qa_quarantines_mock(tmp_path) -> None:
    route = tmp_path / "dummy_route"
    artifacts = tmp_path / "teacher_artifacts" / "da3_fake"
    qa_path = tmp_path / "da3_qa.json"
    generate_dummy_log(route)

    exit_code = run_teacher_main(
        [
            "--teacher",
            "da3",
            "--backend",
            "fake",
            "--log",
            str(route),
            "--out",
            str(artifacts),
        ]
    )
    assert exit_code == 0

    manifest = load_teacher_manifest(artifacts)
    assert manifest["teacher_name"] == "da3"
    assert manifest["backend"] == "fake"
    assert manifest["mock"] is True
    assert manifest["synthetic"] is True
    assert manifest["real_perception"] is False
    assert manifest["artifact_kinds"] == list(DA3_ARTIFACT_KINDS)
    assert manifest["license_review_status"] == "pending_human_review"
    assert manifest["not_robot_frame_truth"] is True

    first = manifest["frames"][0]
    depth = load_array(artifacts / first["artifacts"]["depth"]["path"])
    confidence = load_array(artifacts / first["artifacts"]["confidence"]["path"])
    intrinsics = load_array(artifacts / first["artifacts"]["intrinsics"]["path"])
    extrinsics = load_array(artifacts / first["artifacts"]["extrinsics"]["path"])
    assert depth.shape == (2, 2)
    assert confidence.shape == (2, 2)
    assert intrinsics.shape == (3, 3)
    assert extrinsics.shape == (3, 4)

    validation = validate_teacher_artifacts(artifacts)
    assert validation.teacher_artifact_count == 24
    assert validation.artifact_load_success is True
    assert validation.depth_frame_count == 6

    run_self_calibration_qa(log_dir=route, teacher_artifacts_dir=artifacts, out_path=qa_path)
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    assert qa["frame_count"] == 6
    assert qa["depth_valid_ratio"] == 1.0
    assert qa["intrinsics_valid_ratio"] == 1.0
    assert qa["pose_valid_ratio"] == 1.0
    assert qa["promotable_to_weak_bev"] is False
    assert "mock_or_synthetic_teacher" in qa["quarantine_reasons"]
    assert (tmp_path / "da3_qa_contact_sheets" / "contact_sheet_manifest.json").exists()


def test_da3_to_weak_bev_requires_qa_pass_unless_forced(tmp_path) -> None:
    route = tmp_path / "dummy_route"
    artifacts = tmp_path / "teacher_artifacts" / "da3_fake"
    qa_path = tmp_path / "da3_qa.json"
    bev = tmp_path / "weak_bev"
    generate_dummy_log(route)
    run_teacher_main(["--teacher", "da3", "--backend", "fake", "--log", str(route), "--out", str(artifacts)])
    run_self_calibration_qa(log_dir=route, teacher_artifacts_dir=artifacts, out_path=qa_path)

    with pytest.raises(ValueError, match="QA did not approve"):
        convert_da3_to_weak_bev(log_dir=route, teacher_artifacts_dir=artifacts, qa_path=qa_path, out_dir=bev)

    manifest_path = convert_da3_to_weak_bev(
        log_dir=route,
        teacher_artifacts_dir=artifacts,
        qa_path=qa_path,
        out_dir=bev,
        force_review=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["weak_label"] is True
    assert manifest["control_safe"] is False
    assert manifest["not_robot_frame_truth"] is True
    assert manifest["trainable_for"] == "geometry_pretrain_only"
    validation = validate_bev_artifacts(bev)
    assert validation.load_success is True


def test_video_ingest_uses_cv2_without_intrinsics_or_pose(monkeypatch, tmp_path) -> None:
    video = tmp_path / "walk.mp4"
    route = tmp_path / "video_route"
    video.write_bytes(b"not a real video; fake cv2 handles it")

    class FakeCapture:
        def __init__(self, _path):
            self.index = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == FakeCv2.CAP_PROP_FPS:
                return 10.0
            if prop == FakeCv2.CAP_PROP_FRAME_COUNT:
                return 3
            return 0

        def read(self):
            if self.index >= 3:
                return False, None
            value = self.index * 40
            self.index += 1
            return True, np.full((2, 3, 3), value, dtype=np.uint8)

        def release(self):
            return None

    class FakeCv2:
        CAP_PROP_FPS = 5
        CAP_PROP_FRAME_COUNT = 7
        VideoCapture = FakeCapture

        @staticmethod
        def imwrite(path, frame):
            with open(path, "wb") as handle:
                handle.write(frame.tobytes())
            return True

    monkeypatch.setattr("homebrain.ingest.video._load_cv2", lambda: FakeCv2)
    summary = ingest_video(video_path=video, out_dir=route, camera_name="front_rgb", fps=10, max_frames=2)

    assert summary.frame_count == 2
    metadata = json.loads((route / "route_metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_type"] == "video"
    assert metadata["calibration_class"] == "uncalibrated_visual"
    assert metadata["has_intrinsics"] is False
    assert metadata["pose_source"] == "missing_not_supplied"
