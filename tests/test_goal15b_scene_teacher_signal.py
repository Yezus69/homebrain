import json
from pathlib import Path

import numpy as np
import pytest

from homebrain.ingest.metadata import ROUTE_SOURCE_SCHEMA_VERSION, write_route_metadata
from homebrain.messages.schema import FrameEvent
from homebrain.replay.generate_dummy_log import generate_dummy_log
from homebrain.replay.segment_log import write_segment
from homebrain.teachers.moge_scene_teacher import (
    MoGeSceneTeacherUnavailableError,
    create_moge_scene_teacher,
    run_moge_scene_teacher,
)
from homebrain.teachers.qa_scene_teacher import run_scene_teacher_qa
from homebrain.teachers.run_scene_teacher import main as run_scene_teacher_main
from homebrain.teachers.scene_teacher import SceneTeacherRunConfig, load_scene_teacher_manifest
from homebrain.tools.audit_scene_teacher_signal import audit_scene_teacher_signal


def _owned_frame_route(root: Path, *, frame_count: int = 4) -> Path:
    route = root / "owned_route"
    (route / "frames").mkdir(parents=True)
    events = []
    artifact_files = ["route_metadata.json"]
    for frame_id in range(frame_count):
        data_ref = f"frames/frame_{frame_id:06d}.rgb"
        pixels = np.full((2, 2, 3), frame_id * 17, dtype=np.uint8)
        (route / data_ref).write_bytes(pixels.tobytes(order="C"))
        artifact_files.append(data_ref)
        events.append(
            FrameEvent(
                timestamp_ns=1_700_000_000_000_000_000 + frame_id * 100_000_000,
                sequence_id="owned-route",
                source="owned_fixture_camera",
                camera_id="front_rgb",
                frame_id=frame_id,
                width=2,
                height=2,
                format="rgb8",
                data_ref=data_ref,
                intrinsics={"fx": 2.0, "fy": 2.0, "cx": 0.5, "cy": 0.5},
            )
        )
    write_route_metadata(
        route,
        {
            "schema_version": ROUTE_SOURCE_SCHEMA_VERSION,
            "source_type": "owned_test_fixture",
            "source_path": route.as_posix(),
            "frame_count": frame_count,
            "imported_frame_count": frame_count,
            "has_rgb": True,
            "has_intrinsics": True,
            "has_imu": False,
            "has_wheel_odometry": False,
            "has_odometry": False,
            "has_commands": False,
            "has_camera_to_base_transform": False,
            "has_robot_base_pose": False,
            "owned_or_license_approved": True,
            "user_owned_or_license_unknown": False,
            "robot_frame_truth": False,
            "robot_frame_truth_candidate": False,
            "control_safe": False,
            "missing_sensor_notices": [
                {"sensor": "imu", "status": "unavailable", "reason": "not_measured_by_fixture"},
                {"sensor": "wheel_odometry", "status": "unavailable", "reason": "not_measured_by_fixture"},
                {"sensor": "commands", "status": "unavailable", "reason": "not_measured_by_fixture"},
            ],
        },
    )
    write_segment(route, events, segment_id=route.name, artifact_files=artifact_files)
    return route


def test_fake_moge_scene_teacher_and_signal_audit_are_review_only(tmp_path: Path) -> None:
    route = _owned_frame_route(tmp_path)
    scene = tmp_path / "scene_moge_fake"
    qa_path = tmp_path / "scene_moge_fake_qa.json"
    out_json = tmp_path / "scene_moge_fake_signal.json"
    out_md = tmp_path / "scene_moge_fake_signal.md"

    exit_code = run_scene_teacher_main(
        [
            "--teacher",
            "moge",
            "--backend",
            "fake",
            "--log",
            str(route),
            "--out",
            str(scene),
        ]
    )
    assert exit_code == 0
    run_scene_teacher_qa(artifacts_dir=scene, out_path=qa_path)
    report = audit_scene_teacher_signal(
        log_dir=route,
        scene_teacher_dir=scene,
        qa_path=qa_path,
        out_json=out_json,
        out_md=out_md,
    )

    manifest = load_scene_teacher_manifest(scene)
    assert manifest["teacher_name"] == "moge"
    assert manifest["mock"] is True
    assert manifest["synthetic"] is True
    assert manifest["real_perception"] is False
    assert report["frame_count"] == 4
    assert report["next_allowed_use"] == "review_only"
    assert report["promotable_to_spatial_pack"] is False
    assert report["hard_blockers"] == []
    assert report["route_metadata_sensor_truth"]["truth_pass"] is True
    assert report["route_metadata_sensor_truth"]["owned_or_license_approved"] is True
    assert report["robot_frame_truth"] is False
    assert report["action_supervision_ok"] is False
    assert report["mask_source_distribution"]["floor_traversable_mask_source"]["teacher_output"] == 4
    assert json.loads(out_json.read_text(encoding="utf-8"))["next_allowed_use"] == "review_only"
    assert "next_allowed_use" in out_md.read_text(encoding="utf-8")


def test_signal_audit_blocks_invented_sensor_claims(tmp_path: Path) -> None:
    route = tmp_path / "dummy_route_with_bad_metadata"
    scene = tmp_path / "scene_moge_fake"
    qa_path = tmp_path / "scene_moge_fake_qa.json"
    generate_dummy_log(route)
    write_route_metadata(
        route,
        {
            "schema_version": ROUTE_SOURCE_SCHEMA_VERSION,
            "source_type": "owned_test_fixture",
            "source_path": route.as_posix(),
            "frame_count": 6,
            "imported_frame_count": 6,
            "has_imu": False,
            "has_wheel_odometry": False,
            "has_odometry": False,
            "has_commands": False,
            "has_camera_to_base_transform": False,
            "has_robot_base_pose": False,
            "owned_or_license_approved": True,
            "robot_frame_truth": False,
            "robot_frame_truth_candidate": False,
            "missing_sensor_notices": [],
        },
    )
    run_moge_scene_teacher(route, scene, backend_name="fake")
    run_scene_teacher_qa(artifacts_dir=scene, out_path=qa_path)

    report = audit_scene_teacher_signal(
        log_dir=route,
        scene_teacher_dir=scene,
        qa_path=qa_path,
        out_json=tmp_path / "signal.json",
        out_md=tmp_path / "signal.md",
    )

    assert report["next_allowed_use"] == "blocked"
    blockers = set(report["hard_blockers"])
    assert "imu_events_present_but_metadata_does_not_claim_sensor" in blockers
    assert "wheel_odometry_events_present_but_metadata_does_not_claim_sensor" in blockers
    assert "commands_events_present_but_metadata_does_not_claim_sensor" in blockers


def test_real_moge_backend_fails_clearly_without_local_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    route = _owned_frame_route(tmp_path)
    monkeypatch.delenv("HOMEBRAIN_MOGE_ADAPTER", raising=False)
    monkeypatch.delenv("HOMEBRAIN_MOGE_DIR", raising=False)
    monkeypatch.delenv("HOMEBRAIN_MOGE_CHECKPOINT", raising=False)

    teacher = create_moge_scene_teacher(
        backend_name="real",
        model_dir=tmp_path / "missing_moge",
        checkpoint=tmp_path / "missing_moge.pt",
    )
    with pytest.raises(MoGeSceneTeacherUnavailableError, match="does not clone repositories"):
        teacher.run(SceneTeacherRunConfig(log_dir=route, out_dir=tmp_path / "real_out"))
