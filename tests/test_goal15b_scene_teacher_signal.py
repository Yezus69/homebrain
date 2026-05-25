import json
from pathlib import Path
import sys
import types

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
from homebrain.teachers.scene_teacher import (
    SceneTeacherRunConfig,
    load_scene_teacher_manifest,
    write_scene_teacher_json,
)
from homebrain.tools.audit_scene_teacher_signal import audit_scene_teacher_signal
from homebrain.tools.compare_moge_scene_to_spatial_pack import (
    compare_moge_scene_to_spatial_pack,
    write_aggregate_report,
)
from homebrain.tools.run_owned_geometry_probe import (
    STATUS_BLOCKED_MISSING_TEACHER_SETUP,
    STATUS_REVIEW_ONLY_NOT_TRAINABLE,
    main as probe_main,
    run_owned_geometry_probe,
)


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


def _robot_frame_route(root: Path, *, frame_count: int = 4) -> Path:
    route = _owned_frame_route(root, frame_count=frame_count)
    write_route_metadata(
        route,
        {
            "schema_version": ROUTE_SOURCE_SCHEMA_VERSION,
            "source_type": "openloris_scene",
            "source_path": route.as_posix(),
            "frame_count": frame_count,
            "imported_frame_count": frame_count,
            "has_rgb": True,
            "has_intrinsics": True,
            "has_depth": True,
            "has_imu": True,
            "has_wheel_odometry": False,
            "has_odometry": True,
            "has_groundtruth_pose": True,
            "has_commands": False,
            "has_camera_to_base_transform": True,
            "has_robot_base_pose": False,
            "extrinsics_source": "openloris_static_tf",
            "owned_or_license_approved": True,
            "user_owned_or_license_unknown": False,
            "robot_frame_truth": True,
            "robot_frame_truth_candidate": True,
            "control_safe": False,
            "license_name": "CC BY-ND 4.0",
            "license_review_status": "pending_human_review",
            "sequence": "unit_openloris",
            "scene": "unit",
            "missing_sensor_notices": [],
        },
    )
    return route


def _owned_probe_frames(root: Path, *, frame_count: int = 4) -> Path:
    frames = root / "probe_frames"
    frames.mkdir(parents=True)
    for frame_id in range(frame_count):
        pixels = np.zeros((3, 4, 3), dtype=np.uint8)
        pixels[:, :, 0] = np.uint8(20 + frame_id * 15)
        pixels[:, :, 1] = np.uint8(50 + frame_id * 10)
        pixels[:, :, 2] = np.uint8(90 + frame_id * 5)
        path = frames / f"frame_{frame_id:06d}.ppm"
        with path.open("wb") as handle:
            handle.write(b"P6\n4 3\n255\n")
            handle.write(pixels.tobytes(order="C"))
    return frames


def _spatial_pack_fixture(root: Path, *, frame_count: int = 4, robot_frame_truth: bool = True) -> Path:
    pack = root / "spatial_pack"
    examples_dir = pack / "examples"
    examples_dir.mkdir(parents=True)
    examples = []
    for frame_id in range(frame_count):
        free = np.zeros((4, 4), dtype=np.float32)
        obstacle = np.zeros((4, 4), dtype=np.float32)
        unknown = np.ones((4, 4), dtype=np.float32)
        free[:2, :] = 1.0
        obstacle[3, :2] = 1.0
        unknown[free > 0.5] = 0.0
        unknown[obstacle > 0.5] = 0.0
        example_path = f"examples/d400_color_{frame_id:06d}.npz"
        np.savez(
            pack / example_path,
            frame_id=np.asarray(frame_id),
            bev_free=free,
            bev_obstacle=obstacle,
            bev_unknown=unknown,
            bev_confidence=np.ones((4, 4), dtype=np.float32),
            robot_frame_truth=np.asarray(robot_frame_truth),
            control_safe=np.asarray(False),
        )
        examples.append(
            {
                "frame_id": frame_id,
                "example_path": example_path,
                "robot_frame_truth": robot_frame_truth,
                "robot_frame_truth_candidate": robot_frame_truth,
                "control_safe": False,
                "weak_label": True,
                "source_bev_stats": {
                    "free_ratio": float(free.mean()),
                    "obstacle_ratio": float(obstacle.mean()),
                    "unknown_ratio": float(unknown.mean()),
                    "confidence_mean": 1.0,
                },
            }
        )
    write_scene_teacher_json(
        pack / "manifest.json",
        {
            "schema_version": "homebrain.spatial_dataset.v0",
            "example_count": frame_count,
            "dataset_frame_type": "public_robot_mounted",
            "control_safe": False,
            "examples": examples,
        },
    )
    return pack


def _mark_scene_teacher_real_like(scene: Path) -> None:
    manifest = load_scene_teacher_manifest(scene)
    manifest.update(
        {
            "backend": "real",
            "mock": False,
            "synthetic": False,
            "real_perception": True,
            "scale_status": "metric",
            "robot_frame_truth": False,
            "action_supervision_ok": False,
        }
    )
    for frame in manifest["frames"]:
        frame["scale_status"] = "metric"
    write_scene_teacher_json(scene / "scene_teacher_manifest.json", manifest)


def _assert_no_motion_or_training_claims(root: Path) -> None:
    for path in root.rglob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, value in item.items():
                    if key in {"cmd_vel", "cmd_vel_emitted"}:
                        assert value in {None, False, 0}
                    if key in {"raw_pwm", "raw_pwm_emitted"}:
                        assert value in {None, False, 0}
                    if key in {"control_safe", "control_safe_claim"}:
                        assert value is False
                    if key == "product_training_approved":
                        assert value is False
                    stack.append(value)
            elif isinstance(item, list):
                stack.extend(item)


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


def test_owned_geometry_probe_fake_backend_is_review_only_not_trainable(tmp_path: Path) -> None:
    frames = _owned_probe_frames(tmp_path)
    out = tmp_path / "probe_fake"

    result = run_owned_geometry_probe(
        frames_dir=frames,
        out_dir=out,
        camera_name="front_rgb",
        fps=10.0,
        teacher_name="moge",
        backend_name="fake",
        owned_or_license_approved=True,
        max_frames=3,
    )

    assert result["status"] == STATUS_REVIEW_ONLY_NOT_TRAINABLE
    assert result["steps"]["image_sequence_ingest"]["ok"] is True
    assert result["steps"]["scene_teacher_run"]["ok"] is True
    assert result["steps"]["scene_teacher_qa"]["attempted"] is True
    assert result["steps"]["signal_audit"]["attempted"] is True
    assert result["steps"]["signal_audit"]["next_allowed_use"] == "review_only"
    assert result["steps"]["visual_review"]["attempted"] is True
    assert result["visual_review_exists"] is True
    assert (out / "result.json").exists()
    assert (out / "result.md").exists()
    assert (out / "visual_review" / "scene_teacher_review.ppm").exists()
    assert json.loads((out / "result.json").read_text(encoding="utf-8"))["status"] == STATUS_REVIEW_ONLY_NOT_TRAINABLE
    _assert_no_motion_or_training_claims(out)


def test_owned_geometry_probe_missing_real_moge_setup_blocks_before_qa_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = _owned_probe_frames(tmp_path)
    out = tmp_path / "probe_real_missing"
    monkeypatch.delenv("HOMEBRAIN_MOGE_ADAPTER", raising=False)
    monkeypatch.delenv("HOMEBRAIN_MOGE_DIR", raising=False)
    monkeypatch.delenv("HOMEBRAIN_MOGE_CHECKPOINT", raising=False)
    monkeypatch.delenv("HOMEBRAIN_MOGE_MODEL_ID", raising=False)
    monkeypatch.delenv("HOMEBRAIN_MOGE_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.setitem(sys.modules, "moge", None)
    monkeypatch.delitem(sys.modules, "moge.model", raising=False)
    monkeypatch.delitem(sys.modules, "moge.model.v2", raising=False)

    result = run_owned_geometry_probe(
        frames_dir=frames,
        out_dir=out,
        camera_name="front_rgb",
        fps=10.0,
        teacher_name="moge",
        backend_name="real",
        owned_or_license_approved=True,
        model_dir=tmp_path / "missing_moge",
        checkpoint=tmp_path / "missing_moge.pt",
        max_frames=2,
    )

    assert result["status"] == STATUS_BLOCKED_MISSING_TEACHER_SETUP
    assert result["fake_fallback_used"] is False
    assert result["steps"]["scene_teacher_run"]["attempted"] is True
    assert result["steps"]["scene_teacher_run"]["ok"] is False
    assert result["teacher_artifacts_exist"] is False
    assert result["steps"]["scene_teacher_qa"]["attempted"] is False
    assert result["steps"]["signal_audit"]["attempted"] is False
    assert result["steps"]["visual_review"]["attempted"] is False
    assert not Path(result["paths"]["scene_teacher_manifest"]).exists()
    assert not Path(result["paths"]["qa_path"]).exists()
    assert not Path(result["paths"]["audit_json_path"]).exists()
    missing_fields = {item["field"] for item in result["missing_setup_fields"]}
    assert {"moge_import", "model_source"} <= missing_fields
    assert "MoGe is not installed or importable" in result["steps"]["scene_teacher_run"]["error"]
    assert result["retry_command"].startswith("python -m homebrain.tools.run_owned_geometry_probe")
    assert json.loads((out / "result.json").read_text(encoding="utf-8"))["status"] == STATUS_BLOCKED_MISSING_TEACHER_SETUP
    _assert_no_motion_or_training_claims(out)


def test_owned_geometry_probe_cli_blocked_status_is_nonzero_unless_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = _owned_probe_frames(tmp_path)
    monkeypatch.delenv("HOMEBRAIN_MOGE_ADAPTER", raising=False)
    monkeypatch.delenv("HOMEBRAIN_MOGE_DIR", raising=False)
    monkeypatch.delenv("HOMEBRAIN_MOGE_CHECKPOINT", raising=False)
    monkeypatch.delenv("HOMEBRAIN_MOGE_MODEL_ID", raising=False)
    monkeypatch.delenv("HOMEBRAIN_MOGE_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.setitem(sys.modules, "moge", None)
    monkeypatch.delitem(sys.modules, "moge.model", raising=False)
    monkeypatch.delitem(sys.modules, "moge.model.v2", raising=False)

    base_args = [
        "--frames",
        str(frames),
        "--camera",
        "front_rgb",
        "--fps",
        "10",
        "--teacher",
        "moge",
        "--backend",
        "real",
        "--owned-or-license-approved",
        "--max-frames",
        "2",
    ]

    blocked_out = tmp_path / "probe_cli_blocked"
    assert probe_main([*base_args, "--out", str(blocked_out)]) == 1
    assert json.loads((blocked_out / "result.json").read_text(encoding="utf-8"))["status"] == STATUS_BLOCKED_MISSING_TEACHER_SETUP

    allowed_out = tmp_path / "probe_cli_blocked_allowed"
    assert probe_main([*base_args, "--out", str(allowed_out), "--allow-blocked-exit-zero"]) == 0
    assert json.loads((allowed_out / "result.json").read_text(encoding="utf-8"))["status"] == STATUS_BLOCKED_MISSING_TEACHER_SETUP


def test_owned_geometry_probe_real_moge_official_adapter_with_monkeypatched_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = _owned_probe_frames(tmp_path, frame_count=3)
    out = tmp_path / "probe_real_monkeypatched_moge"
    loaded_sources: list[str] = []

    class FakeMoGeModel:
        @classmethod
        def from_pretrained(cls, source: str) -> "FakeMoGeModel":
            loaded_sources.append(source)
            return cls()

        def to(self, _device: str) -> "FakeMoGeModel":
            return self

        def eval(self) -> "FakeMoGeModel":
            return self

        def infer(self, input_image: object) -> dict[str, np.ndarray]:
            shape = getattr(input_image, "shape")
            height = int(shape[1])
            width = int(shape[2])
            row, col = np.indices((height, width), dtype=np.float32)
            depth = np.float32(1.0) + row * np.float32(0.1) + col * np.float32(0.01)
            points = np.stack([col * np.float32(0.01), row * np.float32(0.01), depth], axis=2)
            return {
                "depth": depth,
                "points": points,
                "mask": np.ones((height, width), dtype=bool),
                "intrinsics": np.asarray(
                    [[4.0, 0.0, 1.5], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
            }

    moge_module = types.ModuleType("moge")
    moge_model_module = types.ModuleType("moge.model")
    moge_v2_module = types.ModuleType("moge.model.v2")
    moge_v2_module.MoGeModel = FakeMoGeModel
    moge_module.model = moge_model_module
    moge_model_module.v2 = moge_v2_module
    monkeypatch.setitem(sys.modules, "moge", moge_module)
    monkeypatch.setitem(sys.modules, "moge.model", moge_model_module)
    monkeypatch.setitem(sys.modules, "moge.model.v2", moge_v2_module)
    monkeypatch.delenv("HOMEBRAIN_MOGE_ADAPTER", raising=False)
    monkeypatch.delenv("HOMEBRAIN_MOGE_CHECKPOINT", raising=False)
    monkeypatch.delenv("HOMEBRAIN_MOGE_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.setenv("HOMEBRAIN_MOGE_MODEL_ID", "unit-test/local-moge")

    result = run_owned_geometry_probe(
        frames_dir=frames,
        out_dir=out,
        camera_name="front_rgb",
        fps=10.0,
        teacher_name="moge",
        backend_name="real",
        owned_or_license_approved=True,
        max_frames=2,
        device="cpu",
    )

    assert loaded_sources == ["unit-test/local-moge"]
    assert result["status"] == "SINGLE_FRAME_GEOMETRY_PRETRAIN_CANDIDATE"
    assert result["teacher_artifacts_exist"] is True
    assert result["qa_exists"] is True
    assert result["audit_exists"] is True
    assert result["visual_review_exists"] is True
    assert result["steps"]["scene_teacher_qa"]["attempted"] is True
    assert result["steps"]["signal_audit"]["attempted"] is True
    assert result["steps"]["visual_review"]["attempted"] is True
    manifest = load_scene_teacher_manifest(Path(result["paths"]["teacher_artifacts_dir"]))
    assert manifest["backend"] == "real"
    assert manifest["mock"] is False
    assert manifest["real_perception"] is True
    assert manifest["scale_status"] == "metric"
    assert manifest["model_id"] == "unit-test/local-moge"
    first_frame = manifest["frames"][0]
    frame_metadata = json.loads((Path(result["paths"]["teacher_artifacts_dir"]) / first_frame["metadata_path"]).read_text())
    assert frame_metadata["backend_metadata"]["adapter"] == "homebrain.teachers.moge_official_adapter:run_scene_teacher"
    assert frame_metadata["backend_metadata"]["scale_status"] == "metric"
    assert frame_metadata["backend_metadata"]["dynamic_motion_mask_source"] == "unavailable_static_zero_placeholder"
    visual_manifest = json.loads((out / "visual_review" / "visual_review_manifest.json").read_text(encoding="utf-8"))
    assert visual_manifest["frames"][0]["panels"][0] == "source_rgb"
    assert (out / "visual_review" / "scene_teacher_review.ppm").exists()
    _assert_no_motion_or_training_claims(out)


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


def test_signal_audit_allows_single_frame_geometry_without_temporal_extrinsics(tmp_path: Path) -> None:
    route = _owned_frame_route(tmp_path)
    scene = tmp_path / "scene_moge_real_like_single_frame"
    qa_path = tmp_path / "scene_moge_real_like_single_frame_qa.json"
    run_moge_scene_teacher(route, scene, backend_name="fake")
    manifest = load_scene_teacher_manifest(scene)
    manifest.update(
        {
            "backend": "real",
            "mock": False,
            "synthetic": False,
            "real_perception": True,
            "scale_status": "metric",
        }
    )
    manifest["artifact_kinds"] = [kind for kind in manifest["artifact_kinds"] if kind != "extrinsics"]
    for frame in manifest["frames"]:
        frame["scale_status"] = "metric"
        frame["artifacts"].pop("extrinsics", None)
        frame["pose_source"] = "missing_from_teacher"
    write_scene_teacher_json(scene / "scene_teacher_manifest.json", manifest)
    write_scene_teacher_json(
        qa_path,
        {
            "schema_version": "homebrain.scene_teacher_qa.v0",
            "backend": "real",
            "mock": False,
            "synthetic": False,
            "real_perception": True,
            "frame_count": 4,
            "missing_artifact_count": 0,
            "artifact_shape_error_count": 0,
            "depth_valid_ratio": 1.0,
            "confidence_valid_ratio": 1.0,
            "pose_valid_ratio": 0.0,
            "track_valid_ratio": 0.0,
            "temporal_geometry_consistency": 0.0,
            "scale_status": "metric",
            "control_safe": False,
            "promotable_to_spatial_pack": False,
            "quarantine_reasons": ["low_pose_valid_ratio"],
            "errors": [],
        },
    )

    report = audit_scene_teacher_signal(
        log_dir=route,
        scene_teacher_dir=scene,
        qa_path=qa_path,
        out_json=tmp_path / "signal.json",
        out_md=tmp_path / "signal.md",
    )

    assert report["next_allowed_use"] == "single_frame_geometry_pretrain_candidate"
    assert report["single_frame_geometry_pretrain_candidate"] is True
    assert report["temporal_memory_pretrain_candidate"] is False
    assert report["temporal_memory_evidence"]["pass"] is False
    assert report["hard_blockers"] == []


def test_signal_audit_allows_public_poc_route_without_owned_flag(tmp_path: Path) -> None:
    route = _owned_frame_route(tmp_path)
    metadata = json.loads((route / "route_metadata.json").read_text(encoding="utf-8"))
    metadata.update(
        {
            "source_type": "openloris_scene",
            "owned_or_license_approved": False,
            "poc_training_eval_allowed": True,
            "usage_policy": {
                "dataset_name": "OpenLORIS-Scene",
                "poc_training_eval_allowed": True,
                "product_training_approved": False,
            },
            "license_name": "CC BY-ND 4.0",
            "license_review_status": "poc_allowed_product_review_later",
        }
    )
    write_route_metadata(route, metadata)
    scene = tmp_path / "scene_moge_real_like_public_poc"
    qa_path = tmp_path / "scene_moge_real_like_public_poc_qa.json"
    run_moge_scene_teacher(route, scene, backend_name="fake")
    manifest = load_scene_teacher_manifest(scene)
    manifest.update(
        {
            "backend": "real",
            "mock": False,
            "synthetic": False,
            "real_perception": True,
            "scale_status": "metric",
        }
    )
    for frame in manifest["frames"]:
        frame["scale_status"] = "metric"
    write_scene_teacher_json(scene / "scene_teacher_manifest.json", manifest)
    write_scene_teacher_json(
        qa_path,
        {
            "schema_version": "homebrain.scene_teacher_qa.v0",
            "backend": "real",
            "mock": False,
            "synthetic": False,
            "real_perception": True,
            "frame_count": 4,
            "missing_artifact_count": 0,
            "artifact_shape_error_count": 0,
            "depth_valid_ratio": 1.0,
            "confidence_valid_ratio": 1.0,
            "pose_valid_ratio": 0.0,
            "track_valid_ratio": 0.0,
            "temporal_geometry_consistency": 0.0,
            "scale_status": "metric",
            "control_safe": False,
            "promotable_to_spatial_pack": False,
            "quarantine_reasons": [],
            "errors": [],
        },
    )

    report = audit_scene_teacher_signal(
        log_dir=route,
        scene_teacher_dir=scene,
        qa_path=qa_path,
        out_json=tmp_path / "signal.json",
        out_md=tmp_path / "signal.md",
    )

    route_truth = report["route_metadata_sensor_truth"]
    assert route_truth["owned_or_license_approved"] is False
    assert route_truth["poc_training_eval_allowed"] is True
    assert route_truth["poc_or_owned_data_allowed"] is True
    assert route_truth["truth_pass"] is True
    assert report["next_allowed_use"] == "single_frame_geometry_pretrain_candidate"


def test_signal_audit_promotes_temporal_memory_only_with_pose_evidence(tmp_path: Path) -> None:
    route = _owned_frame_route(tmp_path)
    scene = tmp_path / "scene_moge_real_like_temporal"
    qa_path = tmp_path / "scene_moge_real_like_temporal_qa.json"
    run_moge_scene_teacher(route, scene, backend_name="fake")
    manifest = load_scene_teacher_manifest(scene)
    manifest.update(
        {
            "backend": "real",
            "mock": False,
            "synthetic": False,
            "real_perception": True,
            "scale_status": "metric",
        }
    )
    for frame in manifest["frames"]:
        frame["scale_status"] = "metric"
    write_scene_teacher_json(scene / "scene_teacher_manifest.json", manifest)
    write_scene_teacher_json(
        qa_path,
        {
            "schema_version": "homebrain.scene_teacher_qa.v0",
            "backend": "real",
            "mock": False,
            "synthetic": False,
            "real_perception": True,
            "frame_count": 4,
            "missing_artifact_count": 0,
            "artifact_shape_error_count": 0,
            "depth_valid_ratio": 1.0,
            "confidence_valid_ratio": 1.0,
            "pose_valid_ratio": 1.0,
            "track_valid_ratio": 0.0,
            "temporal_geometry_consistency": 0.99,
            "scale_status": "metric",
            "control_safe": False,
            "promotable_to_spatial_pack": True,
            "quarantine_reasons": [],
            "errors": [],
        },
    )

    report = audit_scene_teacher_signal(
        log_dir=route,
        scene_teacher_dir=scene,
        qa_path=qa_path,
        out_json=tmp_path / "signal.json",
        out_md=tmp_path / "signal.md",
    )

    assert report["next_allowed_use"] == "temporal_memory_pretrain_candidate"
    assert report["single_frame_geometry_pretrain_candidate"] is True
    assert report["temporal_memory_pretrain_candidate"] is True
    assert report["temporal_memory_evidence"]["teacher_temporal_extrinsics"] is True


def test_goal17a_compare_missing_route_or_spatial_pack_writes_clear_blocker(tmp_path: Path) -> None:
    route = _robot_frame_route(tmp_path)
    scene = tmp_path / "scene_moge_fake"
    run_moge_scene_teacher(route, scene, backend_name="fake")

    report = compare_moge_scene_to_spatial_pack(
        route_dir=tmp_path / "missing_route",
        scene_teacher_dir=scene,
        spatial_pack_dir=tmp_path / "missing_spatial_pack",
        out_json=tmp_path / "compare_missing.json",
        out_md=tmp_path / "compare_missing.md",
        out_viz=tmp_path / "compare_missing.ppm",
    )

    assert report["next_allowed_use"] == "blocked"
    assert "missing_route:" in " ".join(report["hard_blockers"])
    assert "missing_spatial_pack:" in " ".join(report["hard_blockers"])
    assert report["moge_robot_frame_truth"] is False
    assert report["action_supervision_ok"] is False


def test_goal17a_compare_fake_scene_teacher_is_never_promoted(tmp_path: Path) -> None:
    route = _robot_frame_route(tmp_path)
    scene = tmp_path / "scene_moge_fake"
    spatial_pack = _spatial_pack_fixture(tmp_path)
    run_moge_scene_teacher(route, scene, backend_name="fake")

    report = compare_moge_scene_to_spatial_pack(
        route_dir=route,
        scene_teacher_dir=scene,
        spatial_pack_dir=spatial_pack,
        out_json=tmp_path / "compare_fake.json",
        out_md=tmp_path / "compare_fake.md",
        out_viz=tmp_path / "compare_fake.ppm",
    )

    assert report["matched_frame_count"] == 4
    assert report["route_has_robot_frame_truth"] is True
    assert report["scene_teacher"]["mock_or_synthetic"] is True
    assert report["next_allowed_use"] == "review_only"
    assert report["moge_robot_frame_truth"] is False
    assert report["action_supervision_ok"] is False
    assert (tmp_path / "compare_fake.ppm").exists()


def test_goal17a_compare_preserves_moge_not_robot_frame_truth(tmp_path: Path) -> None:
    route = _robot_frame_route(tmp_path)
    scene = tmp_path / "scene_moge_real_like"
    spatial_pack = _spatial_pack_fixture(tmp_path)
    run_moge_scene_teacher(route, scene, backend_name="fake")
    _mark_scene_teacher_real_like(scene)

    report = compare_moge_scene_to_spatial_pack(
        route_dir=route,
        scene_teacher_dir=scene,
        spatial_pack_dir=spatial_pack,
        out_json=tmp_path / "compare_real_like.json",
        out_md=tmp_path / "compare_real_like.md",
        out_viz=tmp_path / "compare_real_like.ppm",
    )

    assert report["next_allowed_use"] == "single_frame_geometry_pretrain_candidate"
    assert report["scene_teacher"]["moge_robot_frame_truth"] is False
    assert report["moge_robot_frame_truth"] is False
    assert report["action_supervision_ok"] is False
    assert report["scene_teacher"]["direct_bev_conversion_attempted"] is False
    assert "single-frame MoGe" in report["scene_teacher"]["projection_blocked_reason"]
    _assert_no_motion_or_training_claims(tmp_path)


def test_goal17a_aggregate_report_handles_partial_route_success_failure(tmp_path: Path) -> None:
    candidate = {
        "inputs": {"route": "route_a"},
        "route": {"sequence": "route_a", "scene": "unit"},
        "scene_teacher": {"backend": "real", "real_perception": True},
        "qa_summary": {"structural_pass": True},
        "matched_frame_count": 4,
        "moge_depth_valid_ratio": 1.0,
        "moge_confidence_valid_ratio": 1.0,
        "route_has_robot_frame_truth": True,
        "next_allowed_use": "single_frame_geometry_pretrain_candidate",
        "hard_blockers": [],
    }
    blocked = {
        "inputs": {"route": "route_b"},
        "route": {"sequence": "route_b", "scene": "unit"},
        "scene_teacher": {"backend": "real", "real_perception": True},
        "qa_summary": {"structural_pass": False},
        "matched_frame_count": 0,
        "moge_depth_valid_ratio": 0.0,
        "moge_confidence_valid_ratio": 0.0,
        "route_has_robot_frame_truth": False,
        "next_allowed_use": "blocked",
        "hard_blockers": ["missing_spatial_pack"],
    }

    aggregate = write_aggregate_report(
        comparison_reports=[candidate, blocked],
        out_json=tmp_path / "report.json",
        out_md=tmp_path / "report.md",
    )

    assert aggregate["route_count"] == 2
    assert aggregate["candidate_route_count"] == 1
    assert aggregate["blocked_route_count"] == 1
    assert aggregate["partial_route_success_or_failure"] is True
    assert aggregate["did_real_moge_run_on_public_robot_frame_routes"] is True
    assert aggregate["did_qa_pass_structurally"] is False
    assert aggregate["aggregate_recommendation"] == "partial_success_review_blocked_routes_before_candidate"
    _assert_no_motion_or_training_claims(tmp_path)


def test_real_moge_backend_fails_clearly_without_local_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    route = _owned_frame_route(tmp_path)
    monkeypatch.delenv("HOMEBRAIN_MOGE_ADAPTER", raising=False)
    monkeypatch.delenv("HOMEBRAIN_MOGE_DIR", raising=False)
    monkeypatch.delenv("HOMEBRAIN_MOGE_CHECKPOINT", raising=False)
    monkeypatch.delenv("HOMEBRAIN_MOGE_MODEL_ID", raising=False)
    monkeypatch.delenv("HOMEBRAIN_MOGE_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.setitem(sys.modules, "moge", None)
    monkeypatch.delitem(sys.modules, "moge.model", raising=False)
    monkeypatch.delitem(sys.modules, "moge.model.v2", raising=False)

    teacher = create_moge_scene_teacher(
        backend_name="real",
        model_dir=tmp_path / "missing_moge",
        checkpoint=tmp_path / "missing_moge.pt",
    )
    with pytest.raises(MoGeSceneTeacherUnavailableError, match="MoGe is not installed or importable"):
        teacher.run(SceneTeacherRunConfig(log_dir=route, out_dir=tmp_path / "real_out"))
