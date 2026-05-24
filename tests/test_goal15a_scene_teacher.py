import json
from pathlib import Path

import numpy as np

from homebrain.geometry.scene_teacher_to_bev import convert_scene_teacher_to_bev
from homebrain.geometry.validate_bev import load_bev_manifest, validate_bev_artifacts
from homebrain.replay.generate_dummy_log import generate_dummy_log
from homebrain.teachers.qa_scene_teacher import qa_scene_teacher_artifacts, run_scene_teacher_qa
from homebrain.teachers.run_scene_teacher import main as run_scene_teacher_main
from homebrain.teachers.scene_teacher import load_scene_teacher_manifest
from homebrain.teachers.vggt_scene_teacher import run_vggt_scene_teacher


def _relative_files(root: Path) -> list[Path]:
    return sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())


def _assert_no_control_outputs(root: Path) -> None:
    for path in root.rglob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                assert item.get("control_safe") is not True
                assert item.get("control_safe_claim") is not True
                assert item.get("raw_pwm_emitted") is not True
                assert "cmd_vel" not in item
                assert "raw_pwm" not in item
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)


def test_fake_vggt_scene_teacher_writes_deterministic_scene_teacher_pack(tmp_path: Path) -> None:
    route = tmp_path / "dummy_route"
    out_a = tmp_path / "scene_a"
    out_b = tmp_path / "scene_b"
    generate_dummy_log(route)

    exit_code = run_scene_teacher_main(
        [
            "--teacher",
            "vggt",
            "--backend",
            "fake",
            "--log",
            str(route),
            "--out",
            str(out_a),
        ]
    )
    assert exit_code == 0
    run_vggt_scene_teacher(route, out_b, backend_name="fake")

    assert _relative_files(out_a) == _relative_files(out_b)
    for relative in _relative_files(out_a):
        assert (out_a / relative).read_bytes() == (out_b / relative).read_bytes()

    manifest = load_scene_teacher_manifest(out_a)
    assert manifest["pack_name"] == "SceneTeacherPack"
    assert manifest["pack_version"] == "v0"
    assert manifest["teacher_name"] == "vggt"
    assert manifest["backend"] == "fake"
    assert manifest["mock"] is True
    assert manifest["synthetic"] is True
    assert manifest["real_perception"] is False
    assert manifest["replay_only"] is True
    assert manifest["not_executed"] is True
    assert manifest["control_safe"] is False
    assert manifest["product_training_approved"] is False
    assert manifest["frame_count"] == 6

    first = manifest["frames"][0]
    artifacts = first["artifacts"]
    assert set(artifacts) >= {
        "depth",
        "point_map",
        "intrinsics",
        "extrinsics",
        "confidence",
        "validity_mask",
        "floor_traversable_mask",
        "obstacle_risk_mask",
        "dynamic_motion_mask",
    }
    assert np.load(out_a / artifacts["depth"]["path"]).shape == (2, 2)
    assert np.load(out_a / artifacts["point_map"]["path"]).shape == (2, 2, 3)
    assert np.load(out_a / artifacts["intrinsics"]["path"]).shape == (3, 3)
    assert np.load(out_a / artifacts["extrinsics"]["path"]).shape == (4, 4)
    assert np.load(out_a / artifacts["validity_mask"]["path"]).dtype == np.uint8

    first_window = manifest["windows"][0]
    tracks = np.load(out_a / first_window["artifacts"]["point_tracks"]["path"])
    track_validity = np.load(out_a / first_window["artifacts"]["track_validity"]["path"])
    assert tracks.shape == (6, 4, 2)
    assert track_validity.shape == (6, 4)


def test_scene_teacher_qa_catches_missing_and_invalid_artifacts(tmp_path: Path) -> None:
    route = tmp_path / "dummy_route"
    artifacts = tmp_path / "scene_teacher"
    qa_path = tmp_path / "scene_qa.json"
    generate_dummy_log(route)
    run_vggt_scene_teacher(route, artifacts, backend_name="fake")

    qa = run_scene_teacher_qa(artifacts_dir=artifacts, out_path=qa_path).metrics
    assert qa["frame_count"] == 6
    assert qa["missing_artifact_count"] == 0
    assert qa["artifact_shape_error_count"] == 0
    assert qa["depth_valid_ratio"] == 1.0
    assert qa["pose_valid_ratio"] == 1.0
    assert qa["track_valid_ratio"] == 1.0
    assert qa["control_safe"] is False
    assert qa["promotable_to_spatial_pack"] is False
    assert "mock_or_synthetic_teacher" in qa["quarantine_reasons"]

    manifest = load_scene_teacher_manifest(artifacts)
    first_depth = artifacts / manifest["frames"][0]["artifacts"]["depth"]["path"]
    first_depth.unlink()
    missing = qa_scene_teacher_artifacts(artifacts)
    assert missing["missing_artifact_count"] >= 1
    assert missing["promotable_to_spatial_pack"] is False

    run_vggt_scene_teacher(route, artifacts, backend_name="fake")
    manifest = load_scene_teacher_manifest(artifacts)
    first_confidence = artifacts / manifest["frames"][0]["artifacts"]["confidence"]["path"]
    with first_confidence.open("wb") as handle:
        np.save(handle, np.zeros((1, 1), dtype=np.float32), allow_pickle=False)
    invalid = qa_scene_teacher_artifacts(artifacts)
    assert invalid["artifact_shape_error_count"] >= 1
    assert invalid["promotable_to_spatial_pack"] is False


def test_scene_teacher_to_bev_preserves_review_only_safety_flags(tmp_path: Path) -> None:
    route = tmp_path / "dummy_route"
    scene = tmp_path / "scene_teacher"
    bev = tmp_path / "scene_teacher_bev"
    generate_dummy_log(route)
    run_vggt_scene_teacher(route, scene, backend_name="fake")

    manifest_path = convert_scene_teacher_to_bev(log_dir=route, scene_teacher_dir=scene, out_dir=bev)
    manifest = load_bev_manifest(bev)
    assert manifest_path == bev / "bev_manifest.json"
    assert manifest["weak_label"] is True
    assert manifest["review_only"] is True
    assert manifest["geometry_pretrain_ok"] is True
    assert manifest["robot_frame_truth"] is False
    assert manifest["not_robot_frame_truth"] is True
    assert manifest["action_supervision_ok"] is False
    assert manifest["replay_only"] is True
    assert manifest["not_executed"] is True
    assert manifest["control_safe"] is False
    assert manifest["product_training_approved"] is False
    assert manifest["raw_pwm_emitted"] is False

    first = manifest["frames"][0]
    metadata = json.loads((bev / first["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["robot_frame_truth"] is False
    assert metadata["action_supervision_ok"] is False
    assert metadata["control_safe"] is False

    validation = validate_bev_artifacts(bev)
    assert validation.load_success is True
    assert validation.control_safe is False


def test_scene_teacher_outputs_make_no_cmd_vel_or_raw_pwm_claims(tmp_path: Path) -> None:
    route = tmp_path / "dummy_route"
    scene = tmp_path / "scene_teacher"
    qa_path = tmp_path / "scene_qa.json"
    bev = tmp_path / "scene_teacher_bev"
    generate_dummy_log(route)
    run_vggt_scene_teacher(route, scene, backend_name="fake")
    run_scene_teacher_qa(artifacts_dir=scene, out_path=qa_path)
    convert_scene_teacher_to_bev(log_dir=route, scene_teacher_dir=scene, out_dir=bev)

    _assert_no_control_outputs(scene)
    _assert_no_control_outputs(bev)
    _assert_no_control_outputs(tmp_path)
