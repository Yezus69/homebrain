from __future__ import annotations

import json
from pathlib import Path

from homebrain.viz.checkpoint_3d_scene_types_v0 import SceneTimelineFrameV0
from homebrain.viz.visualize_checkpoint_3d_v0 import _materialize_rgb_frames, visualize_checkpoint_3d_v0


def test_visualize_checkpoint_3d_fixture_cli_outputs_required_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "viz"
    summary = visualize_checkpoint_3d_v0(fixture="stitched_room_tiny", out=out, write_html=True, write_ply=True, write_jsonl=True)
    summary_json = json.loads((out / "summary.json").read_text(encoding="utf-8"))

    assert summary["accepted_checkpoint_3d_visualizer_v0"] is True
    assert summary_json["safety"] == {
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
    }
    for name in (
        "summary.json",
        "scene_timeline.jsonl",
        "accumulated_scene.json",
        "accumulated_map.ply",
        "camera_trajectory.ply",
        "viewer.html",
        "manifest.json",
        "visualizer_acceptance.json",
    ):
        assert (out / name).exists(), name
    assert (out / "frame_debug").is_dir()
    assert summary_json["geometry_source"] == "fixture_synthetic"
    assert summary_json["dense_3d_claimed"] is False
    assert summary_json["rgb_frame_rendered"] is True
    first_frame = json.loads((out / "scene_timeline.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first_frame["rgb_frame"]["kind"] == "fixture_synthetic_rgb_reference"
    assert "RGB Reference Frame" in (out / "viewer.html").read_text(encoding="utf-8")


def test_missing_extrinsics_fixture_warns_and_fails_acceptance(tmp_path: Path) -> None:
    out = tmp_path / "missing"
    summary = visualize_checkpoint_3d_v0(fixture="missing_extrinsics_real_mode", out=out, write_html=True, write_ply=True, write_jsonl=True)

    assert summary["accepted_checkpoint_3d_visualizer_v0"] is False
    assert summary["camera_pose_rendered"] is False
    assert "missing_camera_to_base_extrinsics" in summary["warnings"]


def test_oracle_pose_is_marked_and_unaccepted_without_debug_flag(tmp_path: Path) -> None:
    summary = visualize_checkpoint_3d_v0(fixture="oracle_pose_debug", out=tmp_path / "oracle", write_html=True, write_ply=True, write_jsonl=True)

    assert summary["pose_source_is_oracle"] is True
    assert summary["accepted_checkpoint_3d_visualizer_v0"] is False
    assert "oracle_pose_requires_debug_oracle_pose" in summary["hard_failures"]


def test_candidate_selection_veto_and_safety_stop_reasons_are_exported(tmp_path: Path) -> None:
    candidate_out = tmp_path / "candidate"
    visualize_checkpoint_3d_v0(fixture="candidate_trajectory_overlay", out=candidate_out, write_html=True, write_ply=True, write_jsonl=True)
    first = json.loads((candidate_out / "scene_timeline.jsonl").read_text(encoding="utf-8").splitlines()[0])
    candidates = first["candidate_trajectories"]

    assert any(item["selected"] is True and item["candidate_id"] == "stop" for item in candidates)
    assert any(item["vetoed"] is True and item["stop_reasons"] == ["hazard"] for item in candidates)

    safety_out = tmp_path / "safety"
    visualize_checkpoint_3d_v0(fixture="safety_stop_overlay", out=safety_out, write_html=True, write_ply=True, write_jsonl=True)
    frames = [json.loads(line) for line in (safety_out / "scene_timeline.jsonl").read_text(encoding="utf-8").splitlines()]

    assert any(frame["safety_debug"]["stop_reasons"] == ["high_risk_near_footprint"] for frame in frames)
    assert "high_risk_near_footprint" in (safety_out / "viewer.html").read_text(encoding="utf-8")


def test_debug_teacher_overlay_is_separate_and_unaccepted(tmp_path: Path) -> None:
    summary = visualize_checkpoint_3d_v0(
        fixture="stitched_room_tiny",
        out=tmp_path / "teacher_overlay",
        write_html=True,
        write_ply=True,
        write_jsonl=True,
        debug_teacher_overlay="teacher_artifacts",
    )
    html = (tmp_path / "teacher_overlay" / "viewer.html").read_text(encoding="utf-8")

    assert summary["accepted_checkpoint_3d_visualizer_v0"] is False
    assert summary["teacher_overlay_enabled"] is True
    assert summary["overlay_is_runtime_truth"] is False
    assert "teacher_overlay_enabled" in html
    assert "overlay_is_runtime_truth" in html


def test_real_rgb_frame_sources_are_materialized_into_artifact(tmp_path: Path) -> None:
    source = tmp_path / "real_rgb.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\n")
    frame = SceneTimelineFrameV0(route_id="r", frame_id=0, rgb_frame={"kind": "route_rgb_frame", "source_path": source.as_posix()})

    frames, artifacts = _materialize_rgb_frames(tmp_path / "viz", [frame])

    assert artifacts == ["rgb_frames/"]
    assert frames[0].rgb_frame is not None
    assert frames[0].rgb_frame["relative_path"] == "rgb_frames/frame_000000.png"
    assert frames[0].rgb_frame["src"] == "rgb_frames/frame_000000.png"
    assert frames[0].rgb_frame["materialized"] is True
    assert (tmp_path / "viz" / "rgb_frames" / "frame_000000.png").read_bytes() == source.read_bytes()
