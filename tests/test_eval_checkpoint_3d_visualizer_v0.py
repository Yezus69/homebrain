from __future__ import annotations

import json
from pathlib import Path

from homebrain.eval.eval_checkpoint_3d_visualizer_v0 import eval_checkpoint_3d_visualizer_v0
from homebrain.viz.visualize_checkpoint_3d_v0 import visualize_checkpoint_3d_v0


def test_eval_accepts_complete_fixture_artifacts(tmp_path: Path) -> None:
    viz = tmp_path / "viz"
    visualize_checkpoint_3d_v0(fixture="stitched_room_tiny", out=viz, write_html=True, write_ply=True, write_jsonl=True)
    out = eval_checkpoint_3d_visualizer_v0(viz_dir=viz, out=viz / "eval.json")
    report = json.loads(out.read_text(encoding="utf-8"))

    assert report["accepted_checkpoint_3d_visualizer_v0"] is True
    assert report["gates_improved"] == ["Gate A", "Gate D"]
    assert report["map_accumulates_over_time"] is True
    assert report["incremental_no_future_leakage_passed"] is True
    assert report["camera_pose_rendered"] is True
    assert report["robot_pose_rendered"] is True
    assert report["trajectory_rendered"] is True
    assert report["accumulated_map_rendered"] is True
    assert report["geometry_source"] == "fixture_synthetic"
    assert report["dense_3d_claimed"] is False
    assert report["safety"] == {
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
    }


def test_eval_rejects_missing_artifacts(tmp_path: Path) -> None:
    viz = tmp_path / "empty"
    viz.mkdir()
    out = eval_checkpoint_3d_visualizer_v0(viz_dir=viz, out=viz / "eval.json")
    report = json.loads(out.read_text(encoding="utf-8"))

    assert report["accepted_checkpoint_3d_visualizer_v0"] is False
    assert "viewer_html_exists" in report["failed_checks"]

