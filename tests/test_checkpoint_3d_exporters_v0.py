from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re

from homebrain.viz.accumulated_scene3d_v0 import AccumulatedScene3DV0
from homebrain.viz.export_checkpoint_3d_artifacts_v0 import (
    file_sha256,
    write_ply_point_cloud,
    write_scene_timeline_jsonl,
    write_viewer_html,
)
from tests.checkpoint_3d_visualizer_fixtures import build_fixture_sequence


def _timeline_with_map():
    fixture = build_fixture_sequence("candidate_trajectory_overlay")
    scene = AccumulatedScene3DV0(fixture.resolution_m, fixture.decay_sec, 1000, fixture.geometry_source)
    frames = []
    maps = []
    for frame in fixture.frames[:2]:
        assert frame.local_bev is not None
        scene.update_from_local_bev(frame.local_bev, camera_frustum=frame.camera_frustum)
        snapshot = scene.snapshot(frame.frame_id, frame.timestamp_ns)
        maps.append(snapshot)
        frames.append(replace(frame, accumulated_map_summary=snapshot.to_dict()))
    return frames, maps


def test_exporters_write_deterministic_jsonl_ply_and_viewer(tmp_path: Path) -> None:
    frames, maps = _timeline_with_map()
    jsonl = write_scene_timeline_jsonl(tmp_path / "scene_timeline.jsonl", frames)
    ply = write_ply_point_cloud(tmp_path / "accumulated_map.ply", maps[-1])
    first_hash = file_sha256(ply)
    write_ply_point_cloud(ply, maps[-1])
    html = write_viewer_html(
        tmp_path / "viewer.html",
        frames=frames,
        accumulated_scene={"final_map": maps[-1].to_dict()},
        summary={"route_id": "fixture", "safety": {"replay_only": True}},
    )

    assert jsonl.read_text(encoding="utf-8").count("\n") == 2
    assert file_sha256(ply) == first_hash
    text = html.read_text(encoding="utf-8")
    assert 'id="timeline"' in text
    assert 'id="play-pause"' in text
    assert 'id="reset-view"' in text
    assert 'id="rgb-frame"' in text
    assert 'id="rgb-link"' in text
    assert 'id="status-panel"' in text
    assert 'id="warning-banner"' in text
    assert 'id="view-mode"' in text
    assert "Raw Replay State" in text
    assert 'id="map-mode"' in text
    assert '<option value="incremental" selected>incremental</option>' in text
    assert '<option value="final">final</option>' in text
    assert 'id="layer-accumulated-hazard"' in text
    assert "candidate trajectories" in text
    assert "overlay_is_runtime_truth" in text
    assert "model_channels_missing" in text
    assert "selected_candidate_score" in text
    assert "const camera3d" in text
    assert "function worldToScreen" in text
    assert "function addBox" in text
    assert "function addTile" in text
    assert "function addLocalBevCells" in text
    assert "function drawPointCloud" in text
    assert "cameraPointToViewer" in text
    assert "camera frame" in text
    assert "renderStatusPanel" in text
    assert "URLSearchParams" in text
    assert "depth point cloud" in text
    assert "renderRgbFrame" in text
    match = re.search(r'<script id="scene-data" type="application/json">(.*?)</script>', text, re.S)
    assert match is not None
    parsed = json.loads(match.group(1))
    assert parsed["frames"]
    assert parsed["frames"][0]["rgb_frame"]["kind"] == "fixture_synthetic_rgb_reference"
    assert parsed["frames"][0]["rgb_frame"]["data_uri"].startswith("data:image/svg+xml")
