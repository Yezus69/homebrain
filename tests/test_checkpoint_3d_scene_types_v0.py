from __future__ import annotations

import json

from homebrain.viz.checkpoint_3d_scene_types_v0 import (
    LocalBEVFrameV0,
    ModelChannelSummaryV0,
    SceneTimelineFrameV0,
    TransformV0,
    deterministic_scene_json,
    safety_flags,
)


def test_scene_timeline_dataclasses_serialize_deterministically() -> None:
    transform = TransformV0(
        frame_from="map",
        frame_to="base_link",
        matrix_4x4=[[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        source="fixture_pose",
        timestamp_ns=7,
    )
    summary = ModelChannelSummaryV0(
        name="bev_free",
        shape=[2, 2],
        min=0.0,
        max=1.0,
        mean=0.5,
        valid_fraction=1.0,
        source="fixture",
        rendered=True,
    )
    local = LocalBEVFrameV0(
        timestamp_ns=7,
        frame_id=1,
        T_map_base=transform,
        resolution_m=0.25,
        origin_convention="test",
        channels={"free": [[1.0, 0.0], [0.0, 1.0]]},
        channel_summaries=[summary],
    )
    frame = SceneTimelineFrameV0(
        route_id="r",
        frame_id=1,
        timestamp_ns=7,
        local_bev=local,
        rgb_frame={"kind": "fixture_synthetic_rgb_reference", "data_uri": "data:image/svg+xml;charset=utf-8,%3Csvg%2F%3E"},
        model_channel_summaries=[summary],
    )

    encoded_a = deterministic_scene_json(frame)
    encoded_b = deterministic_scene_json(frame)
    decoded = json.loads(encoded_a)

    assert encoded_a == encoded_b
    assert decoded["schema_version"] == "Checkpoint3DSceneTimelineV0"
    assert decoded["rgb_frame"]["kind"] == "fixture_synthetic_rgb_reference"
    assert decoded["local_bev"]["channels"]["free"] == [[1.0, 0.0], [0.0, 1.0]]
    assert safety_flags() == {
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
    }
