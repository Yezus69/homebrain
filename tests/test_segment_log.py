from homebrain.messages.schema import CommandEvent, FrameEvent
from homebrain.replay.segment_log import load_manifest, read_events, write_segment


def test_segment_log_write_read_round_trip(tmp_path) -> None:
    events = [
        CommandEvent(
            timestamp_ns=1,
            sequence_id="seq",
            source="test",
            linear_velocity_mps=0.0,
            angular_velocity_radps=0.0,
            command_id="cmd",
        ),
        FrameEvent(
            timestamp_ns=2,
            sequence_id="seq",
            source="camera",
            camera_id="front",
            frame_id=0,
            width=2,
            height=2,
            format="rgb8",
            data_ref="frames/frame_000000.rgb",
            intrinsics=None,
        ),
    ]

    manifest = write_segment(
        tmp_path,
        events,
        segment_id="round-trip",
        artifact_files=["frames/frame_000000.rgb"],
    )

    assert manifest.event_count == 2
    assert load_manifest(tmp_path) == manifest
    assert read_events(tmp_path) == events
