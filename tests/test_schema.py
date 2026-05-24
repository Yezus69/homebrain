from homebrain.messages.schema import (
    BrainOutputEvent,
    CommandEvent,
    EvalEvent,
    FrameEvent,
    ImuEvent,
    OdomEvent,
    PoseEvent,
    SegmentManifest,
    WheelEvent,
    event_from_json_line,
    event_to_json_line,
)


def test_event_schema_round_trip_all_goal0_events() -> None:
    events = [
        FrameEvent(
            timestamp_ns=10,
            sequence_id="seq",
            source="camera",
            camera_id="front",
            frame_id=1,
            width=2,
            height=2,
            format="rgb8",
            data_ref="frames/frame.rgb",
            intrinsics={"fx": 1.0},
        ),
        ImuEvent(
            timestamp_ns=11,
            sequence_id="seq",
            source="imu",
            accel_mps2=(0.0, 0.0, 9.81),
            gyro_radps=(0.0, 0.0, 0.1),
        ),
        WheelEvent(
            timestamp_ns=12,
            sequence_id="seq",
            source="wheel",
            left_ticks=1,
            right_ticks=2,
            left_velocity=0.1,
            right_velocity=0.1,
            odom_delta=(0.01, 0.0, 0.0),
        ),
        OdomEvent(
            timestamp_ns=12,
            sequence_id="seq",
            source="odom",
            position_m=(1.0, 2.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            linear_velocity_mps=(0.1, 0.0, 0.0),
            angular_velocity_radps=(0.0, 0.0, 0.1),
        ),
        PoseEvent(
            timestamp_ns=12,
            sequence_id="seq",
            source="pose",
            position_m=(1.0, 2.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="map",
            child_frame_id="base_link",
            pose_kind="groundtruth",
        ),
        CommandEvent(
            timestamp_ns=13,
            sequence_id="seq",
            source="policy",
            linear_velocity_mps=0.1,
            angular_velocity_radps=0.0,
            command_id="cmd-1",
        ),
        BrainOutputEvent(
            timestamp_ns=14,
            sequence_id="seq",
            source="modeld",
            input_event_ids=["seq:frame:10"],
            pose_delta=(0.0, 0.0, 0.0),
            pose_confidence=0.0,
            candidate_trajectories=[{"trajectory_id": "stop", "mock": True}],
            selected_trajectory_id="stop",
            cmd_vel=(0.0, 0.0),
            uncertainty=1.0,
            stop_reason="test",
            debug={"mock": True},
        ),
        EvalEvent(
            timestamp_ns=15,
            sequence_id="seq",
            source="evald",
            metric_name="event_count",
            metric_value=5,
            metadata={},
        ),
    ]

    for event in events:
        decoded = event_from_json_line(event_to_json_line(event))
        assert decoded == event


def test_segment_manifest_round_trip() -> None:
    manifest = SegmentManifest(
        segment_id="seg",
        created_at_utc="1970-01-01T00:00:00Z",
        schema_version="homebrain.segment.v0",
        event_file="events.jsonl",
        artifact_files=["frames/frame.rgb"],
        start_timestamp_ns=1,
        end_timestamp_ns=2,
        event_count=1,
    )
    assert SegmentManifest.from_dict(manifest.to_dict()) == manifest
