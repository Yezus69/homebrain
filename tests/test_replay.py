from homebrain.messages.schema import BrainOutputEvent, FrameEvent
from homebrain.replay.generate_dummy_log import generate_dummy_log
from homebrain.replay.replayd import replay_log
from homebrain.replay.segment_log import canonical_event_text, load_manifest, read_events


def test_replay_is_deterministic_and_emits_brain_outputs(tmp_path) -> None:
    source = tmp_path / "dummy_route"
    out_a = tmp_path / "replay_a"
    out_b = tmp_path / "replay_b"
    generate_dummy_log(source)

    replay_log(source, out_a)
    replay_log(source, out_b)

    events_a = read_events(out_a)
    events_b = read_events(out_b)
    assert canonical_event_text(events_a) == canonical_event_text(events_b)
    assert sum(isinstance(event, FrameEvent) for event in events_a) == 6
    assert sum(isinstance(event, BrainOutputEvent) for event in events_a) == 6
    assert load_manifest(out_a).event_count == 30
    assert (out_a / "frames" / "frame_000000.rgb").exists()
