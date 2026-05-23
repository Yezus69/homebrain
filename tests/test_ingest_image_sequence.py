import json
import struct
import zlib

from homebrain.eval.run_eval import evaluate_log
from homebrain.ingest.image_sequence import deterministic_frame_paths, ingest_image_sequence
from homebrain.ingest.metadata import ROUTE_SOURCE_METADATA_FILE, load_route_metadata
from homebrain.messages.schema import BrainOutputEvent, FrameEvent
from homebrain.replay.replayd import replay_log
from homebrain.replay.segment_log import load_manifest, read_events
from homebrain.teachers.mock_teacher import run_mock_teacher
from homebrain.teachers.visualize_artifacts import visualize_artifacts


def _write_pgm(path, width=2, height=2) -> None:
    pixels = bytes((index * 17) % 256 for index in range(width * height))
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + pixels)


def _write_ppm(path, width=2, height=2) -> None:
    pixels = bytes((index * 13) % 256 for index in range(width * height * 3))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + pixels)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _write_png(path, width=2, height=2) -> None:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw_rows = b"".join(b"\x00" + bytes([row * 30 + col for col in range(width)]) for row in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw_rows))
        + _png_chunk(b"IEND", b"")
    )


def _write_jpeg_with_size(path, width=2, height=2) -> None:
    sof_payload = (
        b"\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    )
    path.write_bytes(
        b"\xff\xd8"
        + b"\xff\xe0\x00\x02"
        + b"\xff\xc0"
        + (len(sof_payload) + 2).to_bytes(2, "big")
        + sof_payload
        + b"\xff\xd9"
    )


def test_image_sequence_ingest_sorts_and_writes_route_metadata(tmp_path) -> None:
    frames = tmp_path / "frames"
    route = tmp_path / "room_walk_route"
    frames.mkdir()
    _write_ppm(frames / "03.ppm")
    _write_jpeg_with_size(frames / "01.JPG")
    _write_png(frames / "02.png")
    _write_jpeg_with_size(frames / "04.jpeg")
    _write_pgm(frames / "05.pgm")
    (frames / "notes.txt").write_text("ignored", encoding="utf-8")

    assert [path.name for path in deterministic_frame_paths(frames)] == [
        "01.JPG",
        "02.png",
        "03.ppm",
        "04.jpeg",
        "05.pgm",
    ]

    summary = ingest_image_sequence(
        frames_dir=frames,
        out_dir=route,
        camera_name="front_rgb",
        fps=10,
    )

    assert summary.frame_count == 5
    assert summary.image_load_error_count == 0
    manifest = load_manifest(route)
    assert ROUTE_SOURCE_METADATA_FILE in manifest.artifact_files
    assert "frames/frame_000000.jpg" in manifest.artifact_files
    assert "frames/frame_000004.pgm" in manifest.artifact_files

    events = read_events(route)
    frames_out = [event for event in events if isinstance(event, FrameEvent)]
    assert [frame.data_ref for frame in frames_out] == [
        "frames/frame_000000.jpg",
        "frames/frame_000001.png",
        "frames/frame_000002.ppm",
        "frames/frame_000003.jpeg",
        "frames/frame_000004.pgm",
    ]
    assert [frame.format for frame in frames_out] == [
        "encoded_jpeg",
        "encoded_png",
        "encoded_ppm",
        "encoded_jpeg",
        "encoded_pgm",
    ]
    assert all(frame.intrinsics == {"available": False, "status": "missing", "reason": "not_supplied_by_image_sequence_ingest"} for frame in frames_out)

    metadata = load_route_metadata(route)
    assert metadata is not None
    assert metadata["source_type"] == "image_sequence"
    assert metadata["source_path"] == frames.resolve().as_posix()
    assert metadata["camera_name"] == "front_rgb"
    assert metadata["fps"] == 10
    assert metadata["frame_count"] == 5
    assert metadata["width"] == 2
    assert metadata["height"] == 2
    assert metadata["has_imu"] is False
    assert metadata["has_wheel_odometry"] is False
    assert metadata["has_commands"] is False
    assert metadata["user_owned_or_license_unknown"] is True
    assert metadata["image_load_error_count"] == 0
    assert len(metadata["missing_sensor_notices"]) == 3
    assert [record["source_frame_index"] for record in metadata["frames"]] == [0, 1, 2, 3, 4]


def test_image_sequence_ingest_max_frames_stride_and_eval_metrics(tmp_path) -> None:
    frames = tmp_path / "frames"
    route = tmp_path / "route"
    frames.mkdir()
    for frame_id in range(5):
        _write_pgm(frames / f"{frame_id:06d}.pgm")

    ingest_image_sequence(
        frames_dir=frames,
        out_dir=route,
        camera_name="front_rgb",
        fps=10,
        max_frames=2,
        stride=2,
    )

    metadata = load_route_metadata(route)
    assert metadata is not None
    assert metadata["frame_count"] == 2
    assert metadata["selected_frame_count"] == 2
    assert metadata["effective_fps"] == 5
    assert metadata["expected_timestamp_interval_ns"] == 200_000_000
    assert [record["source_frame_index"] for record in metadata["frames"]] == [0, 2]

    frame_events = [event for event in read_events(route) if isinstance(event, FrameEvent)]
    assert [frame.frame_id for frame in frame_events] == [0, 1]
    assert frame_events[1].timestamp_ns - frame_events[0].timestamp_ns == 200_000_000

    metrics = evaluate_log(route, eval_runtime_sec=0.0)
    assert metrics["imported_frame_count"] == 2
    assert metrics["image_load_error_count"] == 0
    assert metrics["timestamp_interval_error_count"] == 0
    assert metrics["missing_sensor_notice_count"] == 3
    assert metrics["dropped_frame_count"] == 0


def test_image_sequence_eval_counts_load_and_timestamp_errors(tmp_path) -> None:
    frames = tmp_path / "frames"
    route = tmp_path / "route"
    frames.mkdir()
    _write_pgm(frames / "000000.pgm")
    (frames / "000001.pgm").write_bytes(b"not a pnm")
    _write_pgm(frames / "000002.pgm")

    ingest_image_sequence(
        frames_dir=frames,
        out_dir=route,
        camera_name="front_rgb",
        fps=10,
    )

    metadata = load_route_metadata(route)
    assert metadata is not None
    assert metadata["frame_count"] == 2
    assert metadata["image_load_error_count"] == 1

    metrics = evaluate_log(route, eval_runtime_sec=0.0)
    assert metrics["imported_frame_count"] == 2
    assert metrics["image_load_error_count"] == 1
    assert metrics["timestamp_interval_error_count"] == 1
    assert metrics["missing_sensor_notice_count"] == 3


def test_replay_teacher_visualization_and_eval_work_on_imported_route(tmp_path) -> None:
    frames = tmp_path / "frames"
    route = tmp_path / "room_walk_route"
    replayed = tmp_path / "replayed"
    artifacts = tmp_path / "teacher_artifacts" / "mock_teacher"
    preview = tmp_path / "preview"
    frames.mkdir()
    _write_ppm(frames / "000000.ppm", width=3, height=2)
    _write_ppm(frames / "000001.ppm", width=3, height=2)

    ingest_image_sequence(
        frames_dir=frames,
        out_dir=route,
        camera_name="front_rgb",
        fps=10,
    )

    replay_log(route, replayed)
    replay_events = read_events(replayed)
    assert sum(isinstance(event, FrameEvent) for event in replay_events) == 2
    assert sum(isinstance(event, BrainOutputEvent) for event in replay_events) == 2
    assert (replayed / ROUTE_SOURCE_METADATA_FILE).exists()

    run_mock_teacher(route, artifacts)
    visualization_manifest = visualize_artifacts(artifacts, preview)
    visualization = json.loads(visualization_manifest.read_text(encoding="utf-8"))
    assert visualization["visualization_written"] is True
    assert visualization["frame_count"] == 2

    metrics = evaluate_log(route, teacher_artifacts=artifacts, eval_runtime_sec=0.0)
    assert metrics["imported_frame_count"] == 2
    assert metrics["teacher_mock_used"] is True
    assert metrics["artifact_load_success"] is True
    assert metrics["frames_with_teacher_artifacts"] == 2

