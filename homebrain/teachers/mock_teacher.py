from __future__ import annotations

from pathlib import Path

import numpy as np

from homebrain.messages.schema import FrameEvent, JsonDict
from homebrain.replay.replayd import order_events_for_replay
from homebrain.replay.segment_log import load_manifest, read_events
from homebrain.teachers.artifacts import (
    BEV_PREVIEW_SHAPE,
    EXPECTED_ARTIFACT_KINDS,
    TEACHER_ARTIFACT_SCHEMA_VERSION,
    expected_shape,
    file_sha256,
    relative_to_root,
    save_array,
    write_json,
    write_teacher_manifest,
)
from homebrain.teachers.base import Teacher, TeacherRunConfig, TeacherRunSummary


class MockTeacher(Teacher):
    name = "mock_teacher"
    version = "mock_teacher.v0"
    mock = True

    def run(self, config: TeacherRunConfig) -> TeacherRunSummary:
        log_dir = Path(config.log_dir)
        out_dir = Path(config.out_dir)
        source_manifest = load_manifest(log_dir)
        events = order_events_for_replay(read_events(log_dir))
        frames = [event for event in events if isinstance(event, FrameEvent)]

        frame_records: list[JsonDict] = []
        for frame in frames:
            frame_records.append(self._write_frame_artifacts(log_dir, out_dir, frame))

        manifest: JsonDict = {
            "schema_version": TEACHER_ARTIFACT_SCHEMA_VERSION,
            "teacher_name": self.name,
            "teacher_version": self.version,
            "mock": True,
            "synthetic": True,
            "real_perception": False,
            "deterministic": True,
            "created_at_utc": "1970-01-01T00:00:00Z",
            "source_log": str(log_dir.as_posix()),
            "source_segment_id": source_manifest.segment_id,
            "source_schema_version": source_manifest.schema_version,
            "frame_count": len(frame_records),
            "artifact_kinds": list(EXPECTED_ARTIFACT_KINDS),
            "frames": frame_records,
        }
        manifest_path = write_teacher_manifest(out_dir, manifest)
        return TeacherRunSummary(
            teacher_name=self.name,
            teacher_version=self.version,
            mock=True,
            frame_count=len(frame_records),
            manifest_path=manifest_path,
        )

    def _write_frame_artifacts(self, log_dir: Path, out_dir: Path, frame: FrameEvent) -> JsonDict:
        frame_dir = out_dir / "frames" / f"{frame.camera_id}_{frame.frame_id:06d}"
        intensity = _frame_intensity(log_dir / frame.data_ref, frame)
        arrays = _mock_arrays(frame, intensity)

        artifact_records: dict[str, JsonDict] = {}
        for kind in EXPECTED_ARTIFACT_KINDS:
            array = arrays[kind]
            expected = expected_shape(kind, frame.width, frame.height)
            if tuple(array.shape) != expected:
                raise ValueError(f"{kind} mock artifact has wrong shape {array.shape}; expected {expected}")
            target = frame_dir / f"{kind}.npy"
            array_record = save_array(target, array)
            artifact_records[kind] = {
                "kind": kind,
                "path": relative_to_root(target, out_dir),
                **array_record,
            }

        metadata: JsonDict = {
            "schema_version": TEACHER_ARTIFACT_SCHEMA_VERSION,
            "teacher_name": self.name,
            "teacher_version": self.version,
            "mock": True,
            "synthetic": True,
            "real_perception": False,
            "sequence_id": frame.sequence_id,
            "camera_id": frame.camera_id,
            "frame_id": frame.frame_id,
            "timestamp_ns": frame.timestamp_ns,
            "width": frame.width,
            "height": frame.height,
            "format": frame.format,
            "source_data_ref": frame.data_ref,
            "note": "Deterministic synthetic teacher output for interface tests; not real perception.",
            "artifact_shapes": {
                kind: artifact_records[kind]["shape"] for kind in EXPECTED_ARTIFACT_KINDS
            },
            "artifact_dtypes": {
                kind: artifact_records[kind]["dtype"] for kind in EXPECTED_ARTIFACT_KINDS
            },
        }
        metadata_path = frame_dir / "metadata.json"
        write_json(metadata_path, metadata)

        return {
            "sequence_id": frame.sequence_id,
            "camera_id": frame.camera_id,
            "frame_id": frame.frame_id,
            "timestamp_ns": frame.timestamp_ns,
            "width": frame.width,
            "height": frame.height,
            "format": frame.format,
            "source_data_ref": frame.data_ref,
            "metadata_path": relative_to_root(metadata_path, out_dir),
            "metadata_sha256": file_sha256(metadata_path),
            "artifacts": artifact_records,
        }


def _frame_intensity(path: Path, frame: FrameEvent) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"frame data_ref is missing: {path}")
    raw = path.read_bytes()
    pixel_count = frame.width * frame.height
    if pixel_count <= 0:
        raise ValueError(f"frame has invalid dimensions: {frame.width}x{frame.height}")

    if frame.format in {"rgb8", "bgr8"}:
        expected_bytes = pixel_count * 3
        if len(raw) != expected_bytes:
            raise ValueError(f"{frame.format} frame expected {expected_bytes} bytes, got {len(raw)}")
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        return pixels.astype(np.float32).mean(axis=2) / np.float32(255.0)

    if frame.format == "gray8":
        if len(raw) != pixel_count:
            raise ValueError(f"gray8 frame expected {pixel_count} bytes, got {len(raw)}")
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(frame.height, frame.width)
        return pixels.astype(np.float32) / np.float32(255.0)

    digest = np.frombuffer(raw or b"\x00", dtype=np.uint8).astype(np.float32)
    tiled = np.resize(digest, pixel_count).reshape(frame.height, frame.width)
    return tiled / np.float32(255.0)


def _mock_arrays(frame: FrameEvent, intensity: np.ndarray) -> dict[str, np.ndarray]:
    height = frame.height
    width = frame.width
    row, col = np.indices((height, width), dtype=np.float32)
    row_norm = row / np.float32(max(height - 1, 1))
    col_norm = col / np.float32(max(width - 1, 1))
    frame_bias = np.float32((frame.frame_id % 97) * 0.01)
    timestamp_bias = np.float32((frame.timestamp_ns % 1_000_000_000) / 1_000_000_000.0)

    depth = (
        np.float32(0.25)
        + np.float32(0.70) * row_norm
        + np.float32(0.35) * col_norm
        + np.float32(0.20) * intensity
        + frame_bias
    ).astype(np.float32)
    depth_confidence = np.clip(
        np.float32(0.95)
        - np.float32(0.20) * np.abs(intensity - np.float32(0.5))
        - np.float32(0.03) * (row_norm + col_norm),
        np.float32(0.0),
        np.float32(1.0),
    ).astype(np.float32)
    dense_features = np.stack(
        [
            row_norm,
            col_norm,
            intensity,
            np.full((height, width), frame_bias + timestamp_bias, dtype=np.float32),
        ],
        axis=2,
    ).astype(np.float32)
    dynamic_mask = (
        ((row.astype(np.int32) + col.astype(np.int32) + frame.frame_id) % 4) == 0
    ).astype(np.uint8)

    bev_h, bev_w = BEV_PREVIEW_SHAPE
    bev_row, bev_col = np.indices((bev_h, bev_w), dtype=np.float32)
    bev_preview = (
        np.float32(0.1)
        + np.float32(0.5) * (bev_row / np.float32(max(bev_h - 1, 1)))
        + np.float32(0.25) * (bev_col / np.float32(max(bev_w - 1, 1)))
        + np.float32(depth.mean()) * np.float32(0.05)
        + frame_bias
    ).astype(np.float32)

    return {
        "depth": depth,
        "depth_confidence": depth_confidence,
        "dense_features": dense_features,
        "dynamic_mask": dynamic_mask,
        "bev_preview": bev_preview,
    }


def run_mock_teacher(log_dir: str | Path, out_dir: str | Path) -> TeacherRunSummary:
    return MockTeacher().run(TeacherRunConfig(log_dir=Path(log_dir), out_dir=Path(out_dir)))
