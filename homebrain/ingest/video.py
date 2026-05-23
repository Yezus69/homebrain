from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Any

from homebrain.ingest.image_sequence import BASE_TIMESTAMP_NS, read_image_size
from homebrain.ingest.metadata import (
    ROUTE_SOURCE_METADATA_FILE,
    ROUTE_SOURCE_SCHEMA_VERSION,
    write_route_metadata,
)
from homebrain.messages.schema import Event, FrameEvent, JsonDict
from homebrain.replay.segment_log import write_segment

VIDEO_SOURCE = "video_ingest"
MISSING_VIDEO_INTRINSICS: JsonDict = {
    "available": False,
    "status": "missing",
    "reason": "not_supplied_by_video_ingest",
}


@dataclass(frozen=True)
class VideoFrameRecord:
    frame_id: int
    source_frame_index: int
    data_ref: str
    timestamp_ns: int
    width: int
    height: int
    format: str


@dataclass(frozen=True)
class VideoIngestSummary:
    out_dir: Path
    frame_count: int
    backend: str
    metadata_path: Path


def ingest_video(
    *,
    video_path: str | Path,
    out_dir: str | Path,
    camera_name: str,
    fps: float,
    max_frames: int | None = None,
) -> VideoIngestSummary:
    if fps <= 0.0:
        raise ValueError("fps must be greater than zero")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1 when supplied")
    source = Path(video_path)
    if not source.exists():
        raise FileNotFoundError(f"video does not exist: {source}")
    output = Path(out_dir)
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cv2_module = _load_cv2()
    if cv2_module is not None:
        records, backend_metadata = _extract_with_cv2(
            cv2_module,
            source=source,
            frames_dir=frames_dir,
            fps=fps,
            max_frames=max_frames,
        )
        backend = "cv2"
    else:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("video ingest requires either opencv-python (cv2) or ffmpeg on PATH")
        records, backend_metadata = _extract_with_ffmpeg(
            ffmpeg=ffmpeg,
            source=source,
            frames_dir=frames_dir,
            fps=fps,
            max_frames=max_frames,
        )
        backend = "ffmpeg"

    if not records:
        raise ValueError(f"video ingest produced no frames from {source}")

    sequence_id = output.name
    events: list[Event] = [
        FrameEvent(
            timestamp_ns=record.timestamp_ns,
            sequence_id=sequence_id,
            source=VIDEO_SOURCE,
            camera_id=camera_name,
            frame_id=record.frame_id,
            width=record.width,
            height=record.height,
            format=record.format,  # type: ignore[arg-type]
            data_ref=record.data_ref,
            intrinsics=dict(MISSING_VIDEO_INTRINSICS),
        )
        for record in records
    ]

    metadata = _build_route_metadata(
        source=source,
        camera_name=camera_name,
        fps=fps,
        max_frames=max_frames,
        records=records,
        backend=backend,
        backend_metadata=backend_metadata,
    )
    metadata_path = write_route_metadata(output, metadata)
    artifact_files = [record.data_ref for record in records]
    artifact_files.append(ROUTE_SOURCE_METADATA_FILE)
    write_segment(output, events, segment_id=sequence_id, artifact_files=artifact_files)

    return VideoIngestSummary(
        out_dir=output,
        frame_count=len(records),
        backend=backend,
        metadata_path=metadata_path,
    )


def _load_cv2() -> Any | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return None
    return cv2


def _extract_with_cv2(
    cv2: Any,
    *,
    source: Path,
    frames_dir: Path,
    fps: float,
    max_frames: int | None,
) -> tuple[list[VideoFrameRecord], JsonDict]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open video: {source}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_interval_s = 1.0 / fps
    next_sample_s = 0.0
    source_index = 0
    records: list[VideoFrameRecord] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if source_fps > 0.0:
                frame_time_s = source_index / source_fps
            else:
                frame_time_s = len(records) * target_interval_s
            if frame_time_s + 1.0e-9 >= next_sample_s:
                frame_id = len(records)
                data_ref = f"frames/frame_{frame_id:06d}.jpg"
                target = frames_dir.parent / data_ref
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(target), frame):
                    raise RuntimeError(f"cv2 failed to write extracted frame: {target}")
                height, width = frame.shape[:2]
                records.append(
                    VideoFrameRecord(
                        frame_id=frame_id,
                        source_frame_index=source_index,
                        data_ref=data_ref,
                        timestamp_ns=BASE_TIMESTAMP_NS + int(round(frame_id * target_interval_s * 1_000_000_000)),
                        width=int(width),
                        height=int(height),
                        format="encoded_jpeg",
                    )
                )
                next_sample_s += target_interval_s
                if max_frames is not None and len(records) >= max_frames:
                    break
            source_index += 1
    finally:
        cap.release()
    return records, {
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "sampled_source_frame_count": source_index + 1 if records else source_index,
    }


def _extract_with_ffmpeg(
    *,
    ffmpeg: str,
    source: Path,
    frames_dir: Path,
    fps: float,
    max_frames: int | None,
) -> tuple[list[VideoFrameRecord], JsonDict]:
    pattern = frames_dir / "frame_%06d.ppm"
    command = [ffmpeg, "-y", "-i", str(source), "-vf", f"fps={fps}"]
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.extend([str(pattern)])
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to extract frames: {completed.stderr[-1000:]}")

    extracted = sorted(frames_dir.glob("frame_*.ppm"))
    records: list[VideoFrameRecord] = []
    target_interval_ns = int(round(1_000_000_000 / fps))
    for index, path in enumerate(extracted):
        canonical = frames_dir / f"frame_{index:06d}.ppm"
        if path != canonical:
            if canonical.exists():
                canonical.unlink()
            path.rename(canonical)
        width, height = read_image_size(canonical)
        records.append(
            VideoFrameRecord(
                frame_id=index,
                source_frame_index=index,
                data_ref=f"frames/frame_{index:06d}.ppm",
                timestamp_ns=BASE_TIMESTAMP_NS + index * target_interval_ns,
                width=width,
                height=height,
                format="encoded_ppm",
            )
        )
    return records, {
        "command": command,
        "stderr_tail": completed.stderr[-2000:],
    }


def _build_route_metadata(
    *,
    source: Path,
    camera_name: str,
    fps: float,
    max_frames: int | None,
    records: list[VideoFrameRecord],
    backend: str,
    backend_metadata: JsonDict,
) -> JsonDict:
    width = records[0].width
    height = records[0].height
    dimensions_consistent = all(record.width == width and record.height == height for record in records)
    missing_sensor_notices: list[JsonDict] = [
        {
            "sensor": "imu",
            "status": "unavailable",
            "reason": "video_source_has_no_imu_stream",
        },
        {
            "sensor": "wheel_odometry",
            "status": "unavailable",
            "reason": "video_source_has_no_wheel_odometry_stream",
        },
        {
            "sensor": "commands",
            "status": "unavailable",
            "reason": "video_source_has_no_command_stream",
        },
    ]
    return {
        "schema_version": ROUTE_SOURCE_SCHEMA_VERSION,
        "source_type": "video",
        "source_path": source.resolve().as_posix(),
        "camera_name": camera_name,
        "fps": fps,
        "effective_fps": fps,
        "max_frames": max_frames,
        "frame_count": len(records),
        "imported_frame_count": len(records),
        "width": width if dimensions_consistent else None,
        "height": height if dimensions_consistent else None,
        "dimensions_consistent": dimensions_consistent,
        "expected_timestamp_interval_ns": int(round(1_000_000_000 / fps)),
        "has_imu": False,
        "has_wheel_odometry": False,
        "has_commands": False,
        "has_intrinsics": False,
        "calibration_class": "uncalibrated_visual",
        "intrinsics_source": "missing_not_supplied",
        "extrinsics_source": "missing_not_supplied",
        "pose_source": "missing_not_supplied",
        "scale_source": "unknown_video_only",
        "gravity_floor_source": "missing_not_supplied",
        "user_owned_or_license_unknown": True,
        "video_ingest_backend": backend,
        "backend_metadata": backend_metadata,
        "missing_sensor_notices": missing_sensor_notices,
        "frames": [
            {
                "frame_id": record.frame_id,
                "source_frame_index": record.source_frame_index,
                "data_ref": record.data_ref,
                "timestamp_ns": record.timestamp_ns,
                "width": record.width,
                "height": record.height,
                "format": record.format,
            }
            for record in records
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import a video as a HomeBrain route log.")
    parser.add_argument("--video", required=True, help="Input video file.")
    parser.add_argument("--out", required=True, help="Output HomeBrain route log directory.")
    parser.add_argument("--camera", required=True, help="Camera name for FrameEvent.camera_id.")
    parser.add_argument("--fps", required=True, type=float, help="Target extraction frame rate.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional imported frame cap.")
    args = parser.parse_args(argv)
    summary = ingest_video(
        video_path=args.video,
        out_dir=args.out,
        camera_name=args.camera,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    print(
        f"imported {summary.frame_count} frame(s) to {summary.out_dir.as_posix()} "
        f"using {summary.backend}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
