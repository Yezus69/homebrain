from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from homebrain.ingest.metadata import (
    ROUTE_SOURCE_METADATA_FILE,
    ROUTE_SOURCE_SCHEMA_VERSION,
    write_route_metadata,
)
from homebrain.messages.schema import Event, FrameEvent, JsonDict
from homebrain.replay.segment_log import write_segment

IMAGE_SEQUENCE_SOURCE = "image_sequence_ingest"
BASE_TIMESTAMP_NS = 1_700_000_000_000_000_000
ACCEPTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pgm", ".ppm"}
JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}
JPEG_STANDALONE_MARKERS = {0x01, *range(0xD0, 0xD8)}
MISSING_INTRINSICS: JsonDict = {
    "available": False,
    "status": "missing",
    "reason": "not_supplied_by_image_sequence_ingest",
}


@dataclass(frozen=True)
class ImageFrameRecord:
    frame_id: int
    source_frame_index: int
    source_path: Path
    data_ref: str
    timestamp_ns: int
    width: int
    height: int
    format: str


@dataclass(frozen=True)
class ImageSequenceIngestSummary:
    out_dir: Path
    frame_count: int
    image_load_error_count: int
    metadata_path: Path


def deterministic_frame_paths(frames_dir: str | Path) -> list[Path]:
    root = Path(frames_dir)
    if not root.exists():
        raise FileNotFoundError(f"frame directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"frames path must be a directory: {root}")
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in ACCEPTED_EXTENSIONS
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )


def ingest_image_sequence(
    *,
    frames_dir: str | Path,
    out_dir: str | Path,
    camera_name: str,
    fps: float,
    max_frames: int | None = None,
    stride: int = 1,
    owned_or_license_approved: bool = False,
) -> ImageSequenceIngestSummary:
    if fps <= 0.0:
        raise ValueError("fps must be greater than zero")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1 when supplied")

    source_root = Path(frames_dir)
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "frames").mkdir(parents=True, exist_ok=True)

    source_paths = deterministic_frame_paths(source_root)
    if not source_paths:
        raise ValueError(f"no supported image frames found in {source_root}")

    selected_source_paths = list(_stride_paths(source_paths, stride))
    if max_frames is not None:
        selected_source_paths = selected_source_paths[:max_frames]

    source_frame_interval_ns = int(round(1_000_000_000 / fps))
    expected_import_interval_ns = source_frame_interval_ns * stride
    records: list[ImageFrameRecord] = []
    image_load_errors: list[JsonDict] = []

    for source_frame_index, source_path in selected_source_paths:
        try:
            width, height = read_image_size(source_path)
        except Exception as exc:  # noqa: BLE001 - import should preserve bad-frame evidence.
            image_load_errors.append(
                {
                    "source_frame_index": source_frame_index,
                    "source_path": source_path.resolve().as_posix(),
                    "error": str(exc),
                }
            )
            continue

        frame_id = len(records)
        suffix = source_path.suffix.lower()
        data_ref = f"frames/frame_{frame_id:06d}{suffix}"
        target = output_root / data_ref
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        records.append(
            ImageFrameRecord(
                frame_id=frame_id,
                source_frame_index=source_frame_index,
                source_path=source_path,
                data_ref=data_ref,
                timestamp_ns=BASE_TIMESTAMP_NS + source_frame_index * source_frame_interval_ns,
                width=width,
                height=height,
                format=_frame_format_for_suffix(suffix),
            )
        )

    if not records:
        raise ValueError(
            "image sequence import produced no valid frames; "
            f"{len(image_load_errors)} selected frame(s) failed header loading"
        )

    sequence_id = output_root.name
    events: list[Event] = [
        FrameEvent(
            timestamp_ns=record.timestamp_ns,
            sequence_id=sequence_id,
            source=IMAGE_SEQUENCE_SOURCE,
            camera_id=camera_name,
            frame_id=record.frame_id,
            width=record.width,
            height=record.height,
            format=record.format,  # type: ignore[arg-type]
            data_ref=record.data_ref,
            intrinsics=dict(MISSING_INTRINSICS),
        )
        for record in records
    ]

    metadata = _build_route_metadata(
        source_root=source_root,
        camera_name=camera_name,
        fps=fps,
        stride=stride,
        max_frames=max_frames,
        source_paths=source_paths,
        selected_source_paths=selected_source_paths,
        records=records,
        image_load_errors=image_load_errors,
        source_frame_interval_ns=source_frame_interval_ns,
        expected_import_interval_ns=expected_import_interval_ns,
        owned_or_license_approved=owned_or_license_approved,
    )
    metadata_path = write_route_metadata(output_root, metadata)
    artifact_files = [record.data_ref for record in records]
    artifact_files.append(ROUTE_SOURCE_METADATA_FILE)
    write_segment(output_root, events, segment_id=sequence_id, artifact_files=artifact_files)

    return ImageSequenceIngestSummary(
        out_dir=output_root,
        frame_count=len(records),
        image_load_error_count=len(image_load_errors),
        metadata_path=metadata_path,
    )


def read_image_size(path: str | Path) -> tuple[int, int]:
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return _read_jpeg_size(image_path)
    if suffix == ".png":
        return _read_png_size(image_path)
    if suffix in {".pgm", ".ppm"}:
        return _read_pnm_size(image_path)
    raise ValueError(f"unsupported image extension: {image_path.suffix}")


def _stride_paths(paths: list[Path], stride: int) -> Iterable[tuple[int, Path]]:
    for source_frame_index, path in enumerate(paths):
        if source_frame_index % stride == 0:
            yield source_frame_index, path


def _frame_format_for_suffix(suffix: str) -> str:
    if suffix in {".jpg", ".jpeg"}:
        return "encoded_jpeg"
    if suffix == ".png":
        return "encoded_png"
    if suffix == ".pgm":
        return "encoded_pgm"
    if suffix == ".ppm":
        return "encoded_ppm"
    raise ValueError(f"unsupported image suffix: {suffix}")


def _read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("invalid PNG header")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return _validate_size(width, height, path)


def _read_jpeg_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError("invalid JPEG start marker")
        while True:
            marker_prefix = handle.read(1)
            while marker_prefix and marker_prefix != b"\xff":
                marker_prefix = handle.read(1)
            if not marker_prefix:
                break

            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if not marker:
                break

            marker_code = marker[0]
            if marker_code in JPEG_STANDALONE_MARKERS:
                continue
            if marker_code in {0xD9, 0xDA}:
                break

            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                raise ValueError("truncated JPEG segment length")
            segment_length = int.from_bytes(length_bytes, "big")
            if segment_length < 2:
                raise ValueError("invalid JPEG segment length")
            payload_length = segment_length - 2

            if marker_code in JPEG_SOF_MARKERS:
                if payload_length < 5:
                    raise ValueError("truncated JPEG SOF segment")
                _precision = handle.read(1)
                height = int.from_bytes(handle.read(2), "big")
                width = int.from_bytes(handle.read(2), "big")
                return _validate_size(width, height, path)

            handle.seek(payload_length, 1)

    raise ValueError("JPEG SOF dimensions not found")


def _read_pnm_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    tokens = _pnm_tokens(data)
    try:
        magic = next(tokens)
        width_token = next(tokens)
        height_token = next(tokens)
    except StopIteration as exc:
        raise ValueError("truncated PNM header") from exc

    if path.suffix.lower() == ".pgm" and magic not in {b"P2", b"P5"}:
        raise ValueError(f"unsupported PGM magic: {magic.decode('ascii', errors='replace')}")
    if path.suffix.lower() == ".ppm" and magic not in {b"P3", b"P6"}:
        raise ValueError(f"unsupported PPM magic: {magic.decode('ascii', errors='replace')}")
    try:
        width = int(width_token)
        height = int(height_token)
    except ValueError as exc:
        raise ValueError("invalid PNM width/height") from exc
    return _validate_size(width, height, path)


def _pnm_tokens(data: bytes) -> Iterable[bytes]:
    index = 0
    length = len(data)
    while index < length:
        while index < length:
            value = data[index]
            if value == ord("#"):
                while index < length and data[index] not in {10, 13}:
                    index += 1
                continue
            if not chr(value).isspace():
                break
            index += 1
        if index >= length:
            return

        start = index
        while index < length:
            value = data[index]
            if value == ord("#") or chr(value).isspace():
                break
            index += 1
        yield data[start:index]


def _validate_size(width: int, height: int, path: Path) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid non-positive image size for {path}: {width}x{height}")
    return width, height


def _build_route_metadata(
    *,
    source_root: Path,
    camera_name: str,
    fps: float,
    stride: int,
    max_frames: int | None,
    source_paths: list[Path],
    selected_source_paths: list[tuple[int, Path]],
    records: list[ImageFrameRecord],
    image_load_errors: list[JsonDict],
    source_frame_interval_ns: int,
    expected_import_interval_ns: int,
    owned_or_license_approved: bool,
) -> JsonDict:
    width = records[0].width
    height = records[0].height
    dimensions_consistent = all(
        record.width == width and record.height == height for record in records
    )
    missing_sensor_notices: list[JsonDict] = [
        {
            "sensor": "imu",
            "status": "unavailable",
            "reason": "image_sequence_source_has_no_imu_stream",
        },
        {
            "sensor": "wheel_odometry",
            "status": "unavailable",
            "reason": "image_sequence_source_has_no_wheel_odometry_stream",
        },
        {
            "sensor": "commands",
            "status": "unavailable",
            "reason": "image_sequence_source_has_no_command_stream",
        },
    ]
    return {
        "schema_version": ROUTE_SOURCE_SCHEMA_VERSION,
        "source_type": "image_sequence",
        "source_path": source_root.resolve().as_posix(),
        "camera_name": camera_name,
        "fps": fps,
        "effective_fps": fps / stride,
        "stride": stride,
        "max_frames": max_frames,
        "source_frame_count": len(source_paths),
        "selected_frame_count": len(selected_source_paths),
        "frame_count": len(records),
        "imported_frame_count": len(records),
        "width": width if dimensions_consistent else None,
        "height": height if dimensions_consistent else None,
        "dimensions_consistent": dimensions_consistent,
        "source_frame_interval_ns": source_frame_interval_ns,
        "expected_timestamp_interval_ns": expected_import_interval_ns,
        "has_imu": False,
        "has_wheel_odometry": False,
        "has_commands": False,
        "owned_or_license_approved": bool(owned_or_license_approved),
        "has_intrinsics": False,
        "calibration_class": "uncalibrated_visual",
        "intrinsics_source": "missing_not_supplied",
        "extrinsics_source": "missing_not_supplied",
        "pose_source": "missing_not_supplied",
        "scale_source": "unknown_image_only",
        "gravity_floor_source": "missing_not_supplied",
        "user_owned_or_license_unknown": not bool(owned_or_license_approved),
        "image_load_error_count": len(image_load_errors),
        "image_load_errors": image_load_errors,
        "missing_sensor_notices": missing_sensor_notices,
        "frames": [
            {
                "frame_id": record.frame_id,
                "source_frame_index": record.source_frame_index,
                "source_path": record.source_path.resolve().as_posix(),
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
    parser = argparse.ArgumentParser(
        description="Import a deterministic indoor image sequence as a HomeBrain route log."
    )
    parser.add_argument("--frames", required=True, help="Directory containing image frames.")
    parser.add_argument("--out", required=True, help="Output HomeBrain route log directory.")
    parser.add_argument("--camera", required=True, help="Camera name for FrameEvent.camera_id.")
    parser.add_argument("--fps", required=True, type=float, help="Source image sequence frame rate.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional imported frame cap.")
    parser.add_argument("--stride", type=int, default=1, help="Import every Nth sorted frame.")
    parser.add_argument(
        "--owned-or-license-approved",
        action="store_true",
        help="Record that the operator explicitly approved this input for local HomeBrain review.",
    )
    args = parser.parse_args(argv)

    summary = ingest_image_sequence(
        frames_dir=args.frames,
        out_dir=args.out,
        camera_name=args.camera,
        fps=args.fps,
        max_frames=args.max_frames,
        stride=args.stride,
        owned_or_license_approved=args.owned_or_license_approved,
    )
    print(
        f"imported {summary.frame_count} frame(s) to {summary.out_dir.as_posix()} "
        f"with {summary.image_load_error_count} image load error(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
