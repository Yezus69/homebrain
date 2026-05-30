from pathlib import Path
import json
import zlib

import numpy as np

from homebrain.datasets.bonn_rgbd_to_route import bonn_rgbd_to_route
from homebrain.datasets.setup_bonn_rgbd import setup_bonn_rgbd
from homebrain.ingest.metadata import load_route_metadata
from homebrain.replay.segment_log import read_events


def test_bonn_missing_root_exits_zero_and_reports_unaccepted(tmp_path: Path) -> None:
    report, code = setup_bonn_rgbd(root=tmp_path / "missing", out_dir=tmp_path / "setup")
    data = json.loads(report.read_text(encoding="utf-8"))

    assert code == 0
    assert data["accepted"] is False
    assert data["reason"] == "missing_bonn_rgbd_root"
    assert data["invented_data"] is False


def test_bonn_importer_writes_shared_route_metadata_and_missing_sensor_notices(tmp_path: Path) -> None:
    source = tmp_path / "bonn_seq"
    _write_bonn_sequence(source, frame_count=3)
    out = bonn_rgbd_to_route(source_dir=source, out_dir=tmp_path / "route")

    metadata = load_route_metadata(out)
    associations = json.loads((out / "bonn_rgbd_associations.json").read_text(encoding="utf-8"))
    events = read_events(out)

    assert metadata is not None
    assert metadata["source_type"] == "bonn_rgbd_dynamic"
    assert metadata["has_depth"] is True
    assert metadata["has_groundtruth_pose"] is True
    assert metadata["control_safe"] is False
    assert metadata["replay_only"] is True
    assert metadata["raw_pwm_emitted"] is False
    assert associations["frames"][0]["rgb_ref"].startswith("frames/")
    assert {notice["sensor"] for notice in associations["missing_sensor_notices"]} >= {"imu", "wheel_odometry", "commands", "camera_to_base"}
    assert any(getattr(event, "event_type", None) == "frame" for event in events)
    assert any(getattr(event, "event_type", None) == "pose" for event in events)


def _write_bonn_sequence(root: Path, *, frame_count: int) -> None:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    rgb_lines = ["# timestamp filename"]
    depth_lines = ["# timestamp filename"]
    gt_lines = ["# timestamp tx ty tz qx qy qz qw"]
    assoc_lines = ["# rgb_timestamp rgb_path depth_timestamp depth_path"]
    for index in range(frame_count):
        timestamp = 1.0 + index * 0.1
        rgb_rel = f"rgb/{timestamp:.6f}.ppm"
        depth_rel = f"depth/{timestamp:.6f}.png"
        _write_ppm(root / rgb_rel, np.full((6, 8, 3), 20 + index * 10, dtype=np.uint8))
        _write_png16(root / depth_rel, np.full((6, 8), 1000 + 50 * index, dtype=np.uint16))
        rgb_lines.append(f"{timestamp:.6f} {rgb_rel}")
        depth_lines.append(f"{timestamp:.6f} {depth_rel}")
        assoc_lines.append(f"{timestamp:.6f} {rgb_rel} {timestamp:.6f} {depth_rel}")
        gt_lines.append(f"{timestamp:.6f} {index * 0.01:.6f} 0 0 0 0 0 1")
    (root / "rgb.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
    (root / "depth.txt").write_text("\n".join(depth_lines) + "\n", encoding="utf-8")
    (root / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    (root / "associations.txt").write_text("\n".join(assoc_lines) + "\n", encoding="utf-8")


def _write_ppm(path: Path, image: np.ndarray) -> None:
    height, width, _channels = image.shape
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + image.astype(np.uint8).tobytes(order="C"))


def _write_png16(path: Path, image: np.ndarray) -> None:
    values = np.asarray(image, dtype=np.uint16)
    height, width = values.shape
    raw_rows = bytearray()
    for row in values.astype(">u2", copy=False):
        raw_rows.append(0)
        raw_rows.extend(row.tobytes(order="C"))
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(_png_chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([16, 0, 0, 0, 0])))
    png.extend(_png_chunk(b"IDAT", zlib.compress(bytes(raw_rows))))
    png.extend(_png_chunk(b"IEND", b""))
    path.write_bytes(bytes(png))


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return len(payload).to_bytes(4, "big") + kind + payload + crc.to_bytes(4, "big")
