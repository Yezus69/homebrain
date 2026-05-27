from pathlib import Path
import zlib

import numpy as np

from homebrain.data.tum_rgbd_route import discover_tum_rgbd_routes, load_tum_rgbd_sequence, route_split_map


def test_tum_rgbd_route_loader_parses_ordered_frames_and_manifest_hashes(tmp_path: Path) -> None:
    route = tmp_path / "route_a"
    _write_tum_sequence(route, frame_count=4)

    sequence = load_tum_rgbd_sequence(route, dataset_name="bonn_rgbd_dynamic")

    assert sequence.route_id == "route_a"
    assert [frame.frame_index for frame in sequence.frames] == [0, 1, 2, 3]
    assert sequence.frames[0].depth_path is not None
    assert sequence.frames[0].pose is not None
    assert sequence.source_manifest["dataset_name"] == "bonn_rgbd_dynamic"
    assert sequence.source_manifest["used_frame_count"] == 4
    assert sequence.source_manifest["sample_file_hashes"]
    assert discover_tum_rgbd_routes(tmp_path) == [route]


def test_route_split_map_rejects_train_val_route_overlap(tmp_path: Path) -> None:
    route = tmp_path / "route_a"
    _write_tum_sequence(route, frame_count=2)
    sequence = load_tum_rgbd_sequence(route)

    try:
        route_split_map([sequence], train_routes=["route_a"], val_routes=["route_a"])
    except ValueError as exc:
        assert "both train and val" in str(exc)
    else:
        raise AssertionError("route split leakage did not fail")


def _write_tum_sequence(root: Path, *, frame_count: int) -> None:
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    rgb_lines = ["# timestamp filename"]
    depth_lines = ["# timestamp filename"]
    gt_lines = ["# timestamp tx ty tz qx qy qz qw"]
    assoc_lines = ["# rgb_timestamp rgb_path depth_timestamp depth_path"]
    for index in range(frame_count):
        timestamp = 1.0 + index * 0.033
        rgb_rel = f"rgb/{timestamp:.6f}.ppm"
        depth_rel = f"depth/{timestamp:.6f}.png"
        _write_ppm(root / rgb_rel, np.full((4, 4, 3), index * 20, dtype=np.uint8))
        _write_png16(root / depth_rel, np.full((4, 4), 1000 + 50 * index, dtype=np.uint16))
        rgb_lines.append(f"{timestamp:.6f} {rgb_rel}")
        depth_lines.append(f"{timestamp:.6f} {depth_rel}")
        assoc_lines.append(f"{timestamp:.6f} {rgb_rel} {timestamp:.6f} {depth_rel}")
        gt_lines.append(f"{timestamp:.6f} {index * 0.01:.6f} 0.0 0.0 0.0 0.0 0.0 1.0")
    (root / "rgb.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
    (root / "depth.txt").write_text("\n".join(depth_lines) + "\n", encoding="utf-8")
    (root / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    (root / "associations.txt").write_text("\n".join(assoc_lines) + "\n", encoding="utf-8")


def _write_ppm(path: Path, image: np.ndarray) -> None:
    height, width, _channels = image.shape
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(image.astype(np.uint8).tobytes(order="C"))


def _write_png16(path: Path, image: np.ndarray) -> None:
    values = np.asarray(image, dtype=np.uint16)
    height, width = values.shape
    raw_rows = bytearray()
    big = values.astype(">u2", copy=False)
    for row in big:
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
