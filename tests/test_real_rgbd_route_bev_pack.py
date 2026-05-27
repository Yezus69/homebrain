from pathlib import Path
import json
import zlib

import numpy as np

from homebrain.brain.direct_bev_student_v0 import DirectBEVStudentV0, DirectBEVStudentV0Config, save_checkpoint
from homebrain.eval.eval_direct_bev_student_v0 import eval_direct_bev_student_v0
from homebrain.train.real_rgbd_route_bev_pack import build_real_rgbd_route_bev_pack


def test_pack_builder_missing_input_writes_unaccepted_manifest(tmp_path: Path) -> None:
    out = tmp_path / "pack"
    manifest_path = build_real_rgbd_route_bev_pack(
        dataset="bonn_rgbd_dynamic",
        input_root=tmp_path / "missing",
        out_dir=out,
        grid_shape=(16, 16),
        meters_per_cell=0.1,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["accepted_real_rgbd_route_bev_pack"] is False
    assert manifest["accepted_real_rgbd_route_bev_student"] is False
    assert manifest["real_dataset_required"] is True
    assert "missing_real_dataset_root_or_tum_rgbd_routes" in manifest["acceptance_reasons"]


def test_pack_builder_rejects_train_val_route_leakage(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_tum_sequence(source / "route_a", frame_count=3)

    try:
        build_real_rgbd_route_bev_pack(
            dataset="bonn_rgbd_dynamic",
            input_root=source,
            out_dir=tmp_path / "pack",
            train_routes=["route_a"],
            val_routes=["route_a"],
            grid_shape=(16, 16),
            meters_per_cell=0.1,
        )
    except ValueError as exc:
        assert "both train and val" in str(exc)
    else:
        raise AssertionError("train/val route leakage did not fail")


def test_tiny_tum_smoke_pack_cannot_pass_real_student_acceptance(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for route_index in range(3):
        _write_tum_sequence(source / f"route_{route_index}", frame_count=4)
    pack = tmp_path / "pack"
    build_real_rgbd_route_bev_pack(
        dataset="bonn_rgbd_dynamic",
        input_root=source,
        out_dir=pack,
        train_routes=["route_0", "route_1"],
        val_routes=["route_2"],
        frame_stride=1,
        grid_shape=(16, 16),
        meters_per_cell=0.1,
    )
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_family"] == "real_rgbd_route"
    assert manifest["no_fixture_data_used"] is True
    assert manifest["heldout_route_ids"] == ["route_2"]

    model = DirectBEVStudentV0(
        DirectBEVStudentV0Config(bev_shape=(16, 16), image_size=(32, 24), hidden_channels=8)
    )
    checkpoint = tmp_path / "direct" / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model,
        metadata={
            "source_dataset_name": "tiny_tum_schema_smoke",
            "train_route_ids": ["route_0", "route_1"],
            "heldout_route_ids": ["route_2"],
            "no_future_labels_used": True,
            "no_teacher_fields_at_runtime": True,
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "raw_pwm_emitted": False,
            "hardware_validated": False,
        },
        metrics={},
    )
    report_path = tmp_path / "eval.json"
    eval_direct_bev_student_v0(checkpoint=checkpoint, pack_dir=pack, split="val", out_path=report_path, device_name="cpu")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["accepted_real_rgbd_route_bev_student"] is False
    assert "real_source_frame_count_gte_1000" in report["acceptance"]["failed_checks"]


def _write_tum_sequence(root: Path, *, frame_count: int) -> None:
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    rgb_lines = ["# timestamp filename"]
    depth_lines = ["# timestamp filename"]
    gt_lines = ["# timestamp tx ty tz qx qy qz qw"]
    for index in range(frame_count):
        timestamp = 1.0 + index * 0.1
        rgb_rel = f"rgb/{timestamp:.6f}.ppm"
        depth_rel = f"depth/{timestamp:.6f}.png"
        rgb = np.zeros((6, 8, 3), dtype=np.uint8)
        rgb[:, :, route_color(index)] = 120
        depth = np.full((6, 8), 900 + 50 * index, dtype=np.uint16)
        _write_ppm(root / rgb_rel, rgb)
        _write_png16(root / depth_rel, depth)
        rgb_lines.append(f"{timestamp:.6f} {rgb_rel}")
        depth_lines.append(f"{timestamp:.6f} {depth_rel}")
        gt_lines.append(f"{timestamp:.6f} {index * 0.01:.6f} 0.0 0.0 0.0 0.0 0.0 1.0")
    (root / "rgb.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
    (root / "depth.txt").write_text("\n".join(depth_lines) + "\n", encoding="utf-8")
    (root / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")


def route_color(index: int) -> int:
    return index % 3


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
