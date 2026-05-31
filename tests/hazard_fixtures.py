from __future__ import annotations

from pathlib import Path
import zlib

import numpy as np

from homebrain.geometry.bev_projector import BEV_SCHEMA_VERSION
from homebrain.teachers.artifacts import file_sha256, relative_to_root, save_array, write_json


def write_tum_sequence(root: Path, *, frame_count: int = 3) -> None:
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
        rgb[:, :, index % 3] = 120
        depth = np.full((6, 8), 1000, dtype=np.uint16)
        write_ppm(root / rgb_rel, rgb)
        write_png16(root / depth_rel, depth)
        rgb_lines.append(f"{timestamp:.6f} {rgb_rel}")
        depth_lines.append(f"{timestamp:.6f} {depth_rel}")
        gt_lines.append(f"{timestamp:.6f} {index * 0.01:.6f} 0.0 0.0 0.0 0.0 0.0 1.0")
    (root / "rgb.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
    (root / "depth.txt").write_text("\n".join(depth_lines) + "\n", encoding="utf-8")
    (root / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")


def write_hazard_bev_source(
    root: Path,
    *,
    route_id: str,
    grid_shape: tuple[int, int] = (16, 16),
    frame_count: int = 3,
    hazard_cell: tuple[int, int] = (10, 8),
    mock: bool = False,
    hand_verified: int = 0,
) -> None:
    frames = []
    positive_cells = 0
    for frame_id in range(frame_count):
        frame_dir = root / "frames" / f"front_rgb_{frame_id:06d}"
        hazard = np.zeros(grid_shape, dtype=np.float32)
        valid = np.ones(grid_shape, dtype=np.uint8)
        if frame_id == 0:
            hazard[hazard_cell] = 1.0
        positive_cells += int(np.count_nonzero(hazard > 0.5))
        artifacts = {}
        for kind, array in {"bev_hazard": hazard, "hazard_valid_mask": valid}.items():
            target = frame_dir / f"{kind}.npy"
            artifacts[kind] = {"kind": kind, "path": relative_to_root(target, root), **save_array(target, array)}
        metadata_path = frame_dir / "metadata.json"
        write_json(
            metadata_path,
            {
                "schema_version": BEV_SCHEMA_VERSION,
                "frame_id": frame_id,
                "weak_label": True,
                "control_safe": False,
                "trainable_for": "hazard_pretrain_only",
            },
        )
        frames.append(
            {
                "sequence_id": route_id,
                "camera_id": "front_rgb",
                "frame_id": frame_id,
                "timestamp_ns": 1_000_000_000 + frame_id,
                "width": 8,
                "height": 6,
                "weak_label": True,
                "control_safe": False,
                "metadata_path": relative_to_root(metadata_path, root),
                "metadata_sha256": file_sha256(metadata_path),
                "hazard_positive_cell_count": int(np.count_nonzero(hazard > 0.5)),
                "hazard_total_cell_count": int(hazard.size),
                "artifacts": artifacts,
            }
        )
    write_json(
        root / "bev_manifest.json",
        {
            "schema_version": BEV_SCHEMA_VERSION,
            "source_segment_id": route_id,
            "source_hazard_teacher_name": "hazard",
            "source_hazard_backend": "fake" if mock else "real",
            "source_hazard_mock": bool(mock),
            "source_hazard_synthetic": bool(mock),
            "source_hazard_real_perception": not mock,
            "hazard_source_model": "fixture_grounding_dino",
            "hazard_license_review_status": "apache-2.0_test_fixture",
            "grid_shape": list(grid_shape),
            "artifact_kinds": ["bev_hazard", "hazard_valid_mask"],
            "weak_label": True,
            "hazard_weak_label": True,
            "robot_frame_truth": False,
            "not_robot_frame_truth": True,
            "action_supervision_ok": False,
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "raw_pwm_emitted": False,
            "hardware_validated": False,
            "trainable_for": "hazard_pretrain_only",
            "hazard_positive_frame_count": 1 if positive_cells else 0,
            "hazard_positive_cell_fraction": float(positive_cells / max(frame_count * grid_shape[0] * grid_shape[1], 1)),
            "hand_verified_hazard_positive_frame_count": int(hand_verified),
            "frame_count": frame_count,
            "frames": frames,
        },
    )


def write_ppm(path: Path, image: np.ndarray) -> None:
    height, width, _channels = image.shape
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(image.astype(np.uint8).tobytes(order="C"))


def write_png16(path: Path, image: np.ndarray) -> None:
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
