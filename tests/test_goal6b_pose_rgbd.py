import json
from pathlib import Path
import zlib

import numpy as np

from homebrain.data.pack_spatial_dataset import pack_spatial_dataset
from homebrain.data.qa_spatial_dataset import qa_spatial_dataset
from homebrain.datasets.tum_rgbd_to_route import tum_rgbd_to_route
from homebrain.geometry.compare_geometry_sources import compare_geometry_sources
from homebrain.geometry.da3_to_weak_bev import convert_da3_to_weak_bev
from homebrain.geometry.inspect_pose_sequence import inspect_pose_sequence
from homebrain.geometry.rgbd_truth_to_bev import rgbd_truth_to_bev
from homebrain.geometry.select_stable_windows import select_route_stable_windows
from homebrain.geometry.validate_bev import validate_bev_artifacts
from homebrain.replay.generate_dummy_log import generate_dummy_log
from homebrain.teachers.artifacts import load_array, save_array, write_json
from homebrain.teachers.run_teacher import main as run_teacher_main
from homebrain.geometry.qa_self_calibration import run_self_calibration_qa


def test_pose_inspection_stable_windows_and_window_gated_da3_bev(tmp_path) -> None:
    route = tmp_path / "dummy_route"
    artifacts = tmp_path / "da3_fake"
    qa_path = tmp_path / "da3_qa.json"
    inspect_dir = tmp_path / "inspect"
    stable_path = tmp_path / "stable_windows.json"
    bev = tmp_path / "da3_window_bev"
    generate_dummy_log(route)
    run_teacher_main(["--teacher", "da3", "--backend", "fake", "--log", str(route), "--out", str(artifacts)])
    _inject_pose_jump(artifacts, target_frame_id=3)

    run_self_calibration_qa(log_dir=route, teacher_artifacts_dir=artifacts, out_path=qa_path)
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    assert qa["pose_jump_outlier_count"] == 1

    pose_deltas_path, outlier_windows_path = inspect_pose_sequence(
        log_dir=route,
        teacher_artifacts_dir=artifacts,
        out_dir=inspect_dir,
    )
    pose_deltas = json.loads(pose_deltas_path.read_text(encoding="utf-8"))
    outlier_windows = json.loads(outlier_windows_path.read_text(encoding="utf-8"))
    assert pose_deltas["pose_jump_outlier_count"] == 1
    assert pose_deltas["outlier_deltas"][0]["previous_frame_id"] == 2
    assert pose_deltas["outlier_deltas"][0]["current_frame_id"] == 3
    assert outlier_windows["outlier_windows"][0]["frame_ids"]
    assert (inspect_dir / "contact_sheets" / "contact_sheet_manifest.json").exists()

    select_route_stable_windows(
        log_dir=route,
        teacher_artifacts_dir=artifacts,
        out_path=stable_path,
        min_window=2,
    )
    stable = json.loads(stable_path.read_text(encoding="utf-8"))
    assert stable["pose_outlier_deltas"][0]["current_frame_id"] == 3
    assert stable["excluded_frame_ids"] == [3]
    assert stable["accepted_window_count"] >= 1

    manifest_path = convert_da3_to_weak_bev(
        log_dir=route,
        teacher_artifacts_dir=artifacts,
        qa_path=qa_path,
        out_dir=bev,
        window_spec_path=stable_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["window_gated"] is True
    assert manifest["control_safe"] is False
    assert {frame["frame_id"] for frame in manifest["frames"]} == {0, 1, 2, 4, 5}
    validation = validate_bev_artifacts(bev)
    assert validation.load_success is True


def test_tum_rgbd_route_truth_bev_pack_qa_and_compare(tmp_path) -> None:
    source = tmp_path / "freiburg1_xyz"
    route = tmp_path / "tum_freiburg1_xyz_route"
    bev = route / "geometry" / "rgbd_truth_bev"
    pack = tmp_path / "tum_freiburg1_xyz_pack"
    qa_path = tmp_path / "tum_freiburg1_xyz_pack_qa.json"
    comparison_path = tmp_path / "geometry_compare.json"
    _write_tum_fixture(source, frame_count=12)

    tum_rgbd_to_route(source_dir=source, out_dir=route, max_frames=12)
    rgbd_truth_to_bev(log_dir=route, out_dir=bev)
    validation = validate_bev_artifacts(bev)
    assert validation.load_success is True
    assert validation.control_safe is False

    pack_spatial_dataset(log_dir=route, bev_dir=bev, out_dir=pack)
    qa = qa_spatial_dataset(pack)
    assert qa["example_count"] == 12
    assert qa["missing_count"] == 0
    assert qa["shape_error_count"] == 0
    assert qa["nan_count"] == 0
    assert qa["control_safe_true_count"] == 0
    assert qa["pose_label_count"] == 11
    assert qa["pose_label_frame"] == "camera_relative_dataset_pose"
    pack_manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    assert pack_manifest["not_robot_frame_truth"] is True
    first_example = np.load(pack / pack_manifest["examples"][0]["example_path"], allow_pickle=False)
    assert first_example["pose_delta_mask"].item() == 1.0
    assert str(first_example["pose_label_frame"].item()) == "camera_relative_dataset_pose"
    assert first_example["pose_delta"].shape == (3,)
    assert first_example["pose_delta"][0] != 0.0
    write_json(qa_path, qa)

    compare_geometry_sources(log_dir=route, source_a=bev, source_b=bev, out_path=comparison_path)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["overlap_frame_count"] == 12
    assert comparison["valid_mask_disagreement_mean"] == 0.0
    assert comparison["occupancy_disagreement_mean"] == 0.0


def _inject_pose_jump(artifacts: Path, *, target_frame_id: int) -> None:
    manifest_path = artifacts / "teacher_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["frames"]:
        if int(record["frame_id"]) < target_frame_id:
            continue
        artifact = record["artifacts"]["extrinsics"]
        path = artifacts / artifact["path"]
        extrinsics = load_array(path)
        extrinsics[0, 3] = extrinsics[0, 3] + np.float32(1.0)
        artifact.update(save_array(path, extrinsics.astype(np.float32)))
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _write_tum_fixture(root: Path, *, frame_count: int) -> None:
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    rgb_lines = ["# timestamp filename"]
    depth_lines = ["# timestamp filename"]
    gt_lines = ["# timestamp tx ty tz qx qy qz qw"]
    assoc_lines = []
    for index in range(frame_count):
        timestamp = 1305031453.0 + index * 0.033
        rgb_rel = f"rgb/{timestamp:.6f}.ppm"
        depth_rel = f"depth/{timestamp:.6f}.png"
        rgb = np.full((4, 4, 3), index * 7 % 255, dtype=np.uint8)
        depth = np.full((4, 4), 10_000 + index, dtype=np.uint16)
        _write_ppm(root / rgb_rel, rgb)
        _write_png16(root / depth_rel, depth)
        rgb_lines.append(f"{timestamp:.6f} {rgb_rel}")
        depth_lines.append(f"{timestamp:.6f} {depth_rel}")
        gt_lines.append(f"{timestamp:.6f} {index * 0.01:.6f} 0.0 0.0 0.0 0.0 0.0 1.0")
        assoc_lines.append(f"{timestamp:.6f} {rgb_rel} {timestamp:.6f} {depth_rel}")
    (root / "rgb.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
    (root / "depth.txt").write_text("\n".join(depth_lines) + "\n", encoding="utf-8")
    (root / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    (root / "associations.txt").write_text("\n".join(assoc_lines) + "\n", encoding="utf-8")


def _write_ppm(path: Path, image: np.ndarray) -> None:
    h, w, _channels = image.shape
    with path.open("wb") as handle:
        handle.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        handle.write(image.astype(np.uint8).tobytes(order="C"))


def _write_png16(path: Path, image: np.ndarray) -> None:
    values = np.asarray(image, dtype=np.uint16)
    h, w = values.shape
    raw_rows = bytearray()
    big = values.astype(">u2", copy=False)
    for row in big:
        raw_rows.append(0)
        raw_rows.extend(row.tobytes(order="C"))
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(_png_chunk(b"IHDR", w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([16, 0, 0, 0, 0])))
    png.extend(_png_chunk(b"IDAT", zlib.compress(bytes(raw_rows))))
    png.extend(_png_chunk(b"IEND", b""))
    path.write_bytes(bytes(png))


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return len(payload).to_bytes(4, "big") + kind + payload + crc.to_bytes(4, "big")
