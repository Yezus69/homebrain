import json
from pathlib import Path
import zlib

import numpy as np
import pytest

from homebrain.data.pack_spatial_dataset import pack_spatial_dataset
from homebrain.datasets.openloris_scene import load_calibration
from homebrain.datasets.openloris_to_route import openloris_to_route
from homebrain.datasets.tum_rgbd_to_route import tum_rgbd_to_route
from homebrain.geometry.rgbd_truth_to_bev import rgbd_truth_to_bev
from homebrain.geometry.qa_robot_frame_bev import qa_robot_frame_bev
from homebrain.geometry.robot_rgbd_to_bev import robot_rgbd_to_bev
from homebrain.geometry.validate_bev import validate_bev_artifacts
from homebrain.policies.audit_bev_action_sanity import audit_bev_action_sanity
from homebrain.policies.build_action_label_pack import build_action_label_pack
from homebrain.policies.compare_robot_frame_action_sources import compare_robot_frame_action_sources
from homebrain.policies.generate_controlled_bev_maps import generate_controlled_bev_maps
from homebrain.policies.qa_action_label_pack import qa_action_label_pack
from homebrain.policies.run_trajectory_scorer import run_trajectory_scorer
from homebrain.replay.segment_log import read_events
from homebrain.messages.schema import ImuEvent, OdomEvent, PoseEvent


def test_openloris_robot_frame_bridge_to_action_pack_v2(tmp_path: Path) -> None:
    openloris_source = tmp_path / "openloris" / "cafe1-1"
    openloris_route = tmp_path / "runs" / "openloris_cafe1_route"
    openloris_bev = openloris_route / "geometry" / "robot_rgbd_bev"
    openloris_pack = tmp_path / "runs" / "openloris_cafe1_robot_spatial_pack"
    openloris_policy = tmp_path / "runs" / "policy_openloris"
    controlled = tmp_path / "runs" / "controlled"
    tum_source = tmp_path / "tum" / "freiburg1_xyz"
    tum_route = tmp_path / "runs" / "tum_route"
    tum_bev = tum_route / "geometry" / "rgbd_truth_bev"
    tum_pack = tmp_path / "runs" / "tum_pack"
    action_pack = tmp_path / "runs" / "actions_v2"
    comparison = tmp_path / "runs" / "comparison"

    _write_openloris_fixture(openloris_source, frame_count=5, include_extrinsics=True)
    openloris_to_route(source_dir=openloris_source, out_dir=openloris_route, max_frames=5)
    route_metadata = json.loads((openloris_route / "route_metadata.json").read_text(encoding="utf-8"))
    associations = json.loads((openloris_route / "openloris_scene_associations.json").read_text(encoding="utf-8"))
    assert route_metadata["artifact_sha256"]["frames/frame_000000.ppm"]
    assert associations["frames"][0]["rgb_sha256"]
    assert associations["frames"][0]["depth_sha256"]
    events = read_events(openloris_route)
    assert any(isinstance(event, ImuEvent) for event in events)
    assert any(isinstance(event, OdomEvent) for event in events)
    assert any(isinstance(event, PoseEvent) for event in events)

    robot_rgbd_to_bev(log_dir=openloris_route, out_dir=openloris_bev, grid_cells=32, meters_per_cell=0.05)
    validation = validate_bev_artifacts(openloris_bev)
    assert validation.load_success is True
    assert validation.control_safe is False
    bev_manifest = json.loads((openloris_bev / "bev_manifest.json").read_text(encoding="utf-8"))
    assert bev_manifest["dataset_frame_type"] == "public_robot_mounted"
    assert bev_manifest["frame_type"] == "public_robot_frame_geometry"
    assert bev_manifest["robot_frame_truth"] is True

    pack_spatial_dataset(log_dir=openloris_route, bev_dir=openloris_bev, out_dir=openloris_pack)
    bev_qa = qa_robot_frame_bev(bev_dir=openloris_bev, spatial_pack=openloris_pack)
    assert bev_qa["control_safe"] is False
    assert bev_qa["qa_pass"] is True
    audit = audit_bev_action_sanity(source=openloris_pack, out_dir=tmp_path / "runs" / "audit_openloris")
    assert audit["control_safe"] is False
    assert audit["action_supervision_ok_fraction"] > 0.0

    run_trajectory_scorer(log_dir=openloris_pack, bev_source="labels", out_dir=openloris_policy)
    policy_eval = json.loads((openloris_policy / "trajectory_eval.json").read_text(encoding="utf-8"))
    assert policy_eval["replay_only"] is True
    assert policy_eval["not_executed"] is True
    assert policy_eval["control_safe"] is False

    generate_controlled_bev_maps(out_dir=controlled, examples_per_scenario=1, grid_size=32, meters_per_cell=0.05)
    _write_tum_fixture(tum_source, frame_count=4)
    tum_rgbd_to_route(source_dir=tum_source, out_dir=tum_route, max_frames=4)
    rgbd_truth_to_bev(log_dir=tum_route, out_dir=tum_bev)
    pack_spatial_dataset(log_dir=tum_route, bev_dir=tum_bev, out_dir=tum_pack)

    build_action_label_pack(
        sources=[controlled, openloris_pack, tum_pack],
        out_dir=action_pack,
        pack_version=2,
    )
    qa = qa_action_label_pack(action_pack)
    action_manifest = json.loads((action_pack / "manifest.json").read_text(encoding="utf-8"))
    assert action_manifest["schema_version"] == "homebrain.action_label_pack.v2"
    assert action_manifest["control_safe"] is False
    assert action_manifest["not_executed"] is True
    assert qa["action_label_pack_qa_pass"] is True
    assert qa["excluded_by_source"].get("tum_route", 0) > 0
    assert qa["per_source"]["tum_route"]["action_supervision_ok_fraction"] == 0.0

    report = compare_robot_frame_action_sources(
        sources=[openloris_pack, tum_pack],
        labels=["openloris_robot", "tum_geometry"],
        out_dir=comparison,
    )
    assert (comparison / "robot_frame_action_comparison.json").exists()
    assert (comparison / "robot_frame_action_comparison_contact_sheet.ppm").exists()
    assert report["per_source"]["openloris_robot"]["action_supervision_ok_fraction"] > 0.0
    assert report["per_source"]["tum_geometry"]["action_supervision_ok_fraction"] == 0.0


def test_robot_rgbd_to_bev_refuses_missing_extrinsics_without_review(tmp_path: Path) -> None:
    source = tmp_path / "openloris" / "cafe1-1"
    route = tmp_path / "route"
    bev = tmp_path / "bev"
    _write_openloris_fixture(source, frame_count=2, include_extrinsics=False)
    with pytest.raises(ValueError, match="requires measured calibration"):
        openloris_to_route(source_dir=source, out_dir=route, max_frames=2)

    openloris_to_route(
        source_dir=source,
        out_dir=route,
        max_frames=2,
        require_robot_frame_calibration=False,
    )

    with pytest.raises(ValueError, match="camera_to_base transform is missing"):
        robot_rgbd_to_bev(log_dir=route, out_dir=bev)

    robot_rgbd_to_bev(log_dir=route, out_dir=bev, review_assumed_extrinsics=True)
    manifest = json.loads((bev / "bev_manifest.json").read_text(encoding="utf-8"))
    assert manifest["robot_frame_truth"] is False
    assert manifest["not_robot_frame_truth"] is True
    assert manifest["control_safe"] is False


def test_openloris_trans_matrix_chain_imports_measured_camera_to_base(tmp_path: Path) -> None:
    source = tmp_path / "openloris" / "market1-1_3"
    route = tmp_path / "route"
    _write_openloris_fixture(source, frame_count=2, include_extrinsics=False)
    (source / "trans_matrix.yaml").write_text(
        """%YAML:1.0
trans_matrix:
   -
      parent_frame: base_link
      child_frame: laser
      matrix: !!opencv-matrix
         rows: 4
         cols: 4
         dt: d
         data: [ 1., 0., 0., 0.2, 0., 1., 0., 0.0, 0., 0., 1., 0.1, 0., 0., 0., 1. ]
   -
      parent_frame: laser
      child_frame: d400_color_optical_frame
      matrix: !!opencv-matrix
         rows: 4
         cols: 4
         dt: d
         data: [ 0., 0., 1., 0.3, -1., 0., 0., 0.0, 0., -1., 0., 0.4, 0., 0., 0., 1. ]
""",
        encoding="utf-8",
    )

    calibration = load_calibration(source, camera_id="d400_color")
    assert calibration["camera_to_base_source"]["transform_chain"] == [
        "base_link",
        "laser",
        "d400_color_optical_frame",
    ]
    assert calibration["camera_to_base"][0][3] == pytest.approx(0.5)

    openloris_to_route(source_dir=source, out_dir=route, max_frames=2)
    associations = json.loads((route / "openloris_scene_associations.json").read_text(encoding="utf-8"))
    assert associations["camera_to_base_source"]["source"] == "trans_matrix.yaml"
    assert associations["camera_to_base_source"]["transform_chain"][-1] == "d400_color_optical_frame"
    assert associations["robot_frame_truth"] is True


def test_robot_rgbd_to_bev_refuses_tum_without_camera_to_base(tmp_path: Path) -> None:
    source = tmp_path / "tum" / "freiburg2_pioneer_slam"
    route = tmp_path / "route"
    bev = tmp_path / "bev"
    _write_tum_fixture(source, frame_count=2)
    tum_rgbd_to_route(source_dir=source, out_dir=route, max_frames=2)

    with pytest.raises(ValueError, match="camera_to_base transform is missing"):
        robot_rgbd_to_bev(log_dir=route, out_dir=bev)


def test_action_label_pack_v3_schema(tmp_path: Path) -> None:
    controlled = tmp_path / "controlled"
    action_pack = tmp_path / "actions_v3"
    generate_controlled_bev_maps(out_dir=controlled, examples_per_scenario=1, grid_size=32, meters_per_cell=0.05)

    build_action_label_pack(sources=[controlled], out_dir=action_pack, pack_version=3)
    qa = qa_action_label_pack(action_pack)
    manifest = json.loads((action_pack / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "homebrain.action_label_pack.v3"
    assert manifest["version"] == 3
    assert manifest["control_safe"] is False
    assert manifest["not_executed"] is True
    assert qa["action_label_pack_qa_pass"] is True


def test_openloris_pose_deltas_and_action_label_pack_v4_future_motion(tmp_path: Path) -> None:
    openloris_source = tmp_path / "openloris" / "cafe1-1"
    route = tmp_path / "runs" / "openloris_route"
    bev = route / "geometry" / "robot_rgbd_bev"
    pack = tmp_path / "runs" / "openloris_pack"
    controlled = tmp_path / "runs" / "controlled"
    action_pack = tmp_path / "runs" / "actions_v4"

    _write_openloris_fixture(openloris_source, frame_count=24, include_extrinsics=True)
    openloris_to_route(source_dir=openloris_source, out_dir=route, max_frames=24)
    robot_rgbd_to_bev(log_dir=route, out_dir=bev, grid_cells=32, meters_per_cell=0.05)
    pack_spatial_dataset(log_dir=route, bev_dir=bev, out_dir=pack)
    pack_manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    assert pack_manifest["pose_label_count"] == 23
    assert pack_manifest["pose_label_frame"] == "robot_base_relative_pose"

    generate_controlled_bev_maps(out_dir=controlled, examples_per_scenario=2, grid_size=32, meters_per_cell=0.05)
    build_action_label_pack(sources=[controlled, pack], out_dir=action_pack, pack_version=4)
    qa = qa_action_label_pack(action_pack)
    manifest = json.loads((action_pack / "manifest.json").read_text(encoding="utf-8"))
    first = np.load(action_pack / manifest["examples"][0]["example_path"], allow_pickle=False)

    assert manifest["schema_version"] == "homebrain.action_label_pack.v4"
    assert manifest["future_motion_supervision"]["enabled"] is True
    assert manifest["balancing"]["enabled"] is True
    assert "raw_selected_distribution" in manifest
    assert "balanced_selected_distribution" in manifest
    assert "future_best_candidate_by_odom" in first
    assert "coverage_expert_selected_candidate_id" in first
    assert qa["future_motion_label_valid_count"] > 0
    assert qa["control_safe"] is False


def _write_openloris_fixture(root: Path, *, frame_count: int, include_extrinsics: bool) -> None:
    (root / "color").mkdir(parents=True, exist_ok=True)
    (root / "aligned_depth").mkdir(parents=True, exist_ok=True)
    color_lines = ["#Time filename"]
    depth_lines = ["#Time filename"]
    gt_lines = ["#Time px py pz qx qy qz qw"]
    odom_lines = ["#Time pose.position.x y z pose.orientation.x y z w twist.linear.x y z twist.angular.x y z"]
    acc_lines = ["#Time Ax Ay Az"]
    gyro_lines = ["#Time Gx Gy Gz"]
    for index in range(frame_count):
        timestamp = 1560000000.0 + index * 0.05
        rgb_rel = f"color/{timestamp:.6f}.ppm"
        depth_rel = f"aligned_depth/{timestamp:.6f}.png"
        rgb = np.full((64, 64, 3), 30 + index * 15, dtype=np.uint8)
        depth = _open_floor_depth(64, 64)
        _write_ppm(root / rgb_rel, rgb)
        _write_png16(root / depth_rel, depth)
        color_lines.append(f"{timestamp:.6f} {rgb_rel}")
        depth_lines.append(f"{timestamp:.6f} {depth_rel}")
        gt_lines.append(f"{timestamp:.8f} {index * 0.02:.6f} 0.0 0.0 0.0 0.0 0.0 1.0")
        odom_lines.append(
            f"{timestamp:.8f} {index * 0.02:.6f} 0.0 0.0 0.0 0.0 0.0 1.0 0.1 0.0 0.0 0.0 0.0 0.0"
        )
        acc_lines.append(f"{timestamp:.8f} 0.0 0.0 9.81")
        gyro_lines.append(f"{timestamp:.8f} 0.0 0.0 0.01")
    (root / "color.txt").write_text("\n".join(color_lines) + "\n", encoding="utf-8")
    (root / "aligned_depth.txt").write_text("\n".join(depth_lines) + "\n", encoding="utf-8")
    (root / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    (root / "odom.txt").write_text("\n".join(odom_lines) + "\n", encoding="utf-8")
    (root / "d400_accelerometer.txt").write_text("\n".join(acc_lines) + "\n", encoding="utf-8")
    (root / "d400_gyroscope.txt").write_text("\n".join(gyro_lines) + "\n", encoding="utf-8")
    calibration = {
        "intrinsics": {"d400_color": {"fx": 45.0, "fy": 45.0, "cx": 31.5, "cy": 31.5, "depth_scale": 1000.0}},
    }
    if include_extrinsics:
        calibration["camera_to_base"] = {
            "d400_color": [
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }
    (root / "openloris_calibration.json").write_text(json.dumps(calibration, sort_keys=True), encoding="utf-8")


def _open_floor_depth(height: int, width: int) -> np.ndarray:
    rows = np.linspace(0.05, 1.55, height, dtype=np.float32)[:, None]
    depth_m = np.repeat(rows, width, axis=1)
    return np.maximum(1, np.round(depth_m * 1000.0)).astype(np.uint16)


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
