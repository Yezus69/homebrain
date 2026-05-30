from pathlib import Path
import json
import zlib

import numpy as np
import pytest
import torch

from homebrain.brain.direct_bev_student_v0 import DirectBEVStudentV0, DirectBEVStudentV0Config, save_checkpoint as save_direct
from homebrain.brain.homebrain_net_v0 import HomeBrainNetV0, HomeBrainNetV0Config, save_checkpoint
from homebrain.brain.modeld import spatial_model_outputs
from homebrain.datasets.bonn_rgbd_to_route import bonn_rgbd_to_route
from homebrain.eval.eval_homebrain_net_v0 import eval_homebrain_net_v0, runtime_field_leakage_passed
from homebrain.replay.segment_log import read_events
from homebrain.train.dynamic_bev_world_pack_v0 import build_dynamic_bev_world_pack
from homebrain.train.real_rgbd_route_bev_pack import build_real_rgbd_route_bev_pack
from homebrain.train.train_homebrain_net_v0 import train_homebrain_net_v0
from dynamic_world_model_fixtures import write_real_rgbd_source_pack_fixture


def test_homebrain_net_v0_cpu_forward_and_head_toggles() -> None:
    model = HomeBrainNetV0(HomeBrainNetV0Config(bev_shape=(8, 8), image_size=(16, 12), hidden_channels=4, history_frames=2, future_horizons_sec=(0.5,)))
    rgb = torch.rand(1, 3, 12, 16)
    depth = torch.rand(1, 1, 12, 16)

    pose = model(rgb, depth=depth, enabled_heads={"pose"})
    occ = model(rgb, depth=depth, enabled_heads={"bev_occupancy"})
    depth_out = model(rgb, depth=depth, enabled_heads={"metric_depth"})
    dynamic = model(rgb, depth=depth, enabled_heads={"dynamic"})
    world = model(rgb, depth=depth, enabled_heads={"world_model"})

    assert pose["pose_delta"].shape == (1, 3)
    assert occ["bev_occupancy_logits"].shape == (1, 4, 8, 8)
    assert depth_out["metric_depth_m"].shape == (1, 1, 12, 16)
    assert dynamic["dynamic_occupancy_logits"].shape == (1, 1, 8, 8)
    assert world["future_occupied_logits"].shape == (1, 1, 8, 8)
    assert "bev_logits" not in pose


def test_homebrain_training_stages_write_metadata_loss_reports_and_warm_start(tmp_path: Path) -> None:
    pack = _tiny_real_pack(tmp_path)
    world_pack = _tiny_world_pack(tmp_path)
    direct = DirectBEVStudentV0(DirectBEVStudentV0Config(bev_shape=(8, 8), image_size=(16, 12), hidden_channels=4))
    direct_checkpoint = tmp_path / "direct" / "checkpoint.pt"
    save_direct(direct_checkpoint, direct, metadata={"replay_only": True}, metrics={})

    report = train_homebrain_net_v0(
        stage="perception",
        real_pack=pack,
        out_dir=tmp_path / "homebrain",
        device_name="cpu",
        batch_size=1,
        max_steps=0,
        image_size=(16, 12),
        hidden_channels=4,
        warm_start_direct_bev=direct_checkpoint,
    )
    checkpoint = torch.load(tmp_path / "homebrain" / "checkpoint.pt", map_location="cpu", weights_only=False)
    loss_report = json.loads((tmp_path / "homebrain" / "train_loss_report.json").read_text(encoding="utf-8"))

    assert report["model_type"] == "HomeBrainNetV0"
    assert checkpoint["metadata"]["stage"] == "perception"
    assert checkpoint["metadata"]["replay_only"] is True
    assert "loss_depth" in loss_report["loss_report"]
    assert checkpoint["metadata"]["warm_start_summary"]["copied_count"] > 0

    stage_expectations = {
        "memory": ({"real_pack": pack}, "loss_flow"),
        "world": ({"world_pack": world_pack}, "loss_candidate_bce"),
        "joint": ({"real_pack": pack}, "loss_pose"),
    }
    for stage, (packs, expected_loss) in stage_expectations.items():
        out_dir = tmp_path / f"homebrain_{stage}"
        train_homebrain_net_v0(
            stage=stage,
            out_dir=out_dir,
            device_name="cpu",
            batch_size=1,
            max_steps=0,
            image_size=(16, 12),
            hidden_channels=4,
            **packs,
        )
        checkpoint = torch.load(out_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        loss_report = json.loads((out_dir / "train_loss_report.json").read_text(encoding="utf-8"))
        assert checkpoint["metadata"]["stage"] == stage
        assert checkpoint["metadata"]["model_type"] == "HomeBrainNetV0"
        assert checkpoint["metadata"]["raw_pwm_emitted"] is False
        assert expected_loss in loss_report["loss_report"]


def test_homebrain_eval_reports_named_baselines_and_overlap_failure(tmp_path: Path) -> None:
    checkpoint = _homebrain_checkpoint(tmp_path)
    fake_pack = tmp_path / "fake_pack"
    fake_pack.mkdir()
    (fake_pack / "manifest.json").write_text(
        json.dumps(
            {
                "grid_shape": [8, 8],
                "synthetic_or_fixture": False,
                "real_source_route_count": 3,
                "real_source_frame_count": 1000,
                "train_route_ids": ["route_a"],
                "heldout_route_ids": ["route_a"],
                "examples": [],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "eval.json"
    eval_homebrain_net_v0(checkpoint=checkpoint, real_pack=fake_pack, split="val", out_path=out, device_name="cpu")
    report = json.loads(out.read_text(encoding="utf-8"))

    assert report["accepted"] is False
    assert report["baselines"]["pose"] == ["identity", "odom_integration"]
    assert report["baselines"]["depth"] == ["median_plane", "constant_depth"]
    assert report["baselines"]["occupancy"] == ["static_copy"]
    assert report["baselines"]["dynamic"] == ["previous_frame_copy", "zero_flow"]
    assert set(report["baselines"]["world"]) == {"static_copy", "previous_frame_copy", "center_prior", "transparent_scorer"}
    assert "route_heldout_no_overlap" in report["acceptance"]["failed_checks"]


def test_homebrain_runtime_leakage_guard_and_modeld_branch(tmp_path: Path) -> None:
    source = tmp_path / "source" / "route_a"
    _write_tum_sequence(source, frame_count=2)
    route = bonn_rgbd_to_route(source_dir=source, out_dir=tmp_path / "route", max_frames=1)
    checkpoint = _homebrain_checkpoint(tmp_path)
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)["metadata"]

    outputs, artifacts = spatial_model_outputs(read_events(route), log_dir=route, out_dir=tmp_path / "runtime", checkpoint=checkpoint, device_name="cpu")

    assert runtime_field_leakage_passed(metadata) is True
    assert outputs[0].source == "homebrain_net_v0"
    assert outputs[0].debug["replay_only"] is True
    assert outputs[0].debug["control_safe"] is False
    assert outputs[0].debug["no_future_labels_used_at_runtime"] is True
    assert artifacts and (tmp_path / "runtime" / artifacts[0]).exists()

    with pytest.raises(ValueError, match="forbids route ground-truth pose"):
        spatial_model_outputs(
            read_events(route),
            log_dir=route,
            out_dir=tmp_path / "runtime_route_pose",
            checkpoint=checkpoint,
            device_name="cpu",
            v1_pose_warp_source="route_pose",
        )


def _homebrain_checkpoint(tmp_path: Path) -> Path:
    model = HomeBrainNetV0(HomeBrainNetV0Config(bev_shape=(8, 8), image_size=(16, 12), hidden_channels=4, history_frames=2, future_horizons_sec=(0.5,)))
    path = tmp_path / "homebrain_ckpt" / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        metadata={
            "runtime_inputs": model.config.to_dict()["runtime_inputs"],
            "no_future_labels_used_at_runtime": True,
            "no_teacher_fields_at_runtime": True,
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "raw_pwm_emitted": False,
            "hardware_validated": False,
        },
        metrics={},
    )
    return path


def _tiny_real_pack(tmp_path: Path) -> Path:
    source = tmp_path / "routes"
    _write_tum_sequence(source / "route_a", frame_count=3)
    pack = tmp_path / "pack"
    build_real_rgbd_route_bev_pack(dataset="bonn_rgbd_dynamic", input_root=source, out_dir=pack, frame_stride=1, grid_shape=(8, 8), meters_per_cell=0.1)
    return pack


def _tiny_world_pack(tmp_path: Path) -> Path:
    source = tmp_path / "world_source"
    write_real_rgbd_source_pack_fixture(source, grid_shape=(8, 8), frame_count=4)
    pack = tmp_path / "world_pack"
    build_dynamic_bev_world_pack(
        input_pack=source,
        out_dir=pack,
        history_frames=2,
        future_horizons_sec=(0.5,),
        max_examples=4,
        robot_radius_m=0.05,
    )
    return pack


def _write_tum_sequence(root: Path, *, frame_count: int) -> None:
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    rgb_lines = ["# timestamp filename"]
    depth_lines = ["# timestamp filename"]
    gt_lines = ["# timestamp tx ty tz qx qy qz qw"]
    assoc_lines = ["# rgb_timestamp rgb_path depth_timestamp depth_path"]
    for index in range(frame_count):
        timestamp = 1.0 + index * 0.1
        rgb_rel = f"rgb/{timestamp:.6f}.ppm"
        depth_rel = f"depth/{timestamp:.6f}.png"
        rgb = np.zeros((6, 8, 3), dtype=np.uint8)
        rgb[:, :, index % 3] = 80 + index * 20
        depth = np.full((6, 8), 900 + 80 * index, dtype=np.uint16)
        _write_ppm(root / rgb_rel, rgb)
        _write_png16(root / depth_rel, depth)
        rgb_lines.append(f"{timestamp:.6f} {rgb_rel}")
        depth_lines.append(f"{timestamp:.6f} {depth_rel}")
        assoc_lines.append(f"{timestamp:.6f} {rgb_rel} {timestamp:.6f} {depth_rel}")
        gt_lines.append(f"{timestamp:.6f} {index * 0.02:.6f} 0 0 0 0 0 1")
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
