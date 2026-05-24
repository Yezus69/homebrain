import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from homebrain.brain.modeld import write_spatial_model_outputs
from homebrain.brain.spatial_memory_v1 import (
    SpatialMemoryNetV1,
    SpatialMemoryNetV1Config,
    load_checkpoint,
    save_checkpoint,
    warp_memory_se2,
)
from homebrain.data.pack_spatial_dataset import pack_spatial_dataset
from homebrain.messages.schema import BrainOutputEvent, FrameEvent
from homebrain.replay.replayd import replay_log
from homebrain.replay.segment_log import read_events, write_segment
from homebrain.teachers.dino_teacher import run_dino_teacher
from homebrain.train.spatial_temporal_dataset import (
    SpatialTemporalTrainDataset,
    build_temporal_memory_targets,
)
from homebrain.train.train_spatial_v1 import train_spatial_v1
from tests.test_data_spatial_dataset import _write_bev, _write_route


def test_se2_warp_forward_lateral_yaw_and_missing_pose() -> None:
    memory = torch.zeros((1, 1, 7, 7), dtype=torch.float32)
    memory[0, 0, 4, 3] = 1.0
    forward = warp_memory_se2(
        memory,
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[1.0]]),
        meters_per_cell=1.0,
        mode="nearest",
    )
    assert forward.tensor[0, 0, 5, 3] == 1.0
    assert bool(forward.pose_warp_valid[0]) is True

    lateral = warp_memory_se2(
        memory,
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([[1.0]]),
        meters_per_cell=1.0,
        mode="nearest",
    )
    assert lateral.tensor[0, 0, 4, 2] == 1.0

    yaw_memory = torch.zeros((1, 1, 7, 7), dtype=torch.float32)
    yaw_memory[0, 0, 5, 3] = 1.0
    yaw = warp_memory_se2(
        yaw_memory,
        torch.tensor([[0.0, 0.0, np.pi / 2.0]], dtype=torch.float32),
        torch.tensor([[1.0]]),
        meters_per_cell=1.0,
        mode="nearest",
    )
    assert yaw.tensor[0, 0, 6, 2] == 1.0

    reset = warp_memory_se2(memory + 2.0, None, None, meters_per_cell=1.0, missing_pose_behavior="reset")
    assert torch.count_nonzero(reset.tensor) == 0
    assert bool(reset.memory_reset[0]) is True
    no_warp = warp_memory_se2(memory, None, None, meters_per_cell=1.0, missing_pose_behavior="no_warp")
    assert torch.equal(no_warp.tensor, memory)
    masked = warp_memory_se2(memory, None, None, meters_per_cell=1.0, missing_pose_behavior="masked_update")
    assert torch.equal(masked.tensor, memory)


def test_temporal_sequence_windows_do_not_invent_pose(tmp_path: Path) -> None:
    route = tmp_path / "route"
    bev = tmp_path / "bev"
    pack = tmp_path / "pack"
    features = tmp_path / "dino_fake"
    _write_route(route, count=12)
    _write_bev(route, bev, count=12, confidence=0.8)
    pack_spatial_dataset(log_dir=route, bev_dir=bev, out_dir=pack)
    run_dino_teacher(route, features, backend_name="fake")

    dataset = SpatialTemporalTrainDataset(pack, feature_dir=features, split="train", window_length=3)
    assert len(dataset) > 0
    item = dataset[0]
    assert item["features"].shape[0] == 3
    assert torch.count_nonzero(item["pose_delta_to_current_mask"]) == 0
    assert item["target_uses_model_predictions"] is False
    assert item["control_safe"].item() is False


def test_temporal_label_merge_warps_previous_labels() -> None:
    labels = torch.zeros((1, 2, 5, 5, 5), dtype=torch.float32)
    unknown = 2
    labels[:, :, unknown] = 1.0
    labels[0, 0, unknown, 2, 2] = 0.0
    labels[0, 0, 0, 2, 2] = 1.0
    labels[0, 1, unknown, 0, 0] = 1.0
    observation = torch.zeros((1, 2, 1, 5, 5), dtype=torch.float32)
    observation[0, 0, 0, 2, 2] = 1.0
    pose = torch.zeros((1, 2, 3), dtype=torch.float32)
    pose[0, 1, 0] = 1.0
    pose_mask = torch.zeros((1, 2, 1), dtype=torch.float32)
    pose_mask[0, 1, 0] = 1.0

    targets = build_temporal_memory_targets(
        labels,
        observation,
        pose,
        pose_mask,
        meters_per_cell=1.0,
    )
    fused = targets["fused_memory_targets"]
    assert fused[0, 1, 0, 3, 2] == 1.0
    assert fused[0, 1, unknown, 3, 2] == 0.0
    assert targets["observed_mask_targets"][0, 1, 0, 3, 2] == 1.0


def test_spatial_memory_v1_forward_save_load(tmp_path: Path) -> None:
    config = SpatialMemoryNetV1Config(feature_dim=8, bev_shape=(6, 6), hidden_channels=16, sensor_dim=5)
    model = SpatialMemoryNetV1(config)
    features = torch.randn((2, 3, 8, 4, 4), dtype=torch.float32)
    timestamps = torch.zeros((2, 3, 1), dtype=torch.float32)
    masks = torch.zeros((2, 3, 4), dtype=torch.float32)
    pose = torch.zeros((2, 3, 3), dtype=torch.float32)
    pose_mask = torch.ones((2, 3, 1), dtype=torch.float32)
    outputs = model.forward_sequence(features, timestamps, masks, pose_delta_to_current=pose, pose_delta_to_current_mask=pose_mask)
    assert outputs["current_bev_logits"].shape == (2, 3, 5, 6, 6)
    assert outputs["fused_memory_bev_logits"].shape == (2, 3, 5, 6, 6)
    assert outputs["uncertainty_grid"].shape == (2, 3, 1, 6, 6)
    assert outputs["debug"]["memory_reset"].shape == (2, 3)

    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, model, metadata={"control_safe": False}, metrics={"ok": True})
    loaded, payload = load_checkpoint(checkpoint)
    loaded_outputs = loaded.forward_sequence(features, timestamps, masks, pose_delta_to_current=pose, pose_delta_to_current_mask=pose_mask)
    assert payload["model_name"] == "SpatialMemoryNetV1"
    assert loaded_outputs["current_bev_logits"].shape == outputs["current_bev_logits"].shape


def test_spatial_v1_tiny_temporal_overfit_and_modeld_replay(tmp_path: Path) -> None:
    route = tmp_path / "route"
    bev = tmp_path / "bev"
    pack = tmp_path / "pack"
    features = tmp_path / "dino_fake"
    train_out = tmp_path / "spatial_v1"
    modeld_route = tmp_path / "modeld_route"
    modeld_features = tmp_path / "modeld_dino_fake"
    modeld_out = tmp_path / "modeld"
    replay_out = tmp_path / "replay"
    _write_route(route, count=12)
    _write_bev(route, bev, count=12, confidence=0.8)
    pack_spatial_dataset(log_dir=route, bev_dir=bev, out_dir=pack)
    run_dino_teacher(route, features, backend_name="fake")

    metrics = train_spatial_v1(
        dataset_dir=pack,
        feature_dir=features,
        out_dir=train_out,
        max_steps=20,
        window_length=2,
        tiny_overfit=True,
        device_name="cpu",
        batch_size=2,
        learning_rate=2e-3,
    )
    assert (train_out / "checkpoint.pt").exists()
    assert metrics["train_loss_end"] <= metrics["train_loss_start"]
    assert metrics["control_safe"] is False

    _write_two_sequence_route(modeld_route)
    run_dino_teacher(modeld_route, modeld_features, backend_name="fake")
    write_spatial_model_outputs(
        modeld_route,
        modeld_out,
        checkpoint=train_out / "checkpoint.pt",
        feature_dir=modeld_features,
        device_name="cpu",
    )
    outputs = [event for event in read_events(modeld_out) if isinstance(event, BrainOutputEvent)]
    assert len(outputs) == 6
    assert outputs[0].source == "spatial_memory_net_v1"
    assert outputs[0].cmd_vel is None
    assert outputs[0].debug["memory_reset"] is True
    assert outputs[1].debug["memory_used"] is True
    assert outputs[3].debug["memory_reset"] is True
    assert outputs[0].debug["control_safe"] is False
    assert outputs[0].debug["replay_only"] is True
    with np.load(modeld_out / outputs[1].local_bev_ref, allow_pickle=False) as data:
        assert data["current_bev_logits"].shape == (5, 4, 4)
        assert data["memory_bev_logits"].shape == (5, 4, 4)
        assert data["memory_used"][0].item() is True

    replay_log(
        modeld_route,
        replay_out,
        checkpoint=train_out / "checkpoint.pt",
        feature_dir=modeld_features,
        device_name="cpu",
    )
    replay_outputs = [event for event in read_events(replay_out) if isinstance(event, BrainOutputEvent)]
    assert len(replay_outputs) == 6
    assert replay_outputs[3].debug["memory_reset"] is True
    assert replay_outputs[0].debug["product_training_approved"] is False


def _write_two_sequence_route(route: Path) -> None:
    frames = []
    for index in range(6):
        sequence_id = "sequence_a" if index < 3 else "sequence_b"
        frame = FrameEvent(
            timestamp_ns=1_700_000_000_000_000_000 + index * 100_000_000,
            sequence_id=sequence_id,
            source="test",
            camera_id="front_rgb",
            frame_id=index,
            width=8,
            height=8,
            format="encoded_ppm",
            data_ref=f"frames/frame_{index:06d}.ppm",
            intrinsics={"available": False},
        )
        target = route / frame.data_ref
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
        frames.append(frame)
    write_segment(route, frames, segment_id=route.name, artifact_files=[frame.data_ref for frame in frames])
