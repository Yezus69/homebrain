import json
from pathlib import Path

import torch

from homebrain.brain.direct_bev_student_v0 import DirectBEVStudentV0, DirectBEVStudentV0Config
from homebrain.train.real_rgbd_route_bev_pack import build_real_rgbd_route_bev_pack
from homebrain.train.train_direct_bev_student_v0 import train_direct_bev_student_v0
from tests.hazard_fixtures import write_hazard_bev_source, write_tum_sequence


def test_direct_bev_student_hazard_head_cpu_forward() -> None:
    model = DirectBEVStudentV0(
        DirectBEVStudentV0Config(
            bev_shape=(8, 8),
            image_size=(16, 12),
            hidden_channels=4,
            compact_feature_channels=2,
        )
    )
    outputs = model(
        torch.zeros((2, 3, 12, 16), dtype=torch.float32),
        depth=torch.ones((2, 1, 12, 16), dtype=torch.float32),
        sensor_mask=torch.ones((2, 4), dtype=torch.float32),
        pose_delta_prev=torch.zeros((2, 3), dtype=torch.float32),
        previous_action=torch.zeros((2, 2), dtype=torch.float32),
    )

    assert outputs["bev_logits"].shape == (2, 5, 8, 8)
    assert outputs["hazard_logits"].shape == (2, 1, 8, 8)


def test_training_smoke_writes_hazard_losses_and_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    hazards = tmp_path / "hazards"
    pack = tmp_path / "pack"
    out = tmp_path / "direct"
    write_tum_sequence(source / "route_a", frame_count=3)
    write_hazard_bev_source(hazards / "route_a", route_id="route_a", frame_count=3, grid_shape=(8, 8), hazard_cell=(6, 4))
    build_real_rgbd_route_bev_pack(
        dataset="tum_rgbd",
        input_root=source,
        out_dir=pack,
        frame_stride=1,
        grid_shape=(8, 8),
        meters_per_cell=0.2,
        hazard_bev_root=hazards,
    )

    checkpoint = train_direct_bev_student_v0(
        pack_dir=pack,
        out_dir=out,
        device_name="cpu",
        batch_size=2,
        max_steps=1,
        val_every=1,
        image_size=(16, 12),
        hidden_channels=4,
    )
    report = json.loads((out / "training_metrics.json").read_text(encoding="utf-8"))

    assert checkpoint.exists()
    assert report["metadata"]["hazard_head"] is True
    assert report["metadata"]["hazard_trained"] is True
    assert "loss_hazard_bce" in report["metrics"]["final"]
    assert "loss_hazard_dice" in report["metrics"]["final"]
    assert "loss_hazard_total" in report["metrics"]["final"]
    assert report["metadata"]["no_teacher_fields_at_runtime"] is True
    assert report["metadata"]["control_safe"] is False

