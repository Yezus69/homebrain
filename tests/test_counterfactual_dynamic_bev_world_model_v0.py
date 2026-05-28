from pathlib import Path
import json

import pytest

torch = pytest.importorskip("torch")

from dynamic_world_model_fixtures import write_real_rgbd_source_pack_fixture
from homebrain.brain.counterfactual_dynamic_bev_world_model_v0 import (
    CounterfactualDynamicBEVWorldModelV0,
    CounterfactualDynamicBEVWorldModelV0Config,
)
from homebrain.train.dynamic_bev_world_pack_v0 import build_dynamic_bev_world_pack
from homebrain.train.train_counterfactual_dynamic_bev_world_model_v0 import (
    DynamicBEVWorldDataset,
    train_counterfactual_dynamic_bev_world_model_v0,
)


def test_model_forward_cpu_tiny_tensors() -> None:
    model = CounterfactualDynamicBEVWorldModelV0(
        CounterfactualDynamicBEVWorldModelV0Config(
            bev_shape=(8, 8),
            history_frames=2,
            hidden_channels=8,
            future_horizons_sec=(0.5, 1.0),
            meters_per_cell=0.1,
            robot_radius_m=0.05,
        )
    )
    outputs = model(
        torch.zeros((1, 2, 7, 8, 8), dtype=torch.float32),
        torch.zeros((1, 2, 3), dtype=torch.float32),
        torch.zeros((1, 3, 4, 3), dtype=torch.float32),
        previous_action_history=torch.zeros((1, 2, 2), dtype=torch.float32),
        sensor_mask=torch.ones((1, 2, 4), dtype=torch.float32),
    )
    assert outputs["future_occupied_logits"].shape == (1, 2, 8, 8)
    assert outputs["future_flow_xy"].shape == (1, 2, 2, 8, 8)
    assert outputs["candidate_risk_logits"].shape == (1, 3)
    assert outputs["candidate_score"].shape == (1, 3)
    assert torch.isfinite(outputs["candidate_score"]).all()


def test_training_smoke_writes_checkpoint_metadata_and_loss_report(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dynamic = tmp_path / "dynamic"
    out = tmp_path / "model"
    write_real_rgbd_source_pack_fixture(source, frame_count=5)
    build_dynamic_bev_world_pack(
        input_pack=source,
        out_dir=dynamic,
        history_frames=2,
        future_horizons_sec=(0.5,),
        max_examples=8,
        robot_radius_m=0.05,
    )
    report = train_counterfactual_dynamic_bev_world_model_v0(
        pack=dynamic,
        out_dir=out,
        device_name="cpu",
        batch_size=2,
        max_steps=1,
        hidden_channels=8,
    )
    loss_report = json.loads((out / "train_loss_report.json").read_text(encoding="utf-8"))
    assert (out / "checkpoint.pt").exists()
    assert loss_report["model_type"] == "CounterfactualDynamicBEVWorldModelV0"
    assert loss_report["no_future_labels_used_at_runtime"] is True
    assert loss_report["control_safe"] is False
    for key in (
        "loss_future_occupied",
        "loss_future_dynamic_risk",
        "loss_future_flow",
        "loss_candidate_bce",
        "loss_candidate_rank",
        "loss_uncertainty",
        "loss_equivariance",
        "loss_total",
    ):
        assert key in report["loss_report"]


def test_dataset_rejects_empty_split(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dynamic = tmp_path / "dynamic"
    write_real_rgbd_source_pack_fixture(source, frame_count=3)
    build_dynamic_bev_world_pack(
        input_pack=source,
        out_dir=dynamic,
        history_frames=2,
        future_horizons_sec=(0.5,),
        max_examples=2,
        robot_radius_m=0.05,
    )
    with pytest.raises(ValueError):
        DynamicBEVWorldDataset(dynamic, split="does_not_exist")
