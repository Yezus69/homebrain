from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from homebrain.datasets.usage_policy import dataset_usage_policy
from homebrain.tools.goal12c_hard_validation import build_goal12c_report
from homebrain.train.spatial_v1_hard_eval import (
    deterministic_occlusion_keep_mask,
    evaluate_deployment_memory,
    evaluate_future_observation_memory,
    evaluate_pose_ablations,
)


def test_openloris_policy_is_poc_allowed_not_product_approved() -> None:
    policy = dataset_usage_policy("OpenLORIS-Scene")
    assert policy["license_name"] == "CC BY-ND 4.0"
    assert policy["poc_training_eval_allowed"] is True
    assert policy["product_training_approved"] is False
    assert policy["runtime_dependency"] is False
    assert policy["derived_dataset_redistribution_allowed"] is False
    assert policy["attribution_required"] is True


def test_deployment_eval_does_not_pass_label_update_masks() -> None:
    model = _FakeHardModel()
    loader = DataLoader([_batch()], batch_size=None)

    metrics = evaluate_deployment_memory(model, loader, device=torch.device("cpu"))

    assert model.observation_masks == [None]
    assert metrics["uses_label_update_masks"] is False
    assert metrics["observation_mask_source"] == "model_predicted_current_bev_confidence"


def test_future_observation_hidden_cell_metric_detects_memory_benefit() -> None:
    model = _FakeHardModel()
    loader = DataLoader([_batch()], batch_size=None)

    metrics = evaluate_future_observation_memory(model, loader, device=torch.device("cpu"), future_horizon=1)

    assert metrics["future_reveal_count"] > 0
    assert metrics["hidden_cell_iou"] > metrics["hidden_current_iou"]
    assert metrics["hidden_memory_benefit_pass"] is True


def test_occlusion_masks_are_deterministic() -> None:
    update = torch.ones((2, 3, 1, 6, 6), dtype=torch.float32)
    frame_ids = torch.arange(6, dtype=torch.int64).reshape(2, 3)

    first = deterministic_occlusion_keep_mask(update, mode="random_block_dropout", frame_ids=frame_ids)
    second = deterministic_occlusion_keep_mask(update, mode="random_block_dropout", frame_ids=frame_ids)
    corridor = deterministic_occlusion_keep_mask(update, mode="forward_corridor_dropout", frame_ids=frame_ids)
    side = deterministic_occlusion_keep_mask(update, mode="side_region_dropout", frame_ids=frame_ids)

    assert torch.equal(first, second)
    assert torch.count_nonzero(first == 0) > 0
    assert torch.count_nonzero(corridor == 0) > 0
    assert torch.count_nonzero(side == 0) > 0


def test_pose_ablation_is_deterministic() -> None:
    model = _FakeHardModel()
    first = evaluate_pose_ablations(model, DataLoader([_batch()], batch_size=None), device=torch.device("cpu"))
    second = evaluate_pose_ablations(model, DataLoader([_batch()], batch_size=None), device=torch.device("cpu"))

    assert first["table"] == second["table"]
    assert "route_pose" in {row["pose_mode"] for row in first["table"]}
    assert "corrupted_route_pose" in {row["pose_mode"] for row in first["table"]}


def test_goal12c_report_structure_contains_true_route_out_and_hard_metrics() -> None:
    hard = {
        "deployment_style": {
            "deployment_memory_iou": 0.4,
            "deployment_current_iou": 0.3,
            "deployment_memory_delta": 0.1,
            "deployment_update_mask_coverage": 0.2,
            "deployment_overwrite_fraction": 0.05,
            "deployment_memory_benefit_pass": True,
        },
        "future_observation": {
            "hidden_cell_iou": 0.4,
            "hidden_current_iou": 0.2,
            "hidden_memory_delta": 0.2,
            "hidden_free_iou": 0.5,
            "hidden_obstacle_iou": 0.3,
            "future_reveal_count": 12,
            "hidden_memory_benefit_pass": True,
        },
        "occlusion_stress": {
            "occluded_current_iou": 0.2,
            "occluded_memory_iou": 0.4,
            "occlusion_recovery_delta": 0.2,
            "occlusion_recovery_pass": True,
        },
        "pose_ablation": {
            "table": [{"pose_mode": "route_pose", "deployment_memory_iou": 0.4, "deployment_current_iou": 0.3, "deployment_memory_delta": 0.1}],
            "pose_ablation_pass": True,
        },
    }
    true_route_out = {
        "available": True,
        "blocked": False,
        "true_leave_one_route_out": True,
        "fold_count": 3,
        "folds": [],
        "errors": [],
    }

    report = build_goal12c_report(
        hard_metrics=hard,
        true_route_out=true_route_out,
        dataset_manifest="manifest.json",
        v1_window1_checkpoint="window1.pt",
        v1_window4_checkpoint="window4.pt",
        v0_checkpoint="v0.pt",
    )

    assert report["schema_version"] == "homebrain.goal12c_spatial_memory_v1_hard_validation_report.v0"
    assert report["openloris_poc_allowed_status"]["poc_training_eval_allowed"] is True
    assert report["openloris_poc_allowed_status"]["product_training_approved"] is False
    assert report["true_route_out_metrics"]["true_leave_one_route_out"] is True
    assert report["pass_fail_gates"]["goal12c_hard_validation_pass"] is True


class _FakeHardModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(meters_per_cell=1.0)
        self.observation_masks: list[object] = []

    def forward_sequence(
        self,
        features: torch.Tensor,
        timestamp_s: torch.Tensor,
        sensor_mask: torch.Tensor,
        *,
        pose_delta_to_current: torch.Tensor | None = None,
        pose_delta_to_current_mask: torch.Tensor | None = None,
        observation_mask: torch.Tensor | None = None,
    ) -> dict[str, object]:
        del timestamp_s, sensor_mask, pose_delta_to_current, pose_delta_to_current_mask
        self.observation_masks.append(observation_mask)
        batch, steps = features.shape[:2]
        height = width = 4
        channels = 5
        current = torch.full((batch, steps, channels, height, width), -4.0)
        current[:, :, 2] = 4.0
        memory = current.clone()
        memory[:, :, 2, 2, 2] = -4.0
        memory[:, :, 0, 2, 2] = 4.0
        memory[:, :, 3, 2, 2] = 4.0
        update = torch.zeros((batch, steps, 1, height, width), dtype=torch.float32)
        if observation_mask is not None:
            update = observation_mask.to(dtype=torch.float32)
        pose = torch.zeros((batch, steps, 3), dtype=torch.float32)
        coverage = update.float().mean(dim=(2, 3, 4))
        debug = {
            "update_mask_coverage": coverage,
            "memory_overwrite_fraction": torch.zeros((batch, steps), dtype=torch.float32),
            "pose_warp_valid": torch.ones((batch, steps), dtype=torch.bool),
            "memory_reset": torch.zeros((batch, steps), dtype=torch.bool),
            "memory_used": torch.ones((batch, steps), dtype=torch.bool),
            "pose_warp_used": torch.ones((batch, steps), dtype=torch.bool),
        }
        return {
            "current_bev_logits": current,
            "fused_memory_bev_logits": memory,
            "update_mask": update,
            "pose_delta": pose,
            "debug": debug,
        }


def _batch() -> dict[str, torch.Tensor]:
    labels = torch.zeros((1, 2, 5, 4, 4), dtype=torch.float32)
    labels[:, :, 2] = 1.0
    labels[:, 1, 2, 2, 2] = 0.0
    labels[:, 1, 0, 2, 2] = 1.0
    labels[:, 1, 3, 2, 2] = 1.0
    pose_mask = torch.zeros((1, 2, 1), dtype=torch.float32)
    pose_mask[:, 1] = 1.0
    return {
        "features": torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32),
        "timestamp_s": torch.zeros((1, 2, 1), dtype=torch.float32),
        "sensor_mask": torch.zeros((1, 2, 4), dtype=torch.float32),
        "bev_labels": labels,
        "bev_label_mask": torch.ones_like(labels),
        "pose_delta_to_current": torch.zeros((1, 2, 3), dtype=torch.float32),
        "pose_delta_to_current_mask": pose_mask,
        "frame_id": torch.tensor([[0, 1]], dtype=torch.int64),
    }
