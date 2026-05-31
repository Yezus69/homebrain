import json
from pathlib import Path

import pytest

from homebrain.brain.direct_bev_student_v0 import DirectBEVStudentV0, DirectBEVStudentV0Config, save_checkpoint
from homebrain.eval.eval_bev_hazard_v0 import eval_bev_hazard_v0
from homebrain.train.real_rgbd_route_bev_pack import build_real_rgbd_route_bev_pack
from tests.hazard_fixtures import write_hazard_bev_source, write_tum_sequence


def test_eval_reports_required_hazard_metrics_and_rejects_fake_acceptance(tmp_path: Path) -> None:
    source = tmp_path / "source"
    hazards = tmp_path / "hazards"
    for route_index in range(3):
        write_tum_sequence(source / f"route_{route_index}", frame_count=3)
    write_hazard_bev_source(
        hazards / "route_2",
        route_id="route_2",
        frame_count=3,
        grid_shape=(8, 8),
        hazard_cell=(6, 4),
        mock=True,
        hand_verified=3,
    )
    pack = tmp_path / "pack"
    build_real_rgbd_route_bev_pack(
        dataset="tum_rgbd",
        input_root=source,
        out_dir=pack,
        train_routes=["route_0", "route_1"],
        val_routes=["route_2"],
        frame_stride=1,
        grid_shape=(8, 8),
        meters_per_cell=0.2,
        hazard_bev_root=hazards,
    )
    model = DirectBEVStudentV0(DirectBEVStudentV0Config(bev_shape=(8, 8), image_size=(16, 12), hidden_channels=4))
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model,
        metadata={
            "source_dataset_name": "hazard_eval_fixture",
            "train_route_ids": ["route_0", "route_1"],
            "heldout_route_ids": ["route_2"],
            "hazard_head": True,
            "hazard_trained": True,
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
    eval_bev_hazard_v0(checkpoint=checkpoint, pack_dir=pack, split="val", out_path=report_path, device_name="cpu")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["accepted_bev_hazard_v0"] is False
    metrics = report["metrics"]
    for key in (
        "hazard_auprc",
        "hazard_f1",
        "hazard_recall_thin_floor_objects",
        "depth_occupancy_only_thin_recall",
        "risky_channel_only_hazard_auprc",
        "free_space_false_positive_rate",
        "candidate_hazard_exposure_mean",
        "candidate_hazard_exposure_vs_depth_only_scorer",
        "route_heldout_count",
        "per_route_hazard_metric_json",
        "worst_route_hazard_recall",
        "runtime_field_leakage_passed",
        "weak_label_provenance_present",
    ):
        assert key in metrics
    assert metrics["runtime_field_leakage_passed"] is True
    assert "hazard_source_not_mock_or_synthetic" in report["acceptance"]["failed_checks"]


def test_hazard_eval_train_val_route_overlap_fails_in_pack_builder(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_tum_sequence(source / "route_a", frame_count=3)

    with pytest.raises(ValueError, match="both train and val"):
        build_real_rgbd_route_bev_pack(
            dataset="tum_rgbd",
            input_root=source,
            out_dir=tmp_path / "pack",
            train_routes=["route_a"],
            val_routes=["route_a"],
            grid_shape=(8, 8),
            meters_per_cell=0.2,
        )

