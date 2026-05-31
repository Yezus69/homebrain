import torch

from homebrain.brain.direct_bev_student_v0 import DirectBEVStudentV0, DirectBEVStudentV0Config, load_checkpoint, save_checkpoint


def test_direct_bev_student_v0_cpu_smoke_and_checkpoint_roundtrip(tmp_path) -> None:
    model = DirectBEVStudentV0(
        DirectBEVStudentV0Config(
            bev_shape=(16, 16),
            image_size=(32, 24),
            hidden_channels=8,
            compact_feature_channels=4,
        )
    )
    outputs = model(
        torch.zeros((2, 3, 24, 32), dtype=torch.float32),
        depth=torch.ones((2, 1, 24, 32), dtype=torch.float32),
        sensor_mask=torch.ones((2, 4), dtype=torch.float32),
        pose_delta_prev=torch.zeros((2, 3), dtype=torch.float32),
        previous_action=torch.zeros((2, 2), dtype=torch.float32),
    )

    assert outputs["bev_logits"].shape == (2, 5, 16, 16)
    assert outputs["hazard_logits"].shape == (2, 1, 16, 16)
    assert outputs["uncertainty_logits"].shape == (2, 1, 16, 16)
    assert outputs["dynamic_risk_logits"].shape == (2, 1, 16, 16)
    assert outputs["compact_features"].shape == (2, 4, 16, 16)

    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model,
        metadata={
            "source_dataset_name": "tiny_schema_smoke",
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
    loaded, payload = load_checkpoint(checkpoint)
    assert loaded.config.bev_shape == (16, 16)
    assert payload["metadata"]["no_future_labels_used"] is True
