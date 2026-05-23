import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from homebrain.brain.modeld import write_spatial_model_outputs
from homebrain.data.pack_spatial_dataset import pack_spatial_dataset
from homebrain.messages.schema import BrainOutputEvent
from homebrain.replay.segment_log import read_events
from homebrain.teachers.artifacts import load_array, load_teacher_manifest, validate_teacher_artifacts
from homebrain.teachers.dino_teacher import run_dino_teacher
from homebrain.train.eval_spatial_v0 import eval_spatial_v0
from homebrain.train.train_spatial_v0 import train_spatial_v0
from tests.test_data_spatial_dataset import _write_bev, _write_route


def test_fake_dino_teacher_is_explicitly_mock_synthetic(tmp_path) -> None:
    route = tmp_path / "route"
    features = tmp_path / "teacher_artifacts" / "dino_fake"
    _write_route(route, count=6)

    run_dino_teacher(route, features, backend_name="fake")

    manifest = load_teacher_manifest(features)
    assert manifest["teacher_name"] == "dino"
    assert manifest["mock"] is True
    assert manifest["synthetic"] is True
    assert manifest["real_perception"] is False
    assert manifest["runtime_dependency"] is False
    assert manifest["license_review_status"] == "pending_human_review"
    assert manifest["artifact_kinds"] == ["patch_features", "cls_feature"]
    first = manifest["frames"][0]
    patch_features = load_array(features / first["artifacts"]["patch_features"]["path"])
    cls_feature = load_array(features / first["artifacts"]["cls_feature"]["path"])
    assert patch_features.shape == (4, 4, 32)
    assert cls_feature.shape == (32,)

    validation = validate_teacher_artifacts(features)
    assert validation.teacher_mock_used is True
    assert validation.teacher_artifact_count == 12
    assert validation.artifact_load_success is True


def test_spatial_v0_tiny_train_eval_and_modeld_outputs(tmp_path) -> None:
    route = tmp_path / "route"
    bev = tmp_path / "bev"
    pack = tmp_path / "pack"
    features = tmp_path / "teacher_artifacts" / "dino_fake"
    train_out = tmp_path / "spatial_v0_overfit"
    eval_out = tmp_path / "eval.json"
    modeld_out = tmp_path / "modeld"
    _write_route(route, count=12)
    _write_bev(route, bev, count=12, confidence=0.8)
    pack_spatial_dataset(log_dir=route, bev_dir=bev, out_dir=pack)
    run_dino_teacher(route, features, backend_name="fake")

    metrics = train_spatial_v0(
        dataset_dir=pack,
        feature_dir=features,
        out_dir=train_out,
        max_steps=20,
        tiny_overfit=True,
        device_name="cpu",
        batch_size=4,
    )
    assert (train_out / "checkpoint.pt").exists()
    assert metrics["control_safe"] is False
    assert metrics["representation_pretraining_only"] is True
    assert metrics["train_loss_end"] <= metrics["train_loss_start"]

    eval_metrics = eval_spatial_v0(
        checkpoint=train_out / "checkpoint.pt",
        dataset_dir=pack,
        feature_dir=features,
        out_path=eval_out,
        device_name="cpu",
    )
    assert eval_metrics["control_safe"] is False
    assert eval_metrics["bev_iou_or_proxy"] >= 0.0
    assert json.loads(eval_out.read_text(encoding="utf-8"))["representation_pretraining_only"] is True

    write_spatial_model_outputs(
        route,
        modeld_out,
        checkpoint=train_out / "checkpoint.pt",
        feature_dir=features,
        device_name="cpu",
    )
    outputs = [event for event in read_events(modeld_out) if isinstance(event, BrainOutputEvent)]
    assert len(outputs) == 12
    assert outputs[0].local_bev_ref is not None
    assert outputs[0].cmd_vel is None
    assert outputs[0].selected_trajectory_id is None
    assert outputs[0].debug["control_safe"] is False
    assert outputs[0].debug["representation_pretraining_only"] is True
    with np.load(modeld_out / outputs[0].local_bev_ref, allow_pickle=False) as data:
        assert data["bev_logits"].shape == (5, 4, 4)
        assert data["uncertainty_grid"].shape == (4, 4)
