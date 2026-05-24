import json

import pytest

torch = pytest.importorskip("torch")

from homebrain.brain.modeld import write_spatial_model_outputs
from homebrain.brain.spatial_memory_v0 import SpatialMemoryNetConfig, SpatialMemoryNetV0
from homebrain.data.pack_spatial_dataset import pack_spatial_dataset
from homebrain.messages.schema import BrainOutputEvent
from homebrain.policies.build_action_label_pack import build_action_label_pack
from homebrain.policies.generate_controlled_bev_maps import generate_controlled_bev_maps
from homebrain.policies.trajectory_scorer_net_v0 import (
    TrajectoryScorerNetConfig,
    TrajectoryScorerNetV0,
    eval_trajectory_scorer_v0,
    save_trajectory_scorer_checkpoint,
    train_trajectory_scorer_v0,
)
from homebrain.replay.segment_log import read_events
from homebrain.teachers.dino_teacher import run_dino_teacher
from homebrain.train.train_spatial_v0 import train_spatial_v0
from tests.test_data_spatial_dataset import _write_bev, _write_route


def test_trajectory_scorer_v0_trains_evals_and_writes_contact_sheets(tmp_path) -> None:
    controlled = tmp_path / "controlled"
    action_pack = tmp_path / "actions"
    train_out = tmp_path / "scorer"
    eval_out = tmp_path / "eval.json"
    viz = tmp_path / "viz"
    generate_controlled_bev_maps(out_dir=controlled, examples_per_scenario=4, grid_size=24)
    build_action_label_pack(sources=[controlled], out_dir=action_pack, pack_version=3)

    train_metrics = train_trajectory_scorer_v0(
        action_pack=action_pack,
        out_dir=train_out,
        max_steps=40,
        batch_size=8,
        device_name="cpu",
    )
    assert (train_out / "checkpoint.pt").exists()
    assert train_metrics["replay_only"] is True
    assert train_metrics["not_executed"] is True
    assert train_metrics["control_safe"] is False
    assert train_metrics["product_training_approved"] is False

    eval_metrics = eval_trajectory_scorer_v0(
        checkpoint=train_out / "checkpoint.pt",
        action_pack=action_pack,
        out_path=eval_out,
        split="val",
        device_name="cpu",
        viz_dir=viz,
    )
    assert eval_metrics["top1_action_agreement"] > eval_metrics["random_top1_action_agreement"]
    assert eval_metrics["beats_random"] is True
    assert eval_metrics["replay_only"] is True
    assert eval_metrics["not_executed"] is True
    assert eval_metrics["control_safe"] is False
    assert eval_metrics["product_training_approved"] is False
    assert (viz / "bev_candidates_expert_learned_contact_sheet.ppm").exists()
    assert (viz / "disagreement_cases_contact_sheet.ppm").exists()
    assert json.loads(eval_out.read_text(encoding="utf-8"))["bev_source"] == "oracle"


def test_modeld_emits_replay_only_learned_trajectory_scores(tmp_path) -> None:
    route = tmp_path / "route"
    bev = tmp_path / "bev"
    pack = tmp_path / "pack"
    features = tmp_path / "dino_fake"
    spatial_out = tmp_path / "spatial"
    modeld_out = tmp_path / "modeld"
    scorer_checkpoint = tmp_path / "scorer" / "checkpoint.pt"
    _write_route(route, count=8)
    _write_bev(route, bev, count=8, confidence=0.8)
    pack_spatial_dataset(log_dir=route, bev_dir=bev, out_dir=pack)
    run_dino_teacher(route, features, backend_name="fake")
    train_spatial_v0(
        dataset_dir=pack,
        feature_dir=features,
        out_dir=spatial_out,
        max_steps=3,
        tiny_overfit=True,
        device_name="cpu",
        batch_size=4,
    )
    scorer = TrajectoryScorerNetV0(TrajectoryScorerNetConfig())
    save_trajectory_scorer_checkpoint(
        scorer_checkpoint,
        scorer,
        metadata={
            "meters_per_cell": 0.5,
            "robot_radius_m": 0.18,
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "product_training_approved": False,
        },
        metrics={
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "product_training_approved": False,
        },
    )

    write_spatial_model_outputs(
        route,
        modeld_out,
        checkpoint=spatial_out / "checkpoint.pt",
        feature_dir=features,
        trajectory_scorer_checkpoint=scorer_checkpoint,
        device_name="cpu",
    )

    outputs = [event for event in read_events(modeld_out) if isinstance(event, BrainOutputEvent)]
    assert outputs
    first = outputs[0]
    assert first.cmd_vel is None
    assert first.selected_trajectory_id is not None
    assert first.candidate_trajectories is not None
    assert len(first.candidate_trajectories) == 9
    assert first.debug["trajectory_scoring"] is True
    assert first.debug["replay_only"] is True
    assert first.debug["not_executed"] is True
    assert first.debug["control_safe"] is False
    assert first.debug["product_training_approved"] is False
    assert all(candidate["trajectory_score"]["replay_only"] is True for candidate in first.candidate_trajectories)
    assert all(candidate["trajectory_score"]["not_executed"] is True for candidate in first.candidate_trajectories)
    assert all(candidate["trajectory_score"]["control_safe"] is False for candidate in first.candidate_trajectories)
    assert all(candidate["trajectory_score"]["product_training_approved"] is False for candidate in first.candidate_trajectories)
