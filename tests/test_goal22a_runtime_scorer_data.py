from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from homebrain.data.spatial_dataset import write_deterministic_npz, write_json
from homebrain.messages.schema import BrainOutputEvent
from homebrain.policies.candidate_trajectories import candidates_hash, generate_default_candidates
from homebrain.policies.calibrate_trajectory_scorer_bias import _fit_bias, _pair_bias_stats
from homebrain.policies.trajectory_scorer_net_v0 import (
    ActionLabelScorerDataset,
    CANDIDATE_FEATURE_NAMES,
    SIGNED_CANDIDATE_FEATURE_NAMES,
)
from homebrain.replay.segment_log import write_segment


def test_action_scorer_dataset_loads_oracle_and_v1_model_bevs(tmp_path) -> None:
    action_pack, modeld = _write_runtime_bev_fixture(tmp_path)

    oracle = ActionLabelScorerDataset(action_pack, split=None, bev_source="oracle")
    current = ActionLabelScorerDataset(action_pack, split=None, bev_source="v1_current_bev", modeld_dir=modeld)
    memory = ActionLabelScorerDataset(action_pack, split=None, bev_source="v1_memory_bev", modeld_dir=modeld)

    assert oracle.samples[0].bev.source == "labels"
    assert current.samples[0].bev.source == "v1_current_bev"
    assert memory.samples[0].bev.source == "v1_memory_bev"
    assert float(oracle.samples[0].bev.free.mean()) == pytest.approx(0.7)
    assert float(current.samples[0].bev.free.mean()) == pytest.approx(0.2)
    assert float(memory.samples[0].bev.free.mean()) == pytest.approx(0.9)
    with pytest.raises(ValueError, match="explicit"):
        ActionLabelScorerDataset(action_pack, split=None, bev_source="model", modeld_dir=modeld)


def test_signed_feature_ablation_is_applied_to_training_samples(tmp_path) -> None:
    action_pack, modeld = _write_runtime_bev_fixture(tmp_path)

    dataset = ActionLabelScorerDataset(
        action_pack,
        split=None,
        bev_source="v1_current_bev",
        modeld_dir=modeld,
        candidate_feature_mode="signed_ablation",
    )

    for name in SIGNED_CANDIDATE_FEATURE_NAMES:
        index = CANDIDATE_FEATURE_NAMES.index(name)
        assert np.allclose(dataset.samples[0].candidate_features[:, index], 0.0)
    assert dataset.feature_alignment["candidate_feature_mode"] == "signed_ablation"


def test_left_right_mirror_augmentation_swaps_expert_label(tmp_path) -> None:
    action_pack, modeld = _write_runtime_bev_fixture(tmp_path, selected_candidate_id="arc_left_small")

    dataset = ActionLabelScorerDataset(
        action_pack,
        split=None,
        bev_source="v1_memory_bev",
        modeld_dir=modeld,
        mirror_left_right=True,
    )

    selected_ids = [dataset.candidate_ids[sample.selected_index] for sample in dataset.samples]
    assert selected_ids == ["arc_left_small", "arc_right_small"]
    assert dataset.feature_alignment["mirror_left_right"] is True


def test_logit_bias_calibration_respects_left_right_pair_cap() -> None:
    candidate_ids = [
        "stop",
        "straight_short",
        "straight_medium",
        "arc_left_small",
        "arc_right_small",
        "arc_left_medium",
        "arc_right_medium",
        "rotate_left",
        "rotate_right",
    ]
    logits = np.zeros((6, len(candidate_ids)), dtype=np.float32)
    labels = np.asarray([2, 2, 4, 4, 5, 6], dtype=np.int64)

    bias, metrics = _fit_bias(
        logits,
        labels,
        candidate_count=len(candidate_ids),
        dominant_max=0.75,
        iterations=100,
        bias_range=2.0,
        max_pair_bias_delta=0.25,
        pair_bias_penalty=0.1,
        seed=22,
        candidate_ids=candidate_ids,
    )

    stats = _pair_bias_stats(candidate_ids, bias)
    assert stats["max_abs_pair_bias_delta"] <= 0.25 + 1e-6
    assert metrics["example_count"] == 6


def _write_runtime_bev_fixture(
    tmp_path,
    *,
    selected_candidate_id: str = "straight_short",
) -> tuple:
    action_pack = tmp_path / "actions"
    action_examples = action_pack / "examples"
    action_examples.mkdir(parents=True)
    source = tmp_path / "source_pack"
    source_examples = source / "examples"
    source_examples.mkdir(parents=True)
    modeld = tmp_path / "modeld"
    shape = (8, 8)
    candidates = generate_default_candidates(grid_shape=shape, meters_per_cell=0.1, robot_radius_m=0.1)
    candidate_ids = [candidate.id for candidate in candidates]
    selected_index = candidate_ids.index(selected_candidate_id)

    write_deterministic_npz(
        source_examples / "d400_color_000001.npz",
        {
            "bev_free": np.full(shape, 0.7, dtype=np.float32),
            "bev_obstacle": np.full(shape, 0.1, dtype=np.float32),
            "bev_unknown": np.full(shape, 0.2, dtype=np.float32),
            "bev_confidence": np.full(shape, 0.8, dtype=np.float32),
            "bev_traversable": np.full(shape, 0.7, dtype=np.float32),
            "bev_risky": np.full(shape, 0.1, dtype=np.float32),
        },
    )
    total_scores = np.full((len(candidate_ids),), 5.0, dtype=np.float32)
    total_scores[selected_index] = 0.0
    selected = np.zeros((len(candidate_ids),), dtype=np.bool_)
    selected[selected_index] = True
    write_deterministic_npz(
        action_examples / "action_000001.npz",
        {
            "total_expert_score": total_scores,
            "selected_by_expert": selected,
            "replay_only": np.asarray(True),
            "not_executed": np.asarray(True),
            "control_safe": np.asarray(False),
        },
    )
    manifest = {
        "package_type": "ActionLabelPack",
        "schema_version": "homebrain.action_label_pack.v5",
        "control_safe": False,
        "source_dirs": [source.as_posix()],
        "candidate_ids": candidate_ids,
        "candidate_hash": candidates_hash(candidates),
        "grid_shape": list(shape),
        "meters_per_cell": 0.1,
        "robot_radius_m": 0.1,
        "examples": [
            {
                "example_path": "examples/action_000001.npz",
                "source_ref": "examples/d400_color_000001.npz",
                "sequence_id": "seq",
                "camera_id": "d400_color",
                "frame_id": 1,
                "source_name": "fixture",
                "source_family": "public_robot_mounted",
                "action_supervision_ok": True,
                "control_safe": False,
            }
        ],
    }
    write_json(action_pack / "manifest.json", manifest, pretty=True)

    artifact_rel = "brain_outputs/spatial_v1/d400_color_000001.npz"
    write_deterministic_npz(
        modeld / artifact_rel,
        {
            "current_bev_free_prob": np.full(shape, 0.2, dtype=np.float32),
            "current_bev_occupied_prob": np.full(shape, 0.3, dtype=np.float32),
            "current_bev_unknown_prob": np.full(shape, 0.5, dtype=np.float32),
            "current_bev_traversable_prob": np.full(shape, 0.2, dtype=np.float32),
            "current_bev_risky_prob": np.full(shape, 0.3, dtype=np.float32),
            "memory_bev_free_prob": np.full(shape, 0.9, dtype=np.float32),
            "memory_bev_occupied_prob": np.full(shape, 0.05, dtype=np.float32),
            "memory_bev_unknown_prob": np.full(shape, 0.05, dtype=np.float32),
            "memory_bev_traversable_prob": np.full(shape, 0.9, dtype=np.float32),
            "memory_bev_risky_prob": np.full(shape, 0.05, dtype=np.float32),
            "uncertainty_grid": np.full(shape, 0.25, dtype=np.float32),
        },
    )
    event = BrainOutputEvent(
        timestamp_ns=1,
        sequence_id="seq",
        source="spatial_memory_net_v1",
        input_event_ids=["frame"],
        pose_confidence=1.0,
        local_bev_ref=artifact_rel,
        uncertainty=0.25,
        debug={"input_frame_id": 1, "camera_id": "d400_color"},
    )
    write_segment(modeld, [event], segment_id="modeld", artifact_files=[artifact_rel])
    return action_pack, modeld
