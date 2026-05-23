import json

import numpy as np

from homebrain.data.spatial_dataset import load_example_npz
from homebrain.policies.audit_stop_heavy import audit_stop_heavy
from homebrain.policies.build_action_label_pack import build_action_label_pack
from homebrain.policies.generate_controlled_bev_maps import DEFAULT_SCENARIOS, generate_controlled_bev_maps
from homebrain.policies.qa_action_label_pack import qa_action_label_pack
from homebrain.policies.run_trajectory_scorer import run_trajectory_scorer


def test_controlled_bev_maps_pack_expected_scenarios(tmp_path) -> None:
    pack = tmp_path / "controlled"
    generate_controlled_bev_maps(out_dir=pack, examples_per_scenario=2, grid_size=24)

    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["controlled_bev_map_pack"] is True
    assert manifest["example_count"] == len(DEFAULT_SCENARIOS) * 2
    assert manifest["replay_only"] is True
    assert manifest["not_executed"] is True
    assert manifest["control_safe"] is False
    assert set(manifest["scenarios"]) == set(DEFAULT_SCENARIOS)


def test_action_label_pack_selects_motion_on_open_maps(tmp_path) -> None:
    controlled = tmp_path / "controlled"
    action_pack = tmp_path / "actions"
    generate_controlled_bev_maps(out_dir=controlled, examples_per_scenario=3, grid_size=24)

    build_action_label_pack(sources=[controlled], out_dir=action_pack)
    qa = qa_action_label_pack(action_pack)

    manifest = json.loads((action_pack / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["package_type"] == "ActionLabelPack"
    assert manifest["replay_only"] is True
    assert manifest["not_executed"] is True
    assert manifest["control_safe"] is False
    assert qa["example_count"] == len(DEFAULT_SCENARIOS) * 3
    assert qa["selected_motion_fraction"] > 0.0

    open_records = [record for record in manifest["examples"] if record["scenario_name"] == "open_room"]
    assert open_records
    for record in open_records:
        example = load_example_npz(action_pack / record["example_path"])
        selected = str(np.asarray(example["selected_candidate_id"]).item())
        assert selected != "stop"
        assert bool(np.asarray(example["replay_only"]).item()) is True
        assert bool(np.asarray(example["not_executed"]).item()) is True
        assert bool(np.asarray(example["control_safe"]).item()) is False


def test_stop_audit_writes_metrics_contact_sheet_and_label_comparison(tmp_path) -> None:
    controlled = tmp_path / "controlled"
    policy = tmp_path / "policy"
    audit = tmp_path / "audit"
    generate_controlled_bev_maps(out_dir=controlled, examples_per_scenario=2, grid_size=24)
    run_trajectory_scorer(log_dir=controlled, out_dir=policy, bev_source="labels")

    metrics = audit_stop_heavy(policy_dir=policy, out_dir=audit, labels=controlled)

    assert (audit / "stop_heavy_audit.json").exists()
    assert (audit / "worst_frame_contact_sheet.ppm").exists()
    assert (audit / "model_vs_label_policy_comparison.json").exists()
    assert metrics["replay_only"] is True
    assert metrics["not_executed"] is True
    assert metrics["control_safe"] is False
    assert "stop_selected_fraction" in metrics
    assert "selected_motion_fraction" in metrics
    assert "likely_root_causes" in metrics
    assert metrics["label_bev_policy_comparison"]["agreement_fraction"] == 1.0
