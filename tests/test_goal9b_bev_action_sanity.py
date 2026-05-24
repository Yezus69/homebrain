import json

import numpy as np

from homebrain.data.spatial_dataset import load_example_npz
from homebrain.policies.audit_bev_action_sanity import audit_bev_action_sanity
from homebrain.policies.build_action_label_pack import build_action_label_pack
from homebrain.policies.build_traversability_view import build_traversability_view
from homebrain.policies.generate_controlled_bev_maps import generate_controlled_bev_maps
from homebrain.policies.qa_action_label_pack import qa_action_label_pack
from homebrain.policies.sweep_scorer_config import sweep_scorer_config


def test_bev_action_sanity_marks_controlled_open_ok(tmp_path) -> None:
    controlled = tmp_path / "controlled"
    audit = tmp_path / "audit"
    generate_controlled_bev_maps(out_dir=controlled, examples_per_scenario=2, grid_size=24, meters_per_cell=0.05)

    report = audit_bev_action_sanity(source=controlled, out_dir=audit)

    assert (audit / "bev_action_sanity.json").exists()
    assert (audit / "worst_frame_contact_sheet.ppm").exists()
    assert report["control_safe"] is False
    assert report["replay_only"] is True
    assert report["action_supervision_ok_fraction"] > 0.0
    open_metrics = report["per_source"]["controlled_bev:open_room"]
    assert open_metrics["action_supervision_ok_fraction"] == 1.0
    assert open_metrics["robot_center_blocked_rate"] == 0.0


def test_traversability_view_preserves_raw_and_requires_review(tmp_path) -> None:
    controlled = tmp_path / "controlled"
    view = tmp_path / "view"
    generate_controlled_bev_maps(out_dir=controlled, examples_per_scenario=1, grid_size=24, meters_per_cell=0.05)

    build_traversability_view(source=controlled, out_dir=view)
    manifest = json.loads((view / "manifest.json").read_text(encoding="utf-8"))
    example = load_example_npz(view / manifest["examples"][0]["example_path"])

    assert manifest["derived"] is True
    assert manifest["review_required"] is True
    assert manifest["control_safe"] is False
    assert bool(np.asarray(example["derived"]).item()) is True
    assert bool(np.asarray(example["action_supervision_ok"]).item()) is False
    assert "raw_bev_free" in example
    assert "bev_traversable" in example
    assert "bev_risky" in example


def test_scorer_sweep_selects_only_controlled_passing_config(tmp_path) -> None:
    controlled = tmp_path / "controlled"
    sweep_dir = tmp_path / "sweep"
    generate_controlled_bev_maps(out_dir=controlled, examples_per_scenario=2, grid_size=24, meters_per_cell=0.05)

    report = sweep_scorer_config(sources=[controlled], out_dir=sweep_dir)

    assert report["selected_config"] is not None
    assert report["selected_metrics"]["controlled_open_motion_rate"] >= 0.95
    assert report["selected_metrics"]["blocked_map_stop_rate"] >= 0.95
    assert report["selected_metrics"]["control_safe"] is False


def test_action_label_pack_v1_filters_and_reports_action_sanity(tmp_path) -> None:
    controlled = tmp_path / "controlled"
    pack = tmp_path / "actions"
    generate_controlled_bev_maps(out_dir=controlled, examples_per_scenario=2, grid_size=24, meters_per_cell=0.05)

    build_action_label_pack(sources=[controlled], out_dir=pack, pack_version=1)
    qa = qa_action_label_pack(pack)
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    example = load_example_npz(pack / manifest["examples"][0]["example_path"])

    assert manifest["schema_version"] == "homebrain.action_label_pack.v1"
    assert manifest["action_sanity_filter"]["enabled"] is True
    assert manifest["action_sanity_filter"]["excluded_frame_count"] >= 0
    assert bool(np.asarray(example["action_supervision_ok"]).item()) is True
    assert float(np.asarray(example["source_weight"]).item()) > 0.0
    assert qa["action_label_pack_qa_pass"] is True
    assert "per_source" in qa
