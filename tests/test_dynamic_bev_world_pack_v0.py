from pathlib import Path
import json

import numpy as np

from dynamic_world_model_fixtures import write_real_rgbd_source_pack_fixture
from homebrain.data.spatial_dataset import load_example_npz
from homebrain.train.dynamic_bev_world_pack_v0 import build_dynamic_bev_world_pack


def test_missing_input_pack_exits_cleanly_with_accepted_false(tmp_path: Path) -> None:
    out = tmp_path / "dynamic"
    manifest_path = build_dynamic_bev_world_pack(input_pack=tmp_path / "missing", out_dir=out)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert manifest["accepted_dynamic_bev_world_pack"] is False
    assert report["accepted_dynamic_bev_world_pack"] is False
    assert "missing_real_rgbd_route_bev_pack" in manifest["acceptance_reasons"]
    assert manifest["replay_only"] is True
    assert manifest["control_safe"] is False


def test_tiny_fixture_pack_cannot_pass_real_acceptance(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_real_rgbd_source_pack_fixture(source, synthetic_or_fixture=True)
    build_dynamic_bev_world_pack(
        input_pack=source,
        out_dir=tmp_path / "dynamic",
        history_frames=3,
        future_horizons_sec=(0.5, 1.0),
        max_examples=12,
        robot_radius_m=0.05,
    )
    manifest = json.loads((tmp_path / "dynamic" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["synthetic_or_fixture"] is True
    assert manifest["accepted_dynamic_bev_world_pack"] is False
    assert "synthetic_or_fixture_pack_not_real_acceptance" in manifest["acceptance_reasons"]


def test_candidate_risk_labels_use_footprint_overlap_with_future_bev(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_real_rgbd_source_pack_fixture(source)
    build_dynamic_bev_world_pack(
        input_pack=source,
        out_dir=tmp_path / "dynamic",
        history_frames=3,
        future_horizons_sec=(0.5,),
        max_examples=3,
        robot_radius_m=0.05,
    )
    manifest = json.loads((tmp_path / "dynamic" / "manifest.json").read_text(encoding="utf-8"))
    record = next(item for item in manifest["examples"] if item["route_id"] == "route_train_a")
    example = load_example_npz(tmp_path / "dynamic" / record["example_path"])
    ids = [str(value) for value in example["candidate_ids"]]
    straight = ids.index("straight_short")
    stop = ids.index("stop")
    assert float(example["candidate_collision_risk"][straight, 0]) > float(example["candidate_collision_risk"][stop, 0])
    assert float(example["candidate_total_teacher_risk"][straight]) >= 0.5
    assert float(example["candidate_safe_label"][stop]) >= 0.5


def test_train_val_route_overlap_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_real_rgbd_source_pack_fixture(source, train_val_overlap=True)
    build_dynamic_bev_world_pack(
        input_pack=source,
        out_dir=tmp_path / "dynamic",
        history_frames=3,
        future_horizons_sec=(0.5,),
        max_examples=3,
        robot_radius_m=0.05,
    )
    manifest = json.loads((tmp_path / "dynamic" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["accepted_dynamic_bev_world_pack"] is False
    assert "train_val_route_overlap" in manifest["acceptance_reasons"]


def test_dynamic_world_pack_shapes_include_required_targets(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_real_rgbd_source_pack_fixture(source)
    build_dynamic_bev_world_pack(
        input_pack=source,
        out_dir=tmp_path / "dynamic",
        history_frames=3,
        future_horizons_sec=(0.5, 1.0),
        max_examples=2,
        robot_radius_m=0.05,
    )
    manifest = json.loads((tmp_path / "dynamic" / "manifest.json").read_text(encoding="utf-8"))
    example = load_example_npz(tmp_path / "dynamic" / manifest["examples"][0]["example_path"])
    assert example["bev_history"].shape == (3, 7, 16, 16)
    assert example["future_flow_xy"].shape == (2, 2, 16, 16)
    assert example["candidate_trajectory_xytheta"].ndim == 3
    assert np.asarray(example["no_future_labels_at_runtime"]).item() is True
