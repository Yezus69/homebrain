from pathlib import Path
import json

import pytest

pytest.importorskip("torch")

from dynamic_world_model_fixtures import write_real_rgbd_source_pack_fixture
from homebrain.eval.eval_counterfactual_dynamic_bev_world_model_v0 import eval_counterfactual_dynamic_bev_world_model_v0
from homebrain.train.dynamic_bev_world_pack_v0 import build_dynamic_bev_world_pack
from homebrain.train.train_counterfactual_dynamic_bev_world_model_v0 import train_counterfactual_dynamic_bev_world_model_v0


def test_eval_compares_against_static_and_previous_frame_baselines(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dynamic = tmp_path / "dynamic"
    model_dir = tmp_path / "model"
    write_real_rgbd_source_pack_fixture(source, frame_count=5)
    build_dynamic_bev_world_pack(
        input_pack=source,
        out_dir=dynamic,
        history_frames=2,
        future_horizons_sec=(0.5,),
        max_examples=12,
        robot_radius_m=0.05,
    )
    train_counterfactual_dynamic_bev_world_model_v0(
        pack=dynamic,
        out_dir=model_dir,
        device_name="cpu",
        batch_size=2,
        max_steps=1,
        hidden_channels=8,
    )
    out = tmp_path / "eval.json"
    eval_counterfactual_dynamic_bev_world_model_v0(
        checkpoint=model_dir / "checkpoint.pt",
        pack=dynamic,
        split="val",
        out_path=out,
        device_name="cpu",
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert "static_copy_baseline" in report["baselines"]
    assert "previous_frame_copy_baseline" in report["baselines"]
    assert "static_copy_baseline_future_occupied_auprc_by_horizon" in report
    assert "previous_frame_copy_baseline_future_dynamic_risk_auprc_by_horizon" in report
    assert "candidate_risk_auroc" in report
    assert report["runtime_field_leakage_passed"] is True
    assert report["accepted_counterfactual_dynamic_bev_world_model_v0"] is False
    assert "synthetic_or_fixture_false" in report["acceptance"]["failed_checks"]
