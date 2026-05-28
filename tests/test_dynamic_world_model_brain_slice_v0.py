from pathlib import Path
import json

import pytest

pytest.importorskip("torch")

from dynamic_world_model_fixtures import write_real_rgbd_source_pack_fixture
from homebrain.eval.eval_dynamic_world_model_brain_slice_v0 import eval_dynamic_world_model_brain_slice_v0
from homebrain.train.dynamic_bev_world_pack_v0 import build_dynamic_bev_world_pack
from homebrain.train.train_counterfactual_dynamic_bev_world_model_v0 import train_counterfactual_dynamic_bev_world_model_v0


def test_brain_slice_writes_selected_candidate_artifacts_and_safety_flags(tmp_path: Path) -> None:
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
    out = tmp_path / "brain_slice.json"
    eval_dynamic_world_model_brain_slice_v0(
        checkpoint=model_dir / "checkpoint.pt",
        pack=dynamic,
        split="val",
        out_path=out,
        device_name="cpu",
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert "selected_candidate_id_distribution" in report
    assert "candidate_change_rate_vs_old_scorer" in report
    assert Path(report["selected_candidate_artifact"]).exists()
    assert report["no_runtime_leakage"] is True
    assert report["replay_only"] is True
    assert report["not_executed"] is True
    assert report["control_safe"] is False
    assert report["raw_pwm_emitted"] is False
