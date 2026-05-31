import json
from pathlib import Path

from homebrain.brain.direct_bev_student_v0 import DirectBEVStudentV0, DirectBEVStudentV0Config, save_checkpoint
from homebrain.train.real_rgbd_route_bev_pack import build_real_rgbd_route_bev_pack
from homebrain.visualization.bev_voxel_viewer import build_bev_voxel_viewer
from tests.hazard_fixtures import write_hazard_bev_source, write_tum_sequence


def test_bev_voxel_viewer_exports_checkpoint_predictions_and_gt(tmp_path: Path) -> None:
    source = tmp_path / "source"
    hazards = tmp_path / "hazards"
    for route_id in ("route_train", "route_val"):
        write_tum_sequence(source / route_id, frame_count=3)
        write_hazard_bev_source(
            hazards / route_id,
            route_id=route_id,
            frame_count=3,
            grid_shape=(8, 8),
            hazard_cell=(6, 4),
            hand_verified=1,
        )
    pack = tmp_path / "pack"
    build_real_rgbd_route_bev_pack(
        dataset="tum_rgbd",
        input_root=source,
        out_dir=pack,
        train_routes=["route_train"],
        val_routes=["route_val"],
        frame_stride=1,
        grid_shape=(8, 8),
        meters_per_cell=0.05,
        hazard_bev_root=hazards,
    )
    model = DirectBEVStudentV0(DirectBEVStudentV0Config(bev_shape=(8, 8), image_size=(16, 12), hidden_channels=4))
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model,
        metadata={
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "raw_pwm_emitted": False,
            "hardware_validated": False,
            "no_teacher_fields_at_runtime": True,
        },
        metrics={},
    )

    index = build_bev_voxel_viewer(
        checkpoint=checkpoint,
        pack_dir=pack,
        out_dir=tmp_path / "viewer",
        split="val",
        max_examples=2,
        device_name="cpu",
    )
    data = json.loads((tmp_path / "viewer" / "data" / "viewer_data.json").read_text(encoding="utf-8"))

    assert index.exists()
    assert (tmp_path / "viewer" / "viewer.js").exists()
    assert data["voxel_size_m"] == 0.01
    assert data["cell_size_m"] == 0.05
    assert data["subdivisions_per_cell"] == 5
    assert data["scene_voxel_frame"]["origin"] == "rgbd_camera"
    assert data["metadata"]["no_teacher_fields_at_runtime"] is True
    assert data["examples"]
    first = data["examples"][0]
    assert set(("hazard", "occupied", "free")).issubset(first["pred"])
    assert set(("hazard", "occupied", "free")).issubset(first["gt"])
    assert first["pred"]["hazard"]["shape"] == [8, 8]
    assert first["gt"]["hazard"]["shape"] == [8, 8]
    assert len(first["pred"]["hazard"]["values"]) == 64
    assert len(first["gt"]["hazard"]["values"]) == 64
    assert first["scene_voxels"]["voxel_size_m"] == 0.01
    assert first["scene_voxels"]["count"] > 0
    assert len(first["scene_voxels"]["coords_cm"]) == first["scene_voxels"]["count"] * 3
