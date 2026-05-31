import json
from pathlib import Path

from homebrain.data.spatial_dataset import load_example_npz
from homebrain.train.real_rgbd_route_bev_pack import build_real_rgbd_route_bev_pack
from tests.hazard_fixtures import write_hazard_bev_source, write_tum_sequence


def test_missing_hazard_artifacts_omits_hazard_channel(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_tum_sequence(source / "route_a", frame_count=3)

    pack = tmp_path / "pack"
    manifest_path = build_real_rgbd_route_bev_pack(
        dataset="tum_rgbd",
        input_root=source,
        out_dir=pack,
        frame_stride=1,
        grid_shape=(16, 16),
        meters_per_cell=0.1,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = load_example_npz(pack / manifest["examples"][0]["example_path"])

    assert manifest["has_hazard_channel"] is False
    assert "target_bev_hazard" not in first
    assert "hazard_valid_mask" not in first


def test_pack_fuses_optional_hazard_channel_without_geometry_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    hazards = tmp_path / "hazards"
    write_tum_sequence(source / "route_a", frame_count=3)
    base_pack = tmp_path / "base_pack"
    build_real_rgbd_route_bev_pack(
        dataset="tum_rgbd",
        input_root=source,
        out_dir=base_pack,
        frame_stride=1,
        grid_shape=(16, 16),
        meters_per_cell=0.1,
    )
    base_manifest = json.loads((base_pack / "manifest.json").read_text(encoding="utf-8"))
    base_first = load_example_npz(base_pack / base_manifest["examples"][0]["example_path"])
    free_cells = (
        (base_first["target_current_bev_free"] > 0.5)
        & (base_first["target_current_bev_occupied"] < 0.5)
        & (base_first["target_current_bev_unknown"] < 0.5)
    )
    row, col = [int(value) for value in next(zip(*free_cells.nonzero()))]
    write_hazard_bev_source(hazards / "route_a", route_id="route_a", frame_count=3, hazard_cell=(row, col))

    pack = tmp_path / "pack"
    build_real_rgbd_route_bev_pack(
        dataset="tum_rgbd",
        input_root=source,
        out_dir=pack,
        frame_stride=1,
        grid_shape=(16, 16),
        meters_per_cell=0.1,
        hazard_bev_root=hazards,
    )
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    first = load_example_npz(pack / manifest["examples"][0]["example_path"])

    assert manifest["has_hazard_channel"] is True
    assert manifest["hazard_positive_frame_count"] == 1
    assert manifest["hazard_weak_label"] is True
    assert first["target_bev_hazard"][row, col] == 1.0
    assert first["target_bev_hazard_valid_mask"][row, col] == 1.0
    assert first["target_current_bev_free"][row, col] == 1.0
    assert first["target_current_bev_occupied"][row, col] == 0.0
    assert first["target_current_bev_unknown"][row, col] == 0.0
    assert (first["target_current_bev_free"] == base_first["target_current_bev_free"]).all()
    assert (first["target_current_bev_occupied"] == base_first["target_current_bev_occupied"]).all()
    assert (first["target_current_bev_unknown"] == base_first["target_current_bev_unknown"]).all()
