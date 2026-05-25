from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from homebrain.artifacts.io import (
    load_array_artifact_optional,
    load_array_artifact_required,
    read_json_object,
    write_json_object,
)
from homebrain.data.spatial_io import load_spatial_example_arrays, scene_frames_by_id, spatial_examples_by_id
from homebrain.visualization.panels import (
    gray_rgb,
    join_with_gap,
    normalize_gray,
    resize_nearest,
    thumbnail_shape,
    tint,
    write_pgm,
    write_ppm,
)


def test_json_and_array_artifact_helpers(tmp_path: Path) -> None:
    report_path = tmp_path / "nested" / "report.json"
    write_json_object(report_path, {"b": 2, "a": 1})

    assert report_path.read_text(encoding="utf-8") == '{"a":1,"b":2}\n'
    assert read_json_object(report_path) == {"a": 1, "b": 2}

    non_object_path = tmp_path / "list.json"
    non_object_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        read_json_object(non_object_path)

    array_path = tmp_path / "depth.npy"
    expected = np.asarray([[1.0, 2.0]], dtype=np.float32)
    with array_path.open("wb") as handle:
        np.save(handle, expected, allow_pickle=False)

    artifacts = {"depth": {"path": "depth.npy"}, "missing_file": {"path": "missing.npy"}}
    np.testing.assert_array_equal(
        load_array_artifact_optional(tmp_path, artifacts, "depth", missing_ok=False),
        expected,
    )
    np.testing.assert_array_equal(load_array_artifact_required(tmp_path, artifacts, "depth"), expected)
    assert load_array_artifact_optional(tmp_path, artifacts, "absent_record", missing_ok=True) is None
    assert load_array_artifact_optional(tmp_path, artifacts, "missing_file", missing_ok=True) is None
    with pytest.raises(FileNotFoundError):
        load_array_artifact_optional(tmp_path, artifacts, "missing_file", missing_ok=False)


def test_visual_panel_helpers_write_deterministic_netpbm(tmp_path: Path) -> None:
    gray = normalize_gray(np.asarray([[0.0, 1.0], [2.0, np.nan]], dtype=np.float32))
    assert gray.shape == (2, 2)
    assert gray.dtype == np.uint8
    assert normalize_gray(np.zeros((2, 2, 3), dtype=np.float32)).shape == (32, 32)
    with pytest.raises(ValueError, match="expected 2D review panel"):
        normalize_gray(np.zeros((2, 2, 3), dtype=np.float32), strict_2d=True)

    rgb = gray_rgb(gray)
    assert rgb.shape == (2, 2, 3)
    assert thumbnail_shape(rgb, max_side=4, min_side=1) == (4, 4)
    assert resize_nearest(rgb, (4, 4)).shape == (4, 4, 3)
    assert tint(gray, (10, 20, 30)).shape == (2, 2, 3)
    assert join_with_gap([rgb, rgb], gap=1, axis=1).shape == (2, 5, 3)
    assert join_with_gap([rgb, rgb], gap=1, axis=0).shape == (5, 2, 3)

    ppm_path = tmp_path / "panel.ppm"
    pgm_path = tmp_path / "panel.pgm"
    write_ppm(ppm_path, rgb)
    write_pgm(pgm_path, gray)
    assert ppm_path.read_bytes().startswith(b"P6\n2 2\n255\n")
    assert pgm_path.read_bytes().startswith(b"P5\n2 2\n255\n")


def test_spatial_manifest_and_example_loading(tmp_path: Path) -> None:
    manifest = {
        "frames": [{"frame_id": "7"}, {"frame_id": None}, {"frame_id": 8.0}],
        "examples": [{"frame_id": 7, "example_path": "examples/e000007.npz"}, {"frame_id": "bad"}],
    }
    assert sorted(scene_frames_by_id(manifest)) == [7, 8]
    assert sorted(spatial_examples_by_id(manifest)) == [7]

    example_dir = tmp_path / "examples"
    example_dir.mkdir()
    example_path = example_dir / "e000007.npz"
    bev_free = np.ones((2, 2), dtype=np.float32)
    with example_path.open("wb") as handle:
        np.savez(
            handle,
            bev_free=bev_free,
            bev_obstacle=np.zeros((2, 2), dtype=np.float32),
            bev_unknown=np.zeros((2, 2), dtype=np.float32),
        )

    arrays = load_spatial_example_arrays(
        tmp_path,
        manifest["examples"][0],
        missing_ok=False,
        require_confidence=True,
    )
    assert arrays is not None
    np.testing.assert_array_equal(arrays["bev_free"], bev_free)
    np.testing.assert_array_equal(arrays["bev_confidence"], np.ones((2, 2), dtype=np.float32))
    assert load_spatial_example_arrays(tmp_path, {"frame_id": 9}, missing_ok=True) is None
    with pytest.raises(ValueError, match="missing example_path"):
        load_spatial_example_arrays(tmp_path, {"frame_id": 9}, missing_ok=False)
