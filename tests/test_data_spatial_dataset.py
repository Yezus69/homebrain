import json
from pathlib import Path

import numpy as np

from homebrain.data.pack_spatial_dataset import pack_spatial_dataset
from homebrain.data.qa_spatial_dataset import qa_spatial_dataset
from homebrain.data.visualize_spatial_dataset import visualize_spatial_dataset
from homebrain.geometry.bev_projector import BEV_ARTIFACT_KINDS, BEV_SCHEMA_VERSION
from homebrain.messages.schema import FrameEvent
from homebrain.replay.segment_log import write_segment
from homebrain.teachers.artifacts import file_sha256, relative_to_root, save_array, write_json


def _write_route(route: Path, count: int = 12) -> None:
    frames = [
        FrameEvent(
            timestamp_ns=1_700_000_000_000_000_000 + index * 100_000_000,
            sequence_id=route.name,
            source="test",
            camera_id="front_rgb",
            frame_id=index,
            width=8,
            height=8,
            format="encoded_ppm",
            data_ref=f"frames/frame_{index:06d}.ppm",
            intrinsics={"available": False},
        )
        for index in range(count)
    ]
    for frame in frames:
        target = route / frame.data_ref
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
    write_segment(route, frames, segment_id=route.name, artifact_files=[frame.data_ref for frame in frames])


def _write_bev(route: Path, bev: Path, count: int = 12, confidence: float = 0.05) -> None:
    frame_records = []
    camera_config = {
        "camera_height_m": 0.45,
        "floor_height_tol_m": 0.08,
        "grid_size_m": 2.0,
        "max_depth_m": 3.5,
        "meters_per_cell": 0.5,
        "obstacle_height_min_m": 0.14,
        "pitch_deg": 25.0,
        "robot_radius_m": 0.18,
        "roll_deg": 0.0,
        "yaw_deg": 0.0,
    }
    for index in range(count):
        frame_dir = bev / "frames" / f"front_rgb_{index:06d}"
        free = np.zeros((4, 4), dtype=np.uint8)
        obstacle = np.zeros((4, 4), dtype=np.uint8)
        unknown = np.ones((4, 4), dtype=np.uint8)
        floor = np.zeros((4, 4), dtype=np.uint8)
        free[0, index % 4] = 1
        unknown[0, index % 4] = 0
        floor[0, index % 4] = 1
        if index % 3 == 0:
            obstacle[1, index % 4] = 1
            unknown[1, index % 4] = 0
        arrays = {
            "bev_free": free,
            "bev_obstacle": obstacle,
            "bev_unknown": unknown,
            "bev_floor_candidate": floor,
            "bev_height": np.zeros((4, 4), dtype=np.float32),
            "bev_confidence": np.where(unknown > 0, 0.0, confidence).astype(np.float32),
        }
        artifacts = {}
        for kind in BEV_ARTIFACT_KINDS:
            target = frame_dir / f"{kind}.npy"
            artifacts[kind] = {"kind": kind, "path": relative_to_root(target, bev), **save_array(target, arrays[kind])}
        metadata_path = frame_dir / "metadata.json"
        metadata = {
            "schema_version": BEV_SCHEMA_VERSION,
            "sequence_id": route.name,
            "camera_id": "front_rgb",
            "frame_id": index,
            "timestamp_ns": 1_700_000_000_000_000_000 + index * 100_000_000,
            "weak_label": True,
            "control_safe": False,
            "stats": {
                "free_ratio": float(np.mean(free)),
                "obstacle_ratio": float(np.mean(obstacle)),
                "unknown_ratio": float(np.mean(unknown)),
                "confidence_mean": float(np.mean(arrays["bev_confidence"])),
                "observed_cell_count": int(np.count_nonzero((free > 0) | (obstacle > 0) | (floor > 0))),
            },
        }
        write_json(metadata_path, metadata)
        frame_records.append(
            {
                "sequence_id": route.name,
                "camera_id": "front_rgb",
                "frame_id": index,
                "timestamp_ns": metadata["timestamp_ns"],
                "width": 8,
                "height": 8,
                "weak_label": True,
                "control_safe": False,
                "metadata_path": relative_to_root(metadata_path, bev),
                "metadata_sha256": file_sha256(metadata_path),
                "artifacts": artifacts,
                "stats": metadata["stats"],
                "assumptions": ["test_camera_config"],
                "warnings": [],
            }
        )
    write_json(
        bev / "bev_manifest.json",
        {
            "schema_version": BEV_SCHEMA_VERSION,
            "source_log": route.as_posix(),
            "source_segment_id": route.name,
            "source_depth_manifest_sha256": "f" * 64,
            "source_depth_teacher_name": "depth_pro",
            "source_depth_backend": "fake",
            "source_depth_mock": True,
            "source_depth_real_perception": False,
            "camera_config": camera_config,
            "grid_shape": [4, 4],
            "artifact_kinds": list(BEV_ARTIFACT_KINDS),
            "weak_label": True,
            "control_safe": False,
            "frame_count": len(frame_records),
            "frames": frame_records,
        },
    )


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_pack_spatial_dataset_writes_deterministic_npz_examples(tmp_path) -> None:
    route = tmp_path / "route"
    bev = tmp_path / "bev"
    out_a = tmp_path / "pack_a"
    out_b = tmp_path / "pack_b"
    _write_route(route)
    _write_bev(route, bev)

    pack_spatial_dataset(log_dir=route, bev_dir=bev, out_dir=out_a)
    pack_spatial_dataset(log_dir=route, bev_dir=bev, out_dir=out_b)

    assert _hash_tree(out_a) == _hash_tree(out_b)
    manifest = json.loads((out_a / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["weak_label"] is True
    assert manifest["control_safe"] is False
    assert manifest["example_count"] == 12
    assert manifest["split_counts"] == {"review": 3, "train": 7, "val": 2}
    splits = [record["split"] for record in manifest["examples"]]
    for previous, current in zip(splits, splits[1:]):
        assert {previous, current} != {"train", "val"}

    first_example = np.load(out_a / manifest["examples"][0]["example_path"], allow_pickle=False)
    assert first_example["weak_label"].item() is True
    assert first_example["control_safe"].item() is False
    assert first_example["not_robot_frame_truth"].item() is True
    assert first_example["pose_delta_mask"].item() == 0.0
    assert first_example["bev_free"].shape == (4, 4)
    assert str(first_example["split"].item()) == "train"


def test_qa_spatial_dataset_quarantines_low_quality_labels(tmp_path) -> None:
    route = tmp_path / "route"
    bev = tmp_path / "bev"
    out = tmp_path / "pack"
    _write_route(route)
    _write_bev(route, bev, confidence=0.05)
    pack_spatial_dataset(log_dir=route, bev_dir=bev, out_dir=out)

    metrics = qa_spatial_dataset(out)

    assert metrics["example_count"] == 12
    assert metrics["missing_count"] == 0
    assert metrics["shape_error_count"] == 0
    assert metrics["nan_count"] == 0
    assert metrics["trainable_candidate"] is False
    assert "mean_confidence_below_0.20" in metrics["quarantine_reasons"]
    assert "label_density_below_0.10" in metrics["quarantine_reasons"]


def test_visualize_spatial_dataset_writes_contact_sheets(tmp_path) -> None:
    route = tmp_path / "route"
    bev = tmp_path / "bev"
    out = tmp_path / "pack"
    viz = tmp_path / "viz"
    _write_route(route)
    _write_bev(route, bev)
    pack_spatial_dataset(log_dir=route, bev_dir=bev, out_dir=out)

    manifest_path = visualize_spatial_dataset(dataset_dir=out, out_dir=viz, sample_count=3)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["visualization_written"] is True
    for sheet in manifest["contact_sheets"]:
        assert (viz / sheet["path"]).exists()
