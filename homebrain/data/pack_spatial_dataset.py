from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.data.spatial_dataset import (
    DETERMINISTIC_CREATED_AT_UTC,
    SPATIAL_DATASET_SCHEMA_VERSION,
    SPATIAL_EXAMPLE_SCHEMA_VERSION,
    SPATIAL_MANIFEST_FILE,
    json_sha256,
    scalar_bool,
    scalar_int,
    scalar_str,
    write_deterministic_npz,
    write_json,
)
from homebrain.geometry.bev_projector import BEV_ARTIFACT_KINDS
from homebrain.geometry.validate_bev import load_bev_manifest
from homebrain.messages.schema import FrameEvent, JsonDict, deterministic_json
from homebrain.replay.replayd import order_events_for_replay
from homebrain.replay.segment_log import DEFAULT_EVENT_FILE, load_manifest, read_events
from homebrain.teachers.artifacts import file_sha256, frame_key, load_array, relative_to_root


def pack_spatial_dataset(
    *,
    log_dir: str | Path,
    bev_dir: str | Path,
    out_dir: str | Path,
    teacher_artifacts: str | Path | None = None,
) -> Path:
    log_root = Path(log_dir)
    bev_root = Path(bev_dir)
    output = Path(out_dir)
    examples_dir = output / "examples"
    _clear_generated_outputs(output)
    examples_dir.mkdir(parents=True, exist_ok=True)

    route_manifest = load_manifest(log_root)
    event_file = log_root / (route_manifest.event_file or DEFAULT_EVENT_FILE)
    bev_manifest = load_bev_manifest(bev_root)
    frames_by_key = _route_frames_by_key(log_root)

    if bev_manifest.get("weak_label") is not True:
        raise ValueError("BEV manifest must be marked weak_label=true")
    if bev_manifest.get("control_safe") is not False:
        raise ValueError("BEV manifest must be marked control_safe=false")

    camera_config = _camera_config(bev_manifest)
    camera_config_hash = json_sha256(camera_config)
    teacher_manifest_hash = _teacher_manifest_hash(bev_manifest, teacher_artifacts)
    source_bev_manifest_hash = file_sha256(bev_root / "bev_manifest.json")
    source_log_manifest_hash = file_sha256(log_root / "manifest.json")
    source_event_file_hash = file_sha256(event_file)

    example_records: list[JsonDict] = []
    split_counts: dict[str, int] = {"train": 0, "val": 0, "review": 0}

    for index, frame_record in enumerate(_manifest_frames(bev_manifest)):
        frame = frames_by_key.get(frame_key(frame_record))
        if frame is None:
            raise ValueError(f"BEV frame is missing from log: {frame_key(frame_record)}")
        _validate_frame_flags(frame_record)

        arrays = _load_bev_arrays(bev_root, frame_record)
        frame_id = int(frame_record["frame_id"])
        camera_id = str(frame_record["camera_id"])
        split = _split_for_index(index)
        split_counts[split] += 1
        rgb_ref = frame.data_ref
        rgb_path = (log_root / rgb_ref).as_posix()
        provenance = _example_provenance(
            log_root=log_root,
            bev_root=bev_root,
            frame=frame,
            frame_record=frame_record,
            source_bev_manifest_hash=source_bev_manifest_hash,
            source_log_manifest_hash=source_log_manifest_hash,
            camera_config_hash=camera_config_hash,
            teacher_manifest_hash=teacher_manifest_hash,
        )

        example_arrays: dict[str, np.ndarray] = {
            "frame_id": scalar_int(frame_id),
            "timestamp_ns": scalar_int(int(frame_record["timestamp_ns"])),
            "timestamp": scalar_int(int(frame_record["timestamp_ns"])),
            "rgb_ref": scalar_str(rgb_ref),
            "rgb_path": scalar_str(rgb_path),
            "bev_free": arrays["bev_free"],
            "bev_obstacle": arrays["bev_obstacle"],
            "bev_unknown": arrays["bev_unknown"],
            "bev_confidence": arrays["bev_confidence"],
            "provenance": scalar_str(deterministic_json(provenance)),
            "weak_label": scalar_bool(True),
            "control_safe": scalar_bool(False),
            "camera_config_hash": scalar_str(camera_config_hash),
            "teacher_manifest_hash": scalar_str(teacher_manifest_hash),
            "split": scalar_str(split),
        }
        if "bev_height" in arrays:
            example_arrays["bev_height"] = arrays["bev_height"]
        if "bev_floor_candidate" in arrays:
            example_arrays["bev_floor_candidate"] = arrays["bev_floor_candidate"]

        example_path = examples_dir / f"{camera_id}_{frame_id:06d}.npz"
        write_deterministic_npz(example_path, example_arrays)
        example_hash = file_sha256(example_path)
        stats = frame_record.get("stats") if isinstance(frame_record.get("stats"), dict) else {}
        example_records.append(
            {
                "sequence_id": frame.sequence_id,
                "camera_id": camera_id,
                "frame_id": frame_id,
                "timestamp_ns": int(frame_record["timestamp_ns"]),
                "rgb_ref": rgb_ref,
                "rgb_path": rgb_path,
                "split": split,
                "weak_label": True,
                "control_safe": False,
                "example_path": relative_to_root(example_path, output),
                "example_sha256": example_hash,
                "camera_config_hash": camera_config_hash,
                "teacher_manifest_hash": teacher_manifest_hash,
                "source_bev_metadata_path": frame_record.get("metadata_path"),
                "source_bev_stats": dict(stats),
            }
        )

    manifest: JsonDict = {
        "schema_version": SPATIAL_DATASET_SCHEMA_VERSION,
        "example_schema_version": SPATIAL_EXAMPLE_SCHEMA_VERSION,
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "package_type": "SpatialTrainPack",
        "quality_review_status": "unreviewed_requires_qa",
        "source_log": log_root.as_posix(),
        "source_segment_id": route_manifest.segment_id,
        "source_log_manifest_sha256": source_log_manifest_hash,
        "source_event_file_sha256": source_event_file_hash,
        "source_bev": bev_root.as_posix(),
        "source_bev_manifest_sha256": source_bev_manifest_hash,
        "source_bev_schema_version": bev_manifest.get("schema_version"),
        "source_depth_teacher_name": bev_manifest.get("source_depth_teacher_name"),
        "source_depth_backend": bev_manifest.get("source_depth_backend"),
        "source_depth_mock": bool(bev_manifest.get("source_depth_mock", False)),
        "source_depth_real_perception": bool(bev_manifest.get("source_depth_real_perception", False)),
        "teacher_artifacts": Path(teacher_artifacts).as_posix() if teacher_artifacts is not None else None,
        "teacher_manifest_hash": teacher_manifest_hash,
        "camera_config_hash": camera_config_hash,
        "camera_config": camera_config,
        "grid_shape": list(bev_manifest.get("grid_shape", [])),
        "artifact_kinds": list(BEV_ARTIFACT_KINDS),
        "example_fields": [
            "frame_id",
            "timestamp_ns",
            "timestamp",
            "rgb_ref",
            "rgb_path",
            "bev_free",
            "bev_obstacle",
            "bev_unknown",
            "bev_confidence",
            "bev_height",
            "bev_floor_candidate",
            "provenance",
            "weak_label",
            "control_safe",
            "camera_config_hash",
            "teacher_manifest_hash",
            "split",
        ],
        "weak_label": True,
        "control_safe": False,
        "control_safety": "not_control_safe_reviewed_training_data_candidate_only",
        "example_count": len(example_records),
        "split_counts": split_counts,
        "frames": example_records,
        "examples": example_records,
    }
    manifest_path = output / SPATIAL_MANIFEST_FILE
    write_json(manifest_path, manifest)
    return manifest_path


def _clear_generated_outputs(output: Path) -> None:
    if (output / "examples").exists():
        shutil.rmtree(output / "examples")
    manifest_path = output / SPATIAL_MANIFEST_FILE
    if manifest_path.exists():
        manifest_path.unlink()


def _route_frames_by_key(log_dir: Path) -> dict[str, FrameEvent]:
    events = order_events_for_replay(read_events(log_dir))
    return {frame_key(event): event for event in events if isinstance(event, FrameEvent)}


def _manifest_frames(manifest: JsonDict) -> list[JsonDict]:
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("BEV manifest frames must be a list")
    return [frame for frame in frames if isinstance(frame, dict)]


def _validate_frame_flags(frame_record: JsonDict) -> None:
    if frame_record.get("weak_label") is not True:
        raise ValueError(f"frame {frame_record.get('frame_id')} must be weak_label=true")
    if frame_record.get("control_safe") is not False:
        raise ValueError(f"frame {frame_record.get('frame_id')} must be control_safe=false")


def _load_bev_arrays(bev_root: Path, frame_record: JsonDict) -> dict[str, np.ndarray]:
    artifacts = frame_record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"frame {frame_record.get('frame_id')} is missing artifacts")
    arrays: dict[str, np.ndarray] = {}
    for kind in BEV_ARTIFACT_KINDS:
        artifact = artifacts.get(kind)
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValueError(f"frame {frame_record.get('frame_id')} missing {kind} artifact")
        arrays[kind] = load_array(bev_root / str(artifact["path"]))
    return arrays


def _camera_config(bev_manifest: JsonDict) -> JsonDict:
    camera_config = bev_manifest.get("camera_config")
    if isinstance(camera_config, dict):
        return dict(camera_config)
    return {}


def _teacher_manifest_hash(bev_manifest: JsonDict, teacher_artifacts: str | Path | None) -> str:
    if teacher_artifacts is not None:
        teacher_manifest_path = Path(teacher_artifacts) / "teacher_manifest.json"
        if not teacher_manifest_path.exists():
            raise FileNotFoundError(f"missing teacher manifest: {teacher_manifest_path}")
        return file_sha256(teacher_manifest_path)
    value = bev_manifest.get("source_depth_manifest_sha256")
    if isinstance(value, str) and value:
        return value
    return "not_supplied"


def _split_for_index(index: int) -> str:
    if index % 10 == 0:
        return "review"
    if index % 5 == 0:
        return "val"
    return "train"


def _example_provenance(
    *,
    log_root: Path,
    bev_root: Path,
    frame: FrameEvent,
    frame_record: JsonDict,
    source_bev_manifest_hash: str,
    source_log_manifest_hash: str,
    camera_config_hash: str,
    teacher_manifest_hash: str,
) -> JsonDict:
    return {
        "schema_version": SPATIAL_EXAMPLE_SCHEMA_VERSION,
        "source_log": log_root.as_posix(),
        "source_bev": bev_root.as_posix(),
        "source_log_manifest_sha256": source_log_manifest_hash,
        "source_bev_manifest_sha256": source_bev_manifest_hash,
        "source_bev_metadata_path": frame_record.get("metadata_path"),
        "sequence_id": frame.sequence_id,
        "camera_id": frame.camera_id,
        "frame_id": frame.frame_id,
        "timestamp_ns": frame.timestamp_ns,
        "rgb_ref": frame.data_ref,
        "weak_label": True,
        "control_safe": False,
        "camera_config_hash": camera_config_hash,
        "teacher_manifest_hash": teacher_manifest_hash,
        "source_bev_stats": frame_record.get("stats") if isinstance(frame_record.get("stats"), dict) else {},
        "source_bev_assumptions": frame_record.get("assumptions") if isinstance(frame_record.get("assumptions"), list) else [],
        "source_bev_warnings": frame_record.get("warnings") if isinstance(frame_record.get("warnings"), list) else [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pack BEV weak labels into a deterministic SpatialTrainPack.")
    parser.add_argument("--log", required=True, help="Input HomeBrain route log directory.")
    parser.add_argument("--bev", required=True, help="Input depth-to-BEV artifact directory.")
    parser.add_argument("--teacher-artifacts", default=None, help="Optional teacher artifact directory.")
    parser.add_argument("--out", required=True, help="Output SpatialTrainPack directory.")
    args = parser.parse_args(argv)
    manifest_path = pack_spatial_dataset(
        log_dir=args.log,
        bev_dir=args.bev,
        teacher_artifacts=args.teacher_artifacts,
        out_dir=args.out,
    )
    print(f"wrote SpatialTrainPack manifest to {manifest_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

