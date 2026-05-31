from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from homebrain.geometry.bev_projector import BEV_MANIFEST_FILE, BEV_SCHEMA_VERSION
from homebrain.geometry.camera_config import CameraConfig, load_camera_config
from homebrain.geometry.hazard_projection import project_hazard_masks_to_bev
from homebrain.geometry.hazard_verification import HazardHandVerification, load_hazard_hand_verification
from homebrain.messages.schema import JsonDict
from homebrain.replay.segment_log import load_manifest
from homebrain.teachers.artifacts import file_sha256, load_array, load_teacher_manifest, relative_to_root, save_array, write_json

HAZARD_BEV_ARTIFACT_KINDS: tuple[str, ...] = ("bev_hazard", "hazard_valid_mask")
HAZARD_PROJECTION_MODEL = "ground_plane_inverse_perspective_hazard_v0"


def convert_hazard_to_bev(
    *,
    log_dir: str | Path,
    hazard_artifacts_dir: str | Path,
    out_dir: str | Path,
    camera_config_path: str | Path | None = None,
    review_assumed_extrinsics: bool = False,
    hand_verification_path: str | Path | None = None,
) -> Path:
    log_root = Path(log_dir)
    hazard_root = Path(hazard_artifacts_dir)
    output = Path(out_dir)
    route_manifest = load_manifest(log_root)
    teacher_manifest = load_teacher_manifest(hazard_root)
    _validate_hazard_teacher_manifest(teacher_manifest)
    config, config_source, assumptions = _load_projection_config(camera_config_path, review_assumed_extrinsics)
    hand_verification = load_hazard_hand_verification(hand_verification_path) if hand_verification_path is not None else None
    output.mkdir(parents=True, exist_ok=True)

    frame_records: list[JsonDict] = []
    warnings: list[str] = []
    for source_record in teacher_manifest["frames"]:
        if not isinstance(source_record, dict):
            warnings.append("skipped non-object hazard teacher frame record")
            continue
        try:
            record = _write_frame(
                output=output,
                hazard_root=hazard_root,
                source_record=source_record,
                config=config,
                config_source=config_source,
                review_assumed_extrinsics=review_assumed_extrinsics,
                hand_verification=hand_verification,
            )
        except Exception as exc:  # noqa: BLE001 - preserve frame-level failures in manifest.
            warnings.append(f"frame_{source_record.get('frame_id')}_failed:{exc}")
            continue
        warnings.extend(str(item) for item in record.get("warnings", []))
        frame_records.append(record)

    if not frame_records:
        raise ValueError("no valid hazard frames were available for BEV projection")

    positive_cells = sum(int(record.get("hazard_positive_cell_count", 0)) for record in frame_records)
    total_cells = sum(int(record.get("hazard_total_cell_count", 0)) for record in frame_records)
    available_frame_ids = {int(record["frame_id"]) for record in frame_records}
    verification_fields = (
        hand_verification.manifest_fields(available_frame_ids=available_frame_ids)
        if hand_verification is not None
        else {
            "hand_verification_schema_version": None,
            "hand_verification_path": None,
            "hand_verification_sha256": None,
            "hand_verified_hazard_positive_frame_count": 0,
            "hand_verified_hazard_positive_frame_ids": [],
            "hand_verified_hazard_missing_frame_ids": [],
        }
    )
    manifest: JsonDict = {
        "schema_version": BEV_SCHEMA_VERSION,
        "source_log": log_root.as_posix(),
        "source_segment_id": route_manifest.segment_id,
        "source_schema_version": route_manifest.schema_version,
        "source_hazard_artifacts": hazard_root.as_posix(),
        "source_hazard_manifest_sha256": file_sha256(hazard_root / "teacher_manifest.json"),
        "source_hazard_teacher_name": teacher_manifest.get("teacher_name"),
        "source_hazard_backend": teacher_manifest.get("backend"),
        "source_hazard_mock": bool(teacher_manifest.get("mock", False)),
        "source_hazard_synthetic": bool(teacher_manifest.get("synthetic", False)),
        "source_hazard_real_perception": bool(teacher_manifest.get("real_perception", False)),
        "hazard_source_model": teacher_manifest.get("model_name") or teacher_manifest.get("model_id"),
        "hazard_license_review_status": teacher_manifest.get("license_review_status"),
        "hazard_class_names": teacher_manifest.get("hazard_class_names", []),
        "grid_shape": list(config.grid_shape),
        "meters_per_cell": float(config.meters_per_cell),
        "grid_orientation": {
            "origin": "bottom_center_robot_base",
            "forward": "decreasing_row",
            "left": "increasing_column",
        },
        "coordinate_frame": "robot_base_ground_plane_projection",
        "projection_model": HAZARD_PROJECTION_MODEL,
        "camera_config_source": config_source,
        "review_assumed_extrinsics": bool(review_assumed_extrinsics),
        "artifact_kinds": list(HAZARD_BEV_ARTIFACT_KINDS),
        "weak_label": True,
        "hazard_weak_label": True,
        "geometry_pretrain_ok": False,
        "pose_pretrain_ok": False,
        "robot_frame_truth_candidate": not review_assumed_extrinsics,
        "robot_frame_truth": False,
        "not_robot_frame_truth": True,
        "action_supervision_ok": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
        "trainable_for": "hazard_pretrain_only",
        "control_safety": "not_control_safe_weak_offline_hazard_label",
        "hazard_positive_frame_count": int(sum(1 for record in frame_records if int(record.get("hazard_positive_cell_count", 0)) > 0)),
        "hazard_positive_cell_fraction": float(positive_cells / max(total_cells, 1)),
        **verification_fields,
        "warnings": sorted(set(warnings)),
        "assumptions": sorted(set(assumptions)),
        "frame_count": len(frame_records),
        "frames": frame_records,
    }
    manifest_path = output / BEV_MANIFEST_FILE
    write_json(manifest_path, manifest)
    return manifest_path


def _write_frame(
    *,
    output: Path,
    hazard_root: Path,
    source_record: JsonDict,
    config: CameraConfig,
    config_source: str,
    review_assumed_extrinsics: bool,
    hand_verification: HazardHandVerification | None,
) -> JsonDict:
    artifacts = source_record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("missing artifacts object")
    masks_record = artifacts.get("hazard_masks")
    if not isinstance(masks_record, dict) or not isinstance(masks_record.get("path"), str):
        raise ValueError("missing hazard_masks artifact")
    masks = load_array(hazard_root / str(masks_record["path"]))
    frame_id = int(source_record["frame_id"])
    camera_id = str(source_record["camera_id"])
    width = int(source_record["width"])
    height = int(source_record["height"])
    intrinsics = source_record.get("intrinsics") if isinstance(source_record.get("intrinsics"), dict) else {}
    bev_hazard, valid_mask, stats = project_hazard_masks_to_bev(
        hazard_masks=masks,
        frame_width=width,
        frame_height=height,
        intrinsics=intrinsics,
        config=config,
    )
    frame_dir = output / "frames" / f"{camera_id}_{frame_id:06d}"

    arrays = {
        "bev_hazard": bev_hazard.astype(np.float32),
        "hazard_valid_mask": valid_mask.astype(np.uint8),
    }
    artifact_records: dict[str, JsonDict] = {}
    for kind in HAZARD_BEV_ARTIFACT_KINDS:
        target = frame_dir / f"{kind}.npy"
        record = save_array(target, arrays[kind])
        artifact_records[kind] = {
            "kind": kind,
            "path": relative_to_root(target, output),
            **record,
        }
    warnings = ["weak_semantic_floor_hazard_not_geometry"]
    if review_assumed_extrinsics:
        warnings.append("review_assumed_extrinsics_not_robot_frame_truth")
    hand_verified_positive = bool(hand_verification is not None and frame_id in hand_verification.positive_frame_ids)

    metadata: JsonDict = {
        "schema_version": BEV_SCHEMA_VERSION,
        "sequence_id": source_record["sequence_id"],
        "camera_id": camera_id,
        "frame_id": frame_id,
        "timestamp_ns": source_record["timestamp_ns"],
        "source_hazard_artifacts": {
            "hazard_masks": str(masks_record.get("path")),
            "hazard_confidence": _artifact_path(artifacts, "hazard_confidence"),
            "hazard_boxes": _artifact_path(artifacts, "hazard_boxes"),
        },
        "source_data_ref": source_record.get("source_data_ref"),
        "width": width,
        "height": height,
        "weak_label": True,
        "hazard_weak_label": True,
        "robot_frame_truth": False,
        "not_robot_frame_truth": True,
        "action_supervision_ok": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
        "trainable_for": "hazard_pretrain_only",
        "projection_model": HAZARD_PROJECTION_MODEL,
        "camera_config_source": config_source,
        "review_assumed_extrinsics": bool(review_assumed_extrinsics),
        "grid_shape": list(config.grid_shape),
        "meters_per_cell": float(config.meters_per_cell),
        "stats": stats,
        "hand_verified_hazard_positive": hand_verified_positive,
        "warnings": warnings,
        "artifact_shapes": {kind: artifact_records[kind]["shape"] for kind in HAZARD_BEV_ARTIFACT_KINDS},
        "artifact_dtypes": {kind: artifact_records[kind]["dtype"] for kind in HAZARD_BEV_ARTIFACT_KINDS},
    }
    metadata_path = frame_dir / "metadata.json"
    write_json(metadata_path, metadata)

    return {
        "sequence_id": source_record["sequence_id"],
        "camera_id": camera_id,
        "frame_id": frame_id,
        "timestamp_ns": source_record["timestamp_ns"],
        "width": width,
        "height": height,
        "weak_label": True,
        "hazard_weak_label": True,
        "robot_frame_truth": False,
        "not_robot_frame_truth": True,
        "action_supervision_ok": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "trainable_for": "hazard_pretrain_only",
        "hand_verified_hazard_positive": hand_verified_positive,
        "hazard_positive_cell_count": stats["hazard_positive_cell_count"],
        "hazard_total_cell_count": int(bev_hazard.size),
        "metadata_path": relative_to_root(metadata_path, output),
        "metadata_sha256": file_sha256(metadata_path),
        "artifacts": artifact_records,
        "stats": stats,
        "warnings": warnings,
    }


def _load_projection_config(
    camera_config_path: str | Path | None,
    review_assumed_extrinsics: bool,
) -> tuple[CameraConfig, str, list[str]]:
    if camera_config_path is not None:
        return load_camera_config(camera_config_path), Path(camera_config_path).as_posix(), []
    if not review_assumed_extrinsics:
        raise ValueError("camera_config_path is required unless --review-assumed-extrinsics is set")
    config = CameraConfig.from_dict(
        {
            "camera_height_m": 0.35,
            "pitch_deg": 45.0,
            "roll_deg": 0.0,
            "yaw_deg": 0.0,
            "grid_size_m": 3.2,
            "meters_per_cell": 0.05,
            "max_depth_m": 4.0,
            "floor_height_tol_m": 0.04,
            "obstacle_height_min_m": 0.12,
            "robot_radius_m": 0.18,
        }
    )
    return config, "review_assumed_default_camera_config", ["camera_to_base_extrinsics_assumed_for_review_only"]


def _validate_hazard_teacher_manifest(manifest: JsonDict) -> None:
    if manifest.get("teacher_name") != "hazard":
        raise ValueError(f"expected hazard teacher manifest, got {manifest.get('teacher_name')!r}")
    if manifest.get("weak_label") is not True:
        raise ValueError("hazard teacher manifest must mark weak_label=true")
    if manifest.get("control_safe") is not False:
        raise ValueError("hazard teacher manifest must mark control_safe=false")
    if manifest.get("mock") is True and manifest.get("real_perception") is True:
        raise ValueError("mock hazard teacher cannot be real_perception=true")


def _artifact_path(artifacts: JsonDict, kind: str) -> str | None:
    record = artifacts.get(kind)
    if isinstance(record, dict) and isinstance(record.get("path"), str):
        return str(record["path"])
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project weak semantic floor-hazard masks into robot-frame BEV.")
    parser.add_argument("--log", required=True, help="Input segment log directory.")
    parser.add_argument("--hazard-artifacts", required=True, help="Hazard teacher artifact directory.")
    parser.add_argument("--out", required=True, help="Output hazard BEV directory.")
    parser.add_argument("--camera-config", default=None, help="Camera-to-base geometry config JSON.")
    parser.add_argument("--review-assumed-extrinsics", action="store_true", help="Allow review-only assumed extrinsics.")
    parser.add_argument("--hand-verification", default=None, help="Optional hand-verified hazard frame JSON.")
    args = parser.parse_args(argv)
    convert_hazard_to_bev(
        log_dir=args.log,
        hazard_artifacts_dir=args.hazard_artifacts,
        out_dir=args.out,
        camera_config_path=args.camera_config,
        review_assumed_extrinsics=args.review_assumed_extrinsics,
        hand_verification_path=args.hand_verification,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
