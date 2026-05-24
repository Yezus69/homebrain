from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from homebrain.data.spatial_dataset import (
    DETERMINISTIC_CREATED_AT_UTC,
    SPATIAL_DATASET_SCHEMA_VERSION,
    SPATIAL_EXAMPLE_SCHEMA_VERSION,
    json_sha256,
    scalar_bool,
    scalar_float,
    scalar_int,
    scalar_str,
    write_deterministic_npz,
    write_json,
)
from homebrain.messages.schema import JsonDict, deterministic_json
from homebrain.policies.bev_action_sanity import load_bev_frames_for_action
from homebrain.teachers.artifacts import file_sha256, relative_to_root

TRAVERSABILITY_VIEW_SCHEMA_VERSION = "homebrain.traversability_view.v0"


@dataclass(frozen=True)
class TraversabilityViewConfig:
    robot_radius_cells: int = 3
    obstacle_inflation_cells: int = 1
    unknown_policy: str = "risky"
    min_free_threshold: float = 0.5
    risky_threshold: float = 0.35

    def to_dict(self) -> JsonDict:
        return {
            "robot_radius_cells": int(self.robot_radius_cells),
            "obstacle_inflation_cells": int(self.obstacle_inflation_cells),
            "unknown_policy": self.unknown_policy,
            "min_free_threshold": float(self.min_free_threshold),
            "risky_threshold": float(self.risky_threshold),
        }


def build_traversability_view(
    *,
    source: str | Path,
    out_dir: str | Path,
    config: TraversabilityViewConfig | None = None,
) -> Path:
    cfg = config or TraversabilityViewConfig()
    if cfg.unknown_policy not in {"risky", "block", "ignore"}:
        raise ValueError("unknown_policy must be one of: risky, block, ignore")
    output = Path(out_dir)
    examples_dir = output / "examples"
    _clear_outputs(output)
    examples_dir.mkdir(parents=True, exist_ok=True)

    frames, metadata = load_bev_frames_for_action(source, scratch_dir=output / "_scratch")
    if not frames:
        raise ValueError(f"no BEV frames found for traversability view: {source}")

    config_hash = json_sha256(cfg.to_dict())
    examples: list[JsonDict] = []
    split_counts: dict[str, int] = {}
    for index, frame in enumerate(frames):
        raw_free = np.clip(frame.record.bev.free.astype(np.float32), 0.0, 1.0)
        raw_obstacle = np.clip(frame.record.bev.occupied.astype(np.float32), 0.0, 1.0)
        raw_unknown = np.clip(frame.record.bev.unknown.astype(np.float32), 0.0, 1.0)
        raw_confidence = (
            np.clip(frame.record.bev.confidence.astype(np.float32), 0.0, 1.0)
            if frame.record.bev.confidence is not None
            else np.where(raw_unknown >= 0.5, 0.05, 0.95).astype(np.float32)
        )
        traversable, risky = _derive_traversability(
            raw_free=raw_free,
            raw_obstacle=raw_obstacle,
            raw_unknown=raw_unknown,
            config=cfg,
        )
        split = str(frame.frame_record.get("split", "review"))
        split_counts[split] = split_counts.get(split, 0) + 1
        example_path = examples_dir / f"traversability_{index:06d}.npz"
        provenance = {
            "schema_version": TRAVERSABILITY_VIEW_SCHEMA_VERSION,
            "source": Path(source).as_posix(),
            "source_ref": frame.record.source_ref,
            "derived": True,
            "config": cfg.to_dict(),
            "control_safe": False,
            "review_required": True,
        }
        write_deterministic_npz(
            example_path,
            {
                "frame_id": scalar_int(frame.record.frame_id),
                "timestamp_ns": scalar_int(frame.record.timestamp_ns),
                "timestamp": scalar_int(frame.record.timestamp_ns),
                "rgb_ref": scalar_str(str(frame.frame_record.get("rgb_ref", ""))),
                "rgb_path": scalar_str(str(frame.frame_record.get("rgb_path", ""))),
                "bev_free": traversable.astype(np.float32),
                "bev_obstacle": risky.astype(np.float32),
                "bev_unknown": raw_unknown.astype(np.float32),
                "bev_confidence": raw_confidence.astype(np.float32),
                "bev_traversable": traversable.astype(np.float32),
                "bev_risky": risky.astype(np.float32),
                "raw_bev_free": raw_free.astype(np.float32),
                "raw_bev_obstacle": raw_obstacle.astype(np.float32),
                "raw_bev_unknown": raw_unknown.astype(np.float32),
                "provenance": scalar_str(deterministic_json(provenance)),
                "weak_label": scalar_bool(True),
                "control_safe": scalar_bool(False),
                "not_robot_frame_truth": scalar_bool(True),
                "derived": scalar_bool(True),
                "review_required": scalar_bool(True),
                "action_supervision_ok": scalar_bool(False),
                "camera_config_hash": scalar_str(config_hash),
                "teacher_manifest_hash": scalar_str("derived_traversability_view_v0"),
                "split": scalar_str(split),
                "split_unit_id": scalar_str(str(frame.frame_record.get("split_unit_id", frame.source_name))),
                "pose_delta": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                "pose_delta_mask": scalar_float(0.0),
                "pose_label_frame": scalar_str("none"),
                "pose_label_source": scalar_str("none"),
                "pose_label_target_frame_id": scalar_int(-1),
                "action_label_mask": scalar_float(0.0),
                "imu_label_mask": scalar_float(0.0),
                "wheel_label_mask": scalar_float(0.0),
                "source_name": scalar_str(f"{frame.source_name}:traversability_view"),
                "source_family": scalar_str("derived_traversability_view"),
                "scenario_name": scalar_str(frame.scenario_name),
                "supervision_grade": scalar_str("derived_traversability_review_required"),
                "replay_only": scalar_bool(True),
                "not_executed": scalar_bool(True),
                "raw_pwm_emitted": scalar_bool(False),
            },
        )
        examples.append(
            {
                "sequence_id": frame.record.sequence_id,
                "camera_id": frame.record.camera_id,
                "frame_id": int(frame.record.frame_id),
                "timestamp_ns": int(frame.record.timestamp_ns),
                "source_ref": frame.record.source_ref,
                "source_name": f"{frame.source_name}:traversability_view",
                "source_family": "derived_traversability_view",
                "scenario_name": frame.scenario_name,
                "supervision_grade": "derived_traversability_review_required",
                "split": split,
                "example_path": relative_to_root(example_path, output),
                "example_sha256": file_sha256(example_path),
                "derived": True,
                "review_required": True,
                "action_supervision_ok": False,
                "weak_label": True,
                "control_safe": False,
            }
        )

    manifest: JsonDict = {
        "schema_version": SPATIAL_DATASET_SCHEMA_VERSION,
        "traversability_view_schema_version": TRAVERSABILITY_VIEW_SCHEMA_VERSION,
        "example_schema_version": SPATIAL_EXAMPLE_SCHEMA_VERSION,
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "package_type": "SpatialTrainPack",
        "source": Path(source).as_posix(),
        "source_metadata": metadata,
        "source_manifest_sha256": file_sha256(Path(source) / "manifest.json") if (Path(source) / "manifest.json").exists() else "missing",
        "source_name": f"{Path(source).name}:traversability_view",
        "source_family": "derived_traversability_view",
        "supervision_grade": "derived_traversability_review_required",
        "robot_supervision_grade": "geometry_only_review_required",
        "config": cfg.to_dict(),
        "config_hash": config_hash,
        "grid_shape": list(frames[0].record.bev.shape),
        "example_count": len(examples),
        "split_counts": split_counts,
        "derived": True,
        "raw_source_preserved": True,
        "review_required": True,
        "action_supervision_ok": False,
        "weak_label": True,
        "control_safe": False,
        "not_executed": True,
        "replay_only": True,
        "raw_pwm_emitted": False,
        "examples": examples,
        "frames": examples,
    }
    write_json(output / "manifest.json", manifest, pretty=True)
    return output / "manifest.json"


def _derive_traversability(
    *,
    raw_free: np.ndarray,
    raw_obstacle: np.ndarray,
    raw_unknown: np.ndarray,
    config: TraversabilityViewConfig,
) -> tuple[np.ndarray, np.ndarray]:
    obstacle = raw_obstacle >= np.float32(config.risky_threshold)
    inflated = _inflate_binary(
        obstacle,
        radius_cells=max(0, int(config.robot_radius_cells) + int(config.obstacle_inflation_cells)),
    )
    unknown = raw_unknown >= np.float32(0.5)
    risky = inflated.copy()
    if config.unknown_policy == "risky":
        risky |= unknown
    traversable = raw_free >= np.float32(config.min_free_threshold)
    traversable &= ~inflated
    if config.unknown_policy == "block":
        traversable &= ~unknown
    return traversable.astype(np.float32), risky.astype(np.float32)


def _inflate_binary(mask: np.ndarray, *, radius_cells: int) -> np.ndarray:
    radius = max(0, int(radius_cells))
    if radius == 0:
        return mask.astype(np.bool_)
    inflated = np.zeros(mask.shape, dtype=np.bool_)
    rows, cols = mask.shape
    for row in range(rows):
        for col in range(cols):
            if not bool(mask[row, col]):
                continue
            row0 = max(0, row - radius)
            row1 = min(rows, row + radius + 1)
            col0 = max(0, col - radius)
            col1 = min(cols, col + radius + 1)
            inflated[row0:row1, col0:col1] = True
    return inflated


def _clear_outputs(output: Path) -> None:
    if (output / "examples").exists():
        shutil.rmtree(output / "examples")
    if (output / "_scratch").exists():
        shutil.rmtree(output / "_scratch")
    manifest = output / "manifest.json"
    if manifest.exists():
        manifest.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a derived replay-only traversability/risk view from raw BEV labels.")
    parser.add_argument("--source", required=True, help="Input SpatialTrainPack or BEV source.")
    parser.add_argument("--out", required=True, help="Output derived SpatialTrainPack directory.")
    parser.add_argument("--robot-radius-cells", type=int, default=3)
    parser.add_argument("--obstacle-inflation-cells", type=int, default=1)
    parser.add_argument("--unknown-policy", choices=("risky", "block", "ignore"), default="risky")
    parser.add_argument("--min-free-threshold", type=float, default=0.5)
    parser.add_argument("--risky-threshold", type=float, default=0.35)
    args = parser.parse_args(argv)
    manifest = build_traversability_view(
        source=args.source,
        out_dir=args.out,
        config=TraversabilityViewConfig(
            robot_radius_cells=args.robot_radius_cells,
            obstacle_inflation_cells=args.obstacle_inflation_cells,
            unknown_policy=args.unknown_policy,
            min_free_threshold=args.min_free_threshold,
            risky_threshold=args.risky_threshold,
        ),
    )
    print(json.dumps({"manifest": manifest.as_posix(), "control_safe": False, "review_required": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
