from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from homebrain.data.spatial_dataset import DETERMINISTIC_CREATED_AT_UTC, write_deterministic_npz, write_json
from homebrain.messages.schema import JsonDict
from homebrain.teachers.artifacts import file_sha256, relative_to_root
from homebrain.train.future_bev_rollout_dataset import (
    DEFAULT_FUTURE_HORIZONS_S,
    FUTURE_BEV_CHANNELS,
    FUTURE_DERIVED_CHANNELS,
    FUTURE_ROLLOUT_EXAMPLE_SCHEMA_VERSION,
    FUTURE_ROLLOUT_PACK_SCHEMA_VERSION,
    ROLLOUT_PROVENANCE_FLAGS,
    build_future_rollout_targets_for_frame,
    default_candidates_for_manifest,
    group_frames_by_sequence,
    load_spatial_rollout_frames,
    meters_per_cell_from_manifest,
    robot_radius_m_from_manifest,
)
from homebrain.train.spatial_dataset import DINOFeatureStore
from homebrain.policies.candidate_trajectories import candidates_hash


def build_future_bev_rollout_pack(
    *,
    sources: list[str | Path],
    out_dir: str | Path,
    feature_dirs: list[str | Path] | None = None,
    horizons_s: tuple[float, ...] = DEFAULT_FUTURE_HORIZONS_S,
    max_examples: int | None = None,
) -> Path:
    if not sources:
        raise ValueError("at least one SpatialTrainPack source is required")
    if not horizons_s or any(float(value) <= 0.0 for value in horizons_s):
        raise ValueError("horizons_s must contain positive values")
    output = Path(out_dir)
    examples_dir = output / "examples"
    _clear_generated_outputs(output)
    examples_dir.mkdir(parents=True, exist_ok=True)

    source_roots = [Path(source) for source in sources]
    feature_roots = _feature_roots(feature_dirs, len(source_roots))
    all_examples: list[JsonDict] = []
    source_records: list[JsonDict] = []
    candidate_ids: list[str] | None = None
    candidate_hash_value: str | None = None
    grid_shape: tuple[int, int] | None = None
    meters_per_cell: float | None = None
    robot_radius_m: float | None = None
    horizon_valid_values: list[float] = []
    candidate_valid_values: list[float] = []
    invalid_reasons: dict[str, int] = {}
    examples_written = 0

    for source_index, source_root in enumerate(source_roots):
        frames, manifest = load_spatial_rollout_frames(source_root)
        if not frames:
            continue
        source_grid = _grid_shape(manifest)
        if grid_shape is None:
            grid_shape = source_grid
            meters_per_cell = meters_per_cell_from_manifest(manifest)
            robot_radius_m = robot_radius_m_from_manifest(manifest)
            candidates = default_candidates_for_manifest(manifest)
            candidate_ids = [candidate.id for candidate in candidates]
            candidate_hash_value = candidates_hash(candidates)
        else:
            if source_grid != grid_shape:
                raise ValueError(f"mixed rollout grid shapes are not supported: {source_grid} != {grid_shape}")
            candidates = default_candidates_for_manifest(manifest)
            if [candidate.id for candidate in candidates] != candidate_ids:
                raise ValueError("mixed candidate sets are not supported")
        assert meters_per_cell is not None
        feature_store = DINOFeatureStore(feature_roots[source_index]) if feature_roots[source_index] is not None else None
        grouped = group_frames_by_sequence(frames)
        source_included = 0
        source_valid_horizons = 0
        for frame in frames:
            if max_examples is not None and examples_written >= max_examples:
                break
            sequence = grouped[(frame.source_name, frame.sequence_id, frame.camera_id)]
            targets = build_future_rollout_targets_for_frame(
                frame=frame,
                sequence_frames=sequence,
                horizons_s=tuple(float(value) for value in horizons_s),
                candidates=candidates,
                meters_per_cell=meters_per_cell,
                feature_store=feature_store,
            )
            example_path = examples_dir / f"future_rollout_{examples_written:06d}.npz"
            write_deterministic_npz(example_path, targets.arrays)
            horizon_valid = [float(value) for value in targets.metadata["horizon_valid"]]
            candidate_valid = np.asarray(targets.arrays["candidate_valid_mask"], dtype=np.float32)
            horizon_valid_values.extend(horizon_valid)
            candidate_valid_values.extend(candidate_valid.tolist())
            source_valid_horizons += sum(1 for value in horizon_valid if value > 0.0)
            for reason in targets.metadata["invalid_reasons"]:
                invalid_reasons[str(reason)] = invalid_reasons.get(str(reason), 0) + 1
            all_examples.append(
                {
                    **targets.metadata,
                    "example_path": relative_to_root(example_path, output),
                    "example_sha256": file_sha256(example_path),
                    "candidate_count": len(candidates),
                    "valid_horizon_count": int(sum(1 for value in horizon_valid if value > 0.0)),
                    "feature_mask": float(np.asarray(targets.arrays["feature_mask"]).item()),
                }
            )
            examples_written += 1
            source_included += 1
        source_records.append(
            {
                "source": source_root.as_posix(),
                "source_manifest_sha256": file_sha256(source_root / "manifest.json"),
                "features": feature_roots[source_index].as_posix() if feature_roots[source_index] is not None else None,
                "feature_manifest_sha256": file_sha256(feature_roots[source_index] / "teacher_manifest.json")
                if feature_roots[source_index] is not None
                else None,
                "source_example_count": len(frames),
                "included_example_count": source_included,
                "valid_horizon_count": source_valid_horizons,
                "source_name": str(manifest.get("source_name", source_root.name)),
                "robot_supervision_grade": str(manifest.get("robot_supervision_grade", "unknown")),
                "dataset_frame_type": str(manifest.get("dataset_frame_type", "unknown")),
                "robot_frame_truth": bool(manifest.get("robot_frame_truth", False)),
                "control_safe": False,
            }
        )
        if max_examples is not None and examples_written >= max_examples:
            break

    if not all_examples or grid_shape is None or candidate_ids is None or candidate_hash_value is None:
        raise ValueError("no rollout examples were written")
    valid_fraction = float(sum(1 for value in horizon_valid_values if value > 0.0) / max(len(horizon_valid_values), 1))
    candidate_valid_fraction = float(
        sum(1 for value in candidate_valid_values if value > 0.0) / max(len(candidate_valid_values), 1)
    )
    manifest: JsonDict = {
        "schema_version": FUTURE_ROLLOUT_PACK_SCHEMA_VERSION,
        "example_schema_version": FUTURE_ROLLOUT_EXAMPLE_SCHEMA_VERSION,
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "package_type": "FutureBEVRolloutPack",
        "version": 1,
        "sources": source_records,
        "source_dirs": [source.as_posix() for source in source_roots],
        "example_count": len(all_examples),
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "meters_per_cell": round(float(meters_per_cell or 0.05), 6),
        "robot_radius_m": round(float(robot_radius_m or 0.18), 6),
        "horizons_s": [round(float(value), 6) for value in horizons_s],
        "future_bev_channels": list(FUTURE_BEV_CHANNELS),
        "future_derived_channels": list(FUTURE_DERIVED_CHANNELS),
        "candidate_count": len(candidate_ids),
        "candidate_ids": candidate_ids,
        "candidate_hash": candidate_hash_value,
        "target_generation": {
            "future_labels_warped_to_current_frame": True,
            "warp_source": "route_pose_delta_future_to_current",
            "warp_mode": "nearest_se2_grid_sample",
            "candidate_label_source": "fixed_candidate_footprints_over_warped_future_bev",
            "uses_model_predictions_as_labels": False,
            "weak_label": True,
            "control_safe": False,
        },
        "valid_horizon_fraction": valid_fraction,
        "candidate_valid_fraction": candidate_valid_fraction,
        "invalid_reason_distribution": dict(sorted(invalid_reasons.items())),
        "route_held_out_split_basis": "split_unit_id",
        "examples": all_examples,
        **ROLLOUT_PROVENANCE_FLAGS,
    }
    write_json(output / "manifest.json", manifest, pretty=True)
    return output / "manifest.json"


def _feature_roots(feature_dirs: list[str | Path] | None, source_count: int) -> list[Path | None]:
    if not feature_dirs:
        return [None for _ in range(source_count)]
    roots = [Path(value) for value in feature_dirs]
    if len(roots) == 1 and source_count > 1:
        return roots * source_count
    if len(roots) != source_count:
        raise ValueError("--features must be supplied once or once per --source")
    return roots


def _grid_shape(manifest: JsonDict) -> tuple[int, int]:
    value = manifest.get("grid_shape")
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("source manifest must include grid_shape")
    return (int(value[0]), int(value[1]))


def _clear_generated_outputs(output: Path) -> None:
    if (output / "examples").exists():
        shutil.rmtree(output / "examples")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def _parse_horizons(value: str) -> tuple[float, ...]:
    horizons = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not horizons:
        raise argparse.ArgumentTypeError("at least one horizon is required")
    if any(item <= 0.0 for item in horizons):
        raise argparse.ArgumentTypeError("horizons must be positive")
    return horizons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic replay-only FutureBEVRolloutPack v1.")
    parser.add_argument("--source", action="append", required=True, help="Input SpatialTrainPack directory.")
    parser.add_argument("--features", action="append", default=None, help="Optional DINO feature artifact directory.")
    parser.add_argument("--out", required=True, help="Output FutureBEVRolloutPack directory.")
    parser.add_argument("--horizons-s", type=_parse_horizons, default=DEFAULT_FUTURE_HORIZONS_S)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args(argv)
    manifest = build_future_bev_rollout_pack(
        sources=[Path(source) for source in args.source],
        out_dir=args.out,
        feature_dirs=[Path(item) for item in args.features] if args.features else None,
        horizons_s=args.horizons_s,
        max_examples=args.max_examples,
    )
    print(json.dumps({"manifest": manifest.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
