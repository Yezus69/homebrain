from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.data.spatial_dataset import DETERMINISTIC_CREATED_AT_UTC, json_sha256, read_json, write_deterministic_npz, write_json
from homebrain.data.tum_rgbd_route import (
    RouteFrame,
    RouteSequence,
    discover_tum_rgbd_routes,
    load_tum_rgbd_sequence,
    pose_delta,
    route_split_map,
)
from homebrain.messages.schema import JsonDict
from homebrain.teachers.rgbd_bev_teacher import RGBDBEVLabels, RGBDBEVTeacherConfig, build_rgbd_bev_labels, load_depth_for_route_frame

REAL_RGBD_BEV_PACK_SCHEMA_VERSION = "homebrain.real_rgbd_route_bev_pack.v0"
REAL_RGBD_EXAMPLE_SCHEMA_VERSION = "homebrain.real_rgbd_route_bev_example.v0"
RUNTIME_ALLOWED_FIELDS: tuple[str, ...] = (
    "rgb_path",
    "depth_path",
    "timestamp_ns",
    "route_id",
    "pose_delta_prev",
    "camera_intrinsics",
    "sensor_mask",
    "previous_action",
)
TEACHER_ONLY_FIELDS: tuple[str, ...] = (
    "target_current_bev_free",
    "target_current_bev_occupied",
    "target_current_bev_unknown",
    "target_current_bev_traversable",
    "target_current_bev_risky",
    "target_uncertainty_map",
    "target_dynamic_residual_risk",
    "pose_delta_next_teacher_only",
)


def build_real_rgbd_route_bev_pack(
    *,
    dataset: str,
    input_root: str | Path,
    out_dir: str | Path,
    train_routes: list[str] | None = None,
    val_routes: list[str] | None = None,
    frame_stride: int = 5,
    grid_shape: tuple[int, int] = (64, 64),
    meters_per_cell: float = 0.05,
) -> Path:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    route_dirs = discover_tum_rgbd_routes(input_root)
    if not route_dirs:
        manifest = _base_manifest(
            dataset=dataset,
            input_root=Path(input_root),
            grid_shape=grid_shape,
            meters_per_cell=meters_per_cell,
            frame_stride=frame_stride,
        )
        manifest.update(
            {
                "accepted_real_rgbd_route_bev_pack": False,
                "accepted_real_rgbd_route_bev_student": False,
                "acceptance_reasons": ["missing_real_dataset_root_or_tum_rgbd_routes"],
                "examples": [],
            }
        )
        write_json(output / "manifest.json", manifest, pretty=True)
        return output / "manifest.json"

    routes = [
        load_tum_rgbd_sequence(route_dir, dataset_name=dataset, route_id=route_dir.name)
        for route_dir in route_dirs
    ]
    splits = route_split_map(routes, train_routes=train_routes, val_routes=val_routes)
    config = RGBDBEVTeacherConfig(grid_shape=grid_shape, meters_per_cell=meters_per_cell)
    examples: list[JsonDict] = []
    source_frame_count = sum(int(route.source_manifest["frame_count"]) for route in routes)
    route_dynamic_frames: dict[str, int] = {route.route_id: 0 for route in routes}
    dynamic_positive_cells = 0
    total_cells = 0
    for route in routes:
        previous_kept: tuple[RouteFrame, RGBDBEVLabels] | None = None
        kept_frames = list(route.frames)[::frame_stride]
        for kept_index, frame in enumerate(kept_frames):
            depth = load_depth_for_route_frame(frame)
            if depth is None:
                continue
            prev_delta = pose_delta(previous_kept[0].pose, frame.pose) if previous_kept is not None else None
            labels = build_rgbd_bev_labels(
                depth_m=depth,
                intrinsics=frame.intrinsics,
                config=config,
                previous_labels=previous_kept[1] if previous_kept is not None else None,
                pose_delta_prev=prev_delta,
            )
            next_delta = _next_pose_delta(kept_frames, kept_index)
            split = splits[route.route_id]
            dynamic_count = int(np.count_nonzero(labels.dynamic_residual_risk > 0.5))
            if dynamic_count > 0:
                route_dynamic_frames[route.route_id] += 1
            dynamic_positive_cells += dynamic_count
            total_cells += int(labels.dynamic_residual_risk.size)
            example_path = _write_example(output, frame, labels, split=split, pose_delta_prev=prev_delta, pose_delta_next=next_delta)
            examples.append(_example_record(output, frame, labels, example_path, split=split, pose_delta_prev=prev_delta, pose_delta_next=next_delta))
            previous_kept = (frame, labels)

    train_ids = sorted([route.route_id for route in routes if splits[route.route_id] == "train"])
    val_ids = sorted([route.route_id for route in routes if splits[route.route_id] == "val"])
    reasons = _pack_acceptance_reasons(
        route_count=len(routes),
        val_route_ids=val_ids,
        route_dynamic_frames=route_dynamic_frames,
        train_ids=train_ids,
    )
    manifest = _base_manifest(
        dataset=dataset,
        input_root=Path(input_root),
        grid_shape=grid_shape,
        meters_per_cell=meters_per_cell,
        frame_stride=frame_stride,
    )
    manifest.update(
        {
            "accepted_real_rgbd_route_bev_pack": not reasons,
            "accepted_real_rgbd_route_bev_student": False,
            "acceptance_reasons": reasons,
            "real_source_frame_count": int(source_frame_count),
            "real_source_route_count": int(len(routes)),
            "used_example_count": int(len(examples)),
            "dynamic_positive_cell_fraction": float(dynamic_positive_cells / max(total_cells, 1)),
            "dynamic_positive_frame_count_by_route": route_dynamic_frames,
            "heldout_route_ids": val_ids,
            "train_route_ids": train_ids,
            "route_held_out_split_basis": "route_id",
            "route_source_manifests": [route.source_manifest for route in routes],
            "examples": examples,
        }
    )
    manifest["manifest_sha256"] = json_sha256({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    write_json(output / "manifest.json", manifest, pretty=True)
    return output / "manifest.json"


def load_pack_manifest(pack_dir: str | Path) -> JsonDict:
    return read_json(Path(pack_dir) / "manifest.json")


def _write_example(
    output: Path,
    frame: RouteFrame,
    labels: RGBDBEVLabels,
    *,
    split: str,
    pose_delta_prev: tuple[float, float, float] | None,
    pose_delta_next: tuple[float, float, float] | None,
) -> Path:
    relative = Path("examples") / split / frame.route_id / f"{frame.frame_index:08d}.npz"
    path = output / relative
    arrays = {
        "schema_version": np.asarray(REAL_RGBD_EXAMPLE_SCHEMA_VERSION),
        "frame_index": np.asarray(frame.frame_index, dtype=np.int64),
        "timestamp_ns": np.asarray(frame.timestamp_ns, dtype=np.int64),
        "route_id": np.asarray(frame.route_id),
        "split": np.asarray(split),
        "pose_delta_prev": np.asarray(pose_delta_prev or (0.0, 0.0, 0.0), dtype=np.float32),
        "pose_delta_prev_mask": np.asarray([1.0 if pose_delta_prev is not None else 0.0], dtype=np.float32),
        "pose_delta_next_teacher_only": np.asarray(pose_delta_next or (0.0, 0.0, 0.0), dtype=np.float32),
        "pose_delta_next_teacher_only_mask": np.asarray([1.0 if pose_delta_next is not None else 0.0], dtype=np.float32),
        "sensor_mask": np.asarray([1.0, 1.0 if frame.depth_path is not None else 0.0, 1.0 if pose_delta_prev is not None else 0.0, 0.0], dtype=np.float32),
        "previous_action": np.zeros((2,), dtype=np.float32),
        "weak_label": np.asarray([True], dtype=np.bool_),
        "product_training_approved": np.asarray([False], dtype=np.bool_),
        "replay_only": np.asarray([True], dtype=np.bool_),
        "not_executed": np.asarray([True], dtype=np.bool_),
        "control_safe": np.asarray([False], dtype=np.bool_),
        "raw_pwm_emitted": np.asarray([False], dtype=np.bool_),
        "hardware_validated": np.asarray([False], dtype=np.bool_),
        **labels.arrays(),
    }
    write_deterministic_npz(path, arrays)
    return path


def _example_record(
    output: Path,
    frame: RouteFrame,
    labels: RGBDBEVLabels,
    example_path: Path,
    *,
    split: str,
    pose_delta_prev: tuple[float, float, float] | None,
    pose_delta_next: tuple[float, float, float] | None,
) -> JsonDict:
    return {
        "schema_version": REAL_RGBD_EXAMPLE_SCHEMA_VERSION,
        "example_path": example_path.relative_to(output).as_posix(),
        "rgb_path": frame.rgb_path.as_posix(),
        "depth_path": frame.depth_path.as_posix() if frame.depth_path is not None else None,
        "timestamp_ns": int(frame.timestamp_ns),
        "route_id": frame.route_id,
        "split_unit_id": frame.split_unit_id,
        "split": split,
        "pose_delta_prev": list(pose_delta_prev) if pose_delta_prev is not None else None,
        "pose_delta_next_teacher_only": list(pose_delta_next) if pose_delta_next is not None else None,
        "camera_intrinsics": dict(frame.intrinsics),
        "runtime_allowed_fields": list(RUNTIME_ALLOWED_FIELDS),
        "teacher_only_fields": list(TEACHER_ONLY_FIELDS),
        "weak_label": True,
        "product_training_approved": False,
        "dynamic_positive": bool(np.count_nonzero(labels.dynamic_residual_risk > 0.5) > 0),
        "target_dynamic_residual_risk_positive_cells": int(np.count_nonzero(labels.dynamic_residual_risk > 0.5)),
    }


def _next_pose_delta(frames: list[RouteFrame], index: int) -> tuple[float, float, float] | None:
    if index + 1 >= len(frames):
        return None
    return pose_delta(frames[index].pose, frames[index + 1].pose)


def _pack_acceptance_reasons(
    *,
    route_count: int,
    val_route_ids: list[str],
    route_dynamic_frames: dict[str, int],
    train_ids: list[str],
) -> list[str]:
    reasons: list[str] = []
    if route_count < 3:
        reasons.append("real_source_route_count_lt_3")
    if not val_route_ids:
        reasons.append("no_heldout_val_route")
    if any(route_id in train_ids for route_id in val_route_ids):
        reasons.append("train_val_route_overlap")
    if sum(route_dynamic_frames.get(route_id, 0) for route_id in val_route_ids) <= 0:
        reasons.append("heldout_val_route_has_zero_dynamic_positive_frames")
    return reasons


def _base_manifest(
    *,
    dataset: str,
    input_root: Path,
    grid_shape: tuple[int, int],
    meters_per_cell: float,
    frame_stride: int,
) -> JsonDict:
    return {
        "schema_version": REAL_RGBD_BEV_PACK_SCHEMA_VERSION,
        "package_type": "RealRGBDRouteBEVPack",
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "dataset_name": dataset,
        "input_root": input_root.as_posix(),
        "real_dataset_required": True,
        "synthetic_or_fixture": False,
        "source_family": "real_rgbd_route",
        "no_fixture_data_used": True,
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "meters_per_cell": float(meters_per_cell),
        "frame_stride": int(frame_stride),
        "runtime_allowed_fields": list(RUNTIME_ALLOWED_FIELDS),
        "teacher_only_fields": list(TEACHER_ONLY_FIELDS),
        "weak_label": True,
        "product_training_approved": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
    }


def _parse_routes(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_grid_shape(value: str) -> tuple[int, int]:
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("grid shape must be HxW, for example 64x64")
    left, right = value.lower().split("x", 1)
    return (int(left), int(right))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a RealRGBDRouteBEVPack from local Bonn/TUM RGB-D route folders. "
            "No dataset is downloaded; missing real input writes an unaccepted manifest."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, for example bonn_rgbd_dynamic or tum_rgbd.")
    parser.add_argument("--input", required=True, help="Local root containing TUM/Bonn sequence folders.")
    parser.add_argument("--out", required=True, help="Output pack directory.")
    parser.add_argument("--train-routes", default=None, help="Comma-separated route IDs for train split.")
    parser.add_argument("--val-routes", default=None, help="Comma-separated route IDs for held-out val split.")
    parser.add_argument("--frame-stride", type=int, default=5, help="Use every Nth frame from each real route.")
    parser.add_argument("--grid-shape", type=_parse_grid_shape, default=(64, 64), help="BEV grid as HxW, for example 64x64.")
    parser.add_argument("--meters-per-cell", type=float, default=0.05, help="BEV meters per cell.")
    args = parser.parse_args(argv)
    build_real_rgbd_route_bev_pack(
        dataset=args.dataset,
        input_root=args.input,
        out_dir=args.out,
        train_routes=_parse_routes(args.train_routes),
        val_routes=_parse_routes(args.val_routes),
        frame_stride=args.frame_stride,
        grid_shape=args.grid_shape,
        meters_per_cell=args.meters_per_cell,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
