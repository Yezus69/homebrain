from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.data.spatial_dataset import (
    DETERMINISTIC_CREATED_AT_UTC,
    json_sha256,
    load_example_npz,
    read_json,
    write_deterministic_npz,
    write_json,
)
from homebrain.messages.schema import JsonDict
from homebrain.policies.candidate_trajectories import CandidateTrajectory, candidates_hash, generate_default_candidates
from homebrain.teachers.artifacts import file_sha256

DYNAMIC_BEV_WORLD_PACK_SCHEMA_VERSION = "homebrain.dynamic_bev_world_pack.v0"
DYNAMIC_BEV_WORLD_EXAMPLE_SCHEMA_VERSION = "homebrain.dynamic_bev_world_example.v0"
DYNAMIC_BEV_HISTORY_CHANNELS: tuple[str, ...] = (
    "free",
    "occupied",
    "unknown",
    "traversable",
    "risky",
    "dynamic_residual_risk",
    "uncertainty",
)
SAFETY_FLAGS: JsonDict = {
    "no_future_labels_at_runtime": True,
    "replay_only": True,
    "not_executed": True,
    "control_safe": False,
    "raw_pwm_emitted": False,
    "hardware_validated": False,
}


def build_dynamic_bev_world_pack(
    *,
    input_pack: str | Path,
    out_dir: str | Path,
    history_frames: int = 6,
    future_horizons_sec: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
    candidate_set: str = "default_low_speed_indoor",
    max_examples: int = 50_000,
    robot_radius_m: float = 0.18,
) -> Path:
    if history_frames <= 0:
        raise ValueError("history_frames must be positive")
    if not future_horizons_sec or any(float(value) <= 0.0 for value in future_horizons_sec):
        raise ValueError("future_horizons_sec must contain positive horizons")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(input_pack)
    manifest_path = source_root / "manifest.json"
    if not manifest_path.exists():
        manifest = _base_manifest(
            source_root=source_root,
            source_manifest=None,
            history_frames=history_frames,
            future_horizons_sec=future_horizons_sec,
            candidate_set=candidate_set,
            candidate_count=0,
            source_pack_sha256=None,
        )
        manifest.update(
            {
                "accepted_dynamic_bev_world_pack": False,
                "acceptance_reasons": ["missing_real_rgbd_route_bev_pack"],
                "report": {"accepted_dynamic_bev_world_pack": False, "reason": "missing_real_rgbd_route_bev_pack"},
                "examples": [],
            }
        )
        _write_manifest_and_report(output, manifest)
        return output / "manifest.json"

    source_manifest = read_json(manifest_path)
    examples = [record for record in source_manifest.get("examples", []) if isinstance(record, dict)]
    if not examples:
        manifest = _base_manifest(
            source_root=source_root,
            source_manifest=source_manifest,
            history_frames=history_frames,
            future_horizons_sec=future_horizons_sec,
            candidate_set=candidate_set,
            candidate_count=0,
            source_pack_sha256=file_sha256(manifest_path),
        )
        manifest.update(
            {
                "accepted_dynamic_bev_world_pack": False,
                "acceptance_reasons": ["missing_real_rgbd_route_bev_pack_examples"],
                "examples": [],
            }
        )
        _write_manifest_and_report(output, manifest)
        return output / "manifest.json"

    grid_shape = _grid_shape(source_manifest, examples, source_root)
    meters_per_cell = float(source_manifest.get("meters_per_cell", 0.05))
    candidates = _candidates(candidate_set, grid_shape=grid_shape, meters_per_cell=meters_per_cell, robot_radius_m=robot_radius_m)
    grouped = _group_examples(examples)
    written: list[JsonDict] = []
    route_dynamic_frames: dict[str, int] = defaultdict(int)
    candidate_positive = 0.0
    candidate_total = 0.0
    for route_id, records in grouped.items():
        loaded = [_LoadedExample(record=record, arrays=load_example_npz(source_root / str(record["example_path"]))) for record in records]
        for index, loaded_example in enumerate(loaded):
            if len(written) >= max_examples:
                break
            arrays = _window_arrays(
                loaded=loaded,
                index=index,
                candidates=candidates,
                history_frames=history_frames,
                horizons_s=future_horizons_sec,
            )
            if bool(np.count_nonzero(arrays["bev_history"][-1, 5] > 0.5)):
                route_dynamic_frames[str(route_id)] += 1
            candidate_positive += float(np.count_nonzero(arrays["candidate_total_teacher_risk"] >= 0.5))
            candidate_total += float(arrays["candidate_total_teacher_risk"].size)
            relative = Path("examples") / str(arrays["split"].item()) / str(route_id) / f"{len(written):08d}.npz"
            write_deterministic_npz(output / relative, arrays)
            written.append(
                {
                    "schema_version": DYNAMIC_BEV_WORLD_EXAMPLE_SCHEMA_VERSION,
                    "example_path": relative.as_posix(),
                    "source_example_path": str(loaded_example.record["example_path"]),
                    "route_id": str(route_id),
                    "split": str(arrays["split"].item()),
                    "timestamp_ns": int(arrays["timestamp_ns"].item()),
                    "history_frames": int(history_frames),
                    "valid_horizon_count": int(np.count_nonzero(arrays["horizon_valid"] > 0.0)),
                    "candidate_count": len(candidates),
                    "dynamic_positive": bool(np.count_nonzero(arrays["future_dynamic_risk"] > 0.5) > 0),
                    "candidate_positive_risk_fraction": float(
                        np.count_nonzero(arrays["candidate_total_teacher_risk"] >= 0.5)
                        / max(arrays["candidate_total_teacher_risk"].size, 1)
                    ),
                }
            )
        if len(written) >= max_examples:
            break

    train_ids = sorted(str(value) for value in source_manifest.get("train_route_ids", []))
    val_ids = sorted(
        str(value)
        for value in source_manifest.get("val_route_ids", source_manifest.get("heldout_route_ids", []))
    )
    if not train_ids or not val_ids:
        train_ids = sorted({str(record.get("route_id", "")) for record in examples if str(record.get("split")) == "train"})
        val_ids = sorted({str(record.get("route_id", "")) for record in examples if str(record.get("split")) == "val"})
    real_frame_count = int(source_manifest.get("real_source_frame_count", len(examples)))
    real_route_count = int(source_manifest.get("real_source_route_count", len(grouped)))
    dynamic_positive_count = int(sum(route_dynamic_frames.values()))
    reasons = _acceptance_reasons(
        source_manifest=source_manifest,
        train_ids=train_ids,
        val_ids=val_ids,
        real_route_count=real_route_count,
        real_frame_count=real_frame_count,
        dynamic_positive_frame_count=dynamic_positive_count,
        example_count=len(written),
    )
    manifest = _base_manifest(
        source_root=source_root,
        source_manifest=source_manifest,
        history_frames=history_frames,
        future_horizons_sec=future_horizons_sec,
        candidate_set=candidate_set,
        candidate_count=len(candidates),
        source_pack_sha256=file_sha256(manifest_path),
    )
    manifest.update(
        {
            "accepted_dynamic_bev_world_pack": not reasons,
            "acceptance_reasons": reasons,
            "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
            "meters_per_cell": float(meters_per_cell),
            "robot_radius_m": float(robot_radius_m),
            "route_heldout_split_basis": str(
                source_manifest.get("route_heldout_split_basis", source_manifest.get("route_held_out_split_basis", "route_id"))
            ),
            "train_route_ids": train_ids,
            "val_route_ids": val_ids,
            "real_source_route_count": real_route_count,
            "real_source_frame_count": real_frame_count,
            "dynamic_positive_frame_count": dynamic_positive_count,
            "dynamic_positive_frame_count_by_route": dict(sorted(route_dynamic_frames.items())),
            "candidate_positive_risk_fraction": float(candidate_positive / max(candidate_total, 1.0)),
            "candidate_ids": [candidate.id for candidate in candidates],
            "candidate_hash": candidates_hash(candidates),
            "history_bev_channels": list(DYNAMIC_BEV_HISTORY_CHANNELS),
            "examples": written,
        }
    )
    manifest["manifest_sha256"] = json_sha256({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    _write_manifest_and_report(output, manifest)
    return output / "manifest.json"


class _LoadedExample:
    def __init__(self, *, record: JsonDict, arrays: dict[str, np.ndarray]) -> None:
        self.record = record
        self.arrays = arrays


def _base_manifest(
    *,
    source_root: Path,
    source_manifest: JsonDict | None,
    history_frames: int,
    future_horizons_sec: tuple[float, ...],
    candidate_set: str,
    candidate_count: int,
    source_pack_sha256: str | None,
) -> JsonDict:
    source_fixture = bool(source_manifest.get("synthetic_or_fixture", False)) if source_manifest else False
    return {
        "schema_version": DYNAMIC_BEV_WORLD_PACK_SCHEMA_VERSION,
        "example_schema_version": DYNAMIC_BEV_WORLD_EXAMPLE_SCHEMA_VERSION,
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "package_type": "DynamicBEVWorldPackV0",
        "input_pack": source_root.as_posix(),
        "source_pack_sha256": source_pack_sha256,
        "history_frames": int(history_frames),
        "future_horizons_sec": [float(value) for value in future_horizons_sec],
        "candidate_set": candidate_set,
        "candidate_count": int(candidate_count),
        "synthetic_or_fixture": source_fixture,
        "runtime_allowed_inputs": [
            "SceneState",
            "LocalBev history",
            "pose_delta_history",
            "previous_action_history",
            "candidate_trajectories",
            "sensor_mask",
        ],
        "teacher_only_targets": [
            "future_free",
            "future_occupied",
            "future_unknown",
            "future_risky",
            "future_dynamic_risk",
            "future_flow_xy",
            "candidate_total_teacher_risk",
        ],
        **SAFETY_FLAGS,
    }


def _grid_shape(source_manifest: JsonDict, examples: list[JsonDict], source_root: Path) -> tuple[int, int]:
    raw = source_manifest.get("grid_shape")
    if isinstance(raw, list) and len(raw) == 2:
        return (int(raw[0]), int(raw[1]))
    first = load_example_npz(source_root / str(examples[0]["example_path"]))
    return tuple(int(value) for value in np.asarray(first["target_current_bev_free"]).shape)  # type: ignore[return-value]


def _candidates(
    candidate_set: str,
    *,
    grid_shape: tuple[int, int],
    meters_per_cell: float,
    robot_radius_m: float,
) -> list[CandidateTrajectory]:
    if candidate_set != "default_low_speed_indoor":
        raise ValueError("only candidate-set default_low_speed_indoor is supported")
    return generate_default_candidates(
        grid_shape=grid_shape,
        meters_per_cell=meters_per_cell,
        robot_radius_m=robot_radius_m,
    )


def _group_examples(examples: list[JsonDict]) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for record in examples:
        grouped[str(record.get("route_id", record.get("split_unit_id", "unknown")))].append(record)
    return {
        route_id: sorted(records, key=lambda item: (int(item.get("timestamp_ns", 0)), str(item.get("example_path", ""))))
        for route_id, records in sorted(grouped.items())
    }


def _window_arrays(
    *,
    loaded: list[_LoadedExample],
    index: int,
    candidates: list[CandidateTrajectory],
    history_frames: int,
    horizons_s: tuple[float, ...],
) -> dict[str, np.ndarray]:
    current = loaded[index]
    history_indices = [max(0, index - history_frames + 1 + offset) for offset in range(history_frames)]
    bev_history = np.stack([_bev_channels(loaded[item].arrays) for item in history_indices], axis=0).astype(np.float32)
    pose_history = np.stack([_pose_delta(loaded[item].arrays, "pose_delta_prev") for item in history_indices], axis=0)
    action_history = np.stack([_vector(loaded[item].arrays, "previous_action", width=2) for item in history_indices], axis=0)
    sensor_history = np.stack([_vector(loaded[item].arrays, "sensor_mask", width=4) for item in history_indices], axis=0)
    observed_history = np.stack([_observed_mask(bev_history[item]) for item in range(history_frames)], axis=0)
    future_targets = [_future_target(loaded, current_index=index, horizon_s=horizon) for horizon in horizons_s]
    shape = tuple(int(value) for value in bev_history.shape[-2:])
    future_free = np.stack([_target_or_fill(item, shape, "target_current_bev_free") for item in future_targets], axis=0)
    future_occupied = np.stack([_target_or_fill(item, shape, "target_current_bev_occupied") for item in future_targets], axis=0)
    future_unknown = np.stack([_target_or_fill(item, shape, "target_current_bev_unknown", fill=1.0) for item in future_targets], axis=0)
    future_risky = np.stack([_target_or_fill(item, shape, "target_current_bev_risky") for item in future_targets], axis=0)
    future_dynamic = np.stack([_target_or_fill(item, shape, "target_dynamic_residual_risk") for item in future_targets], axis=0)
    valid = np.stack([np.ones_like(future_free[0]) if item is not None else np.zeros_like(future_free[0]) for item in future_targets])
    labels = _candidate_labels(
        candidates=candidates,
        future_occupied=future_occupied,
        future_unknown=future_unknown,
        future_risky=future_risky,
        future_dynamic=future_dynamic,
        valid=valid,
    )
    return {
        "schema_version": np.asarray(DYNAMIC_BEV_WORLD_EXAMPLE_SCHEMA_VERSION),
        "route_id": np.asarray(str(current.record.get("route_id", ""))),
        "split": np.asarray(str(current.record.get("split", "train"))),
        "timestamp_ns": np.asarray(int(current.record.get("timestamp_ns", 0)), dtype=np.int64),
        "source_example_index": np.asarray(index, dtype=np.int64),
        "bev_history": bev_history,
        "pose_delta_history": pose_history.astype(np.float32),
        "previous_action_history": action_history.astype(np.float32),
        "sensor_mask_history": sensor_history.astype(np.float32),
        "observed_mask_history": observed_history.astype(np.float32),
        "horizons_s": np.asarray(horizons_s, dtype=np.float32),
        "horizon_valid": np.asarray([1.0 if item is not None else 0.0 for item in future_targets], dtype=np.float32),
        "future_free": future_free.astype(np.float32),
        "future_occupied": future_occupied.astype(np.float32),
        "future_unknown": future_unknown.astype(np.float32),
        "future_risky": future_risky.astype(np.float32),
        "future_dynamic_risk": future_dynamic.astype(np.float32),
        "future_flow_xy": _future_flow(bev_history[-1, 5], future_dynamic).astype(np.float32),
        "future_valid_mask": valid.astype(np.float32),
        "candidate_ids": np.asarray([candidate.id for candidate in candidates]),
        "candidate_trajectory_xytheta": _candidate_trajectory_xytheta(candidates).astype(np.float32),
        "candidate_footprint_masks": _candidate_masks(candidates, future_free.shape[-2:]).astype(np.float32),
        **labels,
        **{key: np.asarray(value, dtype=np.bool_) for key, value in SAFETY_FLAGS.items() if isinstance(value, bool)},
    }


def _bev_channels(example: dict[str, np.ndarray]) -> np.ndarray:
    free = _array(example, "target_current_bev_free")
    occupied = _array(example, "target_current_bev_occupied")
    unknown = _array(example, "target_current_bev_unknown", fill=1.0)
    traversable = _array(example, "target_current_bev_traversable", fallback=free)
    risky = _array(example, "target_current_bev_risky", fallback=occupied)
    dynamic = _array(example, "target_dynamic_residual_risk")
    uncertainty = _array(example, "target_uncertainty_map", fallback=unknown)
    return np.stack([free, occupied, unknown, traversable, risky, dynamic, uncertainty], axis=0).astype(np.float32)


def _array(
    example: dict[str, np.ndarray],
    key: str,
    *,
    fill: float = 0.0,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    if key in example:
        return np.clip(np.asarray(example[key], dtype=np.float32), 0.0, 1.0)
    if fallback is not None:
        return np.asarray(fallback, dtype=np.float32)
    shape = np.asarray(example["target_current_bev_free"]).shape
    return np.full(shape, float(fill), dtype=np.float32)


def _pose_delta(example: dict[str, np.ndarray], key: str) -> np.ndarray:
    return _vector(example, key, width=3)


def _vector(example: dict[str, np.ndarray], key: str, *, width: int) -> np.ndarray:
    value = np.asarray(example.get(key, np.zeros((width,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    out = np.zeros((width,), dtype=np.float32)
    out[: min(width, value.size)] = value[: min(width, value.size)]
    return out


def _observed_mask(bev: np.ndarray) -> np.ndarray:
    observed = (bev[0] > 0.05) | (bev[1] > 0.05) | (bev[3] > 0.05) | (bev[4] > 0.05) | (bev[5] > 0.05)
    observed &= bev[2] < 0.95
    return observed.astype(np.float32)


def _future_target(loaded: list[_LoadedExample], *, current_index: int, horizon_s: float) -> _LoadedExample | None:
    current_ns = int(loaded[current_index].record.get("timestamp_ns", 0))
    target_ns = current_ns + int(round(float(horizon_s) * 1_000_000_000))
    for item in loaded[current_index + 1 :]:
        if int(item.record.get("timestamp_ns", 0)) >= target_ns:
            return item
    return None


def _candidate_labels(
    *,
    candidates: list[CandidateTrajectory],
    future_occupied: np.ndarray,
    future_unknown: np.ndarray,
    future_risky: np.ndarray,
    future_dynamic: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    horizon_count = int(future_occupied.shape[0])
    collision = np.zeros((len(candidates), horizon_count), dtype=np.float32)
    dynamic = np.zeros_like(collision)
    unknown = np.zeros_like(collision)
    for candidate_index, candidate in enumerate(candidates):
        cells = tuple(candidate.footprint_cells)
        for horizon in range(horizon_count):
            if float(np.max(valid[horizon])) <= 0.0:
                continue
            occ_values = _cell_values(np.maximum(future_occupied[horizon], future_risky[horizon]), cells, default=1.0)
            dyn_values = _cell_values(future_dynamic[horizon], cells, default=1.0)
            unk_values = _cell_values(future_unknown[horizon], cells, default=1.0)
            collision[candidate_index, horizon] = _risk_value(occ_values)
            dynamic[candidate_index, horizon] = _risk_value(dyn_values)
            unknown[candidate_index, horizon] = float(np.mean(unk_values))
    total = np.maximum(collision.max(axis=1), dynamic.max(axis=1))
    total = np.maximum(total, 0.35 * unknown.max(axis=1)).astype(np.float32)
    safe = (total < 0.35).astype(np.float32)
    order = np.argsort(total, kind="stable")
    rank = np.zeros_like(total, dtype=np.float32)
    rank[order] = np.arange(len(total), dtype=np.float32) / max(len(total) - 1, 1)
    return {
        "candidate_collision_risk": collision,
        "candidate_dynamic_risk": dynamic,
        "candidate_unknown_exposure": unknown,
        "candidate_total_teacher_risk": total,
        "candidate_safe_label": safe,
        "candidate_rank_target": rank,
    }


def _target_or_fill(target: _LoadedExample | None, shape: tuple[int, int], key: str, fill: float = 0.0) -> np.ndarray:
    return np.full(shape, float(fill), dtype=np.float32) if target is None else _array(target.arrays, key, fill=fill)


def _future_flow(current_dynamic: np.ndarray, future_dynamic: np.ndarray) -> np.ndarray:
    flow = np.zeros((future_dynamic.shape[0], 2, *future_dynamic.shape[-2:]), dtype=np.float32)
    cur_center = _weighted_center(current_dynamic)
    if cur_center is None:
        return flow
    for horizon in range(future_dynamic.shape[0]):
        fut_center = _weighted_center(future_dynamic[horizon])
        if fut_center is None:
            continue
        dy = float(fut_center[0] - cur_center[0])
        dx = float(fut_center[1] - cur_center[1])
        mask = future_dynamic[horizon] > 0.1
        flow[horizon, 0, mask] = dx
        flow[horizon, 1, mask] = dy
    return flow


def _weighted_center(grid: np.ndarray) -> tuple[float, float] | None:
    weight = np.clip(np.asarray(grid, dtype=np.float32), 0.0, 1.0)
    total = float(weight.sum())
    if total <= 1.0e-6:
        return None
    rows, cols = np.indices(weight.shape, dtype=np.float32)
    return (float((rows * weight).sum() / total), float((cols * weight).sum() / total))


def _candidate_trajectory_xytheta(candidates: list[CandidateTrajectory]) -> np.ndarray:
    max_points = max(len(candidate.poses) for candidate in candidates)
    out = np.zeros((len(candidates), max_points, 3), dtype=np.float32)
    for candidate_index, candidate in enumerate(candidates):
        poses = list(candidate.poses)
        if not poses:
            continue
        for pose_index in range(max_points):
            pose = poses[min(pose_index, len(poses) - 1)]
            out[candidate_index, pose_index] = [pose.x_m, pose.y_m, pose.yaw_rad]
    return out


def _candidate_masks(candidates: list[CandidateTrajectory], shape: tuple[int, int]) -> np.ndarray:
    masks = np.zeros((len(candidates), *shape), dtype=np.float32)
    for candidate_index, candidate in enumerate(candidates):
        for row, col in candidate.footprint_cells:
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                masks[candidate_index, row, col] = 1.0
    return masks


def _cell_values(values: np.ndarray, cells: tuple[tuple[int, int], ...], *, default: float) -> np.ndarray:
    if not cells:
        return np.asarray([default], dtype=np.float32)
    out = []
    for row, col in cells:
        if 0 <= row < values.shape[0] and 0 <= col < values.shape[1]:
            out.append(float(values[row, col]))
        else:
            out.append(float(default))
    return np.asarray(out or [default], dtype=np.float32)


def _risk_value(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    return float(np.clip(0.65 * float(np.max(array)) + 0.35 * float(np.mean(array)), 0.0, 1.0))


def _acceptance_reasons(
    *,
    source_manifest: JsonDict,
    train_ids: list[str],
    val_ids: list[str],
    real_route_count: int,
    real_frame_count: int,
    dynamic_positive_frame_count: int,
    example_count: int,
) -> list[str]:
    reasons: list[str] = []
    if bool(source_manifest.get("synthetic_or_fixture", False)):
        reasons.append("synthetic_or_fixture_pack_not_real_acceptance")
    if real_route_count < 3:
        reasons.append("real_source_route_count_lt_3")
    if real_frame_count < 1000:
        reasons.append("real_source_frame_count_lt_1000")
    if not train_ids or not val_ids:
        reasons.append("missing_route_heldout_split")
    if set(train_ids) & set(val_ids):
        reasons.append("train_val_route_overlap")
    if dynamic_positive_frame_count <= 0:
        reasons.append("dynamic_positive_frame_count_zero")
    if example_count <= 0:
        reasons.append("no_dynamic_world_examples")
    return reasons


def _write_manifest_and_report(output: Path, manifest: JsonDict) -> None:
    write_json(output / "manifest.json", manifest, pretty=True)
    write_json(
        output / "report.json",
        {
            "accepted_dynamic_bev_world_pack": bool(manifest.get("accepted_dynamic_bev_world_pack", False)),
            "acceptance_reasons": list(manifest.get("acceptance_reasons", [])),
            "package_type": manifest.get("package_type"),
            **SAFETY_FLAGS,
        },
        pretty=True,
    )


def _parse_horizons(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build replay-only DynamicBEVWorldPackV0 windows from RealRGBDRouteBEVPack.")
    parser.add_argument("--input-pack", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--history-frames", type=int, default=6)
    parser.add_argument("--future-horizons-sec", type=_parse_horizons, default=(0.5, 1.0, 2.0, 3.0))
    parser.add_argument("--candidate-set", default="default_low_speed_indoor")
    parser.add_argument("--max-examples", type=int, default=50_000)
    args = parser.parse_args(argv)
    build_dynamic_bev_world_pack(
        input_pack=args.input_pack,
        out_dir=args.out,
        history_frames=args.history_frames,
        future_horizons_sec=args.future_horizons_sec,
        candidate_set=args.candidate_set,
        max_examples=args.max_examples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
