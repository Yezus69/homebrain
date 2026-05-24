from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from homebrain.brain.modeld import spatial_model_outputs
from homebrain.data.spatial_dataset import load_example_npz, read_json, write_json
from homebrain.geometry.bev_projector import BEV_MANIFEST_FILE
from homebrain.geometry.validate_bev import load_bev_manifest
from homebrain.messages.schema import BrainOutputEvent, Event, FrameEvent, JsonDict, deterministic_json
from homebrain.policies.candidate_trajectories import CandidateTrajectory, generate_default_candidates
from homebrain.policies.trajectory_scorer import (
    CoverageMemory,
    LocalBev,
    TrajectoryDecision,
    risky_candidate_fraction,
    score_trajectories,
)
from homebrain.replay.replayd import order_events_for_replay
from homebrain.replay.segment_log import read_events
from homebrain.teachers.artifacts import load_array

POLICY_ARTIFACT_SCHEMA_VERSION = "homebrain.policy_artifacts.v0"
MODEL_BEV_SOURCES = ("model", "v1_current_bev", "v1_memory_bev")


@dataclass(frozen=True)
class BevDecisionInput:
    sequence_id: str
    camera_id: str
    frame_id: int
    timestamp_ns: int
    bev: LocalBev
    pose_delta: tuple[float, float, float] | None
    source_ref: str


def run_trajectory_scorer(
    *,
    log_dir: str | Path,
    out_dir: str | Path,
    checkpoint: str | Path | None = None,
    features: str | Path | None = None,
    bev_source: str = "model",
    device_name: str | None = None,
    max_frames: int | None = None,
    max_overlay_frames: int = 24,
) -> JsonDict:
    started = time.perf_counter()
    root = Path(log_dir)
    output = Path(out_dir)
    _clear_policy_outputs(output)
    output.mkdir(parents=True, exist_ok=True)

    records, source_metadata = _load_decision_inputs(
        root,
        output,
        checkpoint=checkpoint,
        features=features,
        bev_source=bev_source,
        device_name=device_name,
    )
    if max_frames is not None:
        records = records[:max_frames]
    if not records:
        raise ValueError("no BEV frames found for trajectory scoring")

    meters_per_cell = _meters_per_cell(source_metadata, records[0].bev.shape)
    robot_radius_m = _robot_radius_m(source_metadata)
    candidates = generate_default_candidates(
        grid_shape=records[0].bev.shape,
        meters_per_cell=meters_per_cell,
        robot_radius_m=robot_radius_m,
    )
    coverage_memory = CoverageMemory(records[0].bev.shape, meters_per_cell=meters_per_cell)
    decisions: list[JsonDict] = []
    decision_objects: list[TrajectoryDecision] = []
    overlay_tiles: list[np.ndarray] = []

    for index, record in enumerate(records):
        coverage_memory.align_with_pose_delta(record.pose_delta)
        decision = score_trajectories(
            bev=record.bev,
            candidates=candidates,
            coverage_memory=coverage_memory,
        )
        coverage_memory.update_current_frame(record.bev)
        decision_objects.append(decision)
        decision_record = _decision_record(
            record=record,
            frame_index=index,
            candidates=candidates,
            decision=decision,
            coverage_memory=coverage_memory,
        )
        decisions.append(decision_record)
        if index < max_overlay_frames:
            overlay_tiles.append(_overlay_tile(record.bev, candidates, decision.selected_candidate_id))

    decisions_path = output / "trajectory_decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for decision in decisions:
            handle.write(deterministic_json(decision))
            handle.write("\n")

    overlay_path = output / "trajectory_overlay_contact_sheet.ppm"
    _write_contact_sheet(overlay_path, overlay_tiles)

    metrics = _eval_metrics(
        records=records,
        decisions=decision_objects,
        candidate_count=len(candidates),
        coverage_memory=coverage_memory,
        runtime_ms=(time.perf_counter() - started) * 1000.0,
    )
    eval_path = output / "trajectory_eval.json"
    write_json(eval_path, metrics, pretty=True)

    manifest = {
        "schema_version": POLICY_ARTIFACT_SCHEMA_VERSION,
        "source": root.as_posix(),
        "bev_source": bev_source,
        "checkpoint": Path(checkpoint).as_posix() if checkpoint is not None else None,
        "features": Path(features).as_posix() if features is not None else None,
        "decision_count": len(decisions),
        "candidate_count": len(candidates),
        "meters_per_cell": meters_per_cell,
        "robot_radius_m": robot_radius_m,
        "decision_jsonl": decisions_path.name,
        "eval_json": eval_path.name,
        "overlay_contact_sheet": overlay_path.name,
        "replay_only": True,
        "control_safe": False,
        "not_executed": True,
        "product_training_approved": False,
        "raw_pwm_emitted": False,
        "source_metadata": source_metadata,
    }
    manifest_path = output / "trajectory_policy_manifest.json"
    write_json(manifest_path, manifest, pretty=True)
    return metrics


def _load_decision_inputs(
    root: Path,
    output: Path,
    *,
    checkpoint: str | Path | None,
    features: str | Path | None,
    bev_source: str,
    device_name: str | None,
) -> tuple[list[BevDecisionInput], JsonDict]:
    if checkpoint is not None and features is not None:
        events = order_events_for_replay(read_events(root))
        outputs, _artifacts = spatial_model_outputs(
            events,
            log_dir=root,
            out_dir=output,
            checkpoint=checkpoint,
            feature_dir=features,
            device_name=device_name,
        )
        records = [_record_from_model_event(event, output, bev_source=bev_source) for event in outputs]
        return records, {"source_kind": "spatial_model_inference", "grid_shape": list(records[0].bev.shape) if records else []}
    if bev_source in MODEL_BEV_SOURCES:
        events = read_events(root)
        model_events = [
            event
            for event in events
            if isinstance(event, BrainOutputEvent) and isinstance(event.local_bev_ref, str)
        ]
        records = [_record_from_model_event(event, root, bev_source=bev_source) for event in model_events]
        return records, {
            "source_kind": "modeld_output",
            "model_bev_source": bev_source,
            "grid_shape": list(records[0].bev.shape) if records else [],
        }
    if bev_source == "labels":
        if (root / "manifest.json").exists():
            manifest = read_json(root / "manifest.json")
            if manifest.get("package_type") == "SpatialTrainPack":
                return _records_from_spatial_pack(root, manifest)
        bev_dir = _infer_bev_dir(root)
        manifest = load_bev_manifest(bev_dir)
        return _records_from_bev_manifest(bev_dir, manifest)
    raise ValueError("--bev-source must be model, labels, v1_current_bev, or v1_memory_bev")


def _record_from_model_event(
    event: BrainOutputEvent,
    artifact_root: Path,
    *,
    bev_source: str = "model",
) -> BevDecisionInput:
    if not event.local_bev_ref:
        raise ValueError("BrainOutputEvent is missing local_bev_ref")
    artifact_path = artifact_root / event.local_bev_ref
    with np.load(artifact_path, allow_pickle=False) as data:
        free, occupied, unknown, traversable, risky, uncertainty = _model_bev_arrays(data, bev_source=bev_source)
    frame_id = int(event.debug.get("input_frame_id", -1)) if isinstance(event.debug, dict) else -1
    return BevDecisionInput(
        sequence_id=event.sequence_id,
        camera_id=str(event.debug.get("camera_id", "unknown")) if isinstance(event.debug, dict) else "unknown",
        frame_id=frame_id,
        timestamp_ns=event.timestamp_ns,
        bev=LocalBev(
            free=free,
            occupied=occupied,
            unknown=unknown,
            traversable=traversable,
            risky=risky,
            confidence=(np.float32(1.0) - uncertainty).astype(np.float32) if uncertainty is not None else None,
            uncertainty=uncertainty,
            source=bev_source,
        ),
        pose_delta=event.pose_delta,
        source_ref=event.local_bev_ref,
    )


def _model_bev_arrays(
    data: np.lib.npyio.NpzFile,
    *,
    bev_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if bev_source == "model":
        if "bev_free_prob" not in data.files:
            raise ValueError("v1 modeld artifacts require explicit bev_source='v1_current_bev' or 'v1_memory_bev'")
        prefix = "bev"
    elif bev_source == "v1_current_bev":
        prefix = "current_bev"
    elif bev_source == "v1_memory_bev":
        prefix = "memory_bev"
    else:
        raise ValueError(f"unsupported model BEV source: {bev_source}")
    required = (
        f"{prefix}_free_prob",
        f"{prefix}_occupied_prob",
        f"{prefix}_unknown_prob",
        f"{prefix}_traversable_prob",
        f"{prefix}_risky_prob",
    )
    missing = [name for name in required if name not in data.files]
    if missing:
        raise ValueError(f"model BEV artifact is missing {bev_source} arrays: {missing}")
    uncertainty = np.asarray(data["uncertainty_grid"], dtype=np.float32) if "uncertainty_grid" in data.files else None
    return (
        np.asarray(data[f"{prefix}_free_prob"], dtype=np.float32),
        np.asarray(data[f"{prefix}_occupied_prob"], dtype=np.float32),
        np.asarray(data[f"{prefix}_unknown_prob"], dtype=np.float32),
        np.asarray(data[f"{prefix}_traversable_prob"], dtype=np.float32),
        np.asarray(data[f"{prefix}_risky_prob"], dtype=np.float32),
        uncertainty,
    )


def _records_from_spatial_pack(root: Path, manifest: JsonDict) -> tuple[list[BevDecisionInput], JsonDict]:
    examples = manifest.get("examples") or manifest.get("frames")
    if not isinstance(examples, list):
        raise ValueError("SpatialTrainPack manifest missing examples")
    records: list[BevDecisionInput] = []
    for example_record in examples:
        if not isinstance(example_record, dict) or not isinstance(example_record.get("example_path"), str):
            continue
        example_path = root / str(example_record["example_path"])
        example = load_example_npz(example_path)
        pose_delta = None
        if "pose_delta" in example and float(np.asarray(example.get("pose_delta_mask", 0.0)).item()) > 0.0:
            pose_delta = tuple(float(value) for value in np.asarray(example["pose_delta"], dtype=np.float32).reshape(3))
        records.append(
            BevDecisionInput(
                sequence_id=str(example_record.get("sequence_id", manifest.get("source_segment_id", root.name))),
                camera_id=str(example_record.get("camera_id", "front_rgb")),
                frame_id=int(example_record["frame_id"]),
                timestamp_ns=int(example_record["timestamp_ns"]),
                bev=LocalBev(
                    free=np.asarray(example["bev_free"], dtype=np.float32),
                    occupied=np.asarray(example["bev_obstacle"], dtype=np.float32),
                    unknown=np.asarray(example["bev_unknown"], dtype=np.float32),
                    traversable=np.asarray(example["bev_free"], dtype=np.float32),
                    risky=np.asarray(example["bev_obstacle"], dtype=np.float32),
                    confidence=np.asarray(example["bev_confidence"], dtype=np.float32),
                    source="labels",
                ),
                pose_delta=pose_delta,
                source_ref=str(example_record["example_path"]),
            )
        )
    return records, {
        "source_kind": "spatial_train_pack",
        "source_log": manifest.get("source_log"),
        "grid_shape": manifest.get("grid_shape", []),
        "camera_config": manifest.get("camera_config", {}),
        "weak_label": manifest.get("weak_label"),
        "control_safe": manifest.get("control_safe"),
    }


def _records_from_bev_manifest(bev_dir: Path, manifest: JsonDict) -> tuple[list[BevDecisionInput], JsonDict]:
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("BEV manifest missing frames")
    records: list[BevDecisionInput] = []
    for frame_record in frames:
        if not isinstance(frame_record, dict):
            continue
        artifacts = frame_record.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        arrays = {}
        for kind in ("bev_free", "bev_obstacle", "bev_unknown", "bev_confidence"):
            artifact = artifacts.get(kind)
            if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                raise ValueError(f"BEV frame missing artifact {kind}: {frame_record.get('frame_id')}")
            arrays[kind] = load_array(bev_dir / str(artifact["path"]))
        records.append(
            BevDecisionInput(
                sequence_id=str(frame_record.get("sequence_id", manifest.get("source_segment_id", bev_dir.name))),
                camera_id=str(frame_record.get("camera_id", "front_rgb")),
                frame_id=int(frame_record["frame_id"]),
                timestamp_ns=int(frame_record["timestamp_ns"]),
                bev=LocalBev(
                    free=np.asarray(arrays["bev_free"], dtype=np.float32),
                    occupied=np.asarray(arrays["bev_obstacle"], dtype=np.float32),
                    unknown=np.asarray(arrays["bev_unknown"], dtype=np.float32),
                    traversable=np.asarray(arrays["bev_free"], dtype=np.float32),
                    risky=np.asarray(arrays["bev_obstacle"], dtype=np.float32),
                    confidence=np.asarray(arrays["bev_confidence"], dtype=np.float32),
                    source="labels",
                ),
                pose_delta=None,
                source_ref=f"{bev_dir.as_posix()}:{frame_record.get('frame_id')}",
            )
        )
    return records, {
        "source_kind": "bev_manifest",
        "source_log": manifest.get("source_log"),
        "grid_shape": manifest.get("grid_shape", []),
        "camera_config": manifest.get("camera_config", {}),
        "weak_label": manifest.get("weak_label"),
        "control_safe": manifest.get("control_safe"),
    }


def _infer_bev_dir(root: Path) -> Path:
    if (root / BEV_MANIFEST_FILE).exists():
        return root
    geometry = root / "geometry"
    if not geometry.exists():
        raise FileNotFoundError(f"could not infer BEV labels under {root}")
    candidates = sorted(path.parent for path in geometry.rglob(BEV_MANIFEST_FILE))
    if not candidates:
        raise FileNotFoundError(f"could not infer BEV labels under {geometry}")
    preferred_terms = ("da3_weak_bev_stable", "rgbd_truth_bev", "depth_pro_bev")
    for term in preferred_terms:
        for candidate in candidates:
            if term in candidate.name:
                return candidate
    return candidates[0]


def _decision_record(
    *,
    record: BevDecisionInput,
    frame_index: int,
    candidates: list[CandidateTrajectory],
    decision: TrajectoryDecision,
    coverage_memory: CoverageMemory,
) -> JsonDict:
    selected_score = decision.selected_score()
    scores_by_id = {score.candidate_id: score.to_dict() for score in decision.scores}
    return {
        "schema_version": POLICY_ARTIFACT_SCHEMA_VERSION,
        "event_type": "trajectory_decision",
        "sequence_id": record.sequence_id,
        "camera_id": record.camera_id,
        "frame_id": int(record.frame_id),
        "frame_index": int(frame_index),
        "timestamp_ns": int(record.timestamp_ns),
        "source_ref": record.source_ref,
        "candidate_count": len(candidates),
        "candidates": [
            {**candidate.to_dict(), "score": scores_by_id[candidate.id]}
            for candidate in candidates
        ],
        "selected_candidate_id": decision.selected_candidate_id,
        "selected_score": selected_score.to_dict(),
        "reason": decision.reason,
        "risky_candidate_fraction": risky_candidate_fraction(decision),
        "coverage_memory": coverage_memory.to_dict(),
        "pose_aligned": bool(coverage_memory.pose_aligned),
        "replay_only": True,
        "control_safe": False,
        "not_executed": True,
        "product_training_approved": False,
        "raw_pwm_emitted": False,
    }


def _eval_metrics(
    *,
    records: list[BevDecisionInput],
    decisions: list[TrajectoryDecision],
    candidate_count: int,
    coverage_memory: CoverageMemory,
    runtime_ms: float,
) -> JsonDict:
    selected_scores = [decision.selected_score() for decision in decisions]
    stop_count = sum(1 for decision in decisions if decision.selected_candidate_id == "stop")
    selected_ids = [decision.selected_candidate_id for decision in decisions]
    return {
        "schema_version": "homebrain.trajectory_eval.v0",
        "frame_count": len(records),
        "candidate_count": int(candidate_count),
        "selected_candidate_id": _mode(selected_ids),
        "selected_candidate_ids": selected_ids,
        "risky_candidate_fraction": float(np.mean([risky_candidate_fraction(decision) for decision in decisions])),
        "selected_risk_score": float(np.mean([score.risk_score for score in selected_scores])),
        "selected_coverage_gain_proxy": float(np.mean([score.coverage_gain_proxy for score in selected_scores])),
        "selected_unknown_penalty": float(np.mean([score.unknown_penalty for score in selected_scores])),
        "selected_uncertainty_penalty": float(np.mean([score.uncertainty_penalty for score in selected_scores])),
        "trajectory_eval_runtime_ms": float(runtime_ms),
        "stop_selected_fraction": float(stop_count / max(len(decisions), 1)),
        "coverage_memory_cells_seen": int(coverage_memory.cells_seen),
        "coverage_memory_cells_covered": int(coverage_memory.cells_covered),
        "pose_aligned_final": bool(coverage_memory.pose_aligned),
        "replay_only": True,
        "control_safe": False,
        "not_executed": True,
        "product_training_approved": False,
    }


def _mode(values: list[str]) -> str | None:
    if not values:
        return None
    counts = {value: values.count(value) for value in sorted(set(values))}
    return min(counts, key=lambda value: (-counts[value], value))


def _meters_per_cell(metadata: JsonDict, shape: tuple[int, int]) -> float:
    camera_config = metadata.get("camera_config")
    if isinstance(camera_config, dict):
        value = camera_config.get("meters_per_cell")
        if isinstance(value, (int, float)) and float(value) > 0.0:
            return float(value)
    if min(shape) <= 8:
        return 0.5
    return 0.05


def _robot_radius_m(metadata: JsonDict) -> float:
    camera_config = metadata.get("camera_config")
    if isinstance(camera_config, dict):
        value = camera_config.get("robot_radius_m")
        if isinstance(value, (int, float)) and float(value) >= 0.0:
            return float(value)
    return 0.18


def _overlay_tile(bev: LocalBev, candidates: list[CandidateTrajectory], selected_candidate_id: str) -> np.ndarray:
    free = np.clip(bev.free.astype(np.float32), 0.0, 1.0)
    occupied = np.clip(bev.occupied.astype(np.float32), 0.0, 1.0)
    unknown = np.clip(bev.unknown.astype(np.float32), 0.0, 1.0)
    if bev.uncertainty is not None:
        blue = np.maximum(unknown * 0.5, np.clip(bev.uncertainty.astype(np.float32), 0.0, 1.0))
    else:
        blue = unknown
    rgb = np.stack([occupied, free, blue], axis=2)
    tile = (rgb * np.float32(170.0)).round().astype(np.uint8)
    scale = 6 if min(bev.shape) >= 16 else 16
    tile = np.repeat(np.repeat(tile, scale, axis=0), scale, axis=1)
    for candidate in candidates:
        color = np.asarray([150, 150, 150], dtype=np.uint8)
        if candidate.id == selected_candidate_id:
            color = np.asarray([255, 230, 40], dtype=np.uint8)
        for row, col in candidate.footprint_cells:
            _paint_cell(tile, row, col, scale, color)
    return tile


def _paint_cell(tile: np.ndarray, row: int, col: int, scale: int, color: np.ndarray) -> None:
    row_start = row * scale
    col_start = col * scale
    if row_start < 0 or col_start < 0 or row_start >= tile.shape[0] or col_start >= tile.shape[1]:
        return
    row_end = min(row_start + scale, tile.shape[0])
    col_end = min(col_start + scale, tile.shape[1])
    patch = tile[row_start:row_end, col_start:col_end]
    tile[row_start:row_end, col_start:col_end] = ((patch.astype(np.uint16) + color.astype(np.uint16)) // 2).astype(np.uint8)


def _write_contact_sheet(path: Path, tiles: list[np.ndarray]) -> None:
    if not tiles:
        path.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
        return
    tile_h, tile_w, _ = tiles[0].shape
    cols = min(6, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // cols
        col = index % cols
        sheet[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    path.write_bytes(f"P6\n{sheet.shape[1]} {sheet.shape[0]}\n255\n".encode("ascii") + sheet.tobytes())


def _clear_policy_outputs(output: Path) -> None:
    for name in (
        "trajectory_decisions.jsonl",
        "trajectory_eval.json",
        "trajectory_overlay_contact_sheet.ppm",
        "trajectory_policy_manifest.json",
    ):
        path = output / name
        if path.exists():
            path.unlink()
    generated_brain_outputs = output / "brain_outputs"
    if generated_brain_outputs.exists():
        shutil.rmtree(generated_brain_outputs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score replay-only candidate trajectories from model or label BEV.")
    parser.add_argument("--log", required=True, help="Input route, modeld output, BEV dir, or SpatialTrainPack.")
    parser.add_argument("--checkpoint", default=None, help="Optional SpatialMemoryNet checkpoint.")
    parser.add_argument("--features", default=None, help="Optional DINO feature artifact directory.")
    parser.add_argument("--bev-source", choices=("model", "labels", "v1_current_bev", "v1_memory_bev"), default="model")
    parser.add_argument("--out", required=True, help="Output policy artifact directory.")
    parser.add_argument("--device", default=None, help="Optional torch device for checkpoint inference.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-overlay-frames", type=int, default=24)
    args = parser.parse_args(argv)
    metrics = run_trajectory_scorer(
        log_dir=args.log,
        out_dir=args.out,
        checkpoint=args.checkpoint,
        features=args.features,
        bev_source=args.bev_source,
        device_name=args.device,
        max_frames=args.max_frames,
        max_overlay_frames=args.max_overlay_frames,
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
