from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from homebrain.data.spatial_dataset import write_json
from homebrain.policies.bev_action_sanity import ActionSanityConfig, evaluate_loaded_frame, load_bev_frames_for_action
from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.policies.trajectory_scorer import CoverageMemory, LocalBev, score_trajectories


def compare_robot_frame_action_sources(
    *,
    sources: list[str | Path],
    out_dir: str | Path,
    labels: list[str] | None = None,
    max_contact_frames: int = 24,
) -> dict[str, object]:
    if not sources:
        raise ValueError("at least one source is required")
    output = Path(out_dir)
    _clear_outputs(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = ActionSanityConfig()
    source_labels = labels or [Path(source).name for source in sources]
    if len(source_labels) != len(sources):
        raise ValueError("--label count must match --source count")

    per_source: dict[str, object] = {}
    contact_tiles: list[np.ndarray] = []
    for label, source in zip(source_labels, sources):
        frames, metadata = load_bev_frames_for_action(source, scratch_dir=output / f"scratch_{_safe_name(label)}")
        metrics = []
        selected_ids = []
        candidates_by_shape: dict[tuple[tuple[int, int], float, float], list[object]] = {}
        coverage_by_shape: dict[tuple[int, int], CoverageMemory] = {}
        for frame in frames:
            candidate_key = (frame.record.bev.shape, frame.meters_per_cell, frame.robot_radius_m)
            if candidate_key not in candidates_by_shape:
                candidates_by_shape[candidate_key] = generate_default_candidates(
                    grid_shape=frame.record.bev.shape,
                    meters_per_cell=frame.meters_per_cell,
                    robot_radius_m=frame.robot_radius_m,
                )
            sanity = evaluate_loaded_frame(frame, config=cfg, candidates=candidates_by_shape[candidate_key])  # type: ignore[arg-type]
            metrics.append(sanity)
            coverage = coverage_by_shape.setdefault(
                frame.record.bev.shape,
                CoverageMemory(frame.record.bev.shape, meters_per_cell=frame.meters_per_cell),
            )
            decision = score_trajectories(
                bev=frame.record.bev,
                candidates=candidates_by_shape[candidate_key],  # type: ignore[arg-type]
                coverage_memory=coverage,
            )
            selected_ids.append(decision.selected_candidate_id)
            coverage.update_current_frame(frame.record.bev)
            if len(contact_tiles) < max_contact_frames:
                contact_tiles.append(_tile(frame.record.bev, action_ok=bool(sanity.get("action_supervision_ok"))))
        per_source[label] = _summarize(label=label, source=source, metadata=metadata, metrics=metrics, selected_ids=selected_ids)

    contact_sheet = "robot_frame_action_comparison_contact_sheet.ppm"
    _write_contact_sheet(output / contact_sheet, contact_tiles)
    report: dict[str, object] = {
        "schema_version": "homebrain.robot_frame_action_comparison.v0",
        "sources": [Path(source).as_posix() for source in sources],
        "per_source": per_source,
        "contact_sheet": contact_sheet,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
    }
    write_json(output / "robot_frame_action_comparison.json", report, pretty=True)
    return report


def _summarize(
    *,
    label: str,
    source: str | Path,
    metadata: dict[str, object],
    metrics: list[dict[str, object]],
    selected_ids: list[str],
) -> dict[str, object]:
    stop_count = sum(1 for item in selected_ids if item == "stop")
    return {
        "label": label,
        "source": Path(source).as_posix(),
        "source_metadata": metadata,
        "frame_count": len(metrics),
        "center_blocked_rate": _rate(metrics, "robot_center_blocked"),
        "footprint_blocked_rate": _rate(metrics, "footprint_blocked"),
        "forward_corridor_free_rate": _rate(metrics, "forward_corridor_free"),
        "action_supervision_ok_fraction": _rate(metrics, "action_supervision_ok"),
        "selected_motion_fraction": float((len(selected_ids) - stop_count) / max(len(selected_ids), 1)),
        "selected_stop_fraction": float(stop_count / max(len(selected_ids), 1)),
        "origin_frame_status_distribution": _distribution([str(item.get("origin_frame_status", "unknown")) for item in metrics]),
        "selected_distribution": _distribution(selected_ids),
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
    }


def _rate(metrics: list[dict[str, object]], key: str) -> float:
    if not metrics:
        return 0.0
    return float(sum(1 for item in metrics if bool(item.get(key))) / len(metrics))


def _distribution(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _tile(bev: LocalBev, *, action_ok: bool) -> np.ndarray:
    free = np.clip(bev.free.astype(np.float32), 0.0, 1.0)
    occupied = np.clip(bev.occupied.astype(np.float32), 0.0, 1.0)
    unknown = np.clip(bev.unknown.astype(np.float32), 0.0, 1.0)
    rgb = np.stack([occupied, free, unknown], axis=2)
    tile = (np.clip(rgb, 0.0, 1.0) * np.float32(170.0)).round().astype(np.uint8)
    scale = 6 if min(bev.shape) >= 16 else 16
    tile = np.repeat(np.repeat(tile, scale, axis=0), scale, axis=1)
    color = np.asarray([40, 230, 100], dtype=np.uint8) if action_ok else np.asarray([255, 70, 70], dtype=np.uint8)
    origin = (bev.shape[0] - 1, bev.shape[1] // 2)
    _paint_cell(tile, origin[0], origin[1], scale, color)
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


def _clear_outputs(output: Path) -> None:
    for name in ("robot_frame_action_comparison.json", "robot_frame_action_comparison_contact_sheet.ppm"):
        path = output / name
        if path.exists():
            path.unlink()
    for scratch in output.glob("scratch_*"):
        if scratch.is_dir():
            shutil.rmtree(scratch)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:64] or "source"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare robot-frame public BEVs against geometry-only action sources.")
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-contact-frames", type=int, default=24)
    args = parser.parse_args(argv)
    report = compare_robot_frame_action_sources(
        sources=[Path(source) for source in args.source],
        labels=args.label,
        out_dir=args.out,
        max_contact_frames=args.max_contact_frames,
    )
    printable = {key: value for key, value in report.items() if key != "per_source"}
    print(json.dumps(printable, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
