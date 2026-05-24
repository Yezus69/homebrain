from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from homebrain.data.spatial_dataset import write_json
from homebrain.policies.bev_action_sanity import (
    ActionSanityConfig,
    evaluate_loaded_frame,
    frame_metric_severity,
    load_bev_frames_for_action,
    sanity_summary_json,
)
from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.policies.trajectory_scorer import LocalBev


def audit_bev_action_sanity(
    *,
    source: str | Path,
    out_dir: str | Path,
    config: ActionSanityConfig | None = None,
    max_contact_frames: int = 24,
) -> dict[str, object]:
    cfg = config or ActionSanityConfig()
    output = Path(out_dir)
    _clear_outputs(output)
    output.mkdir(parents=True, exist_ok=True)

    frames, source_metadata = load_bev_frames_for_action(source, scratch_dir=output)
    if not frames:
        raise ValueError(f"no BEV frames found for action sanity audit: {source}")

    candidates_by_shape: dict[tuple[tuple[int, int], float, float], list[object]] = {}
    frame_metrics = []
    for frame in frames:
        key = (frame.record.bev.shape, frame.meters_per_cell, max(frame.robot_radius_m, cfg.robot_radius_cells * frame.meters_per_cell))
        if key not in candidates_by_shape:
            candidates_by_shape[key] = generate_default_candidates(
                grid_shape=frame.record.bev.shape,
                meters_per_cell=frame.meters_per_cell,
                robot_radius_m=max(frame.robot_radius_m, cfg.robot_radius_cells * frame.meters_per_cell),
            )
        metrics = evaluate_loaded_frame(frame, config=cfg, candidates=candidates_by_shape[key])  # type: ignore[arg-type]
        frame_metrics.append(metrics)

    ranked = sorted(zip(frames, frame_metrics), key=lambda item: frame_metric_severity(item[1]), reverse=True)
    contact_sheet_name = "worst_frame_contact_sheet.ppm"
    _write_contact_sheet(
        output / contact_sheet_name,
        [
            _tile(frame.record.bev, metric)
            for frame, metric in ranked[:max(0, int(max_contact_frames))]
        ],
    )
    report = sanity_summary_json(
        source=source,
        source_metadata=source_metadata,
        config=cfg,
        frame_metrics=frame_metrics,
        contact_sheet_name=contact_sheet_name,
    )
    write_json(output / "bev_action_sanity.json", report, pretty=True)
    return report


def _tile(bev: LocalBev, metric: dict[str, object]) -> np.ndarray:
    free = np.clip(bev.free.astype(np.float32), 0.0, 1.0)
    occupied = np.clip(bev.occupied.astype(np.float32), 0.0, 1.0)
    unknown = np.clip(bev.unknown.astype(np.float32), 0.0, 1.0)
    risky = np.clip(bev.risky.astype(np.float32), 0.0, 1.0) if bev.risky is not None else occupied
    rgb = np.stack([np.maximum(occupied, risky), free, unknown], axis=2)
    tile = (np.clip(rgb, 0.0, 1.0) * np.float32(170.0)).round().astype(np.uint8)
    scale = 6 if min(bev.shape) >= 16 else 16
    tile = np.repeat(np.repeat(tile, scale, axis=0), scale, axis=1)
    origin_row = bev.shape[0] - 1
    origin_col = bev.shape[1] // 2
    _paint_cell(tile, origin_row, origin_col, scale, np.asarray([255, 255, 255], dtype=np.uint8))
    color = np.asarray([255, 70, 70], dtype=np.uint8)
    if bool(metric.get("action_supervision_ok")):
        color = np.asarray([40, 230, 100], dtype=np.uint8)
    for row in range(max(0, origin_row - 10), origin_row + 1):
        for col in range(max(0, origin_col - 3), min(bev.shape[1], origin_col + 4)):
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


def _clear_outputs(output: Path) -> None:
    for name in ("bev_action_sanity.json", "worst_frame_contact_sheet.ppm"):
        path = output / name
        if path.exists():
            path.unlink()
    generated_brain_outputs = output / "brain_outputs"
    if generated_brain_outputs.exists():
        shutil.rmtree(generated_brain_outputs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit whether BEV frames satisfy the robot-frame action contract.")
    parser.add_argument("--source", required=True, help="Input SpatialTrainPack, BEV dir, modeld output, or policy dir.")
    parser.add_argument("--out", required=True, help="Output audit directory.")
    parser.add_argument("--robot-radius-cells", type=int, default=3)
    parser.add_argument("--forward-corridor-length-cells", type=int, default=10)
    parser.add_argument("--forward-corridor-half-width-cells", type=int, default=3)
    parser.add_argument("--max-contact-frames", type=int, default=24)
    args = parser.parse_args(argv)
    report = audit_bev_action_sanity(
        source=args.source,
        out_dir=args.out,
        config=ActionSanityConfig(
            robot_radius_cells=args.robot_radius_cells,
            forward_corridor_length_cells=args.forward_corridor_length_cells,
            forward_corridor_half_width_cells=args.forward_corridor_half_width_cells,
        ),
        max_contact_frames=args.max_contact_frames,
    )
    printable = {key: value for key, value in report.items() if key != "frames"}
    print(json.dumps(printable, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
