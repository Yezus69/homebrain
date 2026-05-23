from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from homebrain.messages.schema import BrainOutputEvent
from homebrain.replay.segment_log import read_events


def visualize_spatial_outputs(*, log_dir: str | Path, out_dir: str | Path, max_frames: int = 24) -> Path:
    root = Path(log_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    events = [event for event in read_events(root) if isinstance(event, BrainOutputEvent) and event.local_bev_ref]
    tiles = []
    frame_records = []
    for event in events[:max_frames]:
        artifact_path = root / str(event.local_bev_ref)
        with np.load(artifact_path, allow_pickle=False) as data:
            tile = _bev_tile(data)
        tiles.append(tile)
        frame_records.append(
            {
                "timestamp_ns": event.timestamp_ns,
                "local_bev_ref": event.local_bev_ref,
                "uncertainty": event.uncertainty,
            }
        )
    sheet_path = output / "spatial_v0_contact_sheet.ppm"
    _write_contact_sheet(sheet_path, tiles)
    manifest = {
        "schema_version": "homebrain.spatial_v0_viz.v0",
        "source_log": root.as_posix(),
        "visualization_written": bool(tiles),
        "frame_count": len(tiles),
        "contact_sheet": sheet_path.name,
        "frames": frame_records,
        "representation_pretraining_only": True,
        "control_safe": False,
    }
    manifest_path = output / "visualization_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return manifest_path


def _bev_tile(data: np.lib.npyio.NpzFile) -> np.ndarray:
    free = np.asarray(data["bev_free_prob"], dtype=np.float32)
    occupied = np.asarray(data["bev_occupied_prob"], dtype=np.float32)
    uncertainty = np.asarray(data["uncertainty_grid"], dtype=np.float32)
    rgb = np.stack(
        [
            np.clip(occupied, 0.0, 1.0),
            np.clip(free, 0.0, 1.0),
            np.clip(uncertainty, 0.0, 1.0),
        ],
        axis=2,
    )
    return (rgb * np.float32(255.0)).round().astype(np.uint8)


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
    header = f"P6\n{sheet.shape[1]} {sheet.shape[0]}\n255\n".encode("ascii")
    path.write_bytes(header + sheet.tobytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a contact sheet for SpatialMemoryNet v0 BEV outputs.")
    parser.add_argument("--log", required=True, help="modeld/replay segment log containing BrainOutputEvents.")
    parser.add_argument("--out", required=True, help="Output visualization directory.")
    parser.add_argument("--max-frames", type=int, default=24)
    args = parser.parse_args(argv)
    path = visualize_spatial_outputs(log_dir=args.log, out_dir=args.out, max_frames=args.max_frames)
    print(f"wrote SpatialMemoryNet v0 visualization manifest to {path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
