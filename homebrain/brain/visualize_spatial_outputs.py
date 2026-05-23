from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from homebrain.data.spatial_dataset import load_example_npz, read_json
from homebrain.messages.schema import BrainOutputEvent
from homebrain.replay.segment_log import read_events


def visualize_spatial_outputs(
    *,
    log_dir: str | Path,
    out_dir: str | Path,
    max_frames: int = 24,
    dataset_dir: str | Path | None = None,
) -> Path:
    root = Path(log_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    events = [event for event in read_events(root) if isinstance(event, BrainOutputEvent) and event.local_bev_ref]
    labels_by_frame = _labels_by_frame(dataset_dir) if dataset_dir is not None else {}
    tiles = []
    prediction_label_tiles = []
    frame_records = []
    for event in events[:max_frames]:
        artifact_path = root / str(event.local_bev_ref)
        with np.load(artifact_path, allow_pickle=False) as data:
            tile = _bev_tile(data)
        tiles.append(tile)
        frame_id = int(event.debug.get("input_frame_id", -1)) if isinstance(event.debug, dict) else -1
        label_tile = labels_by_frame.get(frame_id)
        if label_tile is not None:
            prediction_label_tiles.append(np.concatenate([tile, label_tile], axis=1))
        frame_records.append(
            {
                "timestamp_ns": event.timestamp_ns,
                "local_bev_ref": event.local_bev_ref,
                "uncertainty": event.uncertainty,
                "input_frame_id": frame_id,
                "label_available": label_tile is not None,
            }
        )
    sheet_path = output / "spatial_v0_contact_sheet.ppm"
    _write_contact_sheet(sheet_path, tiles)
    prediction_label_path = output / "spatial_v0_prediction_vs_label_contact_sheet.ppm"
    _write_contact_sheet(prediction_label_path, prediction_label_tiles)
    manifest = {
        "schema_version": "homebrain.spatial_v0_viz.v0",
        "source_log": root.as_posix(),
        "dataset": Path(dataset_dir).as_posix() if dataset_dir is not None else None,
        "visualization_written": bool(tiles),
        "prediction_vs_label_written": bool(prediction_label_tiles),
        "frame_count": len(tiles),
        "contact_sheet": sheet_path.name,
        "prediction_vs_label_contact_sheet": prediction_label_path.name,
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


def _labels_by_frame(dataset_dir: str | Path | None) -> dict[int, np.ndarray]:
    if dataset_dir is None:
        return {}
    root = Path(dataset_dir)
    manifest = read_json(root / "manifest.json")
    records = manifest.get("examples") or manifest.get("frames")
    if not isinstance(records, list):
        return {}
    labels: dict[int, np.ndarray] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("example_path"), str):
            continue
        example = load_example_npz(root / str(record["example_path"]))
        labels[int(record["frame_id"])] = _label_tile(example)
    return labels


def _label_tile(example: dict[str, np.ndarray]) -> np.ndarray:
    free = np.asarray(example["bev_free"], dtype=np.float32)
    occupied = np.asarray(example["bev_obstacle"], dtype=np.float32)
    unknown = np.asarray(example["bev_unknown"], dtype=np.float32)
    rgb = np.stack(
        [
            np.clip(occupied, 0.0, 1.0),
            np.clip(free, 0.0, 1.0),
            np.clip(unknown, 0.0, 1.0),
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
    parser.add_argument("--dataset", default=None, help="Optional SpatialTrainPack for prediction-vs-label contact sheet.")
    parser.add_argument("--max-frames", type=int, default=24)
    args = parser.parse_args(argv)
    path = visualize_spatial_outputs(
        log_dir=args.log,
        out_dir=args.out,
        max_frames=args.max_frames,
        dataset_dir=args.dataset,
    )
    print(f"wrote SpatialMemoryNet v0 visualization manifest to {path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
