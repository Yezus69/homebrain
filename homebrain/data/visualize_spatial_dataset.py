from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.data.spatial_dataset import (
    SPATIAL_MANIFEST_FILE,
    SPATIAL_VIZ_SCHEMA_VERSION,
    load_example_npz,
    np_scalar_to_int,
    read_json,
    write_json,
)
from homebrain.visualization.panels import write_ppm


def visualize_spatial_dataset(
    *,
    dataset_dir: str | Path,
    out_dir: str | Path,
    sample_count: int = 5,
) -> Path:
    dataset_root = Path(dataset_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(dataset_root / SPATIAL_MANIFEST_FILE)
    examples = _load_visual_examples(dataset_root, manifest)
    first = examples[:sample_count]
    worst = sorted(examples, key=lambda item: (item["confidence_mean"], item["frame_id"]))[:sample_count]
    median = _median_confidence_examples(examples, sample_count)

    groups = {
        "first_frames": first,
        "median_confidence_frames": median,
        "worst_confidence_frames": worst,
    }
    outputs: list[dict[str, Any]] = []
    for name, group in groups.items():
        target = output / f"{name}.ppm"
        _write_contact_sheet(target, [item["rgb"] for item in group])
        outputs.append(
            {
                "name": name,
                "path": target.relative_to(output).as_posix(),
                "frame_ids": [int(item["frame_id"]) for item in group],
                "confidence_means": [float(item["confidence_mean"]) for item in group],
            }
        )

    viz_manifest = {
        "schema_version": SPATIAL_VIZ_SCHEMA_VERSION,
        "dataset": dataset_root.as_posix(),
        "sample_count": sample_count,
        "visualization_written": True,
        "contact_sheets": outputs,
        "note": "PPM contact sheets render BEV weak labels dimmed by BEV confidence; they are review aids, not ground truth.",
    }
    manifest_path = output / "visualization_manifest.json"
    write_json(manifest_path, viz_manifest, pretty=True)
    return manifest_path


def _load_visual_examples(dataset_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = manifest.get("examples") or manifest.get("frames") or []
    if not isinstance(records, list):
        return []
    examples: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("example_path"), str):
            continue
        path = dataset_root / str(record["example_path"])
        if not path.exists():
            continue
        try:
            data = load_example_npz(path)
            rgb = _label_overlay_rgb(data)
            confidence_mean = float(np.mean(np.asarray(data["bev_confidence"], dtype=np.float32)))
            frame_id = np_scalar_to_int(data["frame_id"])
        except Exception:  # noqa: BLE001 - visualization skips malformed examples; QA reports them.
            continue
        examples.append(
            {
                "frame_id": frame_id,
                "confidence_mean": confidence_mean,
                "rgb": rgb,
            }
        )
    return sorted(examples, key=lambda item: item["frame_id"])


def _median_confidence_examples(examples: list[dict[str, Any]], sample_count: int) -> list[dict[str, Any]]:
    if not examples:
        return []
    confidences = np.asarray([float(item["confidence_mean"]) for item in examples], dtype=np.float32)
    median = float(np.median(confidences))
    ranked = sorted(examples, key=lambda item: (abs(float(item["confidence_mean"]) - median), item["frame_id"]))
    return ranked[:sample_count]


def _label_overlay_rgb(example: dict[str, np.ndarray]) -> np.ndarray:
    free = np.asarray(example["bev_free"]) > 0
    obstacle = np.asarray(example["bev_obstacle"]) > 0
    unknown = np.asarray(example["bev_unknown"]) > 0
    confidence = np.clip(np.asarray(example["bev_confidence"], dtype=np.float32), 0.0, 1.0)
    floor = np.asarray(example.get("bev_floor_candidate", np.zeros_like(example["bev_free"]))) > 0

    rgb = np.zeros((*free.shape, 3), dtype=np.float32)
    rgb[unknown] = np.array([34, 34, 34], dtype=np.float32)
    rgb[floor] = np.array([42, 92, 142], dtype=np.float32)
    rgb[free] = np.array([45, 180, 96], dtype=np.float32)
    rgb[obstacle] = np.array([220, 70, 55], dtype=np.float32)
    brightness = np.float32(0.35) + confidence[..., None] * np.float32(0.65)
    return np.clip(rgb * brightness, 0, 255).astype(np.uint8)


def _write_contact_sheet(path: Path, images: list[np.ndarray]) -> None:
    if not images:
        write_ppm(path, np.zeros((1, 1, 3), dtype=np.uint8))
        return
    thumbs = [_resize_nearest(image, _scale_for_image(image)) for image in images]
    padding = 4
    height = max(int(image.shape[0]) for image in thumbs)
    width = sum(int(image.shape[1]) for image in thumbs) + padding * (len(thumbs) - 1)
    sheet = np.full((height, width, 3), 18, dtype=np.uint8)
    x = 0
    for image in thumbs:
        h, w, _ = image.shape
        sheet[:h, x : x + w] = image
        x += w + padding
    write_ppm(path, sheet)


def _scale_for_image(image: np.ndarray) -> int:
    longest = max(int(image.shape[0]), int(image.shape[1]), 1)
    return max(1, min(16, 160 // longest))


def _resize_nearest(image: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return image.astype(np.uint8)
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write deterministic BEV contact sheets for a SpatialTrainPack.")
    parser.add_argument("--dataset", required=True, help="Input SpatialTrainPack directory.")
    parser.add_argument("--out", required=True, help="Output visualization directory.")
    parser.add_argument("--sample-count", type=int, default=5, help="Frames per contact sheet.")
    args = parser.parse_args(argv)
    manifest_path = visualize_spatial_dataset(
        dataset_dir=args.dataset,
        out_dir=args.out,
        sample_count=args.sample_count,
    )
    print(f"wrote SpatialTrainPack visualization manifest to {manifest_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
