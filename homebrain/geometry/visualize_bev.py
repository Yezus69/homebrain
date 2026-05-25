from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from homebrain.artifacts.io import write_json_object
from homebrain.geometry.validate_bev import load_bev_manifest
from homebrain.messages.schema import JsonDict
from homebrain.teachers.artifacts import load_array
from homebrain.visualization.panels import write_pgm, write_ppm


def visualize_bev(bev_dir: str | Path, out_dir: str | Path) -> Path:
    root = Path(bev_dir)
    output = Path(out_dir)
    manifest = load_bev_manifest(root)
    frame_outputs: list[JsonDict] = []

    for frame_record in manifest["frames"]:
        if not isinstance(frame_record, dict):
            continue
        frame_name = f"{frame_record['camera_id']}_{int(frame_record['frame_id']):06d}"
        frame_dir = output / frame_name
        artifacts = frame_record.get("artifacts", {})
        if not isinstance(artifacts, dict):
            continue
        arrays = {
            kind: load_array(root / str(artifacts[kind]["path"]))
            for kind in (
                "bev_free",
                "bev_obstacle",
                "bev_unknown",
                "bev_floor_candidate",
                "bev_height",
                "bev_confidence",
            )
            if isinstance(artifacts.get(kind), dict)
        }
        if len(arrays) != 6:
            continue

        label_rgb = _label_rgb(arrays)
        label_path = frame_dir / "bev_labels.ppm"
        confidence_path = frame_dir / "bev_confidence.pgm"
        height_path = frame_dir / "bev_height.pgm"
        write_ppm(label_path, label_rgb)
        write_pgm(confidence_path, _normalize_to_uint8(arrays["bev_confidence"]))
        write_pgm(height_path, _normalize_to_uint8(arrays["bev_height"]))

        frame_outputs.append(
            {
                "sequence_id": frame_record["sequence_id"],
                "camera_id": frame_record["camera_id"],
                "frame_id": frame_record["frame_id"],
                "weak_label": True,
                "control_safe": False,
                "previews": {
                    "bev_labels": label_path.relative_to(output).as_posix(),
                    "bev_confidence": confidence_path.relative_to(output).as_posix(),
                    "bev_height": height_path.relative_to(output).as_posix(),
                },
            }
        )

    visualization_manifest: JsonDict = {
        "schema_version": "homebrain.geometry.bev_visualization.v0",
        "source_bev": root.as_posix(),
        "weak_label": True,
        "control_safe": False,
        "visualization_written": True,
        "frame_count": len(frame_outputs),
        "frames": frame_outputs,
    }
    manifest_path = output / "visualization_manifest.json"
    write_json_object(manifest_path, visualization_manifest)
    return manifest_path


def _label_rgb(arrays: dict[str, np.ndarray]) -> np.ndarray:
    unknown = arrays["bev_unknown"] > 0
    floor = arrays["bev_floor_candidate"] > 0
    free = arrays["bev_free"] > 0
    obstacle = arrays["bev_obstacle"] > 0

    rgb = np.zeros((*unknown.shape, 3), dtype=np.uint8)
    rgb[unknown] = np.array([32, 32, 32], dtype=np.uint8)
    rgb[floor] = np.array([40, 90, 140], dtype=np.uint8)
    rgb[free] = np.array([45, 180, 100], dtype=np.uint8)
    rgb[obstacle] = np.array([220, 70, 55], dtype=np.uint8)
    return rgb


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.uint8)
    finite_values = values[finite]
    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    if maximum <= minimum:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = (values - np.float32(minimum)) / np.float32(maximum - minimum)
    return np.where(finite, np.clip(normalized * np.float32(255.0), 0, 255), 0).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create simple previews for BEV weak geometry labels.")
    parser.add_argument("--bev", required=True, help="Input BEV artifact directory.")
    parser.add_argument("--out", required=True, help="Output visualization directory.")
    args = parser.parse_args(argv)
    manifest_path = visualize_bev(args.bev, args.out)
    print(f"wrote BEV visualization manifest to {manifest_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
