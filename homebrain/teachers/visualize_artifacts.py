from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from homebrain.messages.schema import JsonDict, deterministic_json
from homebrain.teachers.artifacts import EXPECTED_ARTIFACT_KINDS, load_array, load_teacher_manifest


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float32)
    minimum = float(values.min()) if values.size else 0.0
    maximum = float(values.max()) if values.size else 0.0
    if maximum <= minimum:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = (values - minimum) / np.float32(maximum - minimum)
    return np.clip(normalized * np.float32(255.0), 0, 255).astype(np.uint8)


def _write_pgm(path: Path, image: np.ndarray) -> None:
    if image.ndim != 2:
        raise ValueError(f"PGM image must be 2D, got shape {image.shape}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    h, w = image.shape
    with target.open("wb") as handle:
        handle.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        handle.write(image.astype(np.uint8).tobytes(order="C"))


def _write_ppm(path: Path, image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"PPM image must be HxWx3, got shape {image.shape}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    h, w, _channels = image.shape
    with target.open("wb") as handle:
        handle.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        handle.write(image.astype(np.uint8).tobytes(order="C"))


def _write_manifest(path: Path, manifest: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(deterministic_json(manifest))
        handle.write("\n")


def _artifact_kinds(manifest: JsonDict) -> tuple[str, ...]:
    kinds = manifest.get("artifact_kinds")
    if not isinstance(kinds, list) or not all(isinstance(kind, str) for kind in kinds):
        return EXPECTED_ARTIFACT_KINDS
    return tuple(kinds)


def visualize_artifacts(artifacts_dir: str | Path, out_dir: str | Path) -> Path:
    root = Path(artifacts_dir)
    output = Path(out_dir)
    manifest = load_teacher_manifest(root)
    frame_outputs: list[JsonDict] = []
    artifact_kinds = _artifact_kinds(manifest)

    for frame_record in manifest["frames"]:
        if not isinstance(frame_record, dict):
            continue
        frame_name = f"{frame_record['camera_id']}_{int(frame_record['frame_id']):06d}"
        frame_dir = output / frame_name
        artifacts = frame_record.get("artifacts", {})
        previews: dict[str, str] = {}
        scalar_values: dict[str, float] = {}

        for kind in artifact_kinds:
            artifact_record = artifacts.get(kind) if isinstance(artifacts, dict) else None
            if not isinstance(artifact_record, dict):
                continue
            array = load_array(root / str(artifact_record["path"]))
            if array.ndim == 0 or (array.ndim == 1 and array.size == 1):
                scalar_values[kind] = float(array.reshape(-1)[0])
                continue
            if kind == "dense_features":
                rgb = _normalize_to_uint8(array[:, :, :3])
                target = frame_dir / f"{kind}.ppm"
                _write_ppm(target, rgb)
            elif array.ndim == 2:
                image = _normalize_to_uint8(array)
                target = frame_dir / f"{kind}.pgm"
                _write_pgm(target, image)
            else:
                continue
            previews[kind] = target.relative_to(output).as_posix()

        frame_outputs.append(
            {
                "sequence_id": frame_record["sequence_id"],
                "camera_id": frame_record["camera_id"],
                "frame_id": frame_record["frame_id"],
                "previews": previews,
                "scalar_values": scalar_values,
            }
        )

    visualization_manifest: JsonDict = {
        "schema_version": "homebrain.teacher_visualization.v0",
        "source_artifacts": str(root.as_posix()),
        "teacher_name": manifest["teacher_name"],
        "mock": bool(manifest["mock"]),
        "artifact_kinds": list(artifact_kinds),
        "visualization_written": True,
        "frame_count": len(frame_outputs),
        "frames": frame_outputs,
    }
    manifest_path = output / "visualization_manifest.json"
    _write_manifest(manifest_path, visualization_manifest)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create simple image previews for teacher artifacts.")
    parser.add_argument("--artifacts", required=True, help="Input teacher artifact directory.")
    parser.add_argument("--out", required=True, help="Output visualization directory.")
    args = parser.parse_args(argv)
    manifest_path = visualize_artifacts(args.artifacts, args.out)
    print(f"wrote teacher artifact visualization manifest to {manifest_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
