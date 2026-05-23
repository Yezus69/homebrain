from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.geometry.bev_projector import BEV_ARTIFACT_KINDS, BEV_MANIFEST_FILE, BEV_SCHEMA_VERSION
from homebrain.messages.schema import JsonDict
from homebrain.teachers.artifacts import load_array


@dataclass(frozen=True)
class BevValidation:
    bev_frame_count: int
    bev_missing_count: int
    bev_shape_error_count: int
    bev_nan_count: int
    free_ratio_mean: float
    obstacle_ratio_mean: float
    unknown_ratio_mean: float
    confidence_mean: float
    temporal_jitter_mean: float
    bev_manifest_frame_count: int
    weak_label: bool
    control_safe: bool
    errors: tuple[str, ...] = ()

    @property
    def load_success(self) -> bool:
        return self.bev_missing_count == 0 and self.bev_shape_error_count == 0 and self.bev_nan_count == 0

    def to_metrics(self) -> dict[str, Any]:
        return {
            "bev_frame_count": self.bev_frame_count,
            "bev_missing_count": self.bev_missing_count,
            "bev_shape_error_count": self.bev_shape_error_count,
            "bev_nan_count": self.bev_nan_count,
            "free_ratio_mean": self.free_ratio_mean,
            "obstacle_ratio_mean": self.obstacle_ratio_mean,
            "unknown_ratio_mean": self.unknown_ratio_mean,
            "confidence_mean": self.confidence_mean,
            "temporal_jitter_mean": self.temporal_jitter_mean,
            "bev_manifest_frame_count": self.bev_manifest_frame_count,
            "bev_load_success": self.load_success,
            "weak_label": self.weak_label,
            "control_safe": self.control_safe,
        }


def load_bev_manifest(bev_dir: str | Path) -> JsonDict:
    path = Path(bev_dir) / BEV_MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"missing BEV manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"BEV manifest must be a JSON object: {path}")
    if data.get("schema_version") != BEV_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported BEV schema {data.get('schema_version')!r}; expected {BEV_SCHEMA_VERSION!r}"
        )
    frames = data.get("frames")
    if not isinstance(frames, list):
        raise ValueError("BEV manifest frames must be a list")
    return data


def validate_bev_artifacts(bev_dir: str | Path) -> BevValidation:
    root = Path(bev_dir)
    errors: list[str] = []
    try:
        manifest = load_bev_manifest(root)
    except Exception as exc:  # noqa: BLE001 - validation reports malformed outputs.
        return BevValidation(
            bev_frame_count=0,
            bev_missing_count=1,
            bev_shape_error_count=0,
            bev_nan_count=0,
            free_ratio_mean=0.0,
            obstacle_ratio_mean=0.0,
            unknown_ratio_mean=0.0,
            confidence_mean=0.0,
            temporal_jitter_mean=0.0,
            bev_manifest_frame_count=0,
            weak_label=False,
            control_safe=True,
            errors=(str(exc),),
        )

    weak_label = manifest.get("weak_label") is True
    control_safe = manifest.get("control_safe") is True
    if not weak_label:
        errors.append("BEV manifest is not marked weak_label=true")
    if control_safe:
        errors.append("BEV manifest must not be marked control_safe=true")

    expected_shape = tuple(int(value) for value in manifest.get("grid_shape", []))
    if len(expected_shape) != 2 or expected_shape[0] <= 0 or expected_shape[1] <= 0:
        expected_shape = (0, 0)
        errors.append("BEV manifest missing valid grid_shape")

    valid_frame_count = 0
    missing_count = 0
    shape_error_count = 0
    nan_count = 0
    free_ratios: list[float] = []
    obstacle_ratios: list[float] = []
    unknown_ratios: list[float] = []
    confidence_means: list[float] = []
    occupancy_classes: list[np.ndarray] = []

    for frame_record in manifest["frames"]:
        if not isinstance(frame_record, dict):
            shape_error_count += 1
            errors.append("BEV frame record is not an object")
            continue
        if frame_record.get("weak_label") is not True:
            shape_error_count += 1
            errors.append(f"frame {frame_record.get('frame_id')} missing weak_label=true")
        if frame_record.get("control_safe") is not False:
            shape_error_count += 1
            errors.append(f"frame {frame_record.get('frame_id')} must have control_safe=false")

        metadata_path = frame_record.get("metadata_path")
        if not isinstance(metadata_path, str) or not (root / metadata_path).exists():
            missing_count += 1
            errors.append(f"missing frame metadata for frame {frame_record.get('frame_id')}")

        artifacts = frame_record.get("artifacts")
        if not isinstance(artifacts, dict):
            missing_count += len(BEV_ARTIFACT_KINDS)
            errors.append(f"missing artifacts object for frame {frame_record.get('frame_id')}")
            continue

        arrays: dict[str, np.ndarray] = {}
        frame_ok = True
        for kind in BEV_ARTIFACT_KINDS:
            artifact = artifacts.get(kind)
            if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                missing_count += 1
                frame_ok = False
                errors.append(f"missing {kind} artifact path for frame {frame_record.get('frame_id')}")
                continue
            artifact_path = root / str(artifact["path"])
            if not artifact_path.exists():
                missing_count += 1
                frame_ok = False
                errors.append(f"missing {kind} artifact file for frame {frame_record.get('frame_id')}")
                continue
            try:
                array = load_array(artifact_path)
            except Exception as exc:  # noqa: BLE001
                shape_error_count += 1
                frame_ok = False
                errors.append(f"failed to load {kind} for frame {frame_record.get('frame_id')}: {exc}")
                continue
            arrays[kind] = array
            if expected_shape != (0, 0) and tuple(array.shape) != expected_shape:
                shape_error_count += 1
                frame_ok = False
                errors.append(
                    f"shape error for {kind} frame {frame_record.get('frame_id')}: "
                    f"{array.shape} != {expected_shape}"
                )
            nan_count += int(np.count_nonzero(~np.isfinite(array)))

        if not frame_ok or any(kind not in arrays for kind in BEV_ARTIFACT_KINDS):
            continue

        valid_frame_count += 1
        total = max(arrays["bev_free"].size, 1)
        free = arrays["bev_free"] > 0
        obstacle = arrays["bev_obstacle"] > 0
        unknown = arrays["bev_unknown"] > 0
        free_ratios.append(float(np.count_nonzero(free) / total))
        obstacle_ratios.append(float(np.count_nonzero(obstacle) / total))
        unknown_ratios.append(float(np.count_nonzero(unknown) / total))
        confidence_means.append(float(np.mean(arrays["bev_confidence"].astype(np.float32))))
        occupancy_classes.append((free.astype(np.uint8) + obstacle.astype(np.uint8) * 2).astype(np.uint8))

    jitter_values: list[float] = []
    for previous, current in zip(occupancy_classes, occupancy_classes[1:]):
        if previous.shape == current.shape and previous.size:
            jitter_values.append(float(np.count_nonzero(previous != current) / previous.size))

    return BevValidation(
        bev_frame_count=valid_frame_count,
        bev_missing_count=missing_count,
        bev_shape_error_count=shape_error_count,
        bev_nan_count=nan_count,
        free_ratio_mean=_mean(free_ratios),
        obstacle_ratio_mean=_mean(obstacle_ratios),
        unknown_ratio_mean=_mean(unknown_ratios),
        confidence_mean=_mean(confidence_means),
        temporal_jitter_mean=_mean(jitter_values),
        bev_manifest_frame_count=len(manifest["frames"]),
        weak_label=weak_label,
        control_safe=control_safe,
        errors=tuple(errors),
    )


def write_bev_metrics(metrics: dict[str, Any], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metrics, handle, sort_keys=True, indent=2)
        handle.write("\n")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate HomeBrain depth-to-BEV weak geometry labels.")
    parser.add_argument("--bev", required=True, help="Input BEV artifact directory.")
    parser.add_argument("--out", required=True, help="Output metrics JSON path.")
    args = parser.parse_args(argv)
    validation = validate_bev_artifacts(args.bev)
    write_bev_metrics(validation.to_metrics(), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

