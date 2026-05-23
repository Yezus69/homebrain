from __future__ import annotations

import io
import json
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.messages.schema import JsonDict, deterministic_json

SPATIAL_DATASET_SCHEMA_VERSION = "homebrain.spatial_dataset.v0"
SPATIAL_EXAMPLE_SCHEMA_VERSION = "homebrain.spatial_example.v0"
SPATIAL_QA_SCHEMA_VERSION = "homebrain.spatial_dataset_qa.v0"
SPATIAL_VIZ_SCHEMA_VERSION = "homebrain.spatial_dataset_viz.v0"
SPATIAL_MANIFEST_FILE = "manifest.json"
DETERMINISTIC_CREATED_AT_UTC = "1970-01-01T00:00:00Z"

REQUIRED_EXAMPLE_FIELDS: tuple[str, ...] = (
    "frame_id",
    "timestamp_ns",
    "timestamp",
    "rgb_ref",
    "rgb_path",
    "bev_free",
    "bev_obstacle",
    "bev_unknown",
    "bev_confidence",
    "provenance",
    "weak_label",
    "control_safe",
    "camera_config_hash",
    "teacher_manifest_hash",
    "split",
)

OPTIONAL_EXAMPLE_FIELDS: tuple[str, ...] = (
    "bev_height",
    "bev_floor_candidate",
)

BEV_LABEL_FIELDS: tuple[str, ...] = (
    "bev_free",
    "bev_obstacle",
    "bev_unknown",
)

BEV_FLOAT_FIELDS: tuple[str, ...] = (
    "bev_confidence",
    "bev_height",
)


def json_sha256(data: JsonDict) -> str:
    return sha256(deterministic_json(data).encode("ascii")).hexdigest()


def text_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def write_json(path: str | Path, data: JsonDict, *, pretty: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        if pretty:
            json.dump(data, handle, sort_keys=True, indent=2)
            handle.write("\n")
        else:
            handle.write(deterministic_json(data))
            handle.write("\n")


def read_json(path: str | Path) -> JsonDict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def scalar_str(value: str) -> np.ndarray:
    return np.asarray(value)


def scalar_int(value: int) -> np.ndarray:
    return np.asarray(value, dtype=np.int64)


def scalar_bool(value: bool) -> np.ndarray:
    return np.asarray(value, dtype=np.bool_)


def write_deterministic_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as output:
        with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(filename=f"{name}.npy")
                info.date_time = (1980, 1, 1, 0, 0, 0)
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o644 << 16
                archive.writestr(info, buffer.getvalue())


def load_example_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def np_scalar_to_str(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def np_scalar_to_int(value: np.ndarray) -> int:
    return int(np.asarray(value).item())


def np_scalar_to_bool(value: np.ndarray) -> bool:
    return bool(np.asarray(value).item())


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if np.isfinite(number):
            return number
    return default

