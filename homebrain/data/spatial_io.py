from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from homebrain.messages.schema import JsonDict


def manifest_records_by_frame_id(manifest: JsonDict, record_key: str) -> dict[int, JsonDict]:
    records: dict[int, JsonDict] = {}
    for record in manifest.get(record_key, []):
        if not isinstance(record, dict):
            continue
        frame_id = int_or_none(record.get("frame_id"))
        if frame_id is not None:
            records[frame_id] = record
    return records


def scene_frames_by_id(manifest: JsonDict) -> dict[int, JsonDict]:
    return manifest_records_by_frame_id(manifest, "frames")


def spatial_examples_by_id(manifest: JsonDict) -> dict[int, JsonDict]:
    return manifest_records_by_frame_id(manifest, "examples")


def load_spatial_example_arrays(
    spatial_root: str | Path,
    example: JsonDict,
    *,
    missing_ok: bool,
    require_confidence: bool = False,
) -> dict[str, np.ndarray] | None:
    path_value = example.get("example_path")
    if not isinstance(path_value, str):
        if missing_ok:
            return None
        raise ValueError(f"spatial example {example.get('frame_id')} is missing example_path")
    path = Path(spatial_root) / path_value
    if missing_ok and not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    if require_confidence and "bev_confidence" not in arrays and "bev_free" in arrays:
        arrays["bev_confidence"] = np.ones_like(np.asarray(arrays["bev_free"], dtype=np.float32))
    return arrays


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None
