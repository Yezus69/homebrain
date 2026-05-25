from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from homebrain.messages.schema import JsonDict, deterministic_json


def read_json_object(path: str | Path) -> JsonDict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def write_json_object(path: str | Path, data: JsonDict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(deterministic_json(data) + "\n", encoding="utf-8", newline="\n")


def load_array(path: str | Path) -> np.ndarray:
    with Path(path).open("rb") as handle:
        return np.load(handle, allow_pickle=False)


def load_array_artifact_optional(
    root: str | Path,
    artifacts: JsonDict,
    kind: str,
    *,
    missing_ok: bool,
) -> np.ndarray | None:
    record = artifacts.get(kind)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return None
    path = Path(root) / str(record["path"])
    if missing_ok and not path.exists():
        return None
    return load_array(path)


def load_array_artifact_required(root: str | Path, artifacts: JsonDict, kind: str) -> np.ndarray:
    array = load_array_artifact_optional(root, artifacts, kind, missing_ok=False)
    if array is None:
        raise ValueError(f"missing artifact {kind}")
    return array
