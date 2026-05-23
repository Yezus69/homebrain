from __future__ import annotations

import json
from pathlib import Path

from homebrain.messages.schema import JsonDict, deterministic_json

ROUTE_SOURCE_METADATA_FILE = "route_metadata.json"
ROUTE_SOURCE_SCHEMA_VERSION = "homebrain.route_source.v0"


def write_route_metadata(route_dir: str | Path, metadata: JsonDict) -> Path:
    path = Path(route_dir) / ROUTE_SOURCE_METADATA_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(deterministic_json(metadata))
        handle.write("\n")
    return path


def load_route_metadata(route_dir: str | Path) -> JsonDict | None:
    path = Path(route_dir) / ROUTE_SOURCE_METADATA_FILE
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"route metadata must be a JSON object: {path}")
    if data.get("schema_version") != ROUTE_SOURCE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported route metadata schema "
            f"{data.get('schema_version')!r}; expected {ROUTE_SOURCE_SCHEMA_VERSION!r}"
        )
    return data

