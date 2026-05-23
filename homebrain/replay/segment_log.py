from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from homebrain.messages.schema import (
    SCHEMA_VERSION,
    Event,
    SegmentManifest,
    deterministic_json,
    event_from_json_line,
    event_to_json_line,
)

MANIFEST_FILE = "manifest.json"
DEFAULT_EVENT_FILE = "events.jsonl"
DETERMINISTIC_CREATED_AT_UTC = "1970-01-01T00:00:00Z"


class SegmentLogError(ValueError):
    pass


def write_segment(
    log_dir: str | Path,
    events: Iterable[Event],
    *,
    segment_id: str | None = None,
    artifact_files: list[str] | None = None,
    event_file: str = DEFAULT_EVENT_FILE,
    created_at_utc: str = DETERMINISTIC_CREATED_AT_UTC,
) -> SegmentManifest:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    materialized_events = list(events)
    timestamps = [event.timestamp_ns for event in materialized_events]
    manifest = SegmentManifest(
        segment_id=segment_id or path.name,
        created_at_utc=created_at_utc,
        schema_version=SCHEMA_VERSION,
        event_file=event_file,
        artifact_files=artifact_files or [],
        start_timestamp_ns=min(timestamps) if timestamps else 0,
        end_timestamp_ns=max(timestamps) if timestamps else 0,
        event_count=len(materialized_events),
    )

    event_path = path / event_file
    with event_path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in materialized_events:
            handle.write(event_to_json_line(event))
            handle.write("\n")

    manifest_path = path / MANIFEST_FILE
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(deterministic_json(manifest.to_dict()))
        handle.write("\n")
    return manifest


def load_manifest(log_dir: str | Path) -> SegmentManifest:
    manifest_path = Path(log_dir) / MANIFEST_FILE
    if not manifest_path.exists():
        raise SegmentLogError(f"missing segment manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SegmentLogError("segment manifest must be a JSON object")
    manifest = SegmentManifest.from_dict(data)
    if manifest.schema_version != SCHEMA_VERSION:
        raise SegmentLogError(
            f"unsupported schema version {manifest.schema_version!r}; expected {SCHEMA_VERSION!r}"
        )
    return manifest


def iter_events(log_dir: str | Path) -> Iterable[Event]:
    path = Path(log_dir)
    manifest = load_manifest(path)
    event_path = path / manifest.event_file
    if not event_path.exists():
        raise SegmentLogError(f"missing segment event file: {event_path}")
    with event_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield event_from_json_line(stripped)
            except Exception as exc:  # noqa: BLE001 - preserve file/line context.
                raise SegmentLogError(f"invalid event at line {line_number}: {exc}") from exc


def read_events(log_dir: str | Path) -> list[Event]:
    events = list(iter_events(log_dir))
    manifest = load_manifest(log_dir)
    if len(events) != manifest.event_count:
        raise SegmentLogError(
            f"manifest event_count {manifest.event_count} does not match {len(events)} records"
        )
    return events


def canonical_event_text(events: Iterable[Event]) -> str:
    return "".join(f"{event_to_json_line(event)}\n" for event in events)
