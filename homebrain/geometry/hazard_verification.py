from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from homebrain.messages.schema import JsonDict
from homebrain.teachers.artifacts import file_sha256

HAZARD_HAND_VERIFICATION_SCHEMA_VERSION = "homebrain.hazard_hand_verification.v0"


@dataclass(frozen=True)
class HazardHandVerification:
    path: Path
    sha256: str
    positive_frame_ids: frozenset[int]
    reviewer: str | None
    review_notes: str | None

    def manifest_fields(self, *, available_frame_ids: set[int]) -> JsonDict:
        matched = sorted(frame_id for frame_id in self.positive_frame_ids if frame_id in available_frame_ids)
        missing = sorted(frame_id for frame_id in self.positive_frame_ids if frame_id not in available_frame_ids)
        return {
            "hand_verification_schema_version": HAZARD_HAND_VERIFICATION_SCHEMA_VERSION,
            "hand_verification_path": self.path.as_posix(),
            "hand_verification_sha256": self.sha256,
            "hand_verification_reviewer": self.reviewer,
            "hand_verification_notes": self.review_notes,
            "hand_verified_hazard_positive_frame_count": int(len(matched)),
            "hand_verified_hazard_positive_frame_ids": matched,
            "hand_verified_hazard_missing_frame_ids": missing,
        }


def load_hazard_hand_verification(path: str | Path) -> HazardHandVerification:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("hazard hand-verification file must contain a JSON object")
    if data.get("hand_verified") is not True:
        raise ValueError("hazard hand-verification file must set hand_verified=true")
    positive_ids = _positive_frame_ids(data)
    return HazardHandVerification(
        path=source,
        sha256=file_sha256(source),
        positive_frame_ids=frozenset(positive_ids),
        reviewer=str(data["reviewer"]) if isinstance(data.get("reviewer"), str) else None,
        review_notes=str(data["notes"]) if isinstance(data.get("notes"), str) else None,
    )


def _positive_frame_ids(data: JsonDict) -> set[int]:
    ids: set[int] = set()
    for value in data.get("positive_frame_ids", []):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            ids.add(int(value))
    frames = data.get("frames", [])
    if isinstance(frames, list):
        for item in frames:
            frame_id = _positive_frame_record_id(item)
            if frame_id is not None:
                ids.add(frame_id)
    if not ids:
        raise ValueError("hazard hand-verification file must include at least one positive frame id")
    return ids


def _positive_frame_record_id(item: Any) -> int | None:
    if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
        return int(item)
    if not isinstance(item, dict):
        return None
    positive = (
        item.get("hazard_present") is True
        or item.get("positive") is True
        or item.get("hand_verified_hazard_positive") is True
    )
    frame_id = item.get("frame_id")
    if positive and isinstance(frame_id, int) and not isinstance(frame_id, bool) and frame_id >= 0:
        return int(frame_id)
    return None
