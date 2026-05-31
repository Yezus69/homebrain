from __future__ import annotations

from pathlib import Path

import numpy as np

from homebrain.data.spatial_dataset import read_json
from homebrain.data.tum_rgbd_route import RouteSequence
from homebrain.messages.schema import JsonDict


def discover_hazard_bev_sources(
    routes: list[RouteSequence],
    *,
    hazard_bev_root: str | Path | None,
) -> dict[str, Path]:
    explicit_root = Path(hazard_bev_root) if hazard_bev_root is not None else None
    sources: dict[str, Path] = {}
    for route in routes:
        candidates: list[Path] = []
        if explicit_root is not None:
            candidates.extend((explicit_root / route.route_id, explicit_root / route.root.name, explicit_root))
        candidates.extend(
            (
                route.root / "teacher_artifacts" / "hazard_bev",
                route.root / "teacher_artifacts" / "hazard" / "bev",
            )
        )
        for candidate in candidates:
            if (candidate / "bev_manifest.json").exists():
                sources[route.route_id] = candidate
                break
    return sources


def load_hazard_bev_lookup(
    sources: dict[str, Path],
    *,
    grid_shape: tuple[int, int],
) -> tuple[dict[tuple[str, int], tuple[np.ndarray, np.ndarray]], JsonDict]:
    if not sources:
        return {}, {
            "has_hazard_channel": False,
            "hazard_source_model": None,
            "hazard_license_review_status": None,
            "hazard_positive_frame_count": 0,
            "hazard_positive_cell_fraction": 0.0,
            "hazard_weak_label": True,
            "hazard_bev_source_count": 0,
            "hazard_source_mock_used": False,
            "hazard_source_synthetic_used": False,
            "hand_verified_hazard_positive_frame_count": 0,
        }

    lookup: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    source_models: set[str] = set()
    license_statuses: set[str] = set()
    positive_frames = 0
    positive_cells = 0
    total_cells = 0
    hand_verified = 0
    mock_used = False
    synthetic_used = False
    for route_id, source in sorted(sources.items()):
        manifest = read_json(source / "bev_manifest.json")
        if manifest.get("hazard_weak_label") is not True and manifest.get("weak_label") is not True:
            raise ValueError(f"hazard BEV manifest must mark weak labels: {source}")
        if manifest.get("control_safe") is not False:
            raise ValueError(f"hazard BEV manifest must remain control_safe=false: {source}")
        if manifest.get("hazard_source_model") is not None:
            source_models.add(str(manifest.get("hazard_source_model")))
        if manifest.get("hazard_license_review_status") is not None:
            license_statuses.add(str(manifest.get("hazard_license_review_status")))
        mock_used = mock_used or bool(manifest.get("source_hazard_mock", False))
        synthetic_used = synthetic_used or bool(manifest.get("source_hazard_synthetic", False))
        hand_verified += int(manifest.get("hand_verified_hazard_positive_frame_count", 0) or 0)
        for frame_record in manifest.get("frames", []):
            _load_frame_hazard(
                lookup=lookup,
                source=source,
                route_id=route_id,
                frame_record=frame_record,
                grid_shape=grid_shape,
            )

    for hazard, _valid in lookup.values():
        if int(np.count_nonzero(hazard > 0.5)) > 0:
            positive_frames += 1
        positive_cells += int(np.count_nonzero(hazard > 0.5))
        total_cells += int(hazard.size)

    return lookup, {
        "has_hazard_channel": True,
        "hazard_source_model": ",".join(sorted(source_models)) if source_models else None,
        "hazard_license_review_status": ",".join(sorted(license_statuses)) if license_statuses else None,
        "hazard_positive_frame_count": int(positive_frames),
        "hazard_positive_cell_fraction": float(positive_cells / max(total_cells, 1)),
        "hazard_weak_label": True,
        "hazard_bev_source_count": int(len(sources)),
        "hazard_bev_sources": {route_id: source.as_posix() for route_id, source in sorted(sources.items())},
        "hazard_source_mock_used": bool(mock_used),
        "hazard_source_synthetic_used": bool(synthetic_used),
        "hand_verified_hazard_positive_frame_count": int(hand_verified),
    }


def _load_frame_hazard(
    *,
    lookup: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    source: Path,
    route_id: str,
    frame_record: object,
    grid_shape: tuple[int, int],
) -> None:
    if not isinstance(frame_record, dict):
        return
    artifacts = frame_record.get("artifacts")
    if not isinstance(artifacts, dict):
        return
    hazard_path = _artifact_relative_path(artifacts, "bev_hazard")
    valid_path = _artifact_relative_path(artifacts, "hazard_valid_mask")
    if hazard_path is None or valid_path is None:
        return
    hazard = np.asarray(np.load(source / hazard_path, allow_pickle=False), dtype=np.float32)
    valid = np.asarray(np.load(source / valid_path, allow_pickle=False), dtype=np.float32)
    if hazard.shape != grid_shape:
        hazard = _resize_nearest(hazard, grid_shape).astype(np.float32)
    if valid.shape != grid_shape:
        valid = _resize_nearest(valid, grid_shape).astype(np.float32)
    lookup[(route_id, int(frame_record["frame_id"]))] = (
        np.clip(hazard, 0.0, 1.0).astype(np.float32),
        (valid > 0.0).astype(np.float32),
    )


def _artifact_relative_path(artifacts: JsonDict, kind: str) -> str | None:
    record = artifacts.get(kind)
    if isinstance(record, dict) and isinstance(record.get("path"), str):
        return str(record["path"])
    return None


def _resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    out_h, out_w = shape
    if values.ndim != 2 or out_h <= 0 or out_w <= 0:
        raise ValueError(f"cannot resize hazard grid {values.shape} to {shape}")
    row_index = np.linspace(0, values.shape[0] - 1, out_h).round().astype(np.int64)
    col_index = np.linspace(0, values.shape[1] - 1, out_w).round().astype(np.int64)
    return values[row_index[:, None], col_index[None, :]]
