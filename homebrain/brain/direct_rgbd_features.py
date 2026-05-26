from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.datasets.openloris_scene import OPENLORIS_DEPTH_SCALE, OPENLORIS_ROUTE_ASSOCIATIONS_FILE
from homebrain.datasets.tum_rgbd import TUM_RGBD_ROUTE_ASSOCIATIONS_FILE, read_depth_png_m
from homebrain.messages.schema import Event, FrameEvent, JsonDict
from homebrain.replay.segment_log import load_manifest, read_events
from homebrain.teachers.artifacts import (
    TEACHER_ARTIFACT_SCHEMA_VERSION,
    file_sha256,
    relative_to_root,
    save_array,
    write_teacher_manifest,
)

DIRECT_RGBD_FEATURE_TEACHER_NAME = "direct_rgbd_student_features"
DIRECT_RGBD_FEATURE_VERSION = "homebrain.direct_rgbd_student_features.v0"
DIRECT_RGBD_FEATURE_DIM = 10
DIRECT_RGBD_PATCH_SHAPE = (16, 16)


@dataclass(frozen=True)
class DepthRef:
    path: str
    scale: float


def write_direct_rgbd_feature_artifacts(
    *,
    log_dir: str | Path,
    out_dir: str | Path,
    feature_dim: int = DIRECT_RGBD_FEATURE_DIM,
    patch_shape: tuple[int, int] = DIRECT_RGBD_PATCH_SHAPE,
) -> Path:
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    log_root = Path(log_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    route_manifest = load_manifest(log_root)
    frames = [event for event in _order_events_for_replay(read_events(log_root)) if isinstance(event, FrameEvent)]
    depth_refs = load_depth_refs(log_root)
    frame_records: list[JsonDict] = []
    missing_depth_count = 0
    depth_valid_ratios: list[float] = []
    for frame in frames:
        rgb = load_rgb_image(log_root / frame.data_ref)
        depth = load_depth_for_frame(log_root, frame, depth_refs)
        if depth is None:
            missing_depth_count += 1
        else:
            depth_valid_ratios.append(float(np.count_nonzero(np.isfinite(depth) & (depth > 0.0)) / max(depth.size, 1)))
        patch_features = rgbd_patch_features(
            rgb=rgb,
            depth_m=depth,
            feature_dim=feature_dim,
            patch_shape=patch_shape,
        )
        cls_feature = patch_features.mean(axis=(0, 1)).astype(np.float32)
        frame_dir = output / "frames" / f"{frame.camera_id}_{frame.frame_id:06d}"
        patch_path = frame_dir / "patch_features.npy"
        cls_path = frame_dir / "cls_feature.npy"
        patch_record = save_array(patch_path, patch_features)
        cls_record = save_array(cls_path, cls_feature)
        frame_records.append(
            {
                "sequence_id": frame.sequence_id,
                "camera_id": frame.camera_id,
                "frame_id": int(frame.frame_id),
                "timestamp_ns": int(frame.timestamp_ns),
                "width": int(frame.width),
                "height": int(frame.height),
                "artifacts": {
                    "patch_features": {
                        "kind": "patch_features",
                        "path": relative_to_root(patch_path, output),
                        **patch_record,
                    },
                    "cls_feature": {
                        "kind": "cls_feature",
                        "path": relative_to_root(cls_path, output),
                        **cls_record,
                    },
                },
                "feature_source": "current_rgbd_sensor_patch_statistics",
                "depth_available": depth is not None,
                "control_safe": False,
            }
        )
    manifest: JsonDict = {
        "schema_version": TEACHER_ARTIFACT_SCHEMA_VERSION,
        "teacher_name": DIRECT_RGBD_FEATURE_TEACHER_NAME,
        "teacher_name_canonical": DIRECT_RGBD_FEATURE_TEACHER_NAME,
        "teacher_version": DIRECT_RGBD_FEATURE_VERSION,
        "backend": "current_rgbd_patch_statistics",
        "mock": False,
        "synthetic": False,
        "real_perception": True,
        "deterministic": True,
        "created_at_utc": "deterministic",
        "source_log": log_root.as_posix(),
        "source_segment_id": route_manifest.segment_id,
        "frame_count": len(frame_records),
        "feature_shape": [int(patch_shape[0]), int(patch_shape[1]), int(feature_dim)],
        "feature_dim": int(feature_dim),
        "patch_shape": [int(patch_shape[0]), int(patch_shape[1])],
        "feature_source": "direct current RGB-D route sensors",
        "runtime_dependency": False,
        "teacher_runtime_dependency": False,
        "student_training_feature_source": True,
        "depth_missing_count": int(missing_depth_count),
        "depth_valid_ratio_mean": float(sum(depth_valid_ratios) / max(len(depth_valid_ratios), 1)),
        "frames": frame_records,
        "control_safe": False,
        "replay_only": True,
        "not_executed": True,
        "product_training_approved": False,
        "raw_pwm_emitted": False,
    }
    return write_teacher_manifest(output, manifest)


def has_direct_rgbd_feature_artifacts(
    feature_dir: str | Path,
    *,
    expected_frame_count: int,
    feature_dim: int = DIRECT_RGBD_FEATURE_DIM,
) -> bool:
    manifest_path = Path(feature_dir) / "teacher_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        import json

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception:  # noqa: BLE001
        return False
    if manifest.get("teacher_name") != DIRECT_RGBD_FEATURE_TEACHER_NAME:
        return False
    if manifest.get("mock") is not False or manifest.get("synthetic") is not False:
        return False
    if manifest.get("real_perception") is not True:
        return False
    if int(manifest.get("frame_count", 0)) != int(expected_frame_count):
        return False
    if int(manifest.get("feature_dim", 0)) != int(feature_dim):
        return False
    for item in manifest.get("frames", []):
        artifacts = item.get("artifacts") if isinstance(item, dict) else None
        if not isinstance(artifacts, dict):
            return False
        for key in ("patch_features", "cls_feature"):
            record = artifacts.get(key)
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                return False
            if not (Path(feature_dir) / str(record["path"])).exists():
                return False
    return True


def load_depth_refs(log_dir: str | Path) -> dict[tuple[str, str, int], DepthRef]:
    root = Path(log_dir)
    associations_path = _association_path(root)
    if associations_path is None:
        return {}
    data = _read_json(associations_path)
    frames = data.get("frames")
    if not isinstance(frames, list):
        return {}
    default_intrinsics = data.get("intrinsics") if isinstance(data.get("intrinsics"), dict) else {}
    refs: dict[tuple[str, str, int], DepthRef] = {}
    for record in frames:
        if not isinstance(record, dict):
            continue
        depth_ref = record.get("depth_ref")
        if not isinstance(depth_ref, str):
            continue
        intrinsics = record.get("intrinsics") if isinstance(record.get("intrinsics"), dict) else default_intrinsics
        scale = float(intrinsics.get("depth_scale", OPENLORIS_DEPTH_SCALE)) if isinstance(intrinsics, dict) else OPENLORIS_DEPTH_SCALE
        refs[
            (
                str(record.get("sequence_id", data.get("source_sequence", ""))),
                str(record.get("camera_id", "front_rgb")),
                int(record.get("frame_id", -1)),
            )
        ] = DepthRef(path=depth_ref, scale=scale)
    return refs


def load_depth_for_frame(
    log_dir: str | Path,
    frame: FrameEvent,
    depth_refs: dict[tuple[str, str, int], DepthRef],
) -> np.ndarray | None:
    ref = depth_refs.get((frame.sequence_id, frame.camera_id, int(frame.frame_id)))
    if ref is None:
        # OpenLORIS associations generated by older importers may not record sequence/camera.
        matches = [value for (sequence_id, camera_id, frame_id), value in depth_refs.items() if frame_id == int(frame.frame_id)]
        ref = matches[0] if matches else None
    if ref is None:
        return None
    path = Path(log_dir) / ref.path
    if not path.exists():
        return None
    try:
        return _read_depth_png_m_fast(path, scale=float(ref.scale))
    except Exception:  # noqa: BLE001
        return None


def load_rgb_image(path: str | Path) -> np.ndarray:
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Pillow is required for direct RGB-D feature extraction") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_depth_png_m_fast(path: Path, *, scale: float) -> np.ndarray:
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return read_depth_png_m(path, scale=scale)
    try:
        with Image.open(path) as image:
            values = np.asarray(image)
    except Exception:  # noqa: BLE001
        return read_depth_png_m(path, scale=scale)
    if values.ndim == 3:
        values = values[:, :, 0]
    depth_raw = np.asarray(values, dtype=np.float32)
    return np.where(depth_raw > np.float32(0.0), depth_raw / np.float32(scale), np.float32(0.0)).astype(np.float32)


def rgbd_patch_features(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray | None,
    feature_dim: int,
    patch_shape: tuple[int, int],
) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("runtime RGB must be HxWx3")
    patch_h, patch_w = patch_shape
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("patch_shape must be positive")
    if depth_m is not None and depth_m.shape[:2] != rgb.shape[:2]:
        depth = resize_nearest_float(depth_m, rgb.shape[:2])
    else:
        depth = depth_m
    regular = _rgbd_patch_features_regular_grid(
        rgb=rgb,
        depth_m=depth,
        feature_dim=feature_dim,
        patch_shape=patch_shape,
    )
    if regular is not None:
        return regular
    rows = _bin_edges(rgb.shape[0], patch_h)
    cols = _bin_edges(rgb.shape[1], patch_w)
    base = np.zeros((patch_h, patch_w, DIRECT_RGBD_FEATURE_DIM), dtype=np.float32)
    rgb_float = rgb.astype(np.float32) / np.float32(255.0)
    for row_index in range(patch_h):
        row_slice = slice(rows[row_index], rows[row_index + 1])
        for col_index in range(patch_w):
            col_slice = slice(cols[col_index], cols[col_index + 1])
            rgb_patch = rgb_float[row_slice, col_slice]
            rgb_mean = rgb_patch.reshape(-1, 3).mean(axis=0) if rgb_patch.size else np.zeros((3,), dtype=np.float32)
            if depth is None:
                depth_valid_ratio = np.float32(0.0)
                depth_mean = np.float32(0.0)
                depth_std = np.float32(0.0)
                depth_near = np.float32(0.0)
                depth_far = np.float32(0.0)
            else:
                depth_patch = np.asarray(depth[row_slice, col_slice], dtype=np.float32)
                valid = np.isfinite(depth_patch) & (depth_patch > np.float32(0.0))
                depth_valid_ratio = np.float32(np.count_nonzero(valid) / max(depth_patch.size, 1))
                if np.any(valid):
                    values = np.clip(depth_patch[valid], 0.0, 6.0).astype(np.float32)
                    depth_mean = np.float32(np.mean(values) / 6.0)
                    depth_std = np.float32(np.std(values) / 3.0)
                    depth_near = np.float32(np.min(values) / 6.0)
                    depth_far = np.float32(np.max(values) / 6.0)
                else:
                    depth_mean = depth_std = depth_near = depth_far = np.float32(0.0)
            base[row_index, col_index] = np.asarray(
                [
                    float(rgb_mean[0]),
                    float(rgb_mean[1]),
                    float(rgb_mean[2]),
                    float(depth_mean),
                    float(depth_std),
                    float(depth_near),
                    float(depth_far),
                    float(depth_valid_ratio),
                    row_index / max(patch_h - 1, 1),
                    col_index / max(patch_w - 1, 1),
                ],
                dtype=np.float32,
            )
    repeats = int(np.ceil(feature_dim / base.shape[-1]))
    tiled = np.tile(base, (1, 1, repeats))[..., :feature_dim]
    return tiled.astype(np.float32)


def _rgbd_patch_features_regular_grid(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray | None,
    feature_dim: int,
    patch_shape: tuple[int, int],
) -> np.ndarray | None:
    patch_h, patch_w = patch_shape
    height, width = rgb.shape[:2]
    if height % patch_h != 0 or width % patch_w != 0:
        return None
    row_bin = height // patch_h
    col_bin = width // patch_w
    if row_bin <= 0 or col_bin <= 0:
        return None
    rgb_float = rgb.astype(np.float32) / np.float32(255.0)
    rgb_mean = rgb_float.reshape(patch_h, row_bin, patch_w, col_bin, 3).mean(axis=(1, 3))
    row_grid = (
        np.arange(patch_h, dtype=np.float32)[:, None] / np.float32(max(patch_h - 1, 1))
    )
    col_grid = (
        np.arange(patch_w, dtype=np.float32)[None, :] / np.float32(max(patch_w - 1, 1))
    )
    row_grid = np.repeat(row_grid, patch_w, axis=1)
    col_grid = np.repeat(col_grid, patch_h, axis=0)
    if depth_m is None:
        depth_mean = np.zeros((patch_h, patch_w), dtype=np.float32)
        depth_std = np.zeros((patch_h, patch_w), dtype=np.float32)
        depth_near = np.zeros((patch_h, patch_w), dtype=np.float32)
        depth_far = np.zeros((patch_h, patch_w), dtype=np.float32)
        depth_valid_ratio = np.zeros((patch_h, patch_w), dtype=np.float32)
    else:
        depth = np.asarray(depth_m, dtype=np.float32).reshape(patch_h, row_bin, patch_w, col_bin)
        valid = np.isfinite(depth) & (depth > np.float32(0.0))
        valid_count = valid.sum(axis=(1, 3)).astype(np.float32)
        denom = np.maximum(valid_count, np.float32(1.0))
        clipped = np.clip(depth, 0.0, 6.0)
        valid_values = np.where(valid, clipped, np.float32(0.0))
        depth_sum = valid_values.sum(axis=(1, 3))
        depth_sumsq = (valid_values * valid_values).sum(axis=(1, 3))
        mean_raw = depth_sum / denom
        variance_raw = np.maximum(depth_sumsq / denom - mean_raw * mean_raw, np.float32(0.0))
        near_raw = np.where(valid, clipped, np.float32(np.inf)).min(axis=(1, 3))
        far_raw = np.where(valid, clipped, np.float32(-np.inf)).max(axis=(1, 3))
        has_valid = valid_count > 0.0
        depth_mean = np.where(has_valid, mean_raw / np.float32(6.0), 0.0).astype(np.float32)
        depth_std = np.where(has_valid, np.sqrt(variance_raw) / np.float32(3.0), 0.0).astype(np.float32)
        depth_near = np.where(has_valid, near_raw / np.float32(6.0), 0.0).astype(np.float32)
        depth_far = np.where(has_valid, far_raw / np.float32(6.0), 0.0).astype(np.float32)
        depth_valid_ratio = (valid_count / np.float32(row_bin * col_bin)).astype(np.float32)
    base = np.concatenate(
        [
            rgb_mean.astype(np.float32),
            depth_mean[..., None],
            depth_std[..., None],
            depth_near[..., None],
            depth_far[..., None],
            depth_valid_ratio[..., None],
            row_grid[..., None],
            col_grid[..., None],
        ],
        axis=-1,
    )
    repeats = int(np.ceil(feature_dim / base.shape[-1]))
    return np.tile(base, (1, 1, repeats))[..., :feature_dim].astype(np.float32)


def depth_valid_observation_mask(depth_m: np.ndarray | None, *, bev_shape: tuple[int, int]) -> np.ndarray | None:
    if depth_m is None:
        return None
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        return None
    valid = (np.isfinite(depth) & (depth > np.float32(0.0))).astype(np.float32)
    if not np.any(valid):
        return None
    resized = resize_nearest_float(valid, bev_shape)
    return (resized > 0.0).astype(np.float32)


def resize_nearest_float(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = np.linspace(0, array.shape[0] - 1, shape[0]).round().astype(np.int64)
    cols = np.linspace(0, array.shape[1] - 1, shape[1]).round().astype(np.int64)
    return np.asarray(array, dtype=np.float32)[rows[:, None], cols[None, :]]


def _association_path(root: Path) -> Path | None:
    for name in (OPENLORIS_ROUTE_ASSOCIATIONS_FILE, TUM_RGBD_ROUTE_ASSOCIATIONS_FILE):
        path = root / name
        if path.exists():
            return path
    return None


def _read_json(path: Path) -> JsonDict:
    import json

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _order_events_for_replay(events: list[Event]) -> list[Event]:
    return [
        event
        for _index, event in sorted(
            enumerate(events),
            key=lambda indexed: (indexed[1].timestamp_ns, indexed[0]),
        )
    ]


def _bin_edges(size: int, bins: int) -> np.ndarray:
    edges = np.linspace(0, size, bins + 1).round().astype(np.int64)
    edges[0] = 0
    edges[-1] = size
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = min(size, edges[index - 1] + 1)
    return np.clip(edges, 0, size)
