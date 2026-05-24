from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from homebrain.messages.schema import FrameEvent, JsonDict, deterministic_json
from homebrain.replay.replayd import order_events_for_replay
from homebrain.replay.segment_log import load_manifest, read_events
from homebrain.teachers.artifacts import file_sha256, relative_to_root, save_array, write_json

SCENE_TEACHER_PACK_SCHEMA_VERSION = "homebrain.scene_teacher_pack.v0"
SCENE_TEACHER_MANIFEST_FILE = "scene_teacher_manifest.json"
SCENE_TEACHER_REQUIRED_GEOMETRY_KINDS = ("depth", "point_map")
SCENE_TEACHER_FRAME_ARTIFACT_KINDS: tuple[str, ...] = (
    "depth",
    "point_map",
    "intrinsics",
    "extrinsics",
    "confidence",
    "validity_mask",
    "floor_traversable_mask",
    "obstacle_risk_mask",
    "dynamic_motion_mask",
)
SCENE_TEACHER_WINDOW_ARTIFACT_KINDS: tuple[str, ...] = (
    "point_tracks",
    "track_validity",
)
SCENE_TEACHER_CONTROL_SAFETY = "not_control_safe_offline_scene_teacher_only"


@dataclass(frozen=True)
class SceneTeacherRunConfig:
    log_dir: Path
    out_dir: Path


@dataclass(frozen=True)
class SceneTeacherRunSummary:
    teacher_name: str
    teacher_version: str
    backend: str
    mock: bool
    frame_count: int
    manifest_path: Path


@dataclass(frozen=True)
class SceneFramePrediction:
    depth: np.ndarray | None = None
    point_map: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    extrinsics: np.ndarray | None = None
    confidence: np.ndarray | None = None
    validity_mask: np.ndarray | None = None
    floor_traversable_mask: np.ndarray | None = None
    obstacle_risk_mask: np.ndarray | None = None
    dynamic_motion_mask: np.ndarray | None = None
    extra_metadata: JsonDict | None = None


@dataclass(frozen=True)
class SceneWindowPrediction:
    frame_indices: tuple[int, ...]
    point_tracks: np.ndarray | None = None
    track_validity: np.ndarray | None = None
    extra_metadata: JsonDict | None = None


@dataclass(frozen=True)
class SceneTeacherBatchPrediction:
    frames: tuple[SceneFramePrediction, ...]
    windows: tuple[SceneWindowPrediction, ...] = ()


class SceneTeacherBackend(Protocol):
    name: str
    mocked: bool
    synthetic: bool
    real_perception: bool
    deterministic: bool
    model_id: str
    model_source: str
    license_review_status: str
    scale_status: str

    def infer(self, log_dir: Path, frames: list[FrameEvent]) -> SceneTeacherBatchPrediction:
        """Return scene/geometry predictions for an ordered frame batch."""

    def dependency_status(self) -> JsonDict:
        """Return dependency and model-load status for manifests."""


class SceneTeacher(ABC):
    name: str
    version: str
    backend: SceneTeacherBackend

    @abstractmethod
    def run(self, config: SceneTeacherRunConfig) -> SceneTeacherRunSummary:
        """Run the scene teacher and write a SceneTeacherPack."""


def write_scene_teacher_pack(
    *,
    teacher_name: str,
    teacher_version: str,
    backend: SceneTeacherBackend,
    log_dir: str | Path,
    out_dir: str | Path,
    max_frames: int | None = None,
    stride: int = 1,
) -> SceneTeacherRunSummary:
    if stride < 1:
        raise ValueError("scene teacher stride must be at least 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("scene teacher max_frames must be at least 1 when supplied")

    log_root = Path(log_dir)
    output = Path(out_dir)
    source_manifest = load_manifest(log_root)
    events = order_events_for_replay(read_events(log_root))
    source_frames = [event for event in events if isinstance(event, FrameEvent)]
    frames = source_frames[::stride]
    if max_frames is not None:
        frames = frames[:max_frames]

    batch = backend.infer(log_root, frames)
    if len(batch.frames) != len(frames):
        raise ValueError(f"backend returned {len(batch.frames)} frame predictions for {len(frames)} frames")

    frame_records: list[JsonDict] = []
    frame_artifact_kinds: set[str] = set()
    for frame, prediction in zip(frames, batch.frames):
        record = _write_frame_prediction(
            log_dir=log_root,
            out_dir=output,
            backend=backend,
            teacher_name=teacher_name,
            teacher_version=teacher_version,
            frame=frame,
            prediction=prediction,
        )
        frame_records.append(record)
        artifacts = record.get("artifacts")
        if isinstance(artifacts, dict):
            frame_artifact_kinds.update(str(kind) for kind in artifacts)

    window_records: list[JsonDict] = []
    window_artifact_kinds: set[str] = set()
    for window_index, window in enumerate(batch.windows):
        record = _write_window_prediction(
            out_dir=output,
            window_index=window_index,
            frames=frames,
            window=window,
        )
        window_records.append(record)
        artifacts = record.get("artifacts")
        if isinstance(artifacts, dict):
            window_artifact_kinds.update(str(kind) for kind in artifacts)

    manifest: JsonDict = {
        "schema_version": SCENE_TEACHER_PACK_SCHEMA_VERSION,
        "pack_name": "SceneTeacherPack",
        "pack_version": "v0",
        "teacher_name": teacher_name,
        "teacher_version": teacher_version,
        "backend": backend.name,
        "model_id": backend.model_id,
        "model_source": backend.model_source,
        "license_review_status": backend.license_review_status,
        "mock": bool(backend.mocked),
        "synthetic": bool(backend.synthetic),
        "real_perception": bool(backend.real_perception),
        "deterministic": bool(backend.deterministic),
        "created_at_utc": _created_at(backend.deterministic),
        "source_log": log_root.as_posix(),
        "source_segment_id": source_manifest.segment_id,
        "source_schema_version": source_manifest.schema_version,
        "source_frame_count": len(source_frames),
        "frame_count": len(frame_records),
        "stride": stride,
        "max_frames": max_frames,
        "artifact_kinds": sorted(frame_artifact_kinds),
        "supported_frame_artifact_kinds": list(SCENE_TEACHER_FRAME_ARTIFACT_KINDS),
        "window_artifact_kinds": sorted(window_artifact_kinds),
        "supported_window_artifact_kinds": list(SCENE_TEACHER_WINDOW_ARTIFACT_KINDS),
        "scale_status": backend.scale_status,
        "calibration_class": "teacher_estimated" if backend.real_perception else "test_only_synthetic",
        "intrinsics_source": "teacher_or_frame_intrinsics_when_available",
        "extrinsics_source": "teacher_estimated_when_available",
        "pose_source": "teacher_estimated_camera_pose_when_available",
        "not_robot_frame_truth": True,
        "robot_frame_truth": False,
        "action_supervision_ok": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "control_safe_claim": False,
        "product_training_approved": False,
        "raw_pwm_emitted": False,
        "control_safety": SCENE_TEACHER_CONTROL_SAFETY,
        "runtime_dependency": False,
        "dependency_status": backend.dependency_status(),
        "frames": frame_records,
        "windows": window_records,
    }
    manifest_path = write_scene_teacher_manifest(output, manifest)
    return SceneTeacherRunSummary(
        teacher_name=teacher_name,
        teacher_version=teacher_version,
        backend=backend.name,
        mock=bool(backend.mocked),
        frame_count=len(frame_records),
        manifest_path=manifest_path,
    )


def write_scene_teacher_manifest(artifacts_dir: str | Path, manifest: JsonDict) -> Path:
    path = Path(artifacts_dir) / SCENE_TEACHER_MANIFEST_FILE
    write_json(path, manifest)
    return path


def load_scene_teacher_manifest(artifacts_dir: str | Path) -> JsonDict:
    path = Path(artifacts_dir) / SCENE_TEACHER_MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"missing scene teacher manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"scene teacher manifest must be a JSON object: {path}")
    if data.get("schema_version") != SCENE_TEACHER_PACK_SCHEMA_VERSION:
        raise ValueError(
            "unsupported scene teacher schema "
            f"{data.get('schema_version')!r}; expected {SCENE_TEACHER_PACK_SCHEMA_VERSION!r}"
        )
    if not isinstance(data.get("frames"), list):
        raise ValueError("scene teacher manifest frames must be a list")
    return data


def write_scene_teacher_json(path: str | Path, data: JsonDict, *, pretty: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        if pretty:
            json.dump(data, handle, sort_keys=True, indent=2)
        else:
            handle.write(deterministic_json(data))
        handle.write("\n")


def _write_frame_prediction(
    *,
    log_dir: Path,
    out_dir: Path,
    backend: SceneTeacherBackend,
    teacher_name: str,
    teacher_version: str,
    frame: FrameEvent,
    prediction: SceneFramePrediction,
) -> JsonDict:
    frame_dir = out_dir / "frames" / f"{frame.camera_id}_{frame.frame_id:06d}"
    arrays, sources = _frame_arrays(log_dir, frame, prediction)
    artifact_records: dict[str, JsonDict] = {}
    for kind in SCENE_TEACHER_FRAME_ARTIFACT_KINDS:
        array = arrays.get(kind)
        if array is None:
            continue
        target = frame_dir / f"{kind}.npy"
        record = save_array(target, array)
        artifact_records[kind] = {
            "kind": kind,
            "path": relative_to_root(target, out_dir),
            **record,
        }

    metadata: JsonDict = {
        "schema_version": SCENE_TEACHER_PACK_SCHEMA_VERSION,
        "pack_name": "SceneTeacherPack",
        "pack_version": "v0",
        "teacher_name": teacher_name,
        "teacher_version": teacher_version,
        "backend": backend.name,
        "model_id": backend.model_id,
        "model_source": backend.model_source,
        "license_review_status": backend.license_review_status,
        "mock": bool(backend.mocked),
        "synthetic": bool(backend.synthetic),
        "real_perception": bool(backend.real_perception),
        "sequence_id": frame.sequence_id,
        "camera_id": frame.camera_id,
        "frame_id": frame.frame_id,
        "timestamp_ns": frame.timestamp_ns,
        "width": frame.width,
        "height": frame.height,
        "format": frame.format,
        "source_data_ref": frame.data_ref,
        "depth_units": "teacher_scale_or_metric_when_backend_documents_it",
        "point_map_frame": "teacher_camera_frame",
        "scale_status": backend.scale_status,
        "calibration_class": "teacher_estimated" if backend.real_perception else "test_only_synthetic",
        "intrinsics_source": sources.get("intrinsics", "missing_from_teacher_and_frame"),
        "extrinsics_source": sources.get("extrinsics", "missing_from_teacher"),
        "pose_source": "teacher_estimated_camera_pose" if "extrinsics" in artifact_records else "missing_from_teacher",
        "floor_traversable_mask_source": sources.get("floor_traversable_mask", "placeholder"),
        "obstacle_risk_mask_source": sources.get("obstacle_risk_mask", "placeholder"),
        "dynamic_motion_mask_source": sources.get("dynamic_motion_mask", "placeholder"),
        "confidence_source": sources.get("confidence", "geometry_finiteness_heuristic"),
        "validity_mask_source": sources.get("validity_mask", "geometry_finiteness_heuristic"),
        "not_robot_frame_truth": True,
        "robot_frame_truth": False,
        "action_supervision_ok": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "control_safe_claim": False,
        "product_training_approved": False,
        "raw_pwm_emitted": False,
        "control_safety": SCENE_TEACHER_CONTROL_SAFETY,
        "dependency_status": backend.dependency_status(),
        "backend_metadata": prediction.extra_metadata or {},
        "artifact_shapes": {kind: artifact_records[kind]["shape"] for kind in artifact_records},
        "artifact_dtypes": {kind: artifact_records[kind]["dtype"] for kind in artifact_records},
    }
    metadata_path = frame_dir / "metadata.json"
    write_json(metadata_path, metadata)

    return {
        "sequence_id": frame.sequence_id,
        "camera_id": frame.camera_id,
        "frame_id": frame.frame_id,
        "timestamp_ns": frame.timestamp_ns,
        "width": frame.width,
        "height": frame.height,
        "format": frame.format,
        "source_data_ref": frame.data_ref,
        "scale_status": backend.scale_status,
        "calibration_class": metadata["calibration_class"],
        "intrinsics_source": metadata["intrinsics_source"],
        "extrinsics_source": metadata["extrinsics_source"],
        "pose_source": metadata["pose_source"],
        "not_robot_frame_truth": True,
        "robot_frame_truth": False,
        "action_supervision_ok": False,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "product_training_approved": False,
        "raw_pwm_emitted": False,
        "metadata_path": relative_to_root(metadata_path, out_dir),
        "metadata_sha256": file_sha256(metadata_path),
        "artifacts": artifact_records,
    }


def _write_window_prediction(
    *,
    out_dir: Path,
    window_index: int,
    frames: list[FrameEvent],
    window: SceneWindowPrediction,
) -> JsonDict:
    if not window.frame_indices:
        raise ValueError("scene teacher window must reference at least one frame")
    invalid = [index for index in window.frame_indices if index < 0 or index >= len(frames)]
    if invalid:
        raise ValueError(f"scene teacher window has invalid frame indices: {invalid}")

    window_dir = out_dir / "windows" / f"window_{window_index:06d}"
    arrays: dict[str, np.ndarray] = {}
    if window.point_tracks is not None:
        arrays["point_tracks"] = np.asarray(window.point_tracks, dtype=np.float32)
    if window.track_validity is not None:
        arrays["track_validity"] = np.asarray(window.track_validity, dtype=np.uint8)

    artifact_records: dict[str, JsonDict] = {}
    for kind in SCENE_TEACHER_WINDOW_ARTIFACT_KINDS:
        array = arrays.get(kind)
        if array is None:
            continue
        target = window_dir / f"{kind}.npy"
        record = save_array(target, array)
        artifact_records[kind] = {
            "kind": kind,
            "path": relative_to_root(target, out_dir),
            **record,
        }

    frame_keys = [_frame_key(frames[index]) for index in window.frame_indices]
    metadata: JsonDict = {
        "schema_version": SCENE_TEACHER_PACK_SCHEMA_VERSION,
        "window_index": window_index,
        "frame_indices": list(window.frame_indices),
        "frame_keys": frame_keys,
        "point_tracks_source": "teacher_estimated_when_available",
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "product_training_approved": False,
        "raw_pwm_emitted": False,
        "backend_metadata": window.extra_metadata or {},
        "artifact_shapes": {kind: artifact_records[kind]["shape"] for kind in artifact_records},
        "artifact_dtypes": {kind: artifact_records[kind]["dtype"] for kind in artifact_records},
    }
    metadata_path = window_dir / "metadata.json"
    write_json(metadata_path, metadata)

    return {
        "window_index": window_index,
        "frame_indices": list(window.frame_indices),
        "frame_keys": frame_keys,
        "start_frame_id": frames[window.frame_indices[0]].frame_id,
        "end_frame_id": frames[window.frame_indices[-1]].frame_id,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "product_training_approved": False,
        "raw_pwm_emitted": False,
        "metadata_path": relative_to_root(metadata_path, out_dir),
        "metadata_sha256": file_sha256(metadata_path),
        "artifacts": artifact_records,
    }


def _frame_arrays(
    log_dir: Path,
    frame: FrameEvent,
    prediction: SceneFramePrediction,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    arrays: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    if prediction.depth is not None:
        arrays["depth"] = _same_shape_float32(prediction.depth, frame, "depth")
        sources["depth"] = "teacher_output"
    if prediction.point_map is not None:
        arrays["point_map"] = _point_map_float32(prediction.point_map, frame)
        sources["point_map"] = "teacher_output"
    if prediction.intrinsics is not None:
        arrays["intrinsics"] = _matrix_float32(prediction.intrinsics, "intrinsics", (3, 3))
        sources["intrinsics"] = "teacher_output"
    else:
        frame_intrinsics = _intrinsics_matrix(frame.intrinsics)
        if frame_intrinsics is not None:
            arrays["intrinsics"] = frame_intrinsics
            sources["intrinsics"] = "frame_intrinsics"
    if prediction.extrinsics is not None:
        arrays["extrinsics"] = _extrinsics_matrix(prediction.extrinsics)
        sources["extrinsics"] = "teacher_output"

    geometry_mask = _geometry_validity(arrays.get("depth"), arrays.get("point_map"), frame)
    if prediction.confidence is not None:
        arrays["confidence"] = np.clip(
            _same_shape_float32(prediction.confidence, frame, "confidence"),
            np.float32(0.0),
            np.float32(1.0),
        ).astype(np.float32)
        sources["confidence"] = "teacher_output"
    else:
        arrays["confidence"] = geometry_mask.astype(np.float32)
        sources["confidence"] = "geometry_finiteness_heuristic"

    if prediction.validity_mask is not None:
        arrays["validity_mask"] = _mask_uint8(prediction.validity_mask, frame, "validity_mask")
        sources["validity_mask"] = "teacher_output"
    else:
        arrays["validity_mask"] = geometry_mask.astype(np.uint8)
        sources["validity_mask"] = "geometry_finiteness_heuristic"

    for kind, source_array, source_name in (
        ("floor_traversable_mask", prediction.floor_traversable_mask, "teacher_output"),
        ("obstacle_risk_mask", prediction.obstacle_risk_mask, "teacher_output"),
        ("dynamic_motion_mask", prediction.dynamic_motion_mask, "teacher_output"),
    ):
        if source_array is not None:
            arrays[kind] = _mask_uint8(source_array, frame, kind)
            sources[kind] = source_name

    placeholder_masks = _placeholder_masks(arrays.get("depth"), arrays["confidence"], arrays["validity_mask"])
    for kind, array in placeholder_masks.items():
        if kind not in arrays:
            arrays[kind] = array
            sources[kind] = "homebrain_placeholder_from_scene_geometry"

    if not any(kind in arrays for kind in SCENE_TEACHER_REQUIRED_GEOMETRY_KINDS):
        raise ValueError(f"frame {frame.frame_id} has no depth or point_map scene geometry")
    return arrays, sources


def _same_shape_float32(array: np.ndarray, frame: FrameEvent, kind: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"{kind} must be 2D after squeeze, got shape {values.shape}")
    expected = (frame.height, frame.width)
    if values.shape == expected:
        return values.astype(np.float32)
    return _resize_nearest(values, expected).astype(np.float32)


def _point_map_float32(array: np.ndarray, frame: FrameEvent) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"point_map must have shape HxWx3 after squeeze, got {values.shape}")
    expected_hw = (frame.height, frame.width)
    if values.shape[:2] == expected_hw:
        return values.astype(np.float32)
    resized = np.stack(
        [_resize_nearest(values[:, :, channel], expected_hw) for channel in range(3)],
        axis=-1,
    )
    return resized.astype(np.float32)


def _mask_uint8(array: np.ndarray, frame: FrameEvent, kind: str) -> np.ndarray:
    values = _same_shape_float32(array, frame, kind)
    return (values > np.float32(0.0)).astype(np.uint8)


def _matrix_float32(array: np.ndarray, kind: str, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    values = np.squeeze(values)
    if values.shape != shape:
        raise ValueError(f"{kind} must have shape {shape}, got {values.shape}")
    return values.astype(np.float32)


def _extrinsics_matrix(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    values = np.squeeze(values)
    if values.shape == (3, 4):
        full = np.eye(4, dtype=np.float32)
        full[:3, :] = values
        return full
    if values.shape != (4, 4):
        raise ValueError(f"extrinsics must have shape 3x4 or 4x4, got {values.shape}")
    return values.astype(np.float32)


def _intrinsics_matrix(intrinsics: JsonDict | None) -> np.ndarray | None:
    if not isinstance(intrinsics, dict):
        return None
    matrix = intrinsics.get("camera_matrix")
    if isinstance(matrix, list) and len(matrix) == 3 and all(isinstance(row, list) and len(row) == 3 for row in matrix):
        values = np.asarray(matrix, dtype=np.float32)
        if np.isfinite(values).all():
            return values
    fx = _optional_number(intrinsics.get("fx") or intrinsics.get("focal_length_px") or intrinsics.get("focallength_px"))
    fy = _optional_number(intrinsics.get("fy"))
    cx = _optional_number(intrinsics.get("cx"))
    cy = _optional_number(intrinsics.get("cy"))
    if fx is None:
        return None
    if fy is None:
        fy = fx
    if cx is None:
        cx = 0.0
    if cy is None:
        cy = 0.0
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _optional_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _geometry_validity(depth: np.ndarray | None, point_map: np.ndarray | None, frame: FrameEvent) -> np.ndarray:
    valid = np.zeros((frame.height, frame.width), dtype=bool)
    if depth is not None:
        valid |= np.isfinite(depth) & (depth > np.float32(0.0))
    if point_map is not None:
        valid |= np.isfinite(point_map).all(axis=2)
    return valid


def _placeholder_masks(
    depth: np.ndarray | None,
    confidence: np.ndarray,
    validity_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    valid = (validity_mask > 0) & np.isfinite(confidence) & (confidence > np.float32(0.05))
    if depth is None or not np.any(valid):
        empty = np.zeros(validity_mask.shape, dtype=np.uint8)
        return {
            "floor_traversable_mask": empty.copy(),
            "obstacle_risk_mask": empty.copy(),
            "dynamic_motion_mask": empty.copy(),
        }
    values = np.asarray(depth, dtype=np.float32)
    finite = values[valid]
    lo = float(np.percentile(finite, 10))
    hi = float(np.percentile(finite, 90))
    scale = max(hi - lo, 1.0e-6)
    norm = np.clip((values - np.float32(lo)) / np.float32(scale), np.float32(0.0), np.float32(1.0))
    floor = valid & (norm >= np.float32(0.35))
    obstacle = valid & (norm <= np.float32(0.20))
    return {
        "floor_traversable_mask": floor.astype(np.uint8),
        "obstacle_risk_mask": obstacle.astype(np.uint8),
        "dynamic_motion_mask": np.zeros(validity_mask.shape, dtype=np.uint8),
    }


def _resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(array)
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"cannot resize non-2D array with shape {values.shape}")
    out_h, out_w = shape
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"invalid resize target shape {shape}")
    if values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError(f"cannot resize empty array with shape {values.shape}")
    row_index = np.linspace(0, values.shape[0] - 1, out_h).round().astype(np.int64)
    col_index = np.linspace(0, values.shape[1] - 1, out_w).round().astype(np.int64)
    return values[row_index[:, None], col_index[None, :]]


def _frame_key(frame: FrameEvent) -> str:
    return f"{frame.sequence_id}:{frame.camera_id}:{frame.frame_id}"


def _created_at(deterministic: bool) -> str:
    if deterministic:
        return "1970-01-01T00:00:00Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
