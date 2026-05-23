from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Protocol

import numpy as np

from homebrain.messages.schema import FrameEvent, JsonDict
from homebrain.replay.replayd import order_events_for_replay
from homebrain.replay.segment_log import load_manifest, read_events
from homebrain.teachers.artifacts import (
    DA3_ARTIFACT_KINDS,
    TEACHER_ARTIFACT_SCHEMA_VERSION,
    expected_shape,
    file_sha256,
    relative_to_root,
    save_array,
    write_json,
    write_teacher_manifest,
)
from homebrain.teachers.base import Teacher, TeacherRunConfig, TeacherRunSummary

DA3_DEFAULT_MODEL_ID = "depth-anything/DA3-SMALL"
DA3_GITHUB_URL = "https://github.com/ByteDance-Seed/depth-anything-3"
DA3_LICENSE_REVIEW_STATUS = "pending_human_review"
DA3_CONTROL_SAFETY = "not_control_safe_training_teacher_only"
DA3_SETUP_STATUS_PATH = Path("external") / "da3_setup_status.json"


class DA3UnavailableError(RuntimeError):
    """Raised when real Depth Anything 3 inference cannot be initialized."""


@dataclass(frozen=True)
class DA3Prediction:
    depth: np.ndarray | None
    confidence: np.ndarray | None
    intrinsics: np.ndarray | None
    extrinsics: np.ndarray | None
    extra_metadata: JsonDict


class DA3Backend(Protocol):
    name: str
    mocked: bool
    synthetic: bool
    real_perception: bool
    deterministic: bool

    def infer(self, log_dir: Path, frames: list[FrameEvent]) -> list[DA3Prediction]:
        """Return DA3 predictions for an ordered frame batch."""

    def dependency_status(self) -> JsonDict:
        """Return dependency and model-load status for the manifest."""


class FakeDA3Backend:
    name = "fake"
    mocked = True
    synthetic = True
    real_perception = False
    deterministic = True

    def __init__(self, *, model_id: str = DA3_DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id

    def dependency_status(self) -> JsonDict:
        return {
            "backend": self.name,
            "depth_anything_3_package": "not_required_for_fake_backend",
            "model_loaded": False,
            "model_id": self.model_id,
            "weights_status": "not_required_for_fake_backend",
            "downloads_attempted_by_homebrain": False,
            "warning": "test backend; not real DA3 inference",
        }

    def infer(self, log_dir: Path, frames: list[FrameEvent]) -> list[DA3Prediction]:
        predictions: list[DA3Prediction] = []
        for index, frame in enumerate(frames):
            luminance = _frame_luminance(log_dir / frame.data_ref, frame)
            row, col = np.indices((frame.height, frame.width), dtype=np.float32)
            row_norm = row / np.float32(max(frame.height - 1, 1))
            col_norm = col / np.float32(max(frame.width - 1, 1))
            depth = (
                np.float32(0.2)
                + np.float32(0.8) * row_norm
                + np.float32(0.4) * col_norm
                + np.float32(0.2) * luminance
                + np.float32(index) * np.float32(0.01)
            ).astype(np.float32)
            confidence = np.clip(
                np.float32(0.9)
                - np.float32(0.15) * np.abs(luminance - np.float32(0.5))
                - np.float32(0.05) * row_norm,
                np.float32(0.0),
                np.float32(1.0),
            ).astype(np.float32)
            focal = np.float32(max(frame.width, frame.height))
            intrinsics = np.asarray(
                [
                    [focal, np.float32(0.0), np.float32((frame.width - 1) / 2.0)],
                    [np.float32(0.0), focal, np.float32((frame.height - 1) / 2.0)],
                    [np.float32(0.0), np.float32(0.0), np.float32(1.0)],
                ],
                dtype=np.float32,
            )
            extrinsics = np.asarray(
                [
                    [np.float32(1.0), np.float32(0.0), np.float32(0.0), np.float32(index) * np.float32(0.02)],
                    [np.float32(0.0), np.float32(1.0), np.float32(0.0), np.float32(0.0)],
                    [np.float32(0.0), np.float32(0.0), np.float32(1.0), np.float32(0.0)],
                ],
                dtype=np.float32,
            )
            predictions.append(
                DA3Prediction(
                    depth=depth,
                    confidence=confidence,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    extra_metadata={
                        "backend": self.name,
                        "warning": "fake deterministic test backend; not real DA3 inference",
                    },
                )
            )
        return predictions


class RealDA3Backend:
    name = "real"
    mocked = False
    synthetic = False
    real_perception = True
    deterministic = False

    def __init__(
        self,
        *,
        model_id: str = DA3_DEFAULT_MODEL_ID,
        model_dir: str | Path | None = None,
        device: str | None = None,
        window_size: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.device = device
        self.window_size = window_size if window_size and window_size > 0 else None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._model_source: str | None = None

    def dependency_status(self) -> JsonDict:
        return {
            "backend": self.name,
            "depth_anything_3_package": "loaded" if self._model is not None else "not_loaded",
            "model_loaded": self._model is not None,
            "model_id": self.model_id,
            "model_source": self._model_source or _model_source_text(self.model_dir, self.model_id),
            "device": self.device or "auto",
            "window_size": self.window_size,
            "downloads_attempted_by_homebrain": False,
            "python_executable": _python_executable(),
        }

    def infer(self, log_dir: Path, frames: list[FrameEvent]) -> list[DA3Prediction]:
        if not frames:
            return []
        self._ensure_loaded()
        assert self._model is not None

        predictions: list[DA3Prediction] = []
        for batch in _batches(frames, self.window_size or len(frames)):
            image_paths = [_frame_image_path(log_dir, frame) for frame in batch]
            try:
                raw_prediction = self._model.inference([str(path) for path in image_paths])
            except Exception as exc:  # noqa: BLE001 - surface model API failures clearly.
                raise DA3UnavailableError(f"DA3 inference failed for a {len(batch)} frame batch: {exc}") from exc
            predictions.extend(_split_da3_prediction(raw_prediction, len(batch)))
        if len(predictions) != len(frames):
            raise DA3UnavailableError(
                f"DA3 returned {len(predictions)} prediction(s) for {len(frames)} input frame(s)"
            )
        return predictions

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from depth_anything_3.api import DepthAnything3  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DA3UnavailableError(
                "Depth Anything 3 API could not be imported in this Python environment. Run "
                "`python -m homebrain.tools.setup_da3_teacher --download`, check "
                "`external/da3_setup_status.json`, or execute the real teacher with the "
                f"repaired DA3 venv Python. Import error: {exc}"
            ) from exc

        model_source = _resolve_model_source(self.model_dir, self.model_id)
        try:
            model = DepthAnything3.from_pretrained(model_source)
        except Exception as exc:  # noqa: BLE001 - include local-cache guidance.
            raise DA3UnavailableError(
                "Depth Anything 3 model could not be loaded from local setup/cache. "
                "Run setup with `--download`, pass `--model-dir` to a local snapshot, "
                "and retry; normal teacher runs do not intentionally download weights."
            ) from exc

        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            torch = None

        if self.device is None and torch is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device is not None and hasattr(model, "to"):
            model = model.to(device=self.device)

        self._model = model
        self._torch = torch
        self._model_source = str(model_source)


class DA3Teacher(Teacher):
    name = "da3"
    version = "da3.teacher.v0"
    mock = False

    def __init__(
        self,
        backend: DA3Backend | None = None,
        *,
        model_id: str = DA3_DEFAULT_MODEL_ID,
        model_dir: str | Path | None = None,
        max_frames: int | None = None,
        stride: int = 1,
        window_size: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.max_frames = max_frames
        self.stride = stride
        self.window_size = window_size
        self.backend = backend if backend is not None else RealDA3Backend(
            model_id=model_id,
            model_dir=model_dir,
            window_size=window_size,
        )

    def run(self, config: TeacherRunConfig) -> TeacherRunSummary:
        if self.stride < 1:
            raise ValueError("DA3 stride must be at least 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("DA3 max_frames must be at least 1 when supplied")

        log_dir = Path(config.log_dir)
        out_dir = Path(config.out_dir)
        source_manifest = load_manifest(log_dir)
        events = order_events_for_replay(read_events(log_dir))
        all_frames = [event for event in events if isinstance(event, FrameEvent)]
        frames = all_frames[:: self.stride]
        if self.max_frames is not None:
            frames = frames[: self.max_frames]

        predictions = self.backend.infer(log_dir, frames)
        frame_records = [
            self._write_frame_artifacts(log_dir, out_dir, frame, prediction)
            for frame, prediction in zip(frames, predictions)
        ]

        manifest: JsonDict = {
            "schema_version": TEACHER_ARTIFACT_SCHEMA_VERSION,
            "teacher_name": self.name,
            "teacher_version": self.version,
            "backend": self.backend.name,
            "mock": bool(self.backend.mocked),
            "synthetic": bool(self.backend.synthetic),
            "real_perception": bool(self.backend.real_perception),
            "deterministic": bool(self.backend.deterministic),
            "created_at_utc": _created_at(self.backend.deterministic),
            "source_log": log_dir.as_posix(),
            "source_segment_id": source_manifest.segment_id,
            "source_schema_version": source_manifest.schema_version,
            "source_frame_count": len(all_frames),
            "frame_count": len(frame_records),
            "stride": self.stride,
            "max_frames": self.max_frames,
            "window_size": self.window_size,
            "artifact_kinds": list(DA3_ARTIFACT_KINDS),
            "model_id": self.model_id,
            "model_path": self.model_dir.as_posix() if self.model_dir is not None else None,
            "model_source": DA3_GITHUB_URL,
            "repo_commit": _setup_repo_commit(),
            "dependency_status": self.backend.dependency_status(),
            "license_review_status": DA3_LICENSE_REVIEW_STATUS,
            "calibration_class": "teacher_estimated" if self.backend.real_perception else "test_only_synthetic",
            "intrinsics_source": "teacher_estimated" if self.backend.real_perception else "synthetic_test_backend",
            "extrinsics_source": "teacher_estimated" if self.backend.real_perception else "synthetic_test_backend",
            "pose_source": "teacher_estimated_camera_pose" if self.backend.real_perception else "synthetic_test_backend",
            "scale_source": "teacher_relative_not_metric",
            "gravity_floor_source": "not_observed",
            "not_robot_frame_truth": True,
            "control_safety": DA3_CONTROL_SAFETY,
            "frames": frame_records,
        }
        manifest_path = write_teacher_manifest(out_dir, manifest)
        return TeacherRunSummary(
            teacher_name=self.name,
            teacher_version=self.version,
            mock=bool(self.backend.mocked),
            frame_count=len(frame_records),
            manifest_path=manifest_path,
        )

    def _write_frame_artifacts(
        self,
        log_dir: Path,
        out_dir: Path,
        frame: FrameEvent,
        prediction: DA3Prediction,
    ) -> JsonDict:
        frame_dir = out_dir / "frames" / f"{frame.camera_id}_{frame.frame_id:06d}"
        arrays = _prediction_arrays(prediction, frame)
        artifact_records: dict[str, JsonDict] = {}

        for kind, array in arrays.items():
            expected = expected_shape(kind, frame.width, frame.height)
            if tuple(array.shape) != expected:
                raise ValueError(f"{kind} DA3 artifact has wrong shape {array.shape}; expected {expected}")
            target = frame_dir / f"{kind}.npy"
            array_record = save_array(target, array)
            artifact_records[kind] = {
                "kind": kind,
                "path": relative_to_root(target, out_dir),
                **array_record,
            }

        missing = [kind for kind in DA3_ARTIFACT_KINDS if kind not in artifact_records]
        metadata: JsonDict = {
            "schema_version": TEACHER_ARTIFACT_SCHEMA_VERSION,
            "teacher_name": self.name,
            "teacher_version": self.version,
            "backend": self.backend.name,
            "mock": bool(self.backend.mocked),
            "synthetic": bool(self.backend.synthetic),
            "real_perception": bool(self.backend.real_perception),
            "license_review_status": DA3_LICENSE_REVIEW_STATUS,
            "sequence_id": frame.sequence_id,
            "camera_id": frame.camera_id,
            "frame_id": frame.frame_id,
            "timestamp_ns": frame.timestamp_ns,
            "width": frame.width,
            "height": frame.height,
            "format": frame.format,
            "source_data_ref": frame.data_ref,
            "model_id": self.model_id,
            "model_path": self.model_dir.as_posix() if self.model_dir is not None else None,
            "depth_units": "relative_teacher_scale",
            "calibration_class": "teacher_estimated" if self.backend.real_perception else "test_only_synthetic",
            "intrinsics_source": "teacher_estimated" if "intrinsics" in artifact_records else "missing_from_teacher_output",
            "extrinsics_source": "teacher_estimated" if "extrinsics" in artifact_records else "missing_from_teacher_output",
            "pose_source": "teacher_estimated_camera_pose" if "extrinsics" in artifact_records else "missing_from_teacher_output",
            "scale_source": "teacher_relative_not_metric",
            "gravity_floor_source": "not_observed",
            "not_robot_frame_truth": True,
            "control_safety": DA3_CONTROL_SAFETY,
            "missing_teacher_outputs": missing,
            "dependency_status": self.backend.dependency_status(),
            "backend_metadata": prediction.extra_metadata,
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
            "calibration_class": metadata["calibration_class"],
            "intrinsics_source": metadata["intrinsics_source"],
            "extrinsics_source": metadata["extrinsics_source"],
            "pose_source": metadata["pose_source"],
            "scale_source": metadata["scale_source"],
            "gravity_floor_source": metadata["gravity_floor_source"],
            "not_robot_frame_truth": True,
            "metadata_path": relative_to_root(metadata_path, out_dir),
            "metadata_sha256": file_sha256(metadata_path),
            "artifacts": artifact_records,
        }


def create_da3_teacher(
    *,
    backend_name: str = "real",
    model_id: str = DA3_DEFAULT_MODEL_ID,
    model_dir: str | Path | None = None,
    device: str | None = None,
    max_frames: int | None = None,
    window_size: int | None = None,
    stride: int = 1,
) -> DA3Teacher:
    if backend_name == "fake":
        return DA3Teacher(
            FakeDA3Backend(model_id=model_id),
            model_id=model_id,
            model_dir=model_dir,
            max_frames=max_frames,
            stride=stride,
            window_size=window_size,
        )
    if backend_name == "real":
        return DA3Teacher(
            RealDA3Backend(model_id=model_id, model_dir=model_dir, device=device, window_size=window_size),
            model_id=model_id,
            model_dir=model_dir,
            max_frames=max_frames,
            stride=stride,
            window_size=window_size,
        )
    raise ValueError(f"unknown DA3 backend {backend_name!r}; expected 'real' or 'fake'")


def run_da3_teacher(
    log_dir: str | Path,
    out_dir: str | Path,
    *,
    backend_name: str = "real",
    model_id: str = DA3_DEFAULT_MODEL_ID,
    model_dir: str | Path | None = None,
    device: str | None = None,
    max_frames: int | None = None,
    window_size: int | None = None,
    stride: int = 1,
) -> TeacherRunSummary:
    teacher = create_da3_teacher(
        backend_name=backend_name,
        model_id=model_id,
        model_dir=model_dir,
        device=device,
        max_frames=max_frames,
        window_size=window_size,
        stride=stride,
    )
    return teacher.run(TeacherRunConfig(log_dir=Path(log_dir), out_dir=Path(out_dir)))


def _prediction_arrays(prediction: DA3Prediction, frame: FrameEvent) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    if prediction.depth is not None:
        arrays["depth"] = _same_shape_float32(prediction.depth, frame, "depth")
    if prediction.confidence is not None:
        confidence = _same_shape_float32(prediction.confidence, frame, "confidence")
        arrays["confidence"] = np.clip(confidence, np.float32(0.0), np.float32(1.0)).astype(np.float32)
    if prediction.intrinsics is not None:
        intrinsics = _matrix_float32(prediction.intrinsics, "intrinsics")
        if intrinsics.shape == (3, 3):
            arrays["intrinsics"] = intrinsics
    if prediction.extrinsics is not None:
        extrinsics = _matrix_float32(prediction.extrinsics, "extrinsics")
        if extrinsics.shape == (4, 4):
            extrinsics = extrinsics[:3, :]
        if extrinsics.shape == (3, 4):
            arrays["extrinsics"] = extrinsics
    return arrays


def _split_da3_prediction(raw_prediction: Any, count: int) -> list[DA3Prediction]:
    depth = _optional_prediction_array(raw_prediction, ("depth", "depths"))
    confidence = _optional_prediction_array(raw_prediction, ("conf", "confidence", "confidence_map"))
    intrinsics = _optional_prediction_array(raw_prediction, ("intrinsics", "K"))
    extrinsics = _optional_prediction_array(raw_prediction, ("extrinsics", "camera_pose", "poses"))
    keys = _prediction_keys(raw_prediction)

    predictions: list[DA3Prediction] = []
    for index in range(count):
        predictions.append(
            DA3Prediction(
                depth=_select_index(depth, index, count),
                confidence=_select_index(confidence, index, count),
                intrinsics=_select_index(intrinsics, index, count),
                extrinsics=_select_index(extrinsics, index, count),
                extra_metadata={
                    "backend": "real",
                    "da3_prediction_keys": keys,
                },
            )
        )
    return predictions


def _optional_prediction_array(raw_prediction: Any, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        value = getattr(raw_prediction, name, None)
        if value is None and isinstance(raw_prediction, dict):
            value = raw_prediction.get(name)
        if value is not None:
            return _to_numpy(value)
    return None


def _prediction_keys(raw_prediction: Any) -> list[str]:
    if isinstance(raw_prediction, dict):
        return sorted(str(key) for key in raw_prediction.keys())
    keys = []
    for name in ("processed_images", "depth", "conf", "confidence", "extrinsics", "intrinsics"):
        if hasattr(raw_prediction, name):
            keys.append(name)
    return keys


def _select_index(array: np.ndarray | None, index: int, count: int) -> np.ndarray | None:
    if array is None:
        return None
    values = np.asarray(array)
    if values.ndim >= 3 and values.shape[0] == count:
        return values[index]
    if values.ndim >= 2 and count == 1:
        return values
    if values.ndim >= 1 and values.shape[0] == count:
        return values[index]
    return values


def _same_shape_float32(array: np.ndarray, frame: FrameEvent, kind: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"{kind} must be 2D after squeeze, got shape {values.shape}")
    expected = (frame.height, frame.width)
    if values.shape == expected:
        return values.astype(np.float32)
    return _resize_nearest(values, expected).astype(np.float32)


def _matrix_float32(array: np.ndarray, kind: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"{kind} must be a 2D matrix after squeeze, got shape {values.shape}")
    return values.astype(np.float32)


def _resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out_h, out_w = shape
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"invalid resize target shape {shape}")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"cannot resize empty array with shape {array.shape}")
    row_index = np.linspace(0, array.shape[0] - 1, out_h).round().astype(np.int64)
    col_index = np.linspace(0, array.shape[1] - 1, out_w).round().astype(np.int64)
    return array[row_index[:, None], col_index[None, :]]


def _frame_image_path(log_dir: Path, frame: FrameEvent) -> Path:
    path = log_dir / frame.data_ref
    if not path.exists():
        raise FileNotFoundError(f"frame data_ref is missing: {path}")
    if frame.format not in {"encoded_jpeg", "encoded_png", "encoded_pgm", "encoded_ppm"}:
        raise DA3UnavailableError(
            f"real DA3 backend requires encoded image frames; frame {frame.frame_id} has {frame.format!r}"
        )
    return path


def _frame_luminance(path: Path, frame: FrameEvent) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"frame data_ref is missing: {path}")
    raw = path.read_bytes()
    pixel_count = frame.width * frame.height
    if pixel_count <= 0:
        raise ValueError(f"frame has invalid dimensions: {frame.width}x{frame.height}")
    if frame.format in {"rgb8", "bgr8", "gray8"}:
        return _raw_luminance(raw, frame)
    digest = np.frombuffer(raw or b"\x00", dtype=np.uint8).astype(np.float32)
    tiled = np.resize(digest, pixel_count).reshape(frame.height, frame.width)
    return tiled / np.float32(255.0)


def _raw_luminance(raw: bytes, frame: FrameEvent) -> np.ndarray:
    pixel_count = frame.width * frame.height
    if frame.format in {"rgb8", "bgr8"}:
        expected_bytes = pixel_count * 3
        if len(raw) != expected_bytes:
            raise ValueError(f"{frame.format} frame expected {expected_bytes} bytes, got {len(raw)}")
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        return pixels.astype(np.float32).mean(axis=2) / np.float32(255.0)
    if len(raw) != pixel_count:
        raise ValueError(f"gray8 frame expected {pixel_count} bytes, got {len(raw)}")
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(frame.height, frame.width)
    return pixels.astype(np.float32) / np.float32(255.0)


def _resolve_model_source(model_dir: Path | None, model_id: str) -> str:
    if model_dir is not None:
        if not model_dir.exists():
            raise DA3UnavailableError(f"DA3 model directory does not exist: {model_dir}")
        return str(model_dir)
    setup_model_dir = _setup_model_dir(model_id)
    if setup_model_dir is not None:
        return str(setup_model_dir)
    raise DA3UnavailableError(
        "No local DA3 model directory was supplied or found in external/da3_setup_status.json. "
        "Run setup with `--download` before real DA3 inference."
    )


def _setup_model_dir(model_id: str) -> Path | None:
    status_path = Path.cwd() / DA3_SETUP_STATUS_PATH
    if not status_path.exists():
        return None
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    model = data.get("model")
    if not isinstance(model, dict):
        return None
    status_model_id = model.get("model_id")
    local_dir = model.get("local_dir")
    if status_model_id != model_id or not isinstance(local_dir, str):
        return None
    path = Path(local_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.exists():
        return path
    return None


def _setup_repo_commit() -> str | None:
    status_path = Path.cwd() / DA3_SETUP_STATUS_PATH
    if not status_path.exists():
        return None
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    repo = data.get("repo")
    if isinstance(repo, dict) and isinstance(repo.get("commit"), str):
        return str(repo["commit"])
    return None


def _model_source_text(model_dir: Path | None, model_id: str) -> str:
    if model_dir is not None:
        return model_dir.as_posix()
    setup_model_dir = _setup_model_dir(model_id)
    if setup_model_dir is not None:
        return setup_model_dir.as_posix()
    return "not_resolved"


def _batches(frames: list[FrameEvent], size: int) -> list[list[FrameEvent]]:
    return [frames[index : index + size] for index in range(0, len(frames), size)]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _created_at(deterministic: bool) -> str:
    if deterministic:
        return "1970-01-01T00:00:00Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _python_executable() -> str:
    return sys.executable
