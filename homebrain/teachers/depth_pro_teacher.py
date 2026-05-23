from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from homebrain.messages.schema import FrameEvent, JsonDict
from homebrain.replay.replayd import order_events_for_replay
from homebrain.replay.segment_log import load_manifest, read_events
from homebrain.teachers.artifacts import (
    BEV_PREVIEW_SHAPE,
    DEPTH_PRO_ARTIFACT_KINDS,
    TEACHER_ARTIFACT_SCHEMA_VERSION,
    expected_shape,
    file_sha256,
    relative_to_root,
    save_array,
    write_json,
    write_teacher_manifest,
)
from homebrain.teachers.base import Teacher, TeacherRunConfig, TeacherRunSummary

DEPTH_PRO_LICENSE_REVIEW_STATUS = "pending_human_review"
DEPTH_PRO_GITHUB_URL = "https://github.com/apple/ml-depth-pro"
DEPTH_PRO_LICENSE_URL = "https://github.com/apple/ml-depth-pro/blob/main/LICENSE"


class DepthProUnavailableError(RuntimeError):
    """Raised when real Depth Pro inference cannot be initialized."""


@dataclass(frozen=True)
class DepthProPrediction:
    depth_m: np.ndarray
    focallength_px: float | None = None
    confidence: np.ndarray | None = None
    extra_metadata: JsonDict | None = None


class DepthProBackend(Protocol):
    name: str
    mocked: bool
    synthetic: bool
    real_perception: bool
    deterministic: bool

    def infer(self, log_dir: Path, frame: FrameEvent) -> DepthProPrediction:
        """Return one Depth Pro-style prediction for a frame."""

    def dependency_status(self) -> JsonDict:
        """Return dependency and model-load status for the manifest."""


class RealDepthProBackend:
    name = "real"
    mocked = False
    synthetic = False
    real_perception = True
    deterministic = False

    def __init__(self, *, device: str | None = None) -> None:
        self.device = device
        self._checkpoint_uri: str | None = None
        self._depth_pro: Any | None = None
        self._model: Any | None = None
        self._transform: Any | None = None
        self._torch: Any | None = None

    def dependency_status(self) -> JsonDict:
        return {
            "backend": self.name,
            "depth_pro_package": "loaded" if self._depth_pro is not None else "not_loaded",
            "model_loaded": self._model is not None,
            "weights_status": "loaded_from_local_depth_pro_install" if self._model is not None else "not_loaded",
            "checkpoint_uri": self._checkpoint_uri or "depth_pro_default",
            "device": self.device or "depth_pro_default",
            "downloads_attempted_by_homebrain": False,
        }

    def infer(self, log_dir: Path, frame: FrameEvent) -> DepthProPrediction:
        self._ensure_loaded()
        assert self._depth_pro is not None
        assert self._model is not None
        assert self._transform is not None

        image, f_px = self._load_image(log_dir, frame)
        tensor = self._transform(image)
        if self.device is not None and hasattr(tensor, "to"):
            tensor = tensor.to(self.device)
        model_f_px = self._model_focal_length(f_px, tensor)

        context = self._torch.no_grad() if self._torch is not None else nullcontext()
        with context:
            prediction = self._model.infer(tensor, f_px=model_f_px)

        depth = _to_numpy(prediction["depth"])
        focallength = _optional_float_from_prediction(prediction.get("focallength_px"))
        if focallength is None:
            focallength = f_px
        confidence = prediction.get("confidence")
        return DepthProPrediction(
            depth_m=depth,
            focallength_px=focallength,
            confidence=_to_numpy(confidence) if confidence is not None else None,
            extra_metadata={
                "backend": self.name,
                "depth_pro_prediction_keys": sorted(str(key) for key in prediction.keys()),
            },
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import depth_pro  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DepthProUnavailableError(
                "Depth Pro is not installed. Install Apple's ml-depth-pro package and local "
                "checkpoints, or rerun with '--backend fake' for tests only."
            ) from exc

        try:
            checkpoint_uri = _resolve_depth_pro_checkpoint_uri(depth_pro)
            model, transform = _create_depth_pro_model_and_transforms(depth_pro, checkpoint_uri)
        except Exception as exc:  # noqa: BLE001 - missing weights surface here.
            raise DepthProUnavailableError(
                "Depth Pro is installed but the model/checkpoints could not be loaded. "
                "Run the Depth Pro checkpoint setup outside tests, set DEPTH_PRO_CHECKPOINT "
                "if needed, then retry; HomeBrain does not download weights automatically."
            ) from exc

        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            torch = None

        model.eval()
        if self.device is not None and hasattr(model, "to"):
            model = model.to(self.device)
        self._checkpoint_uri = checkpoint_uri
        self._depth_pro = depth_pro
        self._model = model
        self._transform = transform
        self._torch = torch

    def _load_image(self, log_dir: Path, frame: FrameEvent) -> tuple[Any, float | None]:
        assert self._depth_pro is not None
        frame_path = log_dir / frame.data_ref
        if not frame_path.exists():
            raise FileNotFoundError(f"frame data_ref is missing: {frame_path}")

        if frame.format.startswith("encoded_"):
            image, _metadata, f_px = self._depth_pro.load_rgb(str(frame_path))
            return image, _optional_number(f_px)

        try:
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DepthProUnavailableError(
                "Pillow is required to run Depth Pro on raw HomeBrain frame artifacts."
            ) from exc

        rgb = _raw_frame_to_rgb(frame_path, frame)
        return Image.fromarray(rgb, mode="RGB"), _intrinsics_focal_length_px(frame.intrinsics)

    def _model_focal_length(self, value: float | None, tensor: Any) -> Any:
        if value is None or self._torch is None:
            return value
        if hasattr(value, "to"):
            return value.to(
                device=getattr(tensor, "device", None),
                dtype=getattr(tensor, "dtype", None),
            )
        return self._torch.as_tensor(
            value,
            device=getattr(tensor, "device", None),
            dtype=getattr(tensor, "dtype", None),
        )


class FakeDepthProBackend:
    name = "fake"
    mocked = True
    synthetic = True
    real_perception = False
    deterministic = True

    def dependency_status(self) -> JsonDict:
        return {
            "backend": self.name,
            "depth_pro_package": "not_required_for_fake_backend",
            "model_loaded": False,
            "weights_status": "not_required_for_fake_backend",
            "downloads_attempted_by_homebrain": False,
            "warning": "test backend; not real Depth Pro inference",
        }

    def infer(self, log_dir: Path, frame: FrameEvent) -> DepthProPrediction:
        luminance = _frame_luminance(log_dir / frame.data_ref, frame)
        row, col = np.indices((frame.height, frame.width), dtype=np.float32)
        row_norm = row / np.float32(max(frame.height - 1, 1))
        col_norm = col / np.float32(max(frame.width - 1, 1))
        depth = (
            np.float32(0.4)
            + np.float32(1.2) * row_norm
            + np.float32(0.5) * col_norm
            + np.float32(0.3) * luminance
            + np.float32(frame.frame_id) * np.float32(0.01)
        ).astype(np.float32)
        focal = _intrinsics_focal_length_px(frame.intrinsics)
        if focal is None:
            focal = float(max(frame.width, frame.height))
        return DepthProPrediction(
            depth_m=depth,
            focallength_px=focal,
            confidence=None,
            extra_metadata={
                "backend": self.name,
                "warning": "fake deterministic test backend; not real Depth Pro inference",
            },
        )


class DepthProTeacher(Teacher):
    name = "depth_pro"
    version = "depth_pro.teacher.v0"
    mock = False

    def __init__(self, backend: DepthProBackend | None = None) -> None:
        self.backend = backend if backend is not None else RealDepthProBackend()

    def run(self, config: TeacherRunConfig) -> TeacherRunSummary:
        log_dir = Path(config.log_dir)
        out_dir = Path(config.out_dir)
        source_manifest = load_manifest(log_dir)
        events = order_events_for_replay(read_events(log_dir))
        frames = [event for event in events if isinstance(event, FrameEvent)]

        frame_records: list[JsonDict] = []
        for frame in frames:
            frame_records.append(self._write_frame_artifacts(log_dir, out_dir, frame))

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
            "source_log": str(log_dir.as_posix()),
            "source_segment_id": source_manifest.segment_id,
            "source_schema_version": source_manifest.schema_version,
            "frame_count": len(frame_records),
            "artifact_kinds": list(DEPTH_PRO_ARTIFACT_KINDS),
            "license_review_status": DEPTH_PRO_LICENSE_REVIEW_STATUS,
            "license_source": DEPTH_PRO_LICENSE_URL,
            "model_source": DEPTH_PRO_GITHUB_URL,
            "dependency_status": self.backend.dependency_status(),
            "control_safety": "not_control_safe_training_teacher_only",
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

    def _write_frame_artifacts(self, log_dir: Path, out_dir: Path, frame: FrameEvent) -> JsonDict:
        frame_dir = out_dir / "frames" / f"{frame.camera_id}_{frame.frame_id:06d}"
        prediction = self.backend.infer(log_dir, frame)
        depth = _depth_array(prediction.depth_m, frame)

        confidence_source = "model_output"
        if prediction.confidence is None:
            confidence = _heuristic_confidence(depth)
            confidence_source = "homebrain_heuristic_from_depth_finiteness_and_edges"
        else:
            confidence = _same_shape_float32(prediction.confidence, frame, "depth_confidence")
            confidence = np.clip(confidence, np.float32(0.0), np.float32(1.0)).astype(np.float32)

        focallength, focal_source = _focallength_value(prediction.focallength_px, frame)
        bev_preview = _rough_bev_preview(depth)
        arrays: dict[str, np.ndarray] = {
            "depth_m": depth,
            "depth_confidence": confidence,
            "focallength_px": np.asarray([focallength], dtype=np.float32),
            "bev_preview": bev_preview,
        }

        artifact_records: dict[str, JsonDict] = {}
        for kind in DEPTH_PRO_ARTIFACT_KINDS:
            array = arrays[kind]
            expected = expected_shape(kind, frame.width, frame.height)
            if tuple(array.shape) != expected:
                raise ValueError(f"{kind} Depth Pro artifact has wrong shape {array.shape}; expected {expected}")
            target = frame_dir / f"{kind}.npy"
            array_record = save_array(target, array)
            artifact_records[kind] = {
                "kind": kind,
                "path": relative_to_root(target, out_dir),
                **array_record,
            }

        extra_metadata = prediction.extra_metadata or {}
        metadata: JsonDict = {
            "schema_version": TEACHER_ARTIFACT_SCHEMA_VERSION,
            "teacher_name": self.name,
            "teacher_version": self.version,
            "backend": self.backend.name,
            "mock": bool(self.backend.mocked),
            "synthetic": bool(self.backend.synthetic),
            "real_perception": bool(self.backend.real_perception),
            "license_review_status": DEPTH_PRO_LICENSE_REVIEW_STATUS,
            "sequence_id": frame.sequence_id,
            "camera_id": frame.camera_id,
            "frame_id": frame.frame_id,
            "timestamp_ns": frame.timestamp_ns,
            "width": frame.width,
            "height": frame.height,
            "format": frame.format,
            "source_data_ref": frame.data_ref,
            "depth_units": "meters",
            "depth_confidence_source": confidence_source,
            "focallength_px_source": focal_source,
            "bev_preview_note": "visualization_only_rough_preview_not_navigation_or_control_label",
            "control_safety": "not_control_safe_training_teacher_only",
            "dependency_status": self.backend.dependency_status(),
            "backend_metadata": extra_metadata,
            "artifact_shapes": {
                kind: artifact_records[kind]["shape"] for kind in DEPTH_PRO_ARTIFACT_KINDS
            },
            "artifact_dtypes": {
                kind: artifact_records[kind]["dtype"] for kind in DEPTH_PRO_ARTIFACT_KINDS
            },
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
            "metadata_path": relative_to_root(metadata_path, out_dir),
            "metadata_sha256": file_sha256(metadata_path),
            "artifacts": artifact_records,
        }


def create_depth_pro_teacher(*, backend_name: str = "real", device: str | None = None) -> DepthProTeacher:
    if backend_name == "real":
        return DepthProTeacher(RealDepthProBackend(device=device))
    if backend_name == "fake":
        return DepthProTeacher(FakeDepthProBackend())
    raise ValueError(f"unknown Depth Pro backend {backend_name!r}; expected 'real' or 'fake'")


def run_depth_pro_teacher(
    log_dir: str | Path,
    out_dir: str | Path,
    *,
    backend_name: str = "real",
    device: str | None = None,
) -> TeacherRunSummary:
    teacher = create_depth_pro_teacher(backend_name=backend_name, device=device)
    return teacher.run(TeacherRunConfig(log_dir=Path(log_dir), out_dir=Path(out_dir)))


def _create_depth_pro_model_and_transforms(depth_pro: Any, checkpoint_uri: str | None) -> tuple[Any, Any]:
    if checkpoint_uri is None:
        return depth_pro.create_model_and_transforms()

    depth_pro_config_module = importlib.import_module("depth_pro.depth_pro")
    config = replace(
        depth_pro_config_module.DEFAULT_MONODEPTH_CONFIG_DICT,
        checkpoint_uri=checkpoint_uri,
    )
    return depth_pro.create_model_and_transforms(config=config)


def _resolve_depth_pro_checkpoint_uri(depth_pro: Any) -> str | None:
    env_checkpoint = os.environ.get("DEPTH_PRO_CHECKPOINT")
    if env_checkpoint:
        return env_checkpoint

    module_file = getattr(depth_pro, "__file__", None)
    if isinstance(module_file, str):
        module_path = Path(module_file).resolve()
        for parent in module_path.parents:
            candidate = parent / "checkpoints" / "depth_pro.pt"
            if candidate.exists():
                return str(candidate)

    cwd_candidate = Path.cwd() / "checkpoints" / "depth_pro.pt"
    if cwd_candidate.exists():
        return str(cwd_candidate)

    return None


def _created_at(deterministic: bool) -> str:
    if deterministic:
        return "1970-01-01T00:00:00Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _optional_float_from_prediction(value: Any) -> float | None:
    if value is None:
        return None
    return _optional_number(_to_numpy(value))


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    number = float(array.reshape(-1)[0])
    if not np.isfinite(number):
        return None
    return number


def _depth_array(depth: np.ndarray, frame: FrameEvent) -> np.ndarray:
    return _same_shape_float32(depth, frame, "depth_m")


def _same_shape_float32(array: np.ndarray, frame: FrameEvent, kind: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"{kind} must be 2D after squeeze, got shape {values.shape}")
    expected = (frame.height, frame.width)
    if values.shape == expected:
        return values.astype(np.float32)
    return _resize_nearest(values, expected).astype(np.float32)


def _resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out_h, out_w = shape
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"invalid resize target shape {shape}")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"cannot resize empty array with shape {array.shape}")
    row_index = np.linspace(0, array.shape[0] - 1, out_h).round().astype(np.int64)
    col_index = np.linspace(0, array.shape[1] - 1, out_w).round().astype(np.int64)
    return array[row_index[:, None], col_index[None, :]]


def _heuristic_confidence(depth: np.ndarray) -> np.ndarray:
    finite_positive = np.isfinite(depth) & (depth > np.float32(0.0))
    if not np.any(finite_positive):
        return np.zeros(depth.shape, dtype=np.float32)

    safe_depth = np.where(finite_positive, depth, np.nan)
    median = float(np.nanmedian(safe_depth))
    scale = max(median, 1.0e-3)
    dy = np.zeros(depth.shape, dtype=np.float32)
    dx = np.zeros(depth.shape, dtype=np.float32)
    dy[1:, :] = np.abs(np.diff(np.where(finite_positive, depth, median), axis=0))
    dx[:, 1:] = np.abs(np.diff(np.where(finite_positive, depth, median), axis=1))
    edge_penalty = np.clip((dx + dy) / np.float32(scale), np.float32(0.0), np.float32(1.0))
    confidence = np.where(
        finite_positive,
        np.float32(1.0) - np.float32(0.5) * edge_penalty,
        np.float32(0.0),
    )
    return np.clip(confidence, np.float32(0.0), np.float32(1.0)).astype(np.float32)


def _focallength_value(value: float | None, frame: FrameEvent) -> tuple[float, str]:
    if value is not None and np.isfinite(value) and value > 0.0:
        return float(value), "depth_pro_prediction"
    intrinsics_value = _intrinsics_focal_length_px(frame.intrinsics)
    if intrinsics_value is not None:
        return intrinsics_value, "frame_intrinsics"
    return float(max(frame.width, frame.height)), "heuristic_max_image_dimension"


def _intrinsics_focal_length_px(intrinsics: JsonDict | None) -> float | None:
    if not isinstance(intrinsics, dict):
        return None
    for key in ("focallength_px", "focal_length_px", "f_px", "fx"):
        value = _optional_number(intrinsics.get(key))
        if value is not None and value > 0.0:
            return value
    fy = _optional_number(intrinsics.get("fy"))
    if fy is not None and fy > 0.0:
        return fy
    camera_matrix = intrinsics.get("camera_matrix")
    if isinstance(camera_matrix, list) and len(camera_matrix) >= 1:
        first_row = camera_matrix[0]
        if isinstance(first_row, list) and first_row:
            value = _optional_number(first_row[0])
            if value is not None and value > 0.0:
                return value
    return None


def _rough_bev_preview(depth: np.ndarray) -> np.ndarray:
    sampled = _resize_nearest(depth, BEV_PREVIEW_SHAPE)
    valid = sampled[np.isfinite(sampled) & (sampled > np.float32(0.0))]
    if valid.size == 0:
        return np.zeros(BEV_PREVIEW_SHAPE, dtype=np.float32)
    near = float(np.percentile(valid, 5))
    far = float(np.percentile(valid, 95))
    if far <= near:
        normalized = np.zeros(sampled.shape, dtype=np.float32)
    else:
        normalized = (sampled.astype(np.float32) - np.float32(near)) / np.float32(far - near)
    inverse_depth_preview = np.float32(1.0) - np.clip(normalized, np.float32(0.0), np.float32(1.0))
    return np.where(np.isfinite(sampled), inverse_depth_preview, np.float32(0.0)).astype(np.float32)


def _raw_frame_to_rgb(path: Path, frame: FrameEvent) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"frame data_ref is missing: {path}")
    raw = path.read_bytes()
    pixel_count = frame.width * frame.height
    if pixel_count <= 0:
        raise ValueError(f"frame has invalid dimensions: {frame.width}x{frame.height}")

    if frame.format in {"rgb8", "bgr8"}:
        expected_bytes = pixel_count * 3
        if len(raw) != expected_bytes:
            raise ValueError(f"{frame.format} frame expected {expected_bytes} bytes, got {len(raw)}")
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        if frame.format == "bgr8":
            pixels = pixels[:, :, ::-1]
        return pixels
    if frame.format == "gray8":
        if len(raw) != pixel_count:
            raise ValueError(f"gray8 frame expected {pixel_count} bytes, got {len(raw)}")
        gray = np.frombuffer(raw, dtype=np.uint8).reshape(frame.height, frame.width)
        return np.repeat(gray[:, :, None], 3, axis=2)
    raise ValueError(f"raw RGB conversion does not support frame format {frame.format!r}")


def _frame_luminance(path: Path, frame: FrameEvent) -> np.ndarray:
    if frame.format in {"rgb8", "bgr8", "gray8"}:
        rgb = _raw_frame_to_rgb(path, frame).astype(np.float32)
        return rgb.mean(axis=2) / np.float32(255.0)
    if not path.exists():
        raise FileNotFoundError(f"frame data_ref is missing: {path}")
    raw = path.read_bytes()
    pixel_count = frame.width * frame.height
    digest = np.frombuffer(raw or b"\x00", dtype=np.uint8).astype(np.float32)
    tiled = np.resize(digest, pixel_count).reshape(frame.height, frame.width)
    return tiled / np.float32(255.0)
