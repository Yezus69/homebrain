from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Protocol

import numpy as np

from homebrain.messages.schema import FrameEvent, JsonDict
from homebrain.replay.replayd import order_events_for_replay
from homebrain.replay.segment_log import load_manifest, read_events
from homebrain.teachers.artifacts import (
    DINO_ARTIFACT_KINDS,
    TEACHER_ARTIFACT_SCHEMA_VERSION,
    file_sha256,
    relative_to_root,
    save_array,
    write_json,
    write_teacher_manifest,
)
from homebrain.teachers.base import Teacher, TeacherRunConfig, TeacherRunSummary

DINO_DEFAULT_MODEL_ID = "dinov2_vits14"
DINO_REPO = "facebookresearch/dinov2"
DINO_GITHUB_URL = "https://github.com/facebookresearch/dinov2"
DINO_LICENSE_REVIEW_STATUS = "pending_human_review"
DINO_TEACHER_VERSION = "dino.teacher.v0"
DINO_FAKE_FEATURE_SHAPE = (4, 4, 32)
DINO_IMAGE_SIZE = 224


class DINOUnavailableError(RuntimeError):
    """Raised when real DINO feature extraction cannot be initialized."""


@dataclass(frozen=True)
class DINOPrediction:
    patch_features: np.ndarray
    cls_feature: np.ndarray
    extra_metadata: JsonDict


class DINOBackend(Protocol):
    name: str
    mocked: bool
    synthetic: bool
    real_perception: bool
    deterministic: bool

    def infer(self, log_dir: Path, frames: list[FrameEvent]) -> list[DINOPrediction]:
        """Return DINO feature predictions for ordered frames."""

    def dependency_status(self) -> JsonDict:
        """Return dependency/model-load status for manifests."""


class FakeDINOBackend:
    name = "fake"
    mocked = True
    synthetic = True
    real_perception = False
    deterministic = True

    def __init__(self, *, model_id: str = DINO_DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id

    def dependency_status(self) -> JsonDict:
        return {
            "backend": self.name,
            "dino_package": "not_required_for_fake_backend",
            "model_loaded": False,
            "model_id": self.model_id,
            "weights_status": "not_required_for_fake_backend",
            "downloads_attempted_by_homebrain": False,
            "warning": "test backend; not real DINO inference",
        }

    def infer(self, log_dir: Path, frames: list[FrameEvent]) -> list[DINOPrediction]:
        predictions: list[DINOPrediction] = []
        patch_h, patch_w, feature_dim = DINO_FAKE_FEATURE_SHAPE
        row, col = np.indices((patch_h, patch_w), dtype=np.float32)
        row_norm = row / np.float32(max(patch_h - 1, 1))
        col_norm = col / np.float32(max(patch_w - 1, 1))
        channel = np.arange(feature_dim, dtype=np.float32).reshape(1, 1, feature_dim)
        for index, frame in enumerate(frames):
            luminance = _frame_luminance(log_dir / frame.data_ref, frame)
            intensity = np.float32(np.mean(luminance))
            frame_bias = np.float32((frame.frame_id % 97) / 97.0)
            patch_features = (
                np.float32(0.25) * row_norm[..., None]
                + np.float32(0.25) * col_norm[..., None]
                + np.float32(0.35) * intensity
                + np.float32(0.10) * frame_bias
                + np.float32(0.05) * np.sin(channel / np.float32(3.0))
            ).astype(np.float32)
            cls_feature = patch_features.mean(axis=(0, 1)).astype(np.float32)
            cls_feature = (cls_feature + np.float32(index) * np.float32(0.001)).astype(np.float32)
            predictions.append(
                DINOPrediction(
                    patch_features=patch_features,
                    cls_feature=cls_feature,
                    extra_metadata={
                        "backend": self.name,
                        "warning": "fake deterministic test backend; not real DINO inference",
                    },
                )
            )
        return predictions


class RealDINOBackend:
    name = "real"
    mocked = False
    synthetic = False
    real_perception = True
    deterministic = False

    def __init__(
        self,
        *,
        model_id: str = DINO_DEFAULT_MODEL_ID,
        model_dir: str | Path | None = None,
        device: str | None = None,
        image_size: int = DINO_IMAGE_SIZE,
    ) -> None:
        self.model_id = model_id
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.device = device
        self.image_size = image_size
        self._model: Any | None = None
        self._torch: Any | None = None

    def dependency_status(self) -> JsonDict:
        return {
            "backend": self.name,
            "torch_package": "loaded" if self._torch is not None else "not_loaded",
            "model_loaded": self._model is not None,
            "model_id": self.model_id,
            "model_path": self.model_dir.as_posix() if self.model_dir is not None else None,
            "model_source": DINO_GITHUB_URL,
            "torch_hub_repo": DINO_REPO,
            "device": self.device or "auto",
            "image_size": self.image_size,
            "downloads_may_occur_for_explicit_real_backend": True,
            "runtime_dependency": False,
            "python_executable": sys.executable,
        }

    def infer(self, log_dir: Path, frames: list[FrameEvent]) -> list[DINOPrediction]:
        self._ensure_loaded()
        assert self._model is not None
        assert self._torch is not None

        predictions: list[DINOPrediction] = []
        for frame in frames:
            image = _load_frame_rgb(log_dir / frame.data_ref, frame)
            tensor = _preprocess_image(image, self._torch, self.image_size).to(self.device)
            with self._torch.no_grad():
                raw = self._model.forward_features(tensor)
            patch_features, cls_feature = _extract_dino_features(raw, self.image_size)
            predictions.append(
                DINOPrediction(
                    patch_features=patch_features,
                    cls_feature=cls_feature,
                    extra_metadata={
                        "backend": self.name,
                        "image_size": self.image_size,
                        "feature_extractor": "torch_hub_forward_features",
                    },
                )
            )
        return predictions

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DINOUnavailableError("PyTorch is required for the real DINO backend.") from exc
        if self.image_size <= 0 or self.image_size % 14 != 0:
            raise DINOUnavailableError("DINO image_size must be a positive multiple of 14")
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = torch.hub.load(DINO_REPO, self.model_id, trust_repo=True)
        except Exception as exc:  # noqa: BLE001 - surface setup/cache failures clearly.
            raise DINOUnavailableError(
                "DINO model could not be loaded through torch.hub. Run "
                "`python -m homebrain.tools.setup_dino_teacher --model-id dinov2_vits14` "
                "or use `--backend fake` for tests."
            ) from exc
        model.eval()
        model.to(self.device)
        self._torch = torch
        self._model = model


class DINOTeacher(Teacher):
    name = "dino"
    version = DINO_TEACHER_VERSION
    mock = False

    def __init__(
        self,
        backend: DINOBackend | None = None,
        *,
        model_id: str = DINO_DEFAULT_MODEL_ID,
        model_dir: str | Path | None = None,
        max_frames: int | None = None,
        stride: int = 1,
        image_size: int = DINO_IMAGE_SIZE,
    ) -> None:
        self.model_id = model_id
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.max_frames = max_frames
        self.stride = stride
        self.image_size = image_size
        self.backend = backend if backend is not None else RealDINOBackend(
            model_id=model_id,
            model_dir=model_dir,
            image_size=image_size,
        )

    def run(self, config: TeacherRunConfig) -> TeacherRunSummary:
        if self.stride < 1:
            raise ValueError("DINO stride must be at least 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("DINO max_frames must be at least 1 when supplied")

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
            self._write_frame_artifacts(out_dir, frame, prediction)
            for frame, prediction in zip(frames, predictions)
        ]
        feature_shape = _manifest_feature_shape(frame_records)

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
            "artifact_kinds": list(DINO_ARTIFACT_KINDS),
            "teacher_name_canonical": "dino",
            "model_id": self.model_id,
            "model_path": self.model_dir.as_posix() if self.model_dir is not None else None,
            "model_source": DINO_GITHUB_URL,
            "feature_shape": feature_shape,
            "license_review_status": DINO_LICENSE_REVIEW_STATUS,
            "runtime_dependency": False,
            "trainable_for": "representation_pretraining_only",
            "control_safe": False,
            "dependency_status": self.backend.dependency_status(),
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
        out_dir: Path,
        frame: FrameEvent,
        prediction: DINOPrediction,
    ) -> JsonDict:
        frame_dir = out_dir / "frames" / f"{frame.camera_id}_{frame.frame_id:06d}"
        patch_features = np.asarray(prediction.patch_features, dtype=np.float32)
        cls_feature = np.asarray(prediction.cls_feature, dtype=np.float32)
        if patch_features.ndim != 3:
            raise ValueError(f"DINO patch_features must be HxWxC, got {patch_features.shape}")
        if cls_feature.ndim != 1:
            raise ValueError(f"DINO cls_feature must be C, got {cls_feature.shape}")
        if patch_features.shape[-1] != cls_feature.shape[0]:
            raise ValueError(
                "DINO patch feature dimension must match cls feature dimension: "
                f"{patch_features.shape[-1]} != {cls_feature.shape[0]}"
            )

        arrays = {
            "patch_features": patch_features,
            "cls_feature": cls_feature,
        }
        artifact_records: dict[str, JsonDict] = {}
        for kind, array in arrays.items():
            target = frame_dir / f"{kind}.npy"
            array_record = save_array(target, array)
            artifact_records[kind] = {
                "kind": kind,
                "path": relative_to_root(target, out_dir),
                **array_record,
            }

        metadata: JsonDict = {
            "schema_version": TEACHER_ARTIFACT_SCHEMA_VERSION,
            "teacher_name": self.name,
            "teacher_version": self.version,
            "backend": self.backend.name,
            "mock": bool(self.backend.mocked),
            "synthetic": bool(self.backend.synthetic),
            "real_perception": bool(self.backend.real_perception),
            "license_review_status": DINO_LICENSE_REVIEW_STATUS,
            "runtime_dependency": False,
            "trainable_for": "representation_pretraining_only",
            "control_safe": False,
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
            "feature_shape": list(patch_features.shape),
            "cls_feature_shape": list(cls_feature.shape),
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
            "metadata_path": relative_to_root(metadata_path, out_dir),
            "metadata_sha256": file_sha256(metadata_path),
            "feature_shape": list(patch_features.shape),
            "cls_feature_shape": list(cls_feature.shape),
            "artifacts": artifact_records,
        }


def create_dino_teacher(
    *,
    backend_name: str = "real",
    model_id: str = DINO_DEFAULT_MODEL_ID,
    model_dir: str | Path | None = None,
    device: str | None = None,
    max_frames: int | None = None,
    stride: int = 1,
    image_size: int = DINO_IMAGE_SIZE,
) -> DINOTeacher:
    if backend_name == "fake":
        return DINOTeacher(
            FakeDINOBackend(model_id=model_id),
            model_id=model_id,
            model_dir=model_dir,
            max_frames=max_frames,
            stride=stride,
            image_size=image_size,
        )
    if backend_name == "real":
        return DINOTeacher(
            RealDINOBackend(model_id=model_id, model_dir=model_dir, device=device, image_size=image_size),
            model_id=model_id,
            model_dir=model_dir,
            max_frames=max_frames,
            stride=stride,
            image_size=image_size,
        )
    raise ValueError(f"unknown DINO backend {backend_name!r}; expected 'real' or 'fake'")


def run_dino_teacher(
    log_dir: str | Path,
    out_dir: str | Path,
    *,
    backend_name: str = "real",
    model_id: str = DINO_DEFAULT_MODEL_ID,
    model_dir: str | Path | None = None,
    device: str | None = None,
    max_frames: int | None = None,
    stride: int = 1,
    image_size: int = DINO_IMAGE_SIZE,
) -> TeacherRunSummary:
    teacher = create_dino_teacher(
        backend_name=backend_name,
        model_id=model_id,
        model_dir=model_dir,
        device=device,
        max_frames=max_frames,
        stride=stride,
        image_size=image_size,
    )
    return teacher.run(TeacherRunConfig(log_dir=Path(log_dir), out_dir=Path(out_dir)))


def _manifest_feature_shape(frame_records: list[JsonDict]) -> list[int]:
    if not frame_records:
        return []
    shape = frame_records[0].get("feature_shape")
    return list(shape) if isinstance(shape, list) else []


def _extract_dino_features(raw: Any, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(raw, dict):
        raise DINOUnavailableError("DINO forward_features did not return a feature dictionary")
    patch_tokens = raw.get("x_norm_patchtokens")
    cls_token = raw.get("x_norm_clstoken")
    if patch_tokens is None or cls_token is None:
        raise DINOUnavailableError("DINO features are missing patch or cls tokens")
    patch_np = _to_numpy(patch_tokens)[0].astype(np.float32)
    cls_np = _to_numpy(cls_token)[0].astype(np.float32)
    patch_grid = image_size // 14
    expected_tokens = patch_grid * patch_grid
    if patch_np.ndim != 2 or patch_np.shape[0] != expected_tokens:
        raise DINOUnavailableError(
            f"unexpected DINO patch token shape {patch_np.shape}; expected {expected_tokens} tokens"
        )
    patch_features = patch_np.reshape(patch_grid, patch_grid, patch_np.shape[-1]).astype(np.float32)
    return patch_features, cls_np.astype(np.float32)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _preprocess_image(image: np.ndarray, torch: Any, image_size: int) -> Any:
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DINOUnavailableError("Pillow is required to resize frames for real DINO inference.") from exc

    resampling = getattr(Image, "Resampling", Image).BICUBIC
    pil = Image.fromarray(image, mode="RGB").resize((image_size, image_size), resampling)
    array = np.asarray(pil, dtype=np.float32) / np.float32(255.0)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    normalized = (array - mean) / std
    chw = np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)
    return torch.from_numpy(chw)


def _load_frame_rgb(path: Path, frame: FrameEvent) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"frame data_ref is missing: {path}")
    raw = path.read_bytes()
    if frame.format in {"rgb8", "bgr8"}:
        expected = frame.width * frame.height * 3
        if len(raw) != expected:
            raise ValueError(f"{frame.format} frame expected {expected} bytes, got {len(raw)}")
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        if frame.format == "bgr8":
            pixels = pixels[..., ::-1]
        return pixels.copy()
    if frame.format == "gray8":
        expected = frame.width * frame.height
        if len(raw) != expected:
            raise ValueError(f"gray8 frame expected {expected} bytes, got {len(raw)}")
        gray = np.frombuffer(raw, dtype=np.uint8).reshape(frame.height, frame.width)
        return np.repeat(gray[..., None], 3, axis=2)
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DINOUnavailableError("Pillow is required to load encoded frames for real DINO inference.") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _frame_luminance(path: Path, frame: FrameEvent) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"frame data_ref is missing: {path}")
    raw = path.read_bytes()
    pixel_count = frame.width * frame.height
    if pixel_count <= 0:
        raise ValueError(f"frame has invalid dimensions: {frame.width}x{frame.height}")
    if frame.format in {"rgb8", "bgr8"}:
        expected = pixel_count * 3
        if len(raw) != expected:
            raise ValueError(f"{frame.format} frame expected {expected} bytes, got {len(raw)}")
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        return pixels.astype(np.float32).mean(axis=2) / np.float32(255.0)
    if frame.format == "gray8":
        if len(raw) != pixel_count:
            raise ValueError(f"gray8 frame expected {pixel_count} bytes, got {len(raw)}")
        return np.frombuffer(raw, dtype=np.uint8).reshape(frame.height, frame.width).astype(np.float32) / np.float32(255.0)
    digest = np.frombuffer(raw or b"\x00", dtype=np.uint8).astype(np.float32)
    tiled = np.resize(digest, pixel_count).reshape(frame.height, frame.width)
    return tiled / np.float32(255.0)


def _created_at(deterministic: bool) -> str:
    if deterministic:
        return "1970-01-01T00:00:00Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
