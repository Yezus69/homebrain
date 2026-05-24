from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from homebrain.messages.schema import FrameEvent, JsonDict
from homebrain.teachers.scene_teacher import (
    SceneFramePrediction,
    SceneTeacher,
    SceneTeacherBackend,
    SceneTeacherBatchPrediction,
    SceneTeacherRunConfig,
    SceneTeacherRunSummary,
    SceneWindowPrediction,
    write_scene_teacher_pack,
)

VGGT_DEFAULT_MODEL_ID = "vggt-local"
VGGT_LICENSE_REVIEW_STATUS = "pending_human_review"
VGGT_MODEL_SOURCE = "local external/vggt install or operator-provided adapter"


class VGGTSceneTeacherUnavailableError(RuntimeError):
    """Raised when real VGGT-style scene inference cannot be initialized."""


@dataclass(frozen=True)
class ExternalVGGTResult:
    frames: tuple[SceneFramePrediction, ...]
    windows: tuple[SceneWindowPrediction, ...]


class FakeVGGTSceneBackend:
    name = "fake"
    mocked = True
    synthetic = True
    real_perception = False
    deterministic = True
    model_source = VGGT_MODEL_SOURCE
    license_review_status = VGGT_LICENSE_REVIEW_STATUS
    scale_status = "synthetic_metric_test_scale"

    def __init__(self, *, model_id: str = VGGT_DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id

    def dependency_status(self) -> JsonDict:
        return {
            "backend": self.name,
            "vggt_package": "not_required_for_fake_backend",
            "model_loaded": False,
            "model_id": self.model_id,
            "weights_status": "not_required_for_fake_backend",
            "downloads_attempted_by_homebrain": False,
            "warning": "test backend; not real VGGT inference",
        }

    def infer(self, log_dir: Path, frames: list[FrameEvent]) -> SceneTeacherBatchPrediction:
        predictions = tuple(self._infer_frame(log_dir, frame, index) for index, frame in enumerate(frames))
        windows: tuple[SceneWindowPrediction, ...] = ()
        if frames:
            windows = (self._tracks_window(frames),)
        return SceneTeacherBatchPrediction(frames=predictions, windows=windows)

    def _infer_frame(self, log_dir: Path, frame: FrameEvent, index: int) -> SceneFramePrediction:
        luminance = _frame_luminance(log_dir / frame.data_ref, frame)
        row, col = np.indices((frame.height, frame.width), dtype=np.float32)
        row_norm = row / np.float32(max(frame.height - 1, 1))
        col_norm = col / np.float32(max(frame.width - 1, 1))
        depth = (
            np.float32(0.35)
            + np.float32(0.70) * row_norm
            + np.float32(0.25) * col_norm
            + np.float32(0.10) * luminance
            + np.float32(index) * np.float32(0.02)
        ).astype(np.float32)
        focal = np.float32(max(frame.width, frame.height))
        cx = np.float32((frame.width - 1) / 2.0)
        cy = np.float32((frame.height - 1) / 2.0)
        x = ((col - cx) / focal * depth).astype(np.float32)
        y = ((row - cy) / focal * depth).astype(np.float32)
        point_map = np.stack([x, y, depth], axis=-1).astype(np.float32)
        confidence = np.clip(
            np.float32(0.90)
            - np.float32(0.10) * np.abs(luminance - np.float32(0.5))
            - np.float32(0.03) * row_norm,
            np.float32(0.0),
            np.float32(1.0),
        ).astype(np.float32)
        intrinsics = np.asarray(
            [
                [focal, np.float32(0.0), cx],
                [np.float32(0.0), focal, cy],
                [np.float32(0.0), np.float32(0.0), np.float32(1.0)],
            ],
            dtype=np.float32,
        )
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[0, 3] = np.float32(index) * np.float32(0.02)
        floor = (depth >= np.float32(np.median(depth))).astype(np.uint8)
        obstacle = (depth <= np.float32(np.percentile(depth, 15))).astype(np.uint8)
        return SceneFramePrediction(
            depth=depth,
            point_map=point_map,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            confidence=confidence,
            validity_mask=np.ones((frame.height, frame.width), dtype=np.uint8),
            floor_traversable_mask=floor,
            obstacle_risk_mask=obstacle,
            dynamic_motion_mask=np.zeros((frame.height, frame.width), dtype=np.uint8),
            extra_metadata={
                "backend": self.name,
                "warning": "fake deterministic test backend; not real VGGT inference",
            },
        )

    def _tracks_window(self, frames: list[FrameEvent]) -> SceneWindowPrediction:
        first = frames[0]
        track_count = min(4, max(first.width * first.height, 1))
        base_points: list[tuple[float, float]] = []
        for index in range(track_count):
            row = float(index // max(first.width, 1))
            col = float(index % max(first.width, 1))
            base_points.append((col, row))
        tracks = np.zeros((len(frames), track_count, 2), dtype=np.float32)
        validity = np.ones((len(frames), track_count), dtype=np.uint8)
        for step_index, _frame in enumerate(frames):
            for track_index, (col, row) in enumerate(base_points):
                tracks[step_index, track_index, 0] = np.float32(col + 0.05 * step_index)
                tracks[step_index, track_index, 1] = np.float32(row + 0.02 * step_index)
        return SceneWindowPrediction(
            frame_indices=tuple(range(len(frames))),
            point_tracks=tracks,
            track_validity=validity,
            extra_metadata={"backend": self.name, "track_count": track_count},
        )


class RealVGGTSceneBackend:
    name = "real"
    mocked = False
    synthetic = False
    real_perception = True
    deterministic = False
    model_source = VGGT_MODEL_SOURCE
    license_review_status = VGGT_LICENSE_REVIEW_STATUS
    scale_status = "teacher_relative_not_metric"

    def __init__(
        self,
        *,
        model_id: str = VGGT_DEFAULT_MODEL_ID,
        model_dir: str | Path | None = None,
        checkpoint: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_dir = Path(model_dir) if model_dir is not None else _default_vggt_dir()
        self.checkpoint = Path(checkpoint) if checkpoint is not None else _default_vggt_checkpoint()
        self.device = device
        self._adapter: Any | None = None

    def dependency_status(self) -> JsonDict:
        return {
            "backend": self.name,
            "model_id": self.model_id,
            "model_dir": self.model_dir.as_posix() if self.model_dir is not None else None,
            "model_dir_exists": self.model_dir.exists() if self.model_dir is not None else False,
            "checkpoint": self.checkpoint.as_posix() if self.checkpoint is not None else None,
            "checkpoint_exists": self.checkpoint.exists() if self.checkpoint is not None else False,
            "adapter": os.environ.get("HOMEBRAIN_VGGT_ADAPTER"),
            "adapter_loaded": self._adapter is not None,
            "device": self.device or "auto",
            "downloads_attempted_by_homebrain": False,
            "python_executable": sys.executable,
        }

    def infer(self, log_dir: Path, frames: list[FrameEvent]) -> SceneTeacherBatchPrediction:
        adapter = self._load_adapter()
        try:
            raw = adapter(
                log_dir=log_dir,
                frames=frames,
                model_id=self.model_id,
                model_dir=self.model_dir,
                checkpoint=self.checkpoint,
                device=self.device,
            )
        except Exception as exc:  # noqa: BLE001 - real backend failures should surface with context.
            raise VGGTSceneTeacherUnavailableError(f"real VGGT scene adapter failed: {exc}") from exc
        return _coerce_external_result(raw, len(frames))

    def _load_adapter(self) -> Any:
        if self._adapter is not None:
            return self._adapter
        if self.model_dir is None or not self.model_dir.exists():
            raise VGGTSceneTeacherUnavailableError(
                "Real VGGT backend requires a local external/vggt checkout or --model-dir. "
                "HomeBrain does not clone repositories or download weights during teacher runs."
            )
        if self.checkpoint is None or not self.checkpoint.exists():
            raise VGGTSceneTeacherUnavailableError(
                "Real VGGT backend requires a local checkpoint via --checkpoint or HOMEBRAIN_VGGT_CHECKPOINT. "
                "HomeBrain does not auto-download VGGT weights."
            )
        if str(self.model_dir) not in sys.path:
            sys.path.insert(0, str(self.model_dir))
        adapter_ref = os.environ.get("HOMEBRAIN_VGGT_ADAPTER", "homebrain_vggt_adapter:run_scene_teacher")
        module_name, sep, attr = adapter_ref.partition(":")
        if not sep or not module_name or not attr:
            raise VGGTSceneTeacherUnavailableError(
                "HOMEBRAIN_VGGT_ADAPTER must have form 'module:function' for real VGGT inference."
            )
        try:
            module = importlib.import_module(module_name)
            adapter = getattr(module, attr)
        except Exception as exc:  # noqa: BLE001 - include setup hint.
            raise VGGTSceneTeacherUnavailableError(
                "Local VGGT assets were found, but no HomeBrain-compatible adapter callable was importable. "
                "Set HOMEBRAIN_VGGT_ADAPTER=module:function; the callable must return SceneFramePrediction "
                "objects or dictionaries with depth/point_map/camera fields."
            ) from exc
        self._adapter = adapter
        return adapter


class VGGTSceneTeacher(SceneTeacher):
    name = "vggt"
    version = "vggt_scene_teacher.v0"

    def __init__(
        self,
        backend: SceneTeacherBackend,
        *,
        max_frames: int | None = None,
        stride: int = 1,
    ) -> None:
        self.backend = backend
        self.max_frames = max_frames
        self.stride = stride

    def run(self, config: SceneTeacherRunConfig) -> SceneTeacherRunSummary:
        return write_scene_teacher_pack(
            teacher_name=self.name,
            teacher_version=self.version,
            backend=self.backend,
            log_dir=config.log_dir,
            out_dir=config.out_dir,
            max_frames=self.max_frames,
            stride=self.stride,
        )


def create_vggt_scene_teacher(
    *,
    backend_name: str = "real",
    model_id: str = VGGT_DEFAULT_MODEL_ID,
    model_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    max_frames: int | None = None,
    stride: int = 1,
) -> VGGTSceneTeacher:
    if backend_name == "fake":
        return VGGTSceneTeacher(FakeVGGTSceneBackend(model_id=model_id), max_frames=max_frames, stride=stride)
    if backend_name == "real":
        return VGGTSceneTeacher(
            RealVGGTSceneBackend(
                model_id=model_id,
                model_dir=model_dir,
                checkpoint=checkpoint,
                device=device,
            ),
            max_frames=max_frames,
            stride=stride,
        )
    raise ValueError(f"unknown VGGT scene backend {backend_name!r}; expected 'real' or 'fake'")


def run_vggt_scene_teacher(
    log_dir: str | Path,
    out_dir: str | Path,
    *,
    backend_name: str = "real",
    model_id: str = VGGT_DEFAULT_MODEL_ID,
    model_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    max_frames: int | None = None,
    stride: int = 1,
) -> SceneTeacherRunSummary:
    teacher = create_vggt_scene_teacher(
        backend_name=backend_name,
        model_id=model_id,
        model_dir=model_dir,
        checkpoint=checkpoint,
        device=device,
        max_frames=max_frames,
        stride=stride,
    )
    return teacher.run(SceneTeacherRunConfig(log_dir=Path(log_dir), out_dir=Path(out_dir)))


def _coerce_external_result(raw: Any, frame_count: int) -> SceneTeacherBatchPrediction:
    if isinstance(raw, SceneTeacherBatchPrediction):
        return raw
    if isinstance(raw, ExternalVGGTResult):
        return SceneTeacherBatchPrediction(frames=raw.frames, windows=raw.windows)
    if isinstance(raw, dict):
        frame_items = raw.get("frames")
        window_items = raw.get("windows", [])
    else:
        frame_items = raw
        window_items = []
    if not isinstance(frame_items, (list, tuple)):
        raise VGGTSceneTeacherUnavailableError("VGGT adapter result must include a list of frame predictions")
    frames = tuple(_coerce_frame_prediction(item) for item in frame_items)
    windows = tuple(_coerce_window_prediction(item) for item in window_items) if isinstance(window_items, (list, tuple)) else ()
    if len(frames) != frame_count:
        raise VGGTSceneTeacherUnavailableError(
            f"VGGT adapter returned {len(frames)} frame prediction(s) for {frame_count} frame(s)"
        )
    return SceneTeacherBatchPrediction(frames=frames, windows=windows)


def _coerce_frame_prediction(item: Any) -> SceneFramePrediction:
    if isinstance(item, SceneFramePrediction):
        return item
    if not isinstance(item, dict):
        raise VGGTSceneTeacherUnavailableError(f"unsupported VGGT frame prediction type: {type(item)!r}")
    return SceneFramePrediction(
        depth=_optional_array(item.get("depth")),
        point_map=_optional_array(item.get("point_map")),
        intrinsics=_optional_array(item.get("intrinsics")),
        extrinsics=_optional_array(item.get("extrinsics")),
        confidence=_optional_array(item.get("confidence")),
        validity_mask=_optional_array(item.get("validity_mask")),
        floor_traversable_mask=_optional_array(item.get("floor_traversable_mask")),
        obstacle_risk_mask=_optional_array(item.get("obstacle_risk_mask")),
        dynamic_motion_mask=_optional_array(item.get("dynamic_motion_mask")),
        extra_metadata=item.get("extra_metadata") if isinstance(item.get("extra_metadata"), dict) else {},
    )


def _coerce_window_prediction(item: Any) -> SceneWindowPrediction:
    if isinstance(item, SceneWindowPrediction):
        return item
    if not isinstance(item, dict):
        raise VGGTSceneTeacherUnavailableError(f"unsupported VGGT window prediction type: {type(item)!r}")
    indices = item.get("frame_indices", ())
    if not isinstance(indices, (list, tuple)):
        raise VGGTSceneTeacherUnavailableError("VGGT window prediction frame_indices must be a sequence")
    return SceneWindowPrediction(
        frame_indices=tuple(int(index) for index in indices),
        point_tracks=_optional_array(item.get("point_tracks")),
        track_validity=_optional_array(item.get("track_validity")),
        extra_metadata=item.get("extra_metadata") if isinstance(item.get("extra_metadata"), dict) else {},
    )


def _optional_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _default_vggt_dir() -> Path | None:
    env_dir = os.environ.get("HOMEBRAIN_VGGT_DIR")
    if env_dir:
        return Path(env_dir)
    candidate = Path.cwd() / "external" / "vggt"
    return candidate if candidate.exists() else None


def _default_vggt_checkpoint() -> Path | None:
    env_checkpoint = os.environ.get("HOMEBRAIN_VGGT_CHECKPOINT")
    if env_checkpoint:
        return Path(env_checkpoint)
    for candidate in (
        Path.cwd() / "external" / "vggt" / "checkpoints" / "vggt.pt",
        Path.cwd() / "external" / "vggt" / "checkpoints" / "model.pt",
    ):
        if candidate.exists():
            return candidate
    return None


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
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(frame.height, frame.width)
        return pixels.astype(np.float32) / np.float32(255.0)
    digest = np.frombuffer(raw or b"\x00", dtype=np.uint8).astype(np.float32)
    tiled = np.resize(digest, pixel_count).reshape(frame.height, frame.width)
    return tiled / np.float32(255.0)
