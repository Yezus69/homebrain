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

MOGE_DEFAULT_MODEL_ID = "Ruicheng/moge-2-vits-normal"
MOGE_LICENSE_REVIEW_STATUS = "pending_human_review"
MOGE_MODEL_SOURCE = "official MoGe install plus local checkpoint/operator model id, or operator-provided adapter"


class MoGeSceneTeacherUnavailableError(RuntimeError):
    """Raised when real MoGe scene inference cannot be initialized."""


@dataclass(frozen=True)
class ExternalMoGeResult:
    frames: tuple[SceneFramePrediction, ...]
    windows: tuple[SceneWindowPrediction, ...] = ()


class FakeMoGeSceneBackend:
    name = "fake"
    mocked = True
    synthetic = True
    real_perception = False
    deterministic = True
    model_source = MOGE_MODEL_SOURCE
    license_review_status = MOGE_LICENSE_REVIEW_STATUS
    scale_status = "synthetic_metric_test_scale"

    def __init__(self, *, model_id: str = MOGE_DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id

    def dependency_status(self) -> JsonDict:
        return {
            "backend": self.name,
            "moge_package": "not_required_for_fake_backend",
            "model_loaded": False,
            "model_id": self.model_id,
            "weights_status": "not_required_for_fake_backend",
            "downloads_attempted_by_homebrain": False,
            "warning": "test backend; not real MoGe inference",
        }

    def infer(self, log_dir: Path, frames: list[FrameEvent]) -> SceneTeacherBatchPrediction:
        predictions = tuple(self._infer_frame(frame, index) for index, frame in enumerate(frames))
        return SceneTeacherBatchPrediction(frames=predictions)

    def _infer_frame(self, frame: FrameEvent, index: int) -> SceneFramePrediction:
        row, col = np.indices((frame.height, frame.width), dtype=np.float32)
        row_norm = row / np.float32(max(frame.height - 1, 1))
        col_norm = col / np.float32(max(frame.width - 1, 1))
        depth = (
            np.float32(0.50)
            + np.float32(0.55) * row_norm
            + np.float32(0.15) * col_norm
            + np.float32(index) * np.float32(0.01)
        ).astype(np.float32)
        focal = np.float32(max(frame.width, frame.height))
        cx = np.float32((frame.width - 1) / 2.0)
        cy = np.float32((frame.height - 1) / 2.0)
        point_map = _depth_to_point_map(depth, _intrinsics(focal, cx, cy))
        confidence = np.clip(
            np.float32(0.92) - np.float32(0.05) * row_norm - np.float32(0.02) * col_norm,
            np.float32(0.0),
            np.float32(1.0),
        ).astype(np.float32)
        floor = (depth >= np.float32(np.median(depth))).astype(np.uint8)
        obstacle = (depth <= np.float32(np.percentile(depth, 20))).astype(np.uint8)
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[2, 3] = np.float32(index) * np.float32(0.01)
        return SceneFramePrediction(
            depth=depth,
            point_map=point_map,
            intrinsics=_intrinsics(focal, cx, cy),
            extrinsics=extrinsics,
            confidence=confidence,
            validity_mask=np.ones((frame.height, frame.width), dtype=np.uint8),
            floor_traversable_mask=floor,
            obstacle_risk_mask=obstacle,
            dynamic_motion_mask=np.zeros((frame.height, frame.width), dtype=np.uint8),
            extra_metadata={
                "backend": self.name,
                "warning": "fake deterministic test backend; not real MoGe inference",
            },
        )


class RealMoGeSceneBackend:
    name = "real"
    mocked = False
    synthetic = False
    real_perception = True
    deterministic = False
    model_source = MOGE_MODEL_SOURCE
    license_review_status = MOGE_LICENSE_REVIEW_STATUS
    scale_status = "metric"

    def __init__(
        self,
        *,
        model_id: str = MOGE_DEFAULT_MODEL_ID,
        model_dir: str | Path | None = None,
        checkpoint: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_dir = Path(model_dir) if model_dir is not None else _default_moge_dir()
        self.checkpoint = Path(checkpoint) if checkpoint is not None else _default_moge_checkpoint()
        self.device = device
        self._adapter: Any | None = None

    def dependency_status(self) -> JsonDict:
        adapter_ref = os.environ.get("HOMEBRAIN_MOGE_ADAPTER") or _default_official_adapter_ref()
        return {
            "backend": self.name,
            "model_id": self.model_id,
            "model_dir": self.model_dir.as_posix() if self.model_dir is not None else None,
            "model_dir_exists": self.model_dir.exists() if self.model_dir is not None else False,
            "checkpoint": self.checkpoint.as_posix() if self.checkpoint is not None else None,
            "checkpoint_exists": self.checkpoint.exists() if self.checkpoint is not None else False,
            "adapter": adapter_ref,
            "adapter_loaded": self._adapter is not None,
            "env_model_id": os.environ.get("HOMEBRAIN_MOGE_MODEL_ID"),
            "allow_default_download": os.environ.get("HOMEBRAIN_MOGE_ALLOW_DOWNLOAD") == "1",
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
        except Exception as exc:  # noqa: BLE001 - real backend failures should surface with setup context.
            raise MoGeSceneTeacherUnavailableError(f"real MoGe scene adapter failed: {exc}") from exc
        result = _coerce_external_result(raw, len(frames))
        self._refresh_scale_status_from_predictions(result.frames)
        return result

    def _load_adapter(self) -> Any:
        if self._adapter is not None:
            return self._adapter
        if self.model_dir is not None and self.model_dir.exists() and str(self.model_dir) not in sys.path:
            sys.path.insert(0, str(self.model_dir))
        adapter_ref = os.environ.get("HOMEBRAIN_MOGE_ADAPTER") or _default_official_adapter_ref()
        module_name, sep, attr = adapter_ref.partition(":")
        if not sep or not module_name or not attr:
            raise MoGeSceneTeacherUnavailableError(
                "HOMEBRAIN_MOGE_ADAPTER must have form 'module:function' for real MoGe inference."
            )
        try:
            module = importlib.import_module(module_name)
            adapter = getattr(module, attr)
        except Exception as exc:  # noqa: BLE001 - include setup hint.
            raise MoGeSceneTeacherUnavailableError(
                f"MoGe adapter callable {adapter_ref!r} was not importable. "
                "The default is homebrain.teachers.moge_official_adapter:run_scene_teacher, "
                "which uses `from moge.model.v2 import MoGeModel`. Set HOMEBRAIN_MOGE_ADAPTER=module:function "
                "only for an operator-supplied override. HomeBrain will not install MoGe or download "
                "checkpoints automatically."
            ) from exc
        self._adapter = adapter
        return adapter

    def _refresh_scale_status_from_predictions(self, frames: tuple[SceneFramePrediction, ...]) -> None:
        statuses = {
            str(frame.extra_metadata.get("scale_status"))
            for frame in frames
            if isinstance(frame.extra_metadata, dict) and frame.extra_metadata.get("scale_status") is not None
        }
        if len(statuses) == 1:
            self.scale_status = statuses.pop()
        model_ids = {
            str(frame.extra_metadata.get("model_id"))
            for frame in frames
            if isinstance(frame.extra_metadata, dict) and frame.extra_metadata.get("model_id") is not None
        }
        if len(model_ids) == 1:
            self.model_id = model_ids.pop()
        model_sources = {
            str(frame.extra_metadata.get("model_source"))
            for frame in frames
            if isinstance(frame.extra_metadata, dict) and frame.extra_metadata.get("model_source") is not None
        }
        if len(model_sources) == 1:
            self.model_source = model_sources.pop()


class MoGeSceneTeacher(SceneTeacher):
    name = "moge"
    version = "moge_scene_teacher.v0"

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


def create_moge_scene_teacher(
    *,
    backend_name: str = "real",
    model_id: str = MOGE_DEFAULT_MODEL_ID,
    model_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    max_frames: int | None = None,
    stride: int = 1,
) -> MoGeSceneTeacher:
    if backend_name == "fake":
        return MoGeSceneTeacher(FakeMoGeSceneBackend(model_id=model_id), max_frames=max_frames, stride=stride)
    if backend_name == "real":
        return MoGeSceneTeacher(
            RealMoGeSceneBackend(
                model_id=model_id,
                model_dir=model_dir,
                checkpoint=checkpoint,
                device=device,
            ),
            max_frames=max_frames,
            stride=stride,
        )
    raise ValueError(f"unknown MoGe scene backend {backend_name!r}; expected 'real' or 'fake'")


def run_moge_scene_teacher(
    log_dir: str | Path,
    out_dir: str | Path,
    *,
    backend_name: str = "real",
    model_id: str = MOGE_DEFAULT_MODEL_ID,
    model_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    max_frames: int | None = None,
    stride: int = 1,
) -> SceneTeacherRunSummary:
    teacher = create_moge_scene_teacher(
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
    if isinstance(raw, ExternalMoGeResult):
        return SceneTeacherBatchPrediction(frames=raw.frames, windows=raw.windows)
    if isinstance(raw, dict):
        frame_items = raw.get("frames")
        window_items = raw.get("windows", [])
    else:
        frame_items = raw
        window_items = []
    if not isinstance(frame_items, (list, tuple)):
        raise MoGeSceneTeacherUnavailableError("MoGe adapter result must include a list of frame predictions")
    frames = tuple(_coerce_frame_prediction(item) for item in frame_items)
    windows = tuple(_coerce_window_prediction(item) for item in window_items) if isinstance(window_items, (list, tuple)) else ()
    if len(frames) != frame_count:
        raise MoGeSceneTeacherUnavailableError(
            f"MoGe adapter returned {len(frames)} frame prediction(s) for {frame_count} frame(s)"
        )
    return SceneTeacherBatchPrediction(frames=frames, windows=windows)


def _coerce_frame_prediction(item: Any) -> SceneFramePrediction:
    if isinstance(item, SceneFramePrediction):
        return item
    if not isinstance(item, dict):
        raise MoGeSceneTeacherUnavailableError(f"unsupported MoGe frame prediction type: {type(item)!r}")
    intrinsics = _first_array(item, ("intrinsics", "camera_intrinsics", "K"))
    depth = _first_array(item, ("depth", "depth_m", "metric_depth"))
    point_map = _first_array(item, ("point_map", "points", "points3d", "xyz"))
    if point_map is None and depth is not None and intrinsics is not None:
        point_map = _depth_to_point_map(np.asarray(depth, dtype=np.float32), np.asarray(intrinsics, dtype=np.float32))
    if depth is None and point_map is not None:
        points = np.asarray(point_map, dtype=np.float32)
        if points.ndim == 3 and points.shape[-1] == 3:
            z = points[:, :, 2]
            norm = np.linalg.norm(points, axis=2)
            depth = np.where(np.isfinite(z) & (z > np.float32(0.0)), z, norm).astype(np.float32)
    metadata = item.get("extra_metadata") if isinstance(item.get("extra_metadata"), dict) else item.get("metadata")
    return SceneFramePrediction(
        depth=_optional_array(depth),
        point_map=_optional_array(point_map),
        intrinsics=_optional_array(intrinsics),
        extrinsics=_first_array(item, ("extrinsics", "camera_extrinsics", "pose", "camera_pose")),
        confidence=_first_array(item, ("confidence", "conf", "confidence_map")),
        validity_mask=_first_array(item, ("validity_mask", "valid_mask", "mask", "valid")),
        floor_traversable_mask=_first_array(item, ("floor_traversable_mask", "floor_mask")),
        obstacle_risk_mask=_first_array(item, ("obstacle_risk_mask", "obstacle_mask")),
        dynamic_motion_mask=_first_array(item, ("dynamic_motion_mask", "dynamic_mask")),
        extra_metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def _coerce_window_prediction(item: Any) -> SceneWindowPrediction:
    if isinstance(item, SceneWindowPrediction):
        return item
    if not isinstance(item, dict):
        raise MoGeSceneTeacherUnavailableError(f"unsupported MoGe window prediction type: {type(item)!r}")
    indices = item.get("frame_indices", ())
    if not isinstance(indices, (list, tuple)):
        raise MoGeSceneTeacherUnavailableError("MoGe window prediction frame_indices must be a sequence")
    return SceneWindowPrediction(
        frame_indices=tuple(int(index) for index in indices),
        point_tracks=_first_array(item, ("point_tracks", "tracks")),
        track_validity=_first_array(item, ("track_validity", "tracks_validity", "track_mask")),
        extra_metadata=item.get("extra_metadata") if isinstance(item.get("extra_metadata"), dict) else {},
    )


def _first_array(item: JsonDict, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in item:
            return _optional_array(item.get(key))
    return None


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


def _intrinsics(focal: np.float32, cx: np.float32, cy: np.float32) -> np.ndarray:
    return np.asarray(
        [
            [focal, np.float32(0.0), cx],
            [np.float32(0.0), focal, cy],
            [np.float32(0.0), np.float32(0.0), np.float32(1.0)],
        ],
        dtype=np.float32,
    )


def _depth_to_point_map(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32).squeeze()
    if values.ndim != 2:
        return np.asarray(depth, dtype=np.float32)
    camera = np.asarray(intrinsics, dtype=np.float32).squeeze()
    if camera.shape != (3, 3):
        return np.zeros((*values.shape, 3), dtype=np.float32)
    fx = np.float32(camera[0, 0] if camera[0, 0] != 0 else 1.0)
    fy = np.float32(camera[1, 1] if camera[1, 1] != 0 else fx)
    cx = np.float32(camera[0, 2])
    cy = np.float32(camera[1, 2])
    row, col = np.indices(values.shape, dtype=np.float32)
    x = ((col - cx) / fx * values).astype(np.float32)
    y = ((row - cy) / fy * values).astype(np.float32)
    return np.stack([x, y, values], axis=-1).astype(np.float32)


def _default_moge_dir() -> Path | None:
    env_dir = os.environ.get("HOMEBRAIN_MOGE_DIR")
    if env_dir:
        return Path(env_dir)
    candidate = Path.cwd() / "external" / "moge"
    return candidate if candidate.exists() else None


def _default_moge_checkpoint() -> Path | None:
    env_checkpoint = os.environ.get("HOMEBRAIN_MOGE_CHECKPOINT")
    if env_checkpoint:
        return Path(env_checkpoint)
    for candidate in (
        Path.cwd() / "external" / "moge" / "checkpoints" / "moge.pt",
        Path.cwd() / "external" / "moge" / "checkpoints" / "model.pt",
    ):
        if candidate.exists():
            return candidate
    return None


def _default_official_adapter_ref() -> str:
    return "homebrain.teachers.moge_official_adapter:run_scene_teacher"
