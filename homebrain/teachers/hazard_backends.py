from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Protocol

import numpy as np

from homebrain.messages.schema import FrameEvent, JsonDict
from homebrain.teachers.hazard_config import (
    HAZARD_DEFAULT_MODEL_ID,
    HAZARD_MODEL_SOURCE,
    HazardBox,
    HazardPrediction,
    HazardPromptConfig,
)


class HazardTeacherUnavailableError(RuntimeError):
    """Raised when the real open-vocabulary hazard backend cannot run locally."""


class HazardBackend(Protocol):
    name: str
    mocked: bool
    synthetic: bool
    real_perception: bool
    deterministic: bool

    def infer(self, log_dir: Path, frames: list[FrameEvent], prompt_config: HazardPromptConfig) -> list[HazardPrediction]:
        """Return per-frame semantic floor-hazard predictions."""

    def dependency_status(self) -> JsonDict:
        """Return dependency/model-load status for manifests."""


class FakeHazardBackend:
    name = "fake"
    mocked = True
    synthetic = True
    real_perception = False
    deterministic = True

    def dependency_status(self) -> JsonDict:
        return {
            "backend": self.name,
            "model_loaded": False,
            "weights_status": "not_required_for_fake_backend",
            "downloads_attempted_by_homebrain": False,
            "warning": "test backend; not real semantic hazard perception",
        }

    def infer(self, log_dir: Path, frames: list[FrameEvent], prompt_config: HazardPromptConfig) -> list[HazardPrediction]:
        del log_dir
        predictions: list[HazardPrediction] = []
        class_count = len(prompt_config.classes)
        for index, frame in enumerate(frames):
            masks = np.zeros((class_count, frame.height, frame.width), dtype=np.float32)
            confidence = np.zeros_like(masks, dtype=np.float32)
            boxes: tuple[HazardBox, ...] = ()
            if class_count:
                class_index = int(frame.frame_id) % class_count
                left, top, right, bottom = _fixture_box(frame, index)
                score = np.float32(0.82)
                masks[class_index, top:bottom, left:right] = score
                confidence[class_index, top:bottom, left:right] = score
                boxes = (
                    HazardBox(
                        class_name=prompt_config.classes[class_index].name,
                        score=float(score),
                        box_xyxy=(float(left), float(top), float(right), float(bottom)),
                    ),
                )
            predictions.append(
                HazardPrediction(
                    masks=masks,
                    confidence=confidence,
                    boxes=boxes,
                    extra_metadata={
                        "backend": self.name,
                        "warning": "fake deterministic test backend; not real semantic hazard perception",
                    },
                )
            )
        return predictions


class RealGroundingDinoHazardBackend:
    name = "real"
    mocked = False
    synthetic = False
    real_perception = True
    deterministic = False

    def __init__(self, *, model_dir: str | Path | None = None, device: str | None = None, model_id: str = HAZARD_DEFAULT_MODEL_ID) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.device = device
        self.model_id = model_id
        self._model: Any | None = None
        self._inference: Any | None = None

    def dependency_status(self) -> JsonDict:
        return {
            "backend": self.name,
            "model_id": self.model_id,
            "model_source": HAZARD_MODEL_SOURCE,
            "model_path": self.model_dir.as_posix() if self.model_dir is not None else None,
            "model_loaded": self._model is not None,
            "groundingdino_package": "loaded" if self._inference is not None else "not_loaded",
            "device": self.device or "auto",
            "downloads_attempted_by_homebrain": False,
            "runtime_dependency": False,
            "python_executable": sys.executable,
        }

    def infer(self, log_dir: Path, frames: list[FrameEvent], prompt_config: HazardPromptConfig) -> list[HazardPrediction]:
        self._ensure_loaded()
        assert self._inference is not None
        predictions: list[HazardPrediction] = []
        for frame in frames:
            image_path = _frame_image_path(log_dir, frame)
            image_source, image = self._inference.load_image(str(image_path))
            boxes, logits, phrases = self._inference.predict(
                model=self._model,
                image=image,
                caption=prompt_config.prompt_text,
                box_threshold=0.20,
                text_threshold=0.20,
                device=self.device,
            )
            masks, confidence, records = _rasterize_grounding_dino_boxes(
                boxes=boxes,
                logits=logits,
                phrases=phrases,
                image_shape=(frame.height, frame.width),
                class_names=prompt_config.class_names,
                prompt_config=prompt_config,
            )
            predictions.append(
                HazardPrediction(
                    masks=masks,
                    confidence=confidence,
                    boxes=tuple(records),
                    extra_metadata={
                        "backend": self.name,
                        "mask_refinement": "box_rasterization_no_sam_refinement",
                        "source_image_shape": list(np.asarray(image_source).shape),
                    },
                )
            )
        return predictions

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self.model_dir is None:
            raise HazardTeacherUnavailableError(
                "Real hazard teacher requires --model-dir with local GroundingDINO config and checkpoint; "
                "HomeBrain does not download weights in tests or normal runs."
            )
        try:
            from groundingdino.util import inference  # type: ignore[import-not-found]
        except ImportError as exc:
            raise HazardTeacherUnavailableError(
                "GroundingDINO is not installed. Install it in an external teacher environment, "
                "or rerun with '--backend fake' for tests only."
            ) from exc
        config_path = _first_existing(self.model_dir / "GroundingDINO_SwinT_OGC.py", self.model_dir / "groundingdino_config.py")
        checkpoint_path = _first_existing(
            self.model_dir / "groundingdino_swint_ogc.pth",
            self.model_dir / "checkpoint.pth",
            self.model_dir / "model.pth",
        )
        if config_path is None or checkpoint_path is None:
            raise HazardTeacherUnavailableError(
                f"model_dir {self.model_dir} must contain a GroundingDINO config and checkpoint; no downloads are attempted"
            )
        self._model = inference.load_model(str(config_path), str(checkpoint_path), device=self.device)
        self._inference = inference


def _fixture_box(frame: FrameEvent, index: int) -> tuple[int, int, int, int]:
    box_w = max(1, frame.width // 3)
    box_h = max(1, frame.height // 4)
    left = int((index * 3 + frame.frame_id) % max(frame.width - box_w + 1, 1))
    top = max(0, int(frame.height * 0.58) - box_h // 2)
    return left, top, min(frame.width, left + box_w), min(frame.height, top + box_h)


def _frame_image_path(log_dir: Path, frame: FrameEvent) -> Path:
    path = log_dir / frame.data_ref
    if not path.exists():
        raise FileNotFoundError(f"frame data_ref is missing: {path}")
    if frame.format.startswith("encoded_"):
        return path
    raise HazardTeacherUnavailableError(
        "Real hazard teacher currently expects encoded image frames so the external detector can load them directly. "
        "Use an encoded route capture or the fake backend for tests."
    )


def _rasterize_grounding_dino_boxes(
    *,
    boxes: Any,
    logits: Any,
    phrases: Any,
    image_shape: tuple[int, int],
    class_names: tuple[str, ...],
    prompt_config: HazardPromptConfig,
) -> tuple[np.ndarray, np.ndarray, list[HazardBox]]:
    height, width = image_shape
    masks = np.zeros((len(class_names), height, width), dtype=np.float32)
    confidence = np.zeros_like(masks, dtype=np.float32)
    records: list[HazardBox] = []
    for box, score, phrase in zip(_to_numpy(boxes).reshape(-1, 4), _to_numpy(logits).reshape(-1), [str(value) for value in phrases]):
        class_index = _class_index_for_phrase(phrase, prompt_config)
        if class_index is None or float(score) < prompt_config.classes[class_index].min_confidence:
            continue
        x0, y0, x1, y1 = _box_cxcywh_to_xyxy(box, width=width, height=height)
        if x1 <= x0 or y1 <= y0:
            continue
        value = np.float32(min(1.0, max(0.5, prompt_config.classes[class_index].risk_weight)))
        masks[class_index, y0:y1, x0:x1] = np.maximum(masks[class_index, y0:y1, x0:x1], value)
        confidence[class_index, y0:y1, x0:x1] = np.maximum(confidence[class_index, y0:y1, x0:x1], np.float32(float(score)))
        records.append(HazardBox(class_names[class_index], float(score), (float(x0), float(y0), float(x1), float(y1))))
    return masks, confidence, records


def _box_cxcywh_to_xyxy(box: np.ndarray, *, width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = [float(value) for value in box]
    return (
        int(max(0.0, (cx - bw / 2.0) * width)),
        int(max(0.0, (cy - bh / 2.0) * height)),
        int(min(float(width), (cx + bw / 2.0) * width)),
        int(min(float(height), (cy + bh / 2.0) * height)),
    )


def _class_index_for_phrase(phrase: str, prompt_config: HazardPromptConfig) -> int | None:
    normalized = phrase.lower().replace("-", "_").replace(" ", "_")
    for index, item in enumerate(prompt_config.classes):
        names = {item.name.lower(), *(prompt.lower().replace("-", "_").replace(" ", "_") for prompt in item.prompts)}
        if any(name in normalized or normalized in name for name in names):
            return index
    return None


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
