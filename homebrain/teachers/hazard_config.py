from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.messages.schema import JsonDict

HAZARD_TEACHER_VERSION = "hazard.teacher.v0"
HAZARD_DEFAULT_MODEL_ID = "grounding-dino"
HAZARD_MODEL_SOURCE = "https://github.com/IDEA-Research/GroundingDINO"
HAZARD_LICENSE_REVIEW_STATUS = "apache-2.0_pending_local_weight_review"
HAZARD_BOXES_FILE = "hazard_boxes.json"


@dataclass(frozen=True)
class HazardClassConfig:
    name: str
    prompts: tuple[str, ...]
    min_confidence: float
    risk_weight: float


@dataclass(frozen=True)
class HazardPromptConfig:
    classes: tuple[HazardClassConfig, ...]

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.classes)

    @property
    def prompt_text(self) -> str:
        phrases: list[str] = []
        for item in self.classes:
            phrases.extend(item.prompts)
        return ". ".join(phrases)


@dataclass(frozen=True)
class HazardBox:
    class_name: str
    score: float
    box_xyxy: tuple[float, float, float, float]

    def to_dict(self) -> JsonDict:
        return {
            "class": self.class_name,
            "score": round(float(self.score), 6),
            "box_xyxy": [round(float(value), 3) for value in self.box_xyxy],
        }


@dataclass(frozen=True)
class HazardPrediction:
    masks: np.ndarray
    confidence: np.ndarray
    boxes: tuple[HazardBox, ...]
    extra_metadata: JsonDict


DEFAULT_HAZARD_CLASSES: tuple[HazardClassConfig, ...] = (
    HazardClassConfig("cable", ("cable", "power cord", "charging cable"), 0.35, 1.0),
    HazardClassConfig("power_cord", ("power cord on floor",), 0.35, 1.0),
    HazardClassConfig("charging_cable", ("charging cable on floor",), 0.35, 1.0),
    HazardClassConfig("phone_charger", ("phone charger on floor",), 0.35, 0.9),
    HazardClassConfig("sock", ("sock on floor",), 0.35, 0.8),
    HazardClassConfig("shoe", ("shoe on floor",), 0.40, 0.9),
    HazardClassConfig("small_toy", ("small toy on floor",), 0.35, 0.9),
    HazardClassConfig("pet_waste", ("pet waste on floor",), 0.30, 1.0),
    HazardClassConfig("liquid_spill", ("liquid spill on floor",), 0.30, 1.0),
    HazardClassConfig("rug_fringe", ("rug fringe on floor",), 0.35, 0.7),
    HazardClassConfig("cloth_on_floor", ("cloth on floor",), 0.35, 0.8),
)


def load_hazard_prompt_config(path: str | Path | None = None) -> HazardPromptConfig:
    if path is None:
        return HazardPromptConfig(classes=DEFAULT_HAZARD_CLASSES)
    data = _read_json_or_yaml(Path(path))
    classes_raw = data.get("classes")
    if isinstance(classes_raw, dict):
        items = [{"name": name, **value} if isinstance(value, dict) else {"name": name} for name, value in classes_raw.items()]
    elif isinstance(classes_raw, list):
        items = [item for item in classes_raw if isinstance(item, dict)]
    else:
        raise ValueError(f"hazard prompt config must include classes in {path}")
    classes = [_class_config(item) for item in items]
    if not classes:
        raise ValueError("hazard prompt config must define at least one class")
    return HazardPromptConfig(classes=tuple(classes))


def _class_config(item: JsonDict) -> HazardClassConfig:
    name = str(item.get("name", "")).strip()
    if not name:
        raise ValueError("hazard class is missing name")
    prompts_raw = item.get("prompts", [name])
    prompts = tuple(str(value).strip() for value in prompts_raw if str(value).strip()) if isinstance(prompts_raw, list) else (name,)
    return HazardClassConfig(
        name=name,
        prompts=prompts or (name,),
        min_confidence=_finite_probability(item.get("min_confidence", 0.35), "min_confidence"),
        risk_weight=_positive_float(item.get("risk_weight", 1.0), "risk_weight"),
    )


def _read_json_or_yaml(path: Path) -> JsonDict:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ValueError(f"{path} is not JSON and PyYAML is not installed") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"expected object in hazard prompt config {path}")
    return data


def _finite_probability(value: Any, name: str) -> float:
    number = _positive_float(value, name)
    if number > 1.0:
        raise ValueError(f"{name} must be <= 1.0")
    return number


def _positive_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number
