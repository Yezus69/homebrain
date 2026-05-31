from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from homebrain.messages.schema import FrameEvent, JsonDict
from homebrain.replay.replayd import order_events_for_replay
from homebrain.replay.segment_log import load_manifest, read_events
from homebrain.teachers.artifacts import (
    HAZARD_ARTIFACT_KINDS,
    TEACHER_ARTIFACT_SCHEMA_VERSION,
    file_sha256,
    relative_to_root,
    save_array,
    write_json,
    write_teacher_manifest,
)
from homebrain.teachers.base import Teacher, TeacherRunConfig, TeacherRunSummary
from homebrain.teachers.hazard_backends import (
    FakeHazardBackend,
    HazardBackend,
    HazardTeacherUnavailableError,
    RealGroundingDinoHazardBackend,
)
from homebrain.teachers.hazard_config import (
    HAZARD_BOXES_FILE,
    HAZARD_DEFAULT_MODEL_ID,
    HAZARD_LICENSE_REVIEW_STATUS,
    HAZARD_MODEL_SOURCE,
    HAZARD_TEACHER_VERSION,
    HazardPrediction,
    load_hazard_prompt_config,
)


class HazardTeacher(Teacher):
    name = "hazard"
    version = HAZARD_TEACHER_VERSION
    mock = False

    def __init__(
        self,
        backend: HazardBackend | None = None,
        *,
        prompts_path: str | Path | None = None,
        model_id: str = HAZARD_DEFAULT_MODEL_ID,
        model_dir: str | Path | None = None,
        max_frames: int | None = None,
        stride: int = 1,
    ) -> None:
        self.prompts_path = Path(prompts_path) if prompts_path is not None else None
        self.prompt_config = load_hazard_prompt_config(self.prompts_path)
        self.model_id = model_id
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.max_frames = max_frames
        self.stride = stride
        self.backend = backend if backend is not None else RealGroundingDinoHazardBackend(model_dir=model_dir, model_id=model_id)

    def run(self, config: TeacherRunConfig) -> TeacherRunSummary:
        if self.stride < 1:
            raise ValueError("hazard teacher stride must be at least 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("hazard teacher max_frames must be at least 1 when supplied")

        log_dir = Path(config.log_dir)
        out_dir = Path(config.out_dir)
        source_manifest = load_manifest(log_dir)
        events = order_events_for_replay(read_events(log_dir))
        all_frames = [event for event in events if isinstance(event, FrameEvent)]
        frames = all_frames[:: self.stride]
        if self.max_frames is not None:
            frames = frames[: self.max_frames]

        predictions = self.backend.infer(log_dir, frames, self.prompt_config)
        frame_records = [
            self._write_frame_artifacts(out_dir, frame, prediction)
            for frame, prediction in zip(frames, predictions)
        ]
        manifest_path = write_teacher_manifest(
            out_dir,
            self._manifest(log_dir, source_manifest, all_frames=all_frames, frame_records=frame_records),
        )
        return TeacherRunSummary(
            teacher_name=self.name,
            teacher_version=self.version,
            mock=bool(self.backend.mocked),
            frame_count=len(frame_records),
            manifest_path=manifest_path,
        )

    def _manifest(
        self,
        log_dir: Path,
        source_manifest: object,
        *,
        all_frames: list[FrameEvent],
        frame_records: list[JsonDict],
    ) -> JsonDict:
        positive_frame_count = sum(1 for record in frame_records if int(record.get("hazard_positive_cell_count", 0)) > 0)
        return {
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
            "artifact_kinds": list(HAZARD_ARTIFACT_KINDS),
            "box_artifact_kind": "hazard_boxes",
            "teacher_name_canonical": "hazard",
            "model_name": self.model_id,
            "model_id": self.model_id,
            "model_path": self.model_dir.as_posix() if self.model_dir is not None else None,
            "model_source": HAZARD_MODEL_SOURCE,
            "license_review_status": HAZARD_LICENSE_REVIEW_STATUS,
            "weak_label": True,
            "control_safe": False,
            "runtime_dependency": False,
            "not_robot_frame_truth": True,
            "trainable_for": "hazard_pretrain_only",
            "hazard_class_names": list(self.prompt_config.class_names),
            "hazard_positive_frame_count": int(positive_frame_count),
            "dependency_status": self.backend.dependency_status(),
            "frames": frame_records,
        }

    def _write_frame_artifacts(self, out_dir: Path, frame: FrameEvent, prediction: HazardPrediction) -> JsonDict:
        frame_dir = out_dir / "frames" / f"{frame.camera_id}_{frame.frame_id:06d}"
        masks, confidence = _validated_prediction_arrays(prediction, class_count=len(self.prompt_config.classes), frame=frame)
        artifact_records = _write_array_artifacts(frame_dir, out_dir, masks=masks, confidence=confidence)
        boxes_path = frame_dir / HAZARD_BOXES_FILE
        write_json(
            boxes_path,
            {
                "schema_version": TEACHER_ARTIFACT_SCHEMA_VERSION,
                "hazard_boxes": [box.to_dict() for box in prediction.boxes],
            },
        )
        artifact_records["hazard_boxes"] = {
            "kind": "hazard_boxes",
            "path": relative_to_root(boxes_path, out_dir),
            "sha256": file_sha256(boxes_path),
        }

        positive_cells = int(np.count_nonzero(np.max(masks, axis=0) >= np.float32(0.5))) if masks.size else 0
        metadata = self._frame_metadata(frame, prediction, positive_cells=positive_cells, artifact_records=artifact_records)
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
            "intrinsics": frame.intrinsics or {},
            "metadata_path": relative_to_root(metadata_path, out_dir),
            "metadata_sha256": file_sha256(metadata_path),
            "hazard_positive_cell_count": positive_cells,
            "artifacts": artifact_records,
        }

    def _frame_metadata(
        self,
        frame: FrameEvent,
        prediction: HazardPrediction,
        *,
        positive_cells: int,
        artifact_records: dict[str, JsonDict],
    ) -> JsonDict:
        return {
            "schema_version": TEACHER_ARTIFACT_SCHEMA_VERSION,
            "teacher_name": self.name,
            "teacher_version": self.version,
            "backend": self.backend.name,
            "mock": bool(self.backend.mocked),
            "synthetic": bool(self.backend.synthetic),
            "real_perception": bool(self.backend.real_perception),
            "license_review_status": HAZARD_LICENSE_REVIEW_STATUS,
            "weak_label": True,
            "control_safe": False,
            "not_robot_frame_truth": True,
            "trainable_for": "hazard_pretrain_only",
            "sequence_id": frame.sequence_id,
            "camera_id": frame.camera_id,
            "frame_id": frame.frame_id,
            "timestamp_ns": frame.timestamp_ns,
            "width": frame.width,
            "height": frame.height,
            "format": frame.format,
            "source_data_ref": frame.data_ref,
            "intrinsics": frame.intrinsics or {},
            "model_name": self.model_id,
            "model_path": self.model_dir.as_posix() if self.model_dir is not None else None,
            "hazard_class_names": list(self.prompt_config.class_names),
            "hazard_positive_cell_count": positive_cells,
            "dependency_status": self.backend.dependency_status(),
            "backend_metadata": prediction.extra_metadata,
            "artifact_shapes": {kind: artifact_records[kind]["shape"] for kind in HAZARD_ARTIFACT_KINDS},
            "artifact_dtypes": {kind: artifact_records[kind]["dtype"] for kind in HAZARD_ARTIFACT_KINDS},
        }


def create_hazard_teacher(
    *,
    backend_name: str = "real",
    device: str | None = None,
    model_id: str | None = None,
    model_dir: str | Path | None = None,
    max_frames: int | None = None,
    stride: int = 1,
    prompts_path: str | Path | None = None,
) -> HazardTeacher:
    resolved_model_id = model_id or HAZARD_DEFAULT_MODEL_ID
    if backend_name == "fake":
        backend: HazardBackend = FakeHazardBackend()
    elif backend_name == "real":
        backend = RealGroundingDinoHazardBackend(model_dir=model_dir, device=device, model_id=resolved_model_id)
    else:
        raise ValueError(f"unknown hazard backend {backend_name!r}; expected 'real' or 'fake'")
    return HazardTeacher(
        backend,
        prompts_path=prompts_path,
        model_id=resolved_model_id,
        model_dir=model_dir,
        max_frames=max_frames,
        stride=stride,
    )


def run_hazard_teacher(
    log_dir: str | Path,
    out_dir: str | Path,
    *,
    backend_name: str = "real",
    device: str | None = None,
    model_id: str | None = None,
    model_dir: str | Path | None = None,
    max_frames: int | None = None,
    stride: int = 1,
    prompts_path: str | Path | None = None,
) -> TeacherRunSummary:
    teacher = create_hazard_teacher(
        backend_name=backend_name,
        device=device,
        model_id=model_id,
        model_dir=model_dir,
        max_frames=max_frames,
        stride=stride,
        prompts_path=prompts_path,
    )
    return teacher.run(TeacherRunConfig(log_dir=Path(log_dir), out_dir=Path(out_dir)))


def _validated_prediction_arrays(
    prediction: HazardPrediction,
    *,
    class_count: int,
    frame: FrameEvent,
) -> tuple[np.ndarray, np.ndarray]:
    masks = np.clip(np.asarray(prediction.masks, dtype=np.float32), 0.0, 1.0)
    confidence = np.clip(np.asarray(prediction.confidence, dtype=np.float32), 0.0, 1.0)
    expected_shape = (class_count, frame.height, frame.width)
    if masks.shape != expected_shape:
        raise ValueError(f"hazard_masks must have shape {expected_shape}, got {masks.shape}")
    if confidence.shape != expected_shape:
        raise ValueError(f"hazard_confidence must have shape {expected_shape}, got {confidence.shape}")
    return masks, confidence


def _write_array_artifacts(
    frame_dir: Path,
    out_dir: Path,
    *,
    masks: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, JsonDict]:
    arrays = {"hazard_masks": masks, "hazard_confidence": confidence}
    artifact_records: dict[str, JsonDict] = {}
    for kind in HAZARD_ARTIFACT_KINDS:
        target = frame_dir / f"{kind}.npy"
        artifact_records[kind] = {
            "kind": kind,
            "path": relative_to_root(target, out_dir),
            **save_array(target, arrays[kind]),
        }
    return artifact_records


def _created_at(deterministic: bool) -> str:
    if deterministic:
        return "1970-01-01T00:00:00Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
