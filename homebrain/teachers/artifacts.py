from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from homebrain.messages.schema import FrameEvent, JsonDict, deterministic_json

TEACHER_ARTIFACT_SCHEMA_VERSION = "homebrain.teacher_artifacts.v0"
TEACHER_MANIFEST_FILE = "teacher_manifest.json"
BEV_PREVIEW_SHAPE = (16, 16)
FEATURE_CHANNELS = 4

EXPECTED_ARTIFACT_KINDS: tuple[str, ...] = (
    "depth",
    "depth_confidence",
    "dense_features",
    "dynamic_mask",
    "bev_preview",
)
DEPTH_PRO_ARTIFACT_KINDS: tuple[str, ...] = (
    "depth_m",
    "depth_confidence",
    "focallength_px",
    "bev_preview",
)
DA3_ARTIFACT_KINDS: tuple[str, ...] = (
    "depth",
    "confidence",
    "intrinsics",
    "extrinsics",
)

EXPECTED_DTYPES: dict[str, str] = {
    "depth": "float32",
    "depth_m": "float32",
    "depth_relative": "float32",
    "depth_confidence": "float32",
    "confidence": "float32",
    "dense_features": "float32",
    "dynamic_mask": "uint8",
    "focallength_px": "float32",
    "intrinsics": "float32",
    "extrinsics": "float32",
    "camera_pose": "float32",
    "bev_preview": "float32",
}


@dataclass(frozen=True)
class TeacherArtifactValidation:
    teacher_artifact_count: int
    teacher_mock_used: bool
    artifact_load_success: bool
    frames_with_teacher_artifacts: int
    missing_artifact_count: int
    artifact_shape_error_count: int
    artifact_determinism_pass: bool
    manifest_frame_count: int
    depth_frame_count: int = 0
    depth_missing_count: int = 0
    depth_nan_count: int = 0
    depth_nonpositive_count: int = 0
    depth_shape_error_count: int = 0
    errors: tuple[str, ...] = ()

    def to_metrics(self) -> dict[str, Any]:
        metrics = {
            "teacher_artifact_count": self.teacher_artifact_count,
            "teacher_mock_used": self.teacher_mock_used,
            "artifact_load_success": self.artifact_load_success,
            "frames_with_teacher_artifacts": self.frames_with_teacher_artifacts,
            "missing_artifact_count": self.missing_artifact_count,
            "artifact_shape_error_count": self.artifact_shape_error_count,
            "artifact_determinism_pass": self.artifact_determinism_pass,
            "teacher_manifest_frame_count": self.manifest_frame_count,
        }
        if (
            self.depth_frame_count
            or self.depth_missing_count
            or self.depth_nan_count
            or self.depth_nonpositive_count
            or self.depth_shape_error_count
        ):
            metrics.update(
                {
                    "depth_frame_count": self.depth_frame_count,
                    "depth_missing_count": self.depth_missing_count,
                    "depth_nan_count": self.depth_nan_count,
                    "depth_nonpositive_count": self.depth_nonpositive_count,
                    "depth_shape_error_count": self.depth_shape_error_count,
                }
            )
        return metrics


def frame_key(frame: FrameEvent | JsonDict) -> str:
    if isinstance(frame, FrameEvent):
        return f"{frame.sequence_id}:{frame.camera_id}:{frame.frame_id}"
    return f"{frame['sequence_id']}:{frame['camera_id']}:{frame['frame_id']}"


def expected_shape(kind: str, width: int, height: int) -> tuple[int, ...]:
    if kind in {"depth", "depth_relative", "depth_m", "depth_confidence", "confidence", "dynamic_mask"}:
        return (height, width)
    if kind == "dense_features":
        return (height, width, FEATURE_CHANNELS)
    if kind == "focallength_px":
        return (1,)
    if kind == "intrinsics":
        return (3, 3)
    if kind in {"extrinsics", "camera_pose"}:
        return (3, 4)
    if kind == "bev_preview":
        return BEV_PREVIEW_SHAPE
    raise ValueError(f"unknown teacher artifact kind {kind!r}")


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_array(path: str | Path, array: np.ndarray) -> JsonDict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": file_sha256(target),
    }


def load_array(path: str | Path) -> np.ndarray:
    with Path(path).open("rb") as handle:
        return np.load(handle, allow_pickle=False)


def write_json(path: str | Path, data: JsonDict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(deterministic_json(data))
        handle.write("\n")


def read_json(path: str | Path) -> JsonDict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def write_teacher_manifest(artifacts_dir: str | Path, manifest: JsonDict) -> Path:
    path = Path(artifacts_dir) / TEACHER_MANIFEST_FILE
    write_json(path, manifest)
    return path


def load_teacher_manifest(artifacts_dir: str | Path) -> JsonDict:
    path = Path(artifacts_dir) / TEACHER_MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"missing teacher manifest: {path}")
    manifest = read_json(path)
    if manifest.get("schema_version") != TEACHER_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported teacher artifact schema "
            f"{manifest.get('schema_version')!r}; expected {TEACHER_ARTIFACT_SCHEMA_VERSION!r}"
        )
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("teacher manifest frames must be a list")
    return manifest


def relative_to_root(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _manifest_frames_by_key(manifest: JsonDict) -> dict[str, JsonDict]:
    frames_by_key: dict[str, JsonDict] = {}
    for frame_record in manifest.get("frames", []):
        if isinstance(frame_record, dict):
            frames_by_key[frame_key(frame_record)] = frame_record
    return frames_by_key


def _expected_frames(frames: Iterable[FrameEvent] | None, manifest: JsonDict) -> list[FrameEvent | JsonDict]:
    if frames is not None:
        return list(frames)
    manifest_frames = manifest.get("frames", [])
    return [frame for frame in manifest_frames if isinstance(frame, dict)]


def _record_width_height(frame: FrameEvent | JsonDict) -> tuple[int, int]:
    if isinstance(frame, FrameEvent):
        return frame.width, frame.height
    return int(frame["width"]), int(frame["height"])


def _artifact_kinds(manifest: JsonDict) -> tuple[str, ...]:
    kinds = manifest.get("artifact_kinds")
    if not isinstance(kinds, list) or not all(isinstance(kind, str) for kind in kinds):
        return EXPECTED_ARTIFACT_KINDS
    return tuple(kinds)


def validate_teacher_artifacts(
    artifacts_dir: str | Path,
    *,
    frames: Iterable[FrameEvent] | None = None,
) -> TeacherArtifactValidation:
    root = Path(artifacts_dir)
    errors: list[str] = []
    try:
        manifest = load_teacher_manifest(root)
    except Exception as exc:  # noqa: BLE001 - eval should report malformed artifacts.
        return TeacherArtifactValidation(
            teacher_artifact_count=0,
            teacher_mock_used=False,
            artifact_load_success=False,
            frames_with_teacher_artifacts=0,
            missing_artifact_count=1,
            artifact_shape_error_count=0,
            artifact_determinism_pass=False,
            manifest_frame_count=0,
            errors=(str(exc),),
        )

    manifest_frames = manifest.get("frames", [])
    manifest_frame_count = len(manifest_frames) if isinstance(manifest_frames, list) else 0
    frames_by_key = _manifest_frames_by_key(manifest)
    expected_frames = _expected_frames(frames, manifest)
    artifact_kinds = _artifact_kinds(manifest)
    depth_kinds = tuple(kind for kind in ("depth_m", "depth", "depth_relative") if kind in artifact_kinds)
    validates_depth = bool(depth_kinds)

    valid_artifact_count = 0
    frames_with_all_artifacts = 0
    missing_artifact_count = 0
    shape_error_count = 0
    depth_frame_count = 0
    depth_missing_count = 0
    depth_nan_count = 0
    depth_nonpositive_count = 0
    depth_shape_error_count = 0
    determinism_pass = bool(manifest.get("deterministic", False))

    for expected_frame in expected_frames:
        key = frame_key(expected_frame)
        frame_record = frames_by_key.get(key)
        if frame_record is None:
            missing_artifact_count += len(artifact_kinds)
            if validates_depth:
                depth_missing_count += 1
            determinism_pass = False
            errors.append(f"missing frame artifact record: {key}")
            continue

        frame_complete = True
        metadata_path = frame_record.get("metadata_path")
        if not isinstance(metadata_path, str) or not (root / metadata_path).exists():
            missing_artifact_count += 1
            frame_complete = False
            determinism_pass = False
            errors.append(f"missing metadata for frame artifact record: {key}")

        artifacts = frame_record.get("artifacts")
        if not isinstance(artifacts, dict):
            missing_artifact_count += len(artifact_kinds)
            if validates_depth:
                depth_missing_count += 1
            determinism_pass = False
            errors.append(f"missing artifacts object for frame artifact record: {key}")
            continue

        width, height = _record_width_height(expected_frame)
        for kind in artifact_kinds:
            artifact_record = artifacts.get(kind)
            if not isinstance(artifact_record, dict):
                missing_artifact_count += 1
                if kind in depth_kinds:
                    depth_missing_count += 1
                frame_complete = False
                determinism_pass = False
                errors.append(f"missing {kind} artifact record for frame {key}")
                continue

            relative_path = artifact_record.get("path")
            if not isinstance(relative_path, str):
                missing_artifact_count += 1
                if kind in depth_kinds:
                    depth_missing_count += 1
                frame_complete = False
                determinism_pass = False
                errors.append(f"missing {kind} artifact path for frame {key}")
                continue

            artifact_path = root / relative_path
            if not artifact_path.exists():
                missing_artifact_count += 1
                if kind in depth_kinds:
                    depth_missing_count += 1
                frame_complete = False
                determinism_pass = False
                errors.append(f"missing {kind} artifact file for frame {key}: {relative_path}")
                continue

            try:
                array = load_array(artifact_path)
            except Exception as exc:  # noqa: BLE001 - keep eval resilient.
                shape_error_count += 1
                if kind in depth_kinds:
                    depth_shape_error_count += 1
                frame_complete = False
                determinism_pass = False
                errors.append(f"failed to load {kind} artifact for frame {key}: {exc}")
                continue

            valid_artifact_count += 1
            try:
                expected = expected_shape(kind, width, height)
                expected_dtype = EXPECTED_DTYPES[kind]
            except KeyError:
                shape_error_count += 1
                if kind in depth_kinds:
                    depth_shape_error_count += 1
                frame_complete = False
                determinism_pass = False
                errors.append(f"unknown artifact dtype for {kind} frame {key}")
                continue
            except ValueError as exc:
                shape_error_count += 1
                if kind in depth_kinds:
                    depth_shape_error_count += 1
                frame_complete = False
                determinism_pass = False
                errors.append(str(exc))
                continue
            manifest_shape = artifact_record.get("shape")
            manifest_dtype = artifact_record.get("dtype")
            if (
                tuple(array.shape) != expected
                or manifest_shape != list(expected)
                or str(array.dtype) != expected_dtype
                or manifest_dtype != expected_dtype
            ):
                shape_error_count += 1
                if kind in depth_kinds:
                    depth_shape_error_count += 1
                frame_complete = False
                determinism_pass = False
                errors.append(
                    f"shape/dtype error for {kind} frame {key}: "
                    f"array shape={array.shape} dtype={array.dtype}"
                )
            elif kind in depth_kinds:
                depth_frame_count += 1
                depth_nan_count += int(np.count_nonzero(~np.isfinite(array)))
                finite = array[np.isfinite(array)]
                depth_nonpositive_count += int(np.count_nonzero(finite <= np.float32(0.0)))

            expected_hash = artifact_record.get("sha256")
            if not isinstance(expected_hash, str) or file_sha256(artifact_path) != expected_hash:
                frame_complete = False
                determinism_pass = False
                errors.append(f"sha256 mismatch for {kind} frame {key}")

        if frame_complete:
            frames_with_all_artifacts += 1

    return TeacherArtifactValidation(
        teacher_artifact_count=valid_artifact_count,
        teacher_mock_used=bool(manifest.get("mock", False)),
        artifact_load_success=missing_artifact_count == 0 and shape_error_count == 0,
        frames_with_teacher_artifacts=frames_with_all_artifacts,
        missing_artifact_count=missing_artifact_count,
        artifact_shape_error_count=shape_error_count,
        artifact_determinism_pass=determinism_pass and missing_artifact_count == 0 and shape_error_count == 0,
        manifest_frame_count=manifest_frame_count,
        depth_frame_count=depth_frame_count,
        depth_missing_count=depth_missing_count,
        depth_nan_count=depth_nan_count,
        depth_nonpositive_count=depth_nonpositive_count,
        depth_shape_error_count=depth_shape_error_count,
        errors=tuple(errors),
    )
