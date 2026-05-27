from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from homebrain.data.spatial_dataset import DETERMINISTIC_CREATED_AT_UTC, read_json, write_deterministic_npz, write_json
from homebrain.messages.schema import JsonDict

PASSIVE_DYNAMIC_TEACHER_SCHEMA_VERSION = "homebrain.passive_dynamic_teacher_sidecar.v1"
PASSIVE_DYNAMIC_TEACHER_MANIFEST_SCHEMA_VERSION = "homebrain.passive_dynamic_teacher_manifest.v1"
PASSIVE_DYNAMIC_DETERMINISTIC_TEACHER_VERSION = "passive_dynamic_deterministic_teacher.v0"
PASSIVE_DYNAMIC_FIXTURE_SCENARIOS: tuple[str, ...] = (
    "moving_blob_crossing_path",
    "person_pet_dynamic_obstacle",
    "occluder_appearing_disappearing",
    "ego_camera_pan_translation",
)


@dataclass(frozen=True)
class PassiveDynamicSidecar:
    frame_id: int
    timestamp_ns: int
    depth: np.ndarray
    dynamic_mask: np.ndarray
    semantic_mask: np.ndarray
    track_id_mask: np.ndarray
    camera_pose_delta: tuple[float, float, float] | None
    confidence: np.ndarray
    scenario: str
    sequence_id: str

    def to_arrays(self) -> dict[str, np.ndarray]:
        pose = (
            np.asarray(self.camera_pose_delta, dtype=np.float32)
            if self.camera_pose_delta is not None
            else np.zeros((3,), dtype=np.float32)
        )
        pose_mask = np.asarray(self.camera_pose_delta is not None, dtype=np.bool_)
        return {
            "schema_version": np.asarray(PASSIVE_DYNAMIC_TEACHER_SCHEMA_VERSION),
            "teacher_name": np.asarray("passive_dynamic_deterministic_teacher"),
            "teacher_version": np.asarray(PASSIVE_DYNAMIC_DETERMINISTIC_TEACHER_VERSION),
            "frame_id": np.asarray(int(self.frame_id), dtype=np.int64),
            "timestamp_ns": np.asarray(int(self.timestamp_ns), dtype=np.int64),
            "scenario": np.asarray(self.scenario),
            "sequence_id": np.asarray(self.sequence_id),
            "depth": np.asarray(self.depth, dtype=np.float32),
            "dynamic_mask": np.asarray(self.dynamic_mask, dtype=np.float32),
            "semantic_mask": np.asarray(self.semantic_mask, dtype=np.int64),
            "track_id_mask": np.asarray(self.track_id_mask, dtype=np.int64),
            "camera_pose_delta": pose,
            "camera_pose_delta_mask": pose_mask,
            "confidence": np.asarray(self.confidence, dtype=np.float32),
            "replay_only": np.asarray(True, dtype=np.bool_),
            "not_executed": np.asarray(True, dtype=np.bool_),
            "control_safe": np.asarray(False, dtype=np.bool_),
            "raw_pwm_emitted": np.asarray(False, dtype=np.bool_),
            "hardware_validated": np.asarray(False, dtype=np.bool_),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "PassiveDynamicSidecar":
        schema = str(np.asarray(arrays.get("schema_version", "")).item())
        if schema and schema != PASSIVE_DYNAMIC_TEACHER_SCHEMA_VERSION:
            raise ValueError(f"unsupported passive dynamic teacher sidecar schema: {schema!r}")
        pose_mask = bool(np.asarray(arrays.get("camera_pose_delta_mask", False)).item())
        pose_value = np.asarray(arrays.get("camera_pose_delta", np.zeros((3,), dtype=np.float32)), dtype=np.float32)
        pose = tuple(float(value) for value in pose_value.reshape(-1)[:3]) if pose_mask else None
        return cls(
            frame_id=int(np.asarray(arrays.get("frame_id", 0)).item()),
            timestamp_ns=int(np.asarray(arrays.get("timestamp_ns", 0)).item()),
            depth=np.asarray(arrays["depth"], dtype=np.float32),
            dynamic_mask=np.asarray(arrays["dynamic_mask"], dtype=np.float32),
            semantic_mask=np.asarray(arrays["semantic_mask"], dtype=np.int64),
            track_id_mask=np.asarray(arrays["track_id_mask"], dtype=np.int64),
            camera_pose_delta=pose,
            confidence=np.asarray(arrays["confidence"], dtype=np.float32),
            scenario=str(np.asarray(arrays.get("scenario", "unknown")).item()),
            sequence_id=str(np.asarray(arrays.get("sequence_id", "sequence")).item()),
        )


class PassiveDynamicDeterministicTeacher:
    name = "passive_dynamic_deterministic_teacher"
    version = PASSIVE_DYNAMIC_DETERMINISTIC_TEACHER_VERSION
    mock = True

    def sidecar_for_frame(
        self,
        *,
        scenario: str,
        sequence_id: str,
        frame_id: int,
        frame_count: int,
        shape: tuple[int, int],
        fps: float,
    ) -> PassiveDynamicSidecar:
        if scenario not in PASSIVE_DYNAMIC_FIXTURE_SCENARIOS:
            raise ValueError(f"unknown passive dynamic fixture scenario: {scenario}")
        height, width = shape
        row, col = np.indices((height, width), dtype=np.float32)
        depth = (0.35 + 0.5 * row / np.float32(max(height - 1, 1)) + 0.05 * frame_id).astype(np.float32)
        confidence = np.full((height, width), 0.92, dtype=np.float32)
        dynamic = np.zeros((height, width), dtype=np.float32)
        semantic = np.zeros((height, width), dtype=np.int64)
        track_id = np.zeros((height, width), dtype=np.int64)
        progress = frame_id / float(max(frame_count - 1, 1))
        camera_pose_delta: tuple[float, float, float] | None = (0.015, 0.0, 0.0)

        if scenario == "moving_blob_crossing_path":
            center = (int(round(height * 0.62)), int(round(width * (0.18 + 0.62 * progress))))
            _paint_disc(dynamic, center, radius=max(1, width // 14), value=1.0)
            _paint_disc(semantic, center, radius=max(1, width // 14), value=10)
            _paint_disc(track_id, center, radius=max(1, width // 14), value=1)
        elif scenario == "person_pet_dynamic_obstacle":
            person = (int(round(height * (0.76 - 0.18 * progress))), int(round(width * 0.46)))
            pet = (int(round(height * 0.70)), int(round(width * (0.72 - 0.28 * progress))))
            _paint_disc(dynamic, person, radius=max(1, width // 13), value=1.0)
            _paint_disc(dynamic, pet, radius=max(1, width // 18), value=0.85)
            _paint_disc(semantic, person, radius=max(1, width // 13), value=11)
            _paint_disc(semantic, pet, radius=max(1, width // 18), value=12)
            _paint_disc(track_id, person, radius=max(1, width // 13), value=2)
            _paint_disc(track_id, pet, radius=max(1, width // 18), value=3)
        elif scenario == "occluder_appearing_disappearing":
            visible = 0.25 <= progress <= 0.75
            if visible:
                top = int(round(height * 0.45))
                bottom = int(round(height * 0.78))
                left = int(round(width * 0.38))
                right = int(round(width * 0.68))
                dynamic[top:bottom, left:right] = 0.45
                semantic[top:bottom, left:right] = 20
                track_id[top:bottom, left:right] = 4
                confidence[top:bottom, left:right] = 0.18
        elif scenario == "ego_camera_pan_translation":
            shift = int(round((progress - 0.5) * width * 0.28))
            center = (int(round(height * 0.58)), int(round(width * 0.58 + shift)))
            _paint_disc(dynamic, center, radius=max(1, width // 12), value=0.9)
            _paint_disc(semantic, center, radius=max(1, width // 12), value=13)
            _paint_disc(track_id, center, radius=max(1, width // 12), value=5)
            camera_pose_delta = (0.01, float(shift) / float(max(width, 1)) * 0.03, 0.025)

        rgb_bias = (0.15 * np.sin(col / np.float32(max(width, 1)) * np.pi) + 0.08 * frame_id).astype(np.float32)
        depth = np.clip(depth + rgb_bias, 0.0, 4.0).astype(np.float32)
        timestamp_ns = int(round(frame_id * 1_000_000_000 / max(float(fps), 1.0e-6)))
        return PassiveDynamicSidecar(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            depth=depth,
            dynamic_mask=np.clip(dynamic, 0.0, 1.0).astype(np.float32),
            semantic_mask=semantic,
            track_id_mask=track_id,
            camera_pose_delta=camera_pose_delta,
            confidence=np.clip(confidence, 0.0, 1.0).astype(np.float32),
            scenario=scenario,
            sequence_id=sequence_id,
        )


def write_passive_dynamic_sidecar(path: str | Path, sidecar: PassiveDynamicSidecar) -> Path:
    target = Path(path)
    write_deterministic_npz(target, sidecar.to_arrays())
    return target


def read_passive_dynamic_sidecar(path: str | Path) -> PassiveDynamicSidecar:
    with np.load(Path(path), allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    return PassiveDynamicSidecar.from_arrays(arrays)


def write_passive_dynamic_fixture_sequences(
    root: str | Path,
    *,
    frame_count: int = 10,
    shape: tuple[int, int] = (16, 16),
    fps: float = 2.0,
) -> Path:
    output = Path(root)
    _clear(output)
    output.mkdir(parents=True, exist_ok=True)
    teacher = PassiveDynamicDeterministicTeacher()
    sequence_records: list[JsonDict] = []
    for split in ("train", "val"):
        for scenario in PASSIVE_DYNAMIC_FIXTURE_SCENARIOS:
            sequence_id = f"{split}_{scenario}"
            sequence_root = output / sequence_id
            frames_dir = sequence_root / "frames"
            sidecars_dir = sequence_root / "sidecars"
            frames_dir.mkdir(parents=True, exist_ok=True)
            sidecars_dir.mkdir(parents=True, exist_ok=True)
            frame_records: list[JsonDict] = []
            for frame_id in range(frame_count):
                sidecar = teacher.sidecar_for_frame(
                    scenario=scenario,
                    sequence_id=sequence_id,
                    frame_id=frame_id,
                    frame_count=frame_count,
                    shape=shape,
                    fps=fps,
                )
                frame_path = frames_dir / f"frame_{frame_id:06d}.ppm"
                sidecar_path = sidecars_dir / f"frame_{frame_id:06d}.npz"
                _write_ppm(frame_path, _rgb_from_sidecar(sidecar))
                write_passive_dynamic_sidecar(sidecar_path, sidecar)
                frame_records.append(
                    {
                        "frame_id": frame_id,
                        "timestamp_ns": sidecar.timestamp_ns,
                        "rgb_path": frame_path.relative_to(sequence_root).as_posix(),
                        "sidecar_path": sidecar_path.relative_to(sequence_root).as_posix(),
                    }
                )
            sequence_manifest: JsonDict = {
                "schema_version": PASSIVE_DYNAMIC_TEACHER_MANIFEST_SCHEMA_VERSION,
                "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
                "teacher_name": teacher.name,
                "teacher_version": teacher.version,
                "mock": True,
                "deterministic": True,
                "sequence_id": sequence_id,
                "scenario": scenario,
                "split": split,
                "fps": float(fps),
                "frame_count": frame_count,
                "shape": [int(shape[0]), int(shape[1])],
                "frames": frame_records,
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
                "raw_pwm_emitted": False,
                "hardware_validated": False,
            }
            write_json(sequence_root / "sequence_manifest.json", sequence_manifest, pretty=True)
            sequence_records.append(
                {
                    "sequence_id": sequence_id,
                    "scenario": scenario,
                    "split": split,
                    "path": sequence_root.relative_to(output).as_posix(),
                    "frame_count": frame_count,
                }
            )
    write_json(
        output / "manifest.json",
        {
            "schema_version": PASSIVE_DYNAMIC_TEACHER_MANIFEST_SCHEMA_VERSION,
            "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
            "teacher_name": teacher.name,
            "teacher_version": teacher.version,
            "fixture_root": True,
            "scenarios": list(PASSIVE_DYNAMIC_FIXTURE_SCENARIOS),
            "sequences": sequence_records,
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
            "raw_pwm_emitted": False,
            "hardware_validated": False,
        },
        pretty=True,
    )
    return output


def sequence_manifest(path: Path) -> JsonDict:
    manifest_path = path / "sequence_manifest.json"
    return read_json(manifest_path) if manifest_path.exists() else {}


def _rgb_from_sidecar(sidecar: PassiveDynamicSidecar) -> np.ndarray:
    depth = np.asarray(sidecar.depth, dtype=np.float32)
    dynamic = np.asarray(sidecar.dynamic_mask, dtype=np.float32)
    confidence = np.asarray(sidecar.confidence, dtype=np.float32)
    base = np.clip(depth / max(float(depth.max()), 1.0e-6), 0.0, 1.0)
    red = np.clip(0.35 + 0.55 * dynamic + 0.25 * base, 0.0, 1.0)
    green = np.clip(0.25 + 0.65 * confidence, 0.0, 1.0)
    blue = np.clip(0.65 - 0.35 * dynamic + 0.20 * base, 0.0, 1.0)
    return (255.0 * np.stack([red, green, blue], axis=2)).astype(np.uint8)


def _write_ppm(path: Path, rgb: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("PPM fixture frames must be RGB")
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    target.write_bytes(header + np.asarray(rgb, dtype=np.uint8).tobytes())


def _paint_disc(array: np.ndarray, center: tuple[int, int], *, radius: int, value: float | int) -> None:
    height, width = array.shape
    center_row, center_col = center
    for row in range(center_row - radius, center_row + radius + 1):
        for col in range(center_col - radius, center_col + radius + 1):
            if row < 0 or row >= height or col < 0 or col >= width:
                continue
            if (row - center_row) ** 2 + (col - center_col) ** 2 <= radius**2:
                array[row, col] = value


def _clear(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.iterdir(), reverse=True):
        if child.is_dir():
            _clear(child)
            child.rmdir()
        else:
            child.unlink()
