from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from homebrain.data.spatial_dataset import SPATIAL_MANIFEST_FILE, load_example_npz, read_json
from homebrain.messages.schema import FrameEvent, JsonDict
from homebrain.teachers.artifacts import frame_key, load_array, load_teacher_manifest

BEV_OUTPUT_CHANNELS: tuple[str, ...] = ("free", "occupied", "unknown", "traversable", "risky")
SENSOR_MASK_FIELDS: tuple[str, ...] = ("pose", "action", "imu", "wheel")


@dataclass(frozen=True)
class FeatureRecord:
    sequence_id: str
    camera_id: str
    frame_id: int
    patch_path: Path
    cls_path: Path
    feature_shape: tuple[int, int, int]


class DINOFeatureStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest = load_teacher_manifest(self.root)
        if self.manifest.get("teacher_name") != "dino":
            raise ValueError(f"expected DINO teacher artifacts, got {self.manifest.get('teacher_name')!r}")
        if self.manifest.get("mock") is True and self.manifest.get("real_perception") is not False:
            raise ValueError("mock DINO artifacts must be marked real_perception=false")
        self.records: dict[str, FeatureRecord] = {}
        for frame_record in self.manifest.get("frames", []):
            if not isinstance(frame_record, dict):
                continue
            artifacts = frame_record.get("artifacts")
            if not isinstance(artifacts, dict):
                continue
            patch = artifacts.get("patch_features")
            cls = artifacts.get("cls_feature")
            if not isinstance(patch, dict) or not isinstance(cls, dict):
                continue
            patch_path = patch.get("path")
            cls_path = cls.get("path")
            if not isinstance(patch_path, str) or not isinstance(cls_path, str):
                continue
            shape_raw = patch.get("shape")
            if not isinstance(shape_raw, list) or len(shape_raw) != 3:
                continue
            key = frame_key(frame_record)
            self.records[key] = FeatureRecord(
                sequence_id=str(frame_record["sequence_id"]),
                camera_id=str(frame_record["camera_id"]),
                frame_id=int(frame_record["frame_id"]),
                patch_path=self.root / patch_path,
                cls_path=self.root / cls_path,
                feature_shape=(int(shape_raw[0]), int(shape_raw[1]), int(shape_raw[2])),
            )

    def load_by_key(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        record = self.records.get(key)
        if record is None:
            raise KeyError(f"missing DINO features for frame {key}")
        patch_features = load_array(record.patch_path).astype(np.float32)
        cls_feature = load_array(record.cls_path).astype(np.float32)
        if patch_features.ndim != 3:
            raise ValueError(f"patch_features must be HxWxC: {record.patch_path}")
        if cls_feature.ndim != 1:
            raise ValueError(f"cls_feature must be C: {record.cls_path}")
        return patch_features, cls_feature

    def load_for_example(self, example_record: JsonDict) -> tuple[np.ndarray, np.ndarray]:
        key = (
            f"{example_record['sequence_id']}:"
            f"{example_record['camera_id']}:"
            f"{example_record['frame_id']}"
        )
        return self.load_by_key(key)

    def load_for_frame(self, frame: FrameEvent) -> tuple[np.ndarray, np.ndarray]:
        return self.load_by_key(frame_key(frame))


class SpatialTrainDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        feature_dir: str | Path | None = None,
        split: str | None = None,
        tiny_overfit: bool = False,
        tiny_limit: int = 8,
        allow_tiny_fixture_features: bool = False,
    ) -> None:
        self.root = Path(dataset_dir)
        self.manifest = read_json(self.root / SPATIAL_MANIFEST_FILE)
        if self.manifest.get("control_safe") is not False:
            raise ValueError("SpatialTrainPack must remain control_safe=false")
        self.robot_supervision_grade = _robot_supervision_grade(self.manifest)
        records = self.manifest.get("examples") or self.manifest.get("frames")
        if not isinstance(records, list):
            raise ValueError("SpatialTrainPack manifest examples must be a list")
        self.records = [record for record in records if isinstance(record, dict)]
        if split is not None:
            self.records = [record for record in self.records if record.get("split") == split]
        if tiny_overfit:
            train_records = [record for record in self.records if record.get("split") == "train"]
            self.records = (train_records or self.records)[:tiny_limit]
        if not self.records:
            raise ValueError(f"no examples found in {self.root}")
        self.feature_store = DINOFeatureStore(feature_dir) if feature_dir is not None else None
        self.allow_tiny_fixture_features = allow_tiny_fixture_features
        grid_shape = self.manifest.get("grid_shape")
        if not isinstance(grid_shape, list) or len(grid_shape) != 2:
            raise ValueError("SpatialTrainPack manifest must include grid_shape")
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.start_timestamp_ns = min(int(record["timestamp_ns"]) for record in self.records)
        self.feature_shape = self._load_first_feature_shape()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        example = load_example_npz(self.root / str(record["example_path"]))
        patch_features, cls_feature, feature_mask = self._features(record, example)
        labels, label_mask = _bev_labels_and_mask(example)
        timestamp_ns = int(np.asarray(example["timestamp_ns"]).item())
        timestamp_s = np.asarray([(timestamp_ns - self.start_timestamp_ns) / 1_000_000_000.0], dtype=np.float32)
        sensor_mask = np.zeros((len(SENSOR_MASK_FIELDS),), dtype=np.float32)
        pose_delta = np.zeros((3,), dtype=np.float32)
        pose_mask = np.asarray([0.0], dtype=np.float32)
        if "pose_delta" in example:
            pose_delta = np.asarray(example["pose_delta"], dtype=np.float32).reshape(3)
            pose_mask = np.asarray([1.0], dtype=np.float32)
            sensor_mask[0] = 1.0

        return {
            "features": torch.from_numpy(np.transpose(patch_features, (2, 0, 1)).copy()),
            "cls_feature": torch.from_numpy(cls_feature.copy()),
            "feature_mask": torch.from_numpy(feature_mask),
            "timestamp_s": torch.from_numpy(timestamp_s),
            "sensor_mask": torch.from_numpy(sensor_mask),
            "bev_labels": torch.from_numpy(labels),
            "bev_label_mask": torch.from_numpy(label_mask),
            "pose_delta": torch.from_numpy(pose_delta),
            "pose_mask": torch.from_numpy(pose_mask),
            "frame_id": torch.tensor(int(record["frame_id"]), dtype=torch.int64),
            "timestamp_ns": torch.tensor(timestamp_ns, dtype=torch.int64),
        }

    def _load_first_feature_shape(self) -> tuple[int, int, int]:
        record = self.records[0]
        example = load_example_npz(self.root / str(record["example_path"]))
        patch_features, _cls_feature, _feature_mask = self._features(record, example)
        return tuple(int(value) for value in patch_features.shape)

    def _features(
        self,
        record: JsonDict,
        example: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.feature_store is not None:
            patch_features, cls_feature = self.feature_store.load_for_example(record)
            return patch_features.astype(np.float32), cls_feature.astype(np.float32), np.asarray([1.0], dtype=np.float32)
        if self.allow_tiny_fixture_features:
            patch_features = _tiny_fixture_features(example)
            cls_feature = patch_features.mean(axis=(0, 1)).astype(np.float32)
            return patch_features, cls_feature, np.asarray([0.0], dtype=np.float32)
        raise ValueError("DINO feature artifacts are required unless tiny fixture features are explicitly enabled")


def _bev_labels_and_mask(example: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    free = (np.asarray(example["bev_free"]) > 0).astype(np.float32)
    occupied = (np.asarray(example["bev_obstacle"]) > 0).astype(np.float32)
    unknown = (np.asarray(example["bev_unknown"]) > 0).astype(np.float32)
    traversable = free.copy()
    risky = occupied.copy()
    labels = np.stack([free, occupied, unknown, traversable, risky], axis=0).astype(np.float32)
    confidence = np.asarray(example["bev_confidence"], dtype=np.float32)
    base_mask = np.isfinite(confidence).astype(np.float32)
    if not np.any(base_mask):
        base_mask = np.ones_like(free, dtype=np.float32)
    mask = np.repeat(base_mask[None, :, :], len(BEV_OUTPUT_CHANNELS), axis=0).astype(np.float32)
    return labels, mask


def _tiny_fixture_features(example: dict[str, np.ndarray]) -> np.ndarray:
    free = np.asarray(example["bev_free"], dtype=np.float32)
    occupied = np.asarray(example["bev_obstacle"], dtype=np.float32)
    unknown = np.asarray(example["bev_unknown"], dtype=np.float32)
    confidence = np.asarray(example["bev_confidence"], dtype=np.float32)
    grid = np.stack([free, occupied, unknown, confidence], axis=-1)
    patch_h = patch_w = 4
    row_idx = np.linspace(0, grid.shape[0] - 1, patch_h).round().astype(np.int64)
    col_idx = np.linspace(0, grid.shape[1] - 1, patch_w).round().astype(np.int64)
    coarse = grid[row_idx[:, None], col_idx[None, :], :]
    repeated = np.concatenate([coarse, coarse, coarse, coarse, coarse, coarse, coarse, coarse], axis=-1)
    return repeated.astype(np.float32)


def _robot_supervision_grade(manifest: dict[str, Any]) -> str:
    existing = manifest.get("robot_supervision_grade")
    if isinstance(existing, str) and existing in {
        "weak_visual_geometry",
        "public_rgbd_anchor",
        "robot_frame_metric",
        "unknown",
    }:
        return existing
    source_name = str(manifest.get("source_depth_teacher_name", "")).lower()
    source_backend = str(manifest.get("source_depth_backend", "")).lower()
    source_bev = str(manifest.get("source_bev", "")).lower()
    if "robot_frame_metric" in {source_name, source_backend}:
        return "robot_frame_metric"
    if (
        "tum" in source_name
        or "rgbd_truth" in source_name
        or "public_rgbd" in source_backend
        or "rgbd_truth" in source_bev
    ):
        return "public_rgbd_anchor"
    if source_name in {"da3", "depth_pro"} or "weak_bev" in source_bev or source_backend in {"real", "fake"}:
        return "weak_visual_geometry"
    return "unknown"
