from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from homebrain.datasets.tum_rgbd import read_depth_png_m, read_png
from homebrain.runtime.direct_bev_runtime import RuntimeDirectBEVStudent, load_runtime_direct_bev_student, predict_local_bev_from_rgbd
from homebrain.train.train_direct_bev_student_v0 import RealRGBDRouteBEVDataset
from homebrain.viz.checkpoint_3d_scene_types_v0 import (
    CandidateTrajectoryVizV0,
    LocalBEVFrameV0,
    ModelChannelSummaryV0,
    SafetyDebugVizV0,
    TransformV0,
)
from homebrain.viz.pose_camera_geometry_v0 import (
    build_T_map_base_from_pose_sequence,
    transform_from_matrix,
    transform_points,
)


KNOWN_CHANNELS = (
    "bev_free",
    "bev_obstacle",
    "bev_unknown",
    "bev_traversable",
    "bev_risky",
    "bev_hazard",
    "bev_confidence",
    "uncertainty",
    "future_free",
    "future_obstacle",
    "future_unknown",
    "future_risk",
    "candidate_scores",
    "candidate_debug",
    "safety_debug",
)


@dataclass(frozen=True)
class AdapterRouteOutputV0:
    frame_id: int
    timestamp_ns: int | None
    T_map_base: TransformV0
    T_base_camera: TransformV0 | None
    intrinsics: dict[str, Any]
    local_bev: LocalBEVFrameV0
    rgb_frame: dict[str, Any] | None
    depth_points_camera: list[list[float]] | None
    candidate_trajectories: list[CandidateTrajectoryVizV0]
    safety_debug: SafetyDebugVizV0
    model_channel_summaries: list[ModelChannelSummaryV0]
    warnings: list[str]


class Checkpoint3DOutputAdapterV0:
    def __init__(self, *, device: str | None = "cpu", review_assumed_extrinsics: bool = False) -> None:
        self.device = device
        self.review_assumed_extrinsics = bool(review_assumed_extrinsics)
        self.runtime: RuntimeDirectBEVStudent | None = None
        self.checkpoint_payload: dict[str, Any] = {}
        self.model_channels_missing = list(KNOWN_CHANNELS)

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        self.runtime = load_runtime_direct_bev_student(path, device=self.device)
        self.checkpoint_payload = {"metadata": self.runtime.metadata, "checkpoint_sha256": self.runtime.checkpoint_sha256}

    def iter_route_outputs(
        self,
        *,
        checkpoint: str | Path,
        pack: str | Path,
        route_id: str,
        max_frames: int | None = None,
        stride: int = 1,
        pose_source: str = "online",
    ) -> Iterator[AdapterRouteOutputV0]:
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.load_checkpoint(checkpoint)
        if self.runtime is None:
            raise RuntimeError("checkpoint failed to load")
        dataset = RealRGBDRouteBEVDataset(pack, split=None, image_size=tuple(self.runtime.model.config.image_size), cache_in_memory=False)
        selected = [(index, record) for index, record in enumerate(dataset.records) if str(record.get("route_id")) == str(route_id)]
        if not selected:
            raise ValueError(f"route_id {route_id!r} not found in pack")
        selected = selected[::stride]
        if max_frames is not None:
            selected = selected[: max(0, int(max_frames))]
        x_m = y_m = yaw_rad = 0.0
        for out_index, (dataset_index, record) in enumerate(selected):
            item = dataset[dataset_index]
            result = predict_local_bev_from_rgbd(
                self.runtime,
                item["rgb"],
                depth=item["depth"],
                sensor_mask=item["sensor_mask"],
                pose_delta_prev=item["pose_delta_prev"],
                previous_action=item["previous_action"],
            )
            delta = record.get("pose_delta_prev") if isinstance(record.get("pose_delta_prev"), list) else None
            if delta is not None and len(delta) >= 3:
                x_m += float(delta[0])
                y_m += float(delta[1])
                yaw_rad += float(delta[2])
            timestamp_ns = int(record.get("timestamp_ns", out_index)) if record.get("timestamp_ns") is not None else None
            source_name = "odom" if pose_source == "online" else str(pose_source)
            T_map_base = build_T_map_base_from_pose_sequence((x_m, y_m, yaw_rad), timestamp_ns=timestamp_ns, source=source_name)
            T_base_camera, warnings = self._camera_extrinsics(record, timestamp_ns)
            local = self.extract_local_bev(result, T_map_base=T_map_base, frame_id=out_index, timestamp_ns=timestamp_ns)
            summaries = self.extract_model_channel_summaries(local)
            yield AdapterRouteOutputV0(
                frame_id=out_index,
                timestamp_ns=timestamp_ns,
                T_map_base=T_map_base,
                T_base_camera=T_base_camera,
                intrinsics=self._intrinsics(record),
                local_bev=local,
                rgb_frame=self.extract_rgb_frame(record, frame_id=out_index, timestamp_ns=timestamp_ns, pack_root=dataset.root),
                depth_points_camera=self.extract_depth_points_camera(record, pack_root=dataset.root),
                candidate_trajectories=self.extract_candidate_trajectories(result, T_map_base=T_map_base, timestamp_ns=timestamp_ns),
                safety_debug=self.extract_safety_debug(result, timestamp_ns=timestamp_ns),
                model_channel_summaries=summaries,
                warnings=warnings,
            )

    def extract_model_channel_summaries(self, local_bev: LocalBEVFrameV0) -> list[ModelChannelSummaryV0]:
        present = set()
        summaries = list(local_bev.channel_summaries)
        for summary in summaries:
            present.add(summary.name)
        self.model_channels_missing = [name for name in KNOWN_CHANNELS if name not in present]
        for name in self.model_channels_missing:
            summaries.append(
                ModelChannelSummaryV0(
                    name=name,
                    shape=[],
                    min=None,
                    max=None,
                    mean=None,
                    valid_fraction=None,
                    source="checkpoint_output_missing",
                    rendered=False,
                    notes=["channel_missing_not_synthesized"],
                )
            )
        return summaries

    def extract_local_bev(
        self,
        output: dict[str, Any],
        *,
        T_map_base: TransformV0,
        frame_id: int,
        timestamp_ns: int | None,
    ) -> LocalBEVFrameV0:
        local = output["local_bev"]
        local.validate()
        channels = {
            "free": _array_list(local.free),
            "obstacle": _array_list(local.occupied),
            "unknown": _array_list(local.unknown),
        }
        for name in ("traversable", "risky", "hazard", "confidence", "uncertainty"):
            value = getattr(local, name)
            if value is not None:
                channels[name] = _array_list(value)
        summaries = [_summary(f"bev_{name}" if name not in {"uncertainty"} else name, np.asarray(value, dtype=np.float32), local.source) for name, value in channels.items()]
        return LocalBEVFrameV0(
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
            T_map_base=T_map_base,
            resolution_m=0.05,
            origin_convention="robot_bottom_center_forward_rows_decrease_left_cols_increase",
            channels=channels,
            channel_summaries=summaries,
            rendered_cells_map=_rendered_cells(channels),
            valid=True,
            warnings=[],
        )

    def extract_candidate_trajectories(
        self,
        output: dict[str, Any],
        *,
        T_map_base: TransformV0,
        timestamp_ns: int | None,
    ) -> list[CandidateTrajectoryVizV0]:
        records = output.get("candidate_records")
        if not isinstance(records, list):
            return []
        selected_id = output.get("selected_candidate_id")
        out = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            points = record.get("points") if isinstance(record.get("points"), list) else [[0.0, 0.0, 0.02]]
            mapped = transform_points(T_map_base, points).tolist()
            out.append(
                CandidateTrajectoryVizV0(
                    timestamp_ns=timestamp_ns,
                    candidate_id=str(record.get("candidate_id", f"candidate_{index}")),
                    points_map=[[round(float(v), 6) for v in point] for point in mapped],
                    score=_optional_float(record.get("world_model_score")),
                    risk=_optional_float(record.get("candidate_risk_probability")),
                    hazard_exposure=_optional_float(record.get("candidate_hazard_exposure")),
                    unknown_exposure=_optional_float(record.get("candidate_unknown_exposure_probability")),
                    selected=str(record.get("candidate_id")) == str(selected_id),
                    vetoed=bool(record.get("vetoed", False)),
                    stop_reasons=[str(item) for item in record.get("stop_reasons", [])] if isinstance(record.get("stop_reasons"), list) else [],
                )
            )
        return out

    def extract_safety_debug(self, output: dict[str, Any], *, timestamp_ns: int | None) -> SafetyDebugVizV0:
        debug = output.get("debug") if isinstance(output.get("debug"), dict) else {}
        return SafetyDebugVizV0(
            timestamp_ns=timestamp_ns,
            accepted=False,
            stop_reasons=[str(item) for item in debug.get("stop_reasons", [])] if isinstance(debug.get("stop_reasons"), list) else [],
            high_uncertainty=None,
            high_risk=None,
            obstacle_too_close=None,
        )

    def extract_rgb_frame(
        self,
        record: dict[str, Any],
        *,
        frame_id: int,
        timestamp_ns: int | None,
        pack_root: Path | None = None,
    ) -> dict[str, Any] | None:
        path_value = record.get("rgb_path") or record.get("rgb_ref")
        if not isinstance(path_value, str) or not path_value:
            return None
        resolved = _resolve_record_path(path_value, pack_root)
        payload: dict[str, Any] = {
            "kind": "route_rgb_frame",
            "frame_id": frame_id,
            "timestamp_ns": timestamp_ns,
            "source_path": path_value,
            "available": resolved.exists(),
        }
        if resolved.exists():
            payload["src"] = resolved.as_uri()
        return payload

    def extract_depth_points_camera(self, record: dict[str, Any], *, pack_root: Path | None = None, max_points: int = 6000) -> list[list[float]] | None:
        path_value = record.get("depth_path")
        if not isinstance(path_value, str) or not path_value:
            return None
        depth_path = _resolve_record_path(path_value, pack_root)
        if not depth_path.exists():
            return None
        intrinsics = record.get("camera_intrinsics") if isinstance(record.get("camera_intrinsics"), dict) else {}
        scale = float(intrinsics.get("depth_scale", 5000.0) or 5000.0)
        depth = read_depth_png_m(depth_path, scale=scale)
        if depth.ndim != 2:
            return None
        height, width = depth.shape
        fx = float(intrinsics.get("fx", 120.0) or 120.0)
        fy = float(intrinsics.get("fy", 120.0) or 120.0)
        cx = float(intrinsics.get("cx", width / 2.0) or width / 2.0)
        cy = float(intrinsics.get("cy", height / 2.0) or height / 2.0)
        step = max(1, int(np.ceil(np.sqrt(float(depth.size) / max(1, max_points)))))
        rows, cols = np.meshgrid(np.arange(0, height, step), np.arange(0, width, step), indexing="ij")
        z = depth[rows, cols]
        valid = np.isfinite(z) & (z > 0.15) & (z < 5.0)
        rows = rows[valid].astype(np.float32)
        cols = cols[valid].astype(np.float32)
        z = z[valid].astype(np.float32)
        if z.size == 0:
            return []
        if z.size > max_points:
            keep = np.linspace(0, z.size - 1, max_points).round().astype(np.int64)
            rows, cols, z = rows[keep], cols[keep], z[keep]
        x = (cols - np.float32(cx)) * z / np.float32(fx)
        y = (rows - np.float32(cy)) * z / np.float32(fy)
        colors = _sample_rgb_colors(record, rows=rows, cols=cols, depth_shape=(height, width), pack_root=pack_root)
        if colors is None:
            return [[round(float(a), 5), round(float(b), 5), round(float(c), 5)] for a, b, c in zip(x, y, z)]
        return [
            [round(float(a), 5), round(float(b), 5), round(float(c), 5), int(color[0]), int(color[1]), int(color[2])]
            for a, b, c, color in zip(x, y, z, colors)
        ]

    def _camera_extrinsics(self, record: dict[str, Any], timestamp_ns: int | None) -> tuple[TransformV0 | None, list[str]]:
        raw = record.get("T_base_camera") or record.get("camera_to_base_extrinsics")
        if isinstance(raw, list):
            return (
                transform_from_matrix(frame_from="base_link", frame_to="camera", matrix_4x4=raw, source="route_camera_extrinsics", timestamp_ns=timestamp_ns),
                [],
            )
        if self.review_assumed_extrinsics:
            return (
                transform_from_matrix(
                    frame_from="base_link",
                    frame_to="camera",
                    matrix_4x4=[
                        [0.0, 0.0, 1.0, 0.12],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.24],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    source="assumed_camera_extrinsics_requires_review",
                    timestamp_ns=timestamp_ns,
                    is_assumed=True,
                ),
                ["assumed_camera_to_base_extrinsics_review_required"],
            )
        return None, ["missing_camera_to_base_extrinsics"]

    def _intrinsics(self, record: dict[str, Any]) -> dict[str, Any]:
        intrinsics = record.get("camera_intrinsics") if isinstance(record.get("camera_intrinsics"), dict) else {}
        return {
            "fx": float(intrinsics.get("fx", 120.0) or 120.0),
            "fy": float(intrinsics.get("fy", 120.0) or 120.0),
            "cx": float(intrinsics.get("cx", 80.0) or 80.0),
            "cy": float(intrinsics.get("cy", 60.0) or 60.0),
            "width": int(intrinsics.get("width", 160) or 160),
            "height": int(intrinsics.get("height", 120) or 120),
        }


def _array_list(array: np.ndarray) -> list[list[float]]:
    values = np.asarray(array, dtype=np.float32)
    return [[round(float(item), 6) for item in row] for row in values.tolist()]


def _summary(name: str, array: np.ndarray, source: str) -> ModelChannelSummaryV0:
    valid = np.isfinite(array)
    values = array[valid]
    return ModelChannelSummaryV0(
        name=name,
        shape=[int(array.shape[0]), int(array.shape[1])],
        min=float(values.min()) if values.size else None,
        max=float(values.max()) if values.size else None,
        mean=float(values.mean()) if values.size else None,
        valid_fraction=float(np.count_nonzero(valid) / max(valid.size, 1)),
        source=source,
        rendered=True,
    )


def _rendered_cells(channels: dict[str, Any], *, threshold: float = 0.5, max_cells: int = 256) -> list[dict[str, Any]]:
    records = []
    for name in sorted(channels):
        values = np.asarray(channels[name], dtype=np.float32)
        rows, cols = np.nonzero(values > threshold)
        for row, col in zip(rows.tolist(), cols.tolist()):
            records.append({"channel": name, "row": int(row), "col": int(col), "value": round(float(values[row, col]), 6)})
            if len(records) >= max_cells:
                return records
    return records


def _optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    return None


def _resolve_record_path(path_value: str, pack_root: Path | None) -> Path:
    path = Path(path_value)
    if path.is_absolute() or path.exists() or pack_root is None:
        return path.expanduser().resolve()
    return (Path(pack_root) / path).expanduser().resolve()


def _sample_rgb_colors(
    record: dict[str, Any],
    *,
    rows: np.ndarray,
    cols: np.ndarray,
    depth_shape: tuple[int, int],
    pack_root: Path | None,
) -> np.ndarray | None:
    path_value = record.get("rgb_path") or record.get("rgb_ref")
    if not isinstance(path_value, str) or not path_value:
        return None
    rgb_path = _resolve_record_path(path_value, pack_root)
    if not rgb_path.exists():
        return None
    rgb = read_png(rgb_path)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    rgb = np.asarray(rgb[:, :, :3], dtype=np.uint8)
    rgb_h, rgb_w = rgb.shape[:2]
    depth_h, depth_w = depth_shape
    scale_r = (rgb_h - 1) / max(1, depth_h - 1)
    scale_c = (rgb_w - 1) / max(1, depth_w - 1)
    rr = np.clip(np.round(rows * scale_r).astype(np.int64), 0, rgb_h - 1)
    cc = np.clip(np.round(cols * scale_c).astype(np.int64), 0, rgb_w - 1)
    return rgb[rr, cc]
