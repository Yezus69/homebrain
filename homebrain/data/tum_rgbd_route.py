from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import math

from homebrain.datasets.tum_rgbd import (
    TUM_RGBD_DEPTH_SCALE,
    TumEntry,
    parse_groundtruth,
    parse_image_list,
    read_png_shape,
    sequence_intrinsics,
)
from homebrain.messages.schema import JsonDict

TUM_RGBD_ROUTE_SCHEMA_VERSION = "homebrain.real_rgbd_route.v0"


@dataclass(frozen=True)
class RouteFrame:
    route_id: str
    split_unit_id: str
    frame_index: int
    timestamp: float
    timestamp_ns: int
    rgb_path: Path
    depth_path: Path | None
    pose: JsonDict | None
    intrinsics: JsonDict
    width: int
    height: int


@dataclass(frozen=True)
class RouteSequence:
    dataset_name: str
    route_id: str
    split_unit_id: str
    root: Path
    frames: tuple[RouteFrame, ...]
    skipped_frame_count: int
    source_manifest: JsonDict


def load_tum_rgbd_sequence(
    sequence_dir: str | Path,
    *,
    dataset_name: str = "tum_rgbd",
    route_id: str | None = None,
    split_unit_id: str | None = None,
    max_association_s: float = 0.02,
    intrinsics: JsonDict | None = None,
) -> RouteSequence:
    root = Path(sequence_dir)
    rid = route_id or root.name
    sid = split_unit_id or rid
    if not root.exists():
        raise FileNotFoundError(f"TUM/Bonn RGB-D route directory does not exist: {root}")
    missing_required = [name for name in ("rgb.txt",) if not (root / name).exists()]
    if missing_required:
        raise FileNotFoundError(f"TUM/Bonn RGB-D route is missing: {', '.join(missing_required)}")

    rgb_entries = parse_image_list(root / "rgb.txt")
    depth_entries = parse_image_list(root / "depth.txt") if (root / "depth.txt").exists() else []
    pose_entries = parse_groundtruth(root / "groundtruth.txt") if (root / "groundtruth.txt").exists() else []
    if (root / "associations.txt").exists():
        associated = _associate_from_file(root / "associations.txt", pose_entries, max_association_s=max_association_s)
    else:
        associated = associate_rgb_depth_pose(
            rgb_entries,
            depth_entries,
            pose_entries,
            max_association_s=max_association_s,
        )
    default_intrinsics = _sequence_intrinsics(root, dataset_name, intrinsics)
    frames: list[RouteFrame] = []
    skipped = 0
    for index, item in enumerate(associated):
        rgb_path = root / item["rgb"].path
        if not rgb_path.exists():
            skipped += 1
            continue
        depth_entry = item.get("depth")
        depth_path = root / depth_entry.path if depth_entry is not None else None
        if depth_path is not None and not depth_path.exists():
            depth_path = None
        width, height = _image_shape(rgb_path)
        frames.append(
            RouteFrame(
                route_id=rid,
                split_unit_id=sid,
                frame_index=len(frames),
                timestamp=float(item["rgb"].timestamp),
                timestamp_ns=_timestamp_ns(float(item["rgb"].timestamp)),
                rgb_path=rgb_path,
                depth_path=depth_path,
                pose=_pose_dict(item.get("pose")),
                intrinsics=dict(default_intrinsics),
                width=int(width),
                height=int(height),
            )
        )

    manifest = source_manifest_for_sequence(
        dataset_name=dataset_name,
        route_id=rid,
        root=root,
        frame_count=len(rgb_entries),
        used_frame_count=len(frames),
        skipped_frame_count=skipped + max(0, len(rgb_entries) - len(associated)),
    )
    return RouteSequence(
        dataset_name=dataset_name,
        route_id=rid,
        split_unit_id=sid,
        root=root,
        frames=tuple(frames),
        skipped_frame_count=int(manifest["skipped_frame_count"]),
        source_manifest=manifest,
    )


def associate_rgb_depth_pose(
    rgb_entries: list[Any],
    depth_entries: list[Any],
    pose_entries: list[Any],
    *,
    max_association_s: float = 0.02,
) -> list[JsonDict]:
    used_depth: set[int] = set()
    associations: list[JsonDict] = []
    depth_times = [float(entry.timestamp) for entry in depth_entries]
    pose_times = [float(entry.timestamp) for entry in pose_entries]
    for rgb in sorted(rgb_entries, key=lambda entry: (float(entry.timestamp), str(entry.path))):
        depth_index = _nearest_index(depth_times, float(rgb.timestamp), max_association_s, used_depth)
        pose_index = _nearest_index(pose_times, float(rgb.timestamp), max_association_s, None)
        depth = depth_entries[depth_index] if depth_index is not None else None
        pose = pose_entries[pose_index] if pose_index is not None else None
        if depth_index is not None:
            used_depth.add(depth_index)
        associations.append({"rgb": rgb, "depth": depth, "pose": pose})
    return associations


def discover_tum_rgbd_routes(input_root: str | Path) -> list[Path]:
    root = Path(input_root)
    if not root.exists():
        return []
    if (root / "rgb.txt").exists():
        return [root]
    routes = [child for child in root.iterdir() if child.is_dir() and (child / "rgb.txt").exists()]
    return sorted(routes, key=lambda path: path.name)


def source_manifest_for_sequence(
    *,
    dataset_name: str,
    route_id: str,
    root: Path,
    frame_count: int,
    used_frame_count: int,
    skipped_frame_count: int,
    sample_limit: int = 12,
) -> JsonDict:
    sample_files = []
    for list_name in ("rgb.txt", "depth.txt", "groundtruth.txt", "associations.txt"):
        path = root / list_name
        if path.exists():
            sample_files.append(path)
    for child in sorted((root / "rgb").glob("*"))[: max(0, sample_limit - len(sample_files))]:
        if child.is_file():
            sample_files.append(child)
    hashes = [
        {
            "path": _relative_or_name(path, root),
            "sha256": _file_sha256(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in sample_files[:sample_limit]
    ]
    return {
        "schema_version": TUM_RGBD_ROUTE_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "route_id": route_id,
        "source_root": root.as_posix(),
        "frame_count": int(frame_count),
        "used_frame_count": int(used_frame_count),
        "skipped_frame_count": int(skipped_frame_count),
        "sample_file_hashes": hashes,
    }


def route_split_map(
    routes: list[RouteSequence],
    *,
    train_routes: list[str] | None = None,
    val_routes: list[str] | None = None,
) -> dict[str, str]:
    route_ids = [route.route_id for route in routes]
    if train_routes is None and val_routes is None:
        if len(route_ids) <= 1:
            return {route_id: "train" for route_id in route_ids}
        return {route_id: ("val" if index == len(route_ids) - 1 else "train") for index, route_id in enumerate(route_ids)}
    train = set(train_routes or [])
    val = set(val_routes or [])
    overlap = train & val
    if overlap:
        raise ValueError(f"route ids cannot appear in both train and val: {sorted(overlap)}")
    unknown = (train | val) - set(route_ids)
    if unknown:
        raise ValueError(f"requested split route ids are missing from input: {sorted(unknown)}")
    return {route_id: ("val" if route_id in val else "train") for route_id in route_ids}


def pose_delta(origin: JsonDict | None, target: JsonDict | None) -> tuple[float, float, float] | None:
    if origin is None or target is None:
        return None
    dx_world = float(target["tx"]) - float(origin["tx"])
    dy_world = float(target["ty"]) - float(origin["ty"])
    yaw_origin = _yaw(origin)
    dx = math.cos(yaw_origin) * dx_world + math.sin(yaw_origin) * dy_world
    dy = -math.sin(yaw_origin) * dx_world + math.cos(yaw_origin) * dy_world
    return (float(dx), float(dy), _wrap_angle(_yaw(target) - yaw_origin))


def _sequence_intrinsics(root: Path, dataset_name: str, override: JsonDict | None) -> JsonDict:
    if override is not None:
        values = dict(override)
    else:
        meta = _load_metadata(root)
        values = dict(meta.get("intrinsics", {})) if isinstance(meta.get("intrinsics"), dict) else {}
    if not values:
        values = sequence_intrinsics(root.name)
    if str(dataset_name).startswith("bonn") and values.get("source", "").startswith("unknown"):
        values.update({"source": "bonn_rgbd_tum_format_default"})
    values.setdefault("depth_scale", TUM_RGBD_DEPTH_SCALE)
    return values


def _load_metadata(root: Path) -> JsonDict:
    for name in ("metadata.json", "camera.json", "intrinsics.json"):
        path = root / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def _nearest_index(
    timestamps: list[float],
    target: float,
    max_difference: float,
    used: set[int] | None,
) -> int | None:
    best_index: int | None = None
    best_diff = float(max_difference)
    for index, timestamp in enumerate(timestamps):
        if used is not None and index in used:
            continue
        diff = abs(float(timestamp) - float(target))
        if diff <= best_diff:
            best_index = index
            best_diff = diff
    return best_index


def _associate_from_file(
    path: Path,
    pose_entries: list[Any],
    *,
    max_association_s: float,
) -> list[JsonDict]:
    associations: list[JsonDict] = []
    pose_times = [float(entry.timestamp) for entry in pose_entries]
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        rgb = TumEntry(timestamp=float(parts[0]), path=parts[1])
        depth = TumEntry(timestamp=float(parts[2]), path=parts[3])
        pose_index = _nearest_index(pose_times, float(rgb.timestamp), max_association_s, None)
        pose = pose_entries[pose_index] if pose_index is not None else None
        associations.append({"rgb": rgb, "depth": depth, "pose": pose})
    return sorted(associations, key=lambda item: (float(item["rgb"].timestamp), str(item["rgb"].path)))


def _pose_dict(pose: Any | None) -> JsonDict | None:
    if pose is None:
        return None
    return {
        "timestamp": float(pose.timestamp),
        "tx": float(pose.tx),
        "ty": float(pose.ty),
        "tz": float(pose.tz),
        "qx": float(pose.qx),
        "qy": float(pose.qy),
        "qz": float(pose.qz),
        "qw": float(pose.qw),
    }


def _image_shape(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".png":
            return read_png_shape(path)
        if suffix in {".ppm", ".pgm"}:
            return _pnm_shape(path)
    except Exception:
        return (0, 0)
    return (0, 0)


def _pnm_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        magic = handle.readline().strip()
        if magic not in {b"P5", b"P6"}:
            raise ValueError(f"unsupported PNM magic in {path}")
        line = handle.readline().strip()
        while line.startswith(b"#"):
            line = handle.readline().strip()
        width_s, height_s = line.split()[:2]
        return int(width_s), int(height_s)


def _timestamp_ns(timestamp_s: float) -> int:
    return int(round(float(timestamp_s) * 1_000_000_000))


def _relative_or_name(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaw(pose: JsonDict) -> float:
    qx = float(pose["qx"])
    qy = float(pose["qy"])
    qz = float(pose["qz"])
    qw = float(pose["qw"])
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _wrap_angle(value: float) -> float:
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return float(value)
