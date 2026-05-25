from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import re
import shutil
import ssl
import subprocess
import tarfile
import urllib.request
import zipfile

import numpy as np

from homebrain.messages.schema import JsonDict

OPENLORIS_SCHEMA_VERSION = "homebrain.datasets.openloris_scene.v0"
OPENLORIS_ROUTE_ASSOCIATIONS_FILE = "openloris_scene_associations.json"
OPENLORIS_SOURCE_URL = "https://lifelong-robotic-vision.github.io/dataset/scene.html"
OPENLORIS_DOWNLOAD_PAGE = "https://github.com/lifelong-robotic-vision/OpenLORIS-Scene/blob/master/download.md"
OPENLORIS_HF_REPO = "shixuesong/openloris-scene"
OPENLORIS_HF_PACKAGE_URL = "https://huggingface.co/datasets/shixuesong/openloris-scene/tree/main/package"
OPENLORIS_TOOLS_URL = "https://github.com/lifelong-robotic-vision/openloris-scene-tools"
OPENLORIS_TOOLS_COMMIT_INSPECTED = "ce6a4839f618bf036d3f3dbae14561bfc7413641"
OPENLORIS_LICENSE_NAME = "CC BY-ND 4.0"
OPENLORIS_LICENSE_REVIEW_STATUS = "poc_allowed_product_review_later"
OPENLORIS_DEPTH_SCALE = 1000.0

DEFAULT_MAX_DOWNLOAD_GB = 2.0

# Small subset of the official OpenLORIS static transforms, transcribed from
# openloris-scene-tools/benchmark/openloris_tf_data.py (MIT licensed).
_TSINGHUA_CAMERA_TO_BASE: dict[str, tuple[float, float, float, float, float, float, float]] = {
    "d400_color": (
        0.2264836849091656,
        -0.05114194035652147,
        0.916,
        -0.49676229968284136,
        0.4998795887129771,
        -0.49510681269354095,
        0.5081504289345848,
    ),
    "d400_depth": (
        0.22722047136786871,
        -0.06601512576001335,
        0.9157872415132173,
        -0.49538006860636935,
        0.49954735142309487,
        -0.49840674562280746,
        0.5065982108450463,
    ),
}
_MARKET_CAMERA_TO_BASE: dict[str, tuple[float, float, float, float, float, float, float]] = {
    "d400_color": (
        0.398363566676407,
        0.029117046016620643,
        1.198511534258444,
        0.5508509426177262,
        -0.5167305893488635,
        0.44976770953124534,
        -0.47671977566632984,
    ),
    "d400_depth": (
        0.399546294466062,
        0.014453458565135675,
        1.1984746547500458,
        0.5499697614834611,
        -0.5171593202226934,
        0.4537274051736184,
        -0.47350917705472384,
    ),
}

OPENLORIS_STATIC_TRANSFORM_SOURCE = {
    "source_url": (
        "https://github.com/lifelong-robotic-vision/openloris-scene-tools/"
        "blob/master/benchmark/openloris_tf_data.py"
    ),
    "tools_license": "MIT",
    "tools_commit_inspected": OPENLORIS_TOOLS_COMMIT_INSPECTED,
    "transform_direction": "camera_points_to_base_link",
}
OPENLORIS_TRANS_MATRIX_SOURCE = {
    "source": "trans_matrix.yaml",
    "frame_source": "OpenLORIS extracted static transform file",
    "transform_direction": "camera_points_to_base_link",
}


@dataclass(frozen=True)
class OpenLorisImageEntry:
    timestamp: float
    path: str


@dataclass(frozen=True)
class OpenLorisPose:
    timestamp: float
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float

    def to_dict(self) -> JsonDict:
        return {
            "timestamp": self.timestamp,
            "tx": self.tx,
            "ty": self.ty,
            "tz": self.tz,
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz,
            "qw": self.qw,
        }


@dataclass(frozen=True)
class OpenLorisOdom:
    timestamp: float
    pose: OpenLorisPose
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]

    def to_dict(self) -> JsonDict:
        return {
            "timestamp": self.timestamp,
            "pose": self.pose.to_dict(),
            "linear_velocity": list(self.linear_velocity),
            "angular_velocity": list(self.angular_velocity),
        }


@dataclass(frozen=True)
class OpenLorisImuSample:
    timestamp: float
    values: tuple[float, float, float]


@dataclass(frozen=True)
class OpenLorisImuPair:
    timestamp: float
    accel: tuple[float, float, float]
    gyro: tuple[float, float, float]


@dataclass(frozen=True)
class OpenLorisAssociation:
    rgb: OpenLorisImageEntry
    depth: OpenLorisImageEntry | None
    pose: OpenLorisPose | None
    odom: OpenLorisOdom | None


@dataclass(frozen=True)
class OpenLorisRemoteFile:
    path: str
    size_bytes: int

    @property
    def size_gb(self) -> float:
        return float(self.size_bytes / (1024.0**3))


def openloris_sequence_dir(out_dir: str | Path, sequence: str) -> Path:
    return Path(out_dir) / sequence


def validate_sequence_dir(sequence_dir: str | Path, *, require_depth: bool = False) -> None:
    root = Path(sequence_dir)
    missing = [name for name in ("color.txt",) if not (root / name).exists()]
    if require_depth and not _depth_list_path(root):
        missing.append("aligned_depth.txt or depth.txt")
    if missing:
        raise FileNotFoundError(f"OpenLORIS-Scene sequence is missing required files: {', '.join(missing)}")


def discover_sequence_root(root: str | Path) -> Path:
    path = Path(root)
    if (path / "color.txt").exists():
        return path
    candidates = sorted(parent for parent in path.rglob("color.txt") if parent.is_file())
    if not candidates:
        raise FileNotFoundError(f"could not find OpenLORIS color.txt under {path}")
    return candidates[0].parent


def sequence_scene(sequence: str | Path) -> str:
    name = Path(sequence).name.lower()
    for scene in ("office", "corridor", "home", "cafe", "market"):
        if name.startswith(scene):
            return scene
    return name.split("1", 1)[0].split("-", 1)[0] or "unknown"


def parse_image_list(path: str | Path) -> list[OpenLorisImageEntry]:
    entries: list[OpenLorisImageEntry] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        entries.append(OpenLorisImageEntry(timestamp=float(parts[0]), path=parts[1]))
    return entries


def parse_pose_file(path: str | Path) -> list[OpenLorisPose]:
    source = Path(path)
    if not source.exists():
        return []
    entries: list[OpenLorisPose] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 8:
            continue
        entries.append(
            OpenLorisPose(
                timestamp=float(parts[0]),
                tx=float(parts[1]),
                ty=float(parts[2]),
                tz=float(parts[3]),
                qx=float(parts[4]),
                qy=float(parts[5]),
                qz=float(parts[6]),
                qw=float(parts[7]),
            )
        )
    return entries


def parse_odom_file(path: str | Path) -> list[OpenLorisOdom]:
    source = Path(path)
    if not source.exists():
        return []
    entries: list[OpenLorisOdom] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 14:
            continue
        timestamp = float(parts[0])
        entries.append(
            OpenLorisOdom(
                timestamp=timestamp,
                pose=OpenLorisPose(
                    timestamp=timestamp,
                    tx=float(parts[1]),
                    ty=float(parts[2]),
                    tz=float(parts[3]),
                    qx=float(parts[4]),
                    qy=float(parts[5]),
                    qz=float(parts[6]),
                    qw=float(parts[7]),
                ),
                linear_velocity=(float(parts[8]), float(parts[9]), float(parts[10])),
                angular_velocity=(float(parts[11]), float(parts[12]), float(parts[13])),
            )
        )
    return entries


def parse_imu_samples(path: str | Path) -> list[OpenLorisImuSample]:
    source = Path(path)
    if not source.exists():
        return []
    samples: list[OpenLorisImuSample] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        samples.append(
            OpenLorisImuSample(
                timestamp=float(parts[0]),
                values=(float(parts[1]), float(parts[2]), float(parts[3])),
            )
        )
    return samples


def merge_imu_samples(
    accel: list[OpenLorisImuSample],
    gyro: list[OpenLorisImuSample],
    *,
    max_difference: float = 0.02,
) -> list[OpenLorisImuPair]:
    if not accel or not gyro:
        return []
    gyro_timestamps = [sample.timestamp for sample in gyro]
    pairs: list[OpenLorisImuPair] = []
    used_gyro: set[int] = set()
    for acc in accel:
        gyro_index = _nearest_index(gyro_timestamps, acc.timestamp, max_difference=max_difference, used=used_gyro)
        if gyro_index is None:
            continue
        used_gyro.add(gyro_index)
        gyr = gyro[gyro_index]
        pairs.append(
            OpenLorisImuPair(
                timestamp=(acc.timestamp + gyr.timestamp) * 0.5,
                accel=acc.values,
                gyro=gyr.values,
            )
        )
    return pairs


def load_associations(
    sequence_dir: str | Path,
    *,
    max_difference: float = 0.03,
    max_frames: int | None = None,
) -> list[OpenLorisAssociation]:
    root = Path(sequence_dir)
    rgb_entries = parse_image_list(root / "color.txt")
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive when supplied")
        rgb_entries = rgb_entries[:max_frames]
    depth_entries = parse_image_list(_depth_list_path(root)) if _depth_list_path(root) else []
    pose_entries = parse_pose_file(root / "groundtruth.txt")
    odom_entries = parse_odom_file(root / "odom.txt")
    depth_timestamps = [entry.timestamp for entry in depth_entries]
    pose_timestamps = [entry.timestamp for entry in pose_entries]
    odom_timestamps = [entry.timestamp for entry in odom_entries]
    used_depth: set[int] = set()
    associations: list[OpenLorisAssociation] = []
    for rgb in rgb_entries:
        depth_index = _nearest_index(depth_timestamps, rgb.timestamp, max_difference=max_difference, used=used_depth)
        pose_index = _nearest_index(pose_timestamps, rgb.timestamp, max_difference=max_difference * 4.0, used=None)
        odom_index = _nearest_index(odom_timestamps, rgb.timestamp, max_difference=max_difference * 4.0, used=None)
        depth = None
        if depth_index is not None:
            used_depth.add(depth_index)
            depth = depth_entries[depth_index]
        associations.append(
            OpenLorisAssociation(
                rgb=rgb,
                depth=depth,
                pose=pose_entries[pose_index] if pose_index is not None else None,
                odom=odom_entries[odom_index] if odom_index is not None else None,
            )
        )
    return associations


def load_calibration(sequence_dir: str | Path, *, camera_id: str = "d400_color") -> JsonDict:
    root = Path(sequence_dir)
    json_calibration = _read_json_calibration(root, camera_id)
    intrinsics = (
        json_calibration.get("intrinsics")
        if isinstance(json_calibration.get("intrinsics"), dict)
        else _parse_intrinsics_from_sensors_yaml(root / "sensors.yaml", camera_id)
    )
    camera_to_base = json_calibration.get("camera_to_base")
    camera_to_base_source = json_calibration.get("camera_to_base_source")
    if not _valid_matrix(camera_to_base):
        parsed = _parse_camera_to_base_from_trans_matrix_yaml(root / "trans_matrix.yaml", camera_id)
        if parsed is not None:
            camera_to_base, camera_to_base_source = parsed
    if not _valid_matrix(camera_to_base) and environ_truthy("HOMEBRAIN_OPENLORIS_ALLOW_OFFICIAL_STATIC_TF"):
        camera_to_base = _official_camera_to_base(sequence_scene(root), camera_id)
        camera_to_base_source = {
            **OPENLORIS_STATIC_TRANSFORM_SOURCE,
            "operator_review_required": True,
            "enabled_by_env": "HOMEBRAIN_OPENLORIS_ALLOW_OFFICIAL_STATIC_TF",
        }
    return {
        "intrinsics": intrinsics,
        "intrinsics_source": str(intrinsics.get("source", "missing")) if intrinsics.get("available") else "missing",
        "camera_to_base": camera_to_base if _valid_matrix(camera_to_base) else None,
        "camera_to_base_source": camera_to_base_source if _valid_matrix(camera_to_base) else None,
    }


def list_remote_package_files() -> list[OpenLorisRemoteFile]:
    url = "https://huggingface.co/api/datasets/shixuesong/openloris-scene/tree/main/package"
    try:
        with urllib.request.urlopen(url, context=_ssl_context(), timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return [
            OpenLorisRemoteFile("package/cafe1-1_2-package.tar", int(7.47 * 1024**3)),
            OpenLorisRemoteFile("package/office1-1_7-package.tar", int(10.6 * 1024**3)),
            OpenLorisRemoteFile("package/corridor1-1.7z", int(13.9 * 1024**3)),
            OpenLorisRemoteFile("package/home1-1_5-package.tar", int(19.0 * 1024**3)),
            OpenLorisRemoteFile("package/market1-1_3-package.tar", int(41.5 * 1024**3)),
        ]
    if not isinstance(data, list):
        return []
    files: list[OpenLorisRemoteFile] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        size = item.get("size")
        if isinstance(path, str) and isinstance(size, int):
            files.append(OpenLorisRemoteFile(path=path, size_bytes=size))
    return sorted(files, key=lambda item: (item.size_bytes, item.path))


def choose_remote_package(sequence: str, files: list[OpenLorisRemoteFile]) -> OpenLorisRemoteFile | None:
    if not files:
        return None
    normalized = sequence.lower().replace("_", "-")
    for remote in files:
        if normalized and normalized in remote.path.lower():
            return remote
    scene = sequence_scene(sequence)
    for remote in files:
        if scene != "unknown" and Path(remote.path).name.lower().startswith(scene):
            return remote
    return files[0]


def download_remote_package(remote: OpenLorisRemoteFile, out_dir: str | Path) -> Path:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "_downloads" / Path(remote.path).name
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    quoted = "/".join(urllib.request.pathname2url(part) for part in remote.path.split("/"))
    url = f"https://huggingface.co/datasets/{OPENLORIS_HF_REPO}/resolve/main/{quoted}?download=true"
    with urllib.request.urlopen(url, context=_ssl_context(), timeout=120) as response:
        with target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return target


def extract_package(archive_path: str | Path, out_dir: str | Path, sequence: str) -> Path:
    archive = Path(archive_path)
    root = Path(out_dir)
    extract_root = root / "_extracting" / sequence
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive.suffixes).lower()
    if suffixes.endswith(".tar") or suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        with tarfile.open(archive, "r:*") as tar:
            _safe_extract_tar(tar, extract_root)
        _extract_nested_7z_archives(extract_root)
    elif suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive) as zipped:
            _safe_extract_zip(zipped, extract_root)
        _extract_nested_7z_archives(extract_root)
    elif suffixes.endswith(".7z"):
        _extract_7z_archive(archive, extract_root / archive.stem)
    else:
        raise RuntimeError(f"unsupported OpenLORIS archive type: {archive.name}")
    discovered = discover_sequence_root(extract_root)
    target = openloris_sequence_dir(root, sequence)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(discovered, target)
    shutil.rmtree(extract_root.parent)
    return target


def timestamp_ns(timestamp: float) -> int:
    return int(round(float(timestamp) * 1_000_000_000))


def transform_to_matrix(values: tuple[float, float, float, float, float, float, float]) -> list[list[float]]:
    tx, ty, tz, qx, qy, qz, qw = values
    rotation = quat_to_rotation(qx, qy, qz, qw)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [tx, ty, tz]
    return [[float(value) for value in row] for row in matrix.tolist()]


def quat_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        raise ValueError("quaternion has zero norm")
    x = qx / norm
    y = qy / norm
    z = qz / norm
    w = qw / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _read_json_calibration(root: Path, camera_id: str) -> JsonDict:
    for name in ("openloris_calibration.json", "calibration.json", "sensors.json"):
        path = root / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            continue
        intrinsics = _intrinsics_from_json(data, camera_id)
        camera_to_base = _matrix_from_json(data, "camera_to_base", camera_id)
        return {
            "intrinsics": intrinsics,
            "intrinsics_source": name if intrinsics.get("available") else "missing",
            "camera_to_base": camera_to_base,
            "camera_to_base_source": {"source": name, "transform_direction": "camera_points_to_base_link"}
            if camera_to_base
            else None,
        }
    return {}


def _intrinsics_from_json(data: JsonDict, camera_id: str) -> JsonDict:
    candidate: object = data.get("intrinsics")
    if isinstance(candidate, dict):
        if camera_id in candidate and isinstance(candidate[camera_id], dict):
            candidate = candidate[camera_id]
        if all(key in candidate for key in ("fx", "fy", "cx", "cy")):
            return {
                "available": True,
                "fx": float(candidate["fx"]),  # type: ignore[index]
                "fy": float(candidate["fy"]),  # type: ignore[index]
                "cx": float(candidate["cx"]),  # type: ignore[index]
                "cy": float(candidate["cy"]),  # type: ignore[index]
                "depth_scale": float(candidate.get("depth_scale", OPENLORIS_DEPTH_SCALE)),  # type: ignore[union-attr]
                "source": "openloris_calibration_json",
            }
    return {"available": False, "status": "missing", "depth_scale": OPENLORIS_DEPTH_SCALE}


def _matrix_from_json(data: JsonDict, key: str, camera_id: str) -> list[list[float]] | None:
    candidate: object = data.get(key)
    if isinstance(candidate, dict):
        candidate = candidate.get(camera_id)
    if isinstance(candidate, list) and len(candidate) == 4 and all(isinstance(row, list) and len(row) == 4 for row in candidate):
        return [[float(value) for value in row] for row in candidate]
    return None


def _parse_camera_to_base_from_trans_matrix_yaml(
    path: Path,
    camera_id: str,
) -> tuple[list[list[float]], JsonDict] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    child_aliases = _camera_frame_aliases(camera_id)
    entries = re.finditer(
        r"parent_frame:\s*([^\n\r]+).*?child_frame:\s*([^\n\r]+).*?data:\s*\[([^\]]+)\]",
        text,
        flags=re.S,
    )
    for match in entries:
        parent = match.group(1).strip().strip("'\"")
        child = match.group(2).strip().strip("'\"")
        if parent != "base_link" or child not in child_aliases:
            continue
        values = _float_list(match.group(3))
        if len(values) != 16:
            continue
        matrix = [
            [float(values[row * 4 + col]) for col in range(4)]
            for row in range(4)
        ]
        if not _valid_matrix(matrix):
            continue
        return (
            matrix,
            {
                **OPENLORIS_TRANS_MATRIX_SOURCE,
                "path": path.as_posix(),
                "parent_frame": parent,
                "child_frame": child,
            },
        )
    return None


def _camera_frame_aliases(camera_id: str) -> set[str]:
    aliases = {camera_id}
    if camera_id.endswith("_optical_frame"):
        aliases.add(camera_id.removesuffix("_optical_frame"))
    else:
        aliases.add(f"{camera_id}_optical_frame")
    if camera_id == "d400_color":
        aliases.add("d400_color_optical_frame")
    if camera_id == "d400_depth":
        aliases.add("d400_depth_optical_frame")
    return aliases


def _parse_intrinsics_from_sensors_yaml(path: Path, camera_id: str) -> JsonDict:
    if not path.exists():
        return {"available": False, "status": "missing", "depth_scale": OPENLORIS_DEPTH_SCALE}
    text = path.read_text(encoding="utf-8", errors="replace")
    search_text = _sensor_yaml_section(text, camera_id)
    matrices = re.findall(r"intrinsics:\s*!!opencv-matrix.*?data:\s*\[([^\]]+)\]", search_text, flags=re.S)
    for matrix_text in matrices:
        values = _float_list(matrix_text)
        if len(values) == 4 and values[0] > 0.0 and values[2] > 0.0:
            return {
                "available": True,
                "fx": float(values[0]),
                "fy": float(values[2]),
                "cx": float(values[1]),
                "cy": float(values[3]),
                "depth_scale": OPENLORIS_DEPTH_SCALE,
                "source": "sensors.yaml_camera_intrinsics",
            }
        if len(values) >= 9 and values[0] > 0.0 and values[4] > 0.0:
            return {
                "available": True,
                "fx": float(values[0]),
                "fy": float(values[4]),
                "cx": float(values[2]),
                "cy": float(values[5]),
                "depth_scale": OPENLORIS_DEPTH_SCALE,
                "source": "sensors.yaml_camera_matrix",
            }
    return {"available": False, "status": "not_found_in_sensors_yaml", "depth_scale": OPENLORIS_DEPTH_SCALE}


def _sensor_yaml_section(text: str, camera_id: str) -> str:
    names = [camera_id]
    if not camera_id.endswith("_optical_frame"):
        names.append(f"{camera_id}_optical_frame")
    for name in names:
        match = re.search(rf"(?m)^{re.escape(name)}:\s*$", text)
        if match is None:
            continue
        start = match.start()
        next_match = re.search(r"(?m)^[A-Za-z0-9_]+:\s*$", text[match.end() :])
        end = match.end() + next_match.start() if next_match is not None else len(text)
        return text[start:end]
    camera_position = text.find(camera_id)
    return text[camera_position:] if camera_position >= 0 else text


def _float_list(value: str) -> list[float]:
    return [float(part) for part in re.split(r"[,\s]+", value.strip()) if part]


def _official_camera_to_base(scene: str, camera_id: str) -> list[list[float]] | None:
    table = _MARKET_CAMERA_TO_BASE if scene == "market" else _TSINGHUA_CAMERA_TO_BASE
    values = table.get(camera_id)
    if values is None and camera_id == "d400_color":
        values = table.get("d400_depth")
    if values is None:
        return None
    return transform_to_matrix(values)


def _valid_matrix(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in value)
    )


def _depth_list_path(root: Path) -> Path | None:
    for name in ("aligned_depth.txt", "depth.txt"):
        path = root / name
        if path.exists():
            return path
    return None


def _nearest_index(
    timestamps: list[float],
    target: float,
    *,
    max_difference: float,
    used: set[int] | None,
) -> int | None:
    best_index: int | None = None
    best_diff = max_difference
    for index, timestamp in enumerate(timestamps):
        if used is not None and index in used:
            continue
        diff = abs(timestamp - target)
        if diff <= best_diff:
            best_index = index
            best_diff = diff
    return best_index


def _safe_extract_tar(archive: tarfile.TarFile, out_dir: Path) -> None:
    out_root = out_dir.resolve()
    for member in archive.getmembers():
        target = (out_dir / member.name).resolve()
        if out_root not in target.parents and target != out_root:
            raise ValueError(f"unsafe archive member path: {member.name}")
    archive.extractall(out_dir)


def _safe_extract_zip(archive: zipfile.ZipFile, out_dir: Path) -> None:
    out_root = out_dir.resolve()
    for member in archive.infolist():
        target = (out_dir / member.filename).resolve()
        if out_root not in target.parents and target != out_root:
            raise ValueError(f"unsafe archive member path: {member.filename}")
    archive.extractall(out_dir)


def _extract_nested_7z_archives(root: Path) -> None:
    for archive in sorted(root.rglob("*.7z")):
        _extract_7z_archive(archive, archive.with_suffix(""))


def _extract_7z_archive(archive: Path, out_dir: Path) -> None:
    executable = _find_7z_executable()
    if executable is None:
        raise RuntimeError(
            "OpenLORIS package contains .7z archives, but no 7z executable was found; "
            "install 7-Zip or extract the package manually before import"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [executable.as_posix(), "x", "-y", f"-o{out_dir.as_posix()}", archive.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        summary = (result.stderr or result.stdout or "").strip().splitlines()
        detail = summary[-1] if summary else "no extractor output"
        raise RuntimeError(f"7z failed to extract {archive.name}: {detail}")


def _find_7z_executable() -> Path | None:
    path_value = shutil.which("7z") or shutil.which("7zz") or shutil.which("7za")
    if path_value:
        return Path(path_value)
    for candidate in (
        Path("C:/Program Files/7-Zip/7z.exe"),
        Path("C:/Program Files/NVIDIA Corporation/NVIDIA GeForce Experience/7z.exe"),
        Path("C:/Program Files/Unity/Editor/Data/Tools/7z.exe"),
    ):
        if candidate.exists():
            return candidate
    return None


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore[import-not-found]
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def write_json(path: str | Path, data: JsonDict, *, pretty: bool = True) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        if pretty:
            json.dump(data, handle, sort_keys=True, indent=2)
            handle.write("\n")
        else:
            json.dump(data, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")


def local_disk_free_gb(path: str | Path) -> float:
    usage = shutil.disk_usage(Path(path).resolve() if Path(path).exists() else Path(path).resolve().parent)
    return float(usage.free / (1024.0**3))


def environ_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
