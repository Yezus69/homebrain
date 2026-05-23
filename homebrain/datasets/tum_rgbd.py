from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ssl
import shutil
import tarfile
import urllib.request
import zlib

import numpy as np

from homebrain.messages.schema import JsonDict

TUM_RGBD_SCHEMA_VERSION = "homebrain.datasets.tum_rgbd.v0"
TUM_RGBD_ROUTE_ASSOCIATIONS_FILE = "tum_rgbd_associations.json"
TUM_RGBD_SOURCE_URL = "https://cvg.cit.tum.de/data/datasets/rgbd-dataset"
TUM_RGBD_DOWNLOADS: dict[str, str] = {
    "freiburg1_xyz": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
}
TUM_RGBD_ARCHIVE_ROOTS: dict[str, str] = {
    "freiburg1_xyz": "rgbd_dataset_freiburg1_xyz",
}
TUM_RGBD_LICENSE_NAME = "CC BY 4.0"
TUM_RGBD_LICENSE_REVIEW_STATUS = "pending_human_review"
TUM_RGBD_DEPTH_SCALE = 5000.0
TUM_RGBD_INTRINSICS: dict[str, JsonDict] = {
    "freiburg1_xyz": {
        "fx": 517.3,
        "fy": 516.5,
        "cx": 318.6,
        "cy": 255.3,
        "depth_scale": TUM_RGBD_DEPTH_SCALE,
        "source": "TUM_RGBD_freiburg1_calibration_table",
    },
}


@dataclass(frozen=True)
class TumEntry:
    timestamp: float
    path: str


@dataclass(frozen=True)
class TumGroundTruth:
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
class TumAssociation:
    rgb: TumEntry
    depth: TumEntry
    groundtruth: TumGroundTruth


def tum_sequence_dir(out_dir: str | Path, sequence: str) -> Path:
    return Path(out_dir) / sequence


def download_and_extract_sequence(out_dir: str | Path, sequence: str) -> Path:
    if sequence not in TUM_RGBD_DOWNLOADS:
        raise ValueError(f"unsupported TUM RGB-D sequence {sequence!r}")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = tum_sequence_dir(root, sequence)
    if _has_sequence_files(target):
        return target

    archive_dir = root / "_downloads"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{TUM_RGBD_ARCHIVE_ROOTS[sequence]}.tgz"
    if not archive_path.exists():
        download_url(TUM_RGBD_DOWNLOADS[sequence], archive_path)

    extract_root = root / "_extracting" / sequence
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        _safe_extract(archive, extract_root)

    extracted = _find_sequence_root(extract_root)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(extracted, target)
    shutil.rmtree(extract_root.parent)
    return target


def download_url(url: str, out_path: str | Path) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    context = _ssl_context()
    with urllib.request.urlopen(url, context=context, timeout=120) as response:
        with target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return target


def validate_sequence_dir(sequence_dir: str | Path) -> None:
    root = Path(sequence_dir)
    required = ("rgb.txt", "depth.txt", "groundtruth.txt")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"TUM RGB-D sequence is missing required files: {', '.join(missing)}")


def parse_image_list(path: str | Path) -> list[TumEntry]:
    entries: list[TumEntry] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        entries.append(TumEntry(timestamp=float(parts[0]), path=parts[1]))
    return entries


def parse_groundtruth(path: str | Path) -> list[TumGroundTruth]:
    entries: list[TumGroundTruth] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 8:
            continue
        entries.append(
            TumGroundTruth(
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


def load_associations(sequence_dir: str | Path, *, max_difference: float = 0.02) -> list[TumAssociation]:
    root = Path(sequence_dir)
    rgb_entries = parse_image_list(root / "rgb.txt")
    depth_entries = parse_image_list(root / "depth.txt")
    groundtruth_entries = parse_groundtruth(root / "groundtruth.txt")
    association_path = root / "associations.txt"
    if association_path.exists():
        associations = _load_associations_file(association_path, groundtruth_entries, max_difference=max_difference)
        if associations:
            return associations
    return associate_streams(
        rgb_entries,
        depth_entries,
        groundtruth_entries,
        max_difference=max_difference,
    )


def associate_streams(
    rgb_entries: list[TumEntry],
    depth_entries: list[TumEntry],
    groundtruth_entries: list[TumGroundTruth],
    *,
    max_difference: float,
) -> list[TumAssociation]:
    associations: list[TumAssociation] = []
    used_depth: set[int] = set()
    for rgb in rgb_entries:
        depth_index = _nearest_index(
            [entry.timestamp for entry in depth_entries],
            rgb.timestamp,
            max_difference=max_difference,
            used=used_depth,
        )
        gt_index = _nearest_index(
            [entry.timestamp for entry in groundtruth_entries],
            rgb.timestamp,
            max_difference=max_difference,
            used=None,
        )
        if depth_index is None or gt_index is None:
            continue
        used_depth.add(depth_index)
        associations.append(TumAssociation(rgb=rgb, depth=depth_entries[depth_index], groundtruth=groundtruth_entries[gt_index]))
    return associations


def read_depth_png_m(path: str | Path, *, scale: float = TUM_RGBD_DEPTH_SCALE) -> np.ndarray:
    values = read_png(Path(path))
    if values.ndim == 3:
        values = values[:, :, 0]
    depth_raw = np.asarray(values, dtype=np.float32)
    return np.where(depth_raw > np.float32(0.0), depth_raw / np.float32(scale), np.float32(0.0)).astype(np.float32)


def read_png_shape(path: str | Path) -> tuple[int, int]:
    with Path(path).open("rb") as handle:
        data = handle.read(24)
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"invalid PNG header: {path}")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height


def read_png(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"invalid PNG signature: {path}")
    offset = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError(f"truncated PNG chunk header: {path}")
        length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + length
        if payload_end + 4 > len(data):
            raise ValueError(f"truncated PNG chunk payload: {path}")
        payload = data[payload_start:payload_end]
        if chunk_type == b"IHDR":
            width = int.from_bytes(payload[0:4], "big")
            height = int.from_bytes(payload[4:8], "big")
            bit_depth = payload[8]
            color_type = payload[9]
        elif chunk_type == b"IDAT":
            idat.extend(payload)
        elif chunk_type == b"IEND":
            break
        offset = payload_end + 4

    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError(f"PNG missing IHDR: {path}")
    channels = _png_channels(color_type)
    if bit_depth not in {8, 16}:
        raise ValueError(f"unsupported PNG bit depth {bit_depth} in {path}")
    bytes_per_sample = bit_depth // 8
    bpp = max(1, channels * bytes_per_sample)
    row_bytes = width * channels * bytes_per_sample
    raw = zlib.decompress(bytes(idat))
    rows = []
    previous = bytearray(row_bytes)
    cursor = 0
    for _row in range(height):
        filter_type = raw[cursor]
        cursor += 1
        scanline = bytearray(raw[cursor : cursor + row_bytes])
        cursor += row_bytes
        recon = _unfilter_png_row(filter_type, scanline, previous, bpp)
        rows.append(bytes(recon))
        previous = recon
    reconstructed = b"".join(rows)
    if bit_depth == 16:
        array = np.frombuffer(reconstructed, dtype=">u2").astype(np.uint16)
    else:
        array = np.frombuffer(reconstructed, dtype=np.uint8)
    if channels == 1:
        return array.reshape(height, width)
    return array.reshape(height, width, channels)


def sequence_intrinsics(sequence: str) -> JsonDict:
    if sequence in TUM_RGBD_INTRINSICS:
        return dict(TUM_RGBD_INTRINSICS[sequence])
    if sequence.startswith("freiburg1"):
        return dict(TUM_RGBD_INTRINSICS["freiburg1_xyz"])
    return {
        "fx": 0.0,
        "fy": 0.0,
        "cx": 0.0,
        "cy": 0.0,
        "depth_scale": TUM_RGBD_DEPTH_SCALE,
        "source": "unknown_sequence_defaults_missing",
    }


def _load_associations_file(
    path: Path,
    groundtruth_entries: list[TumGroundTruth],
    *,
    max_difference: float,
) -> list[TumAssociation]:
    associations: list[TumAssociation] = []
    gt_timestamps = [entry.timestamp for entry in groundtruth_entries]
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        rgb = TumEntry(timestamp=float(parts[0]), path=parts[1])
        depth = TumEntry(timestamp=float(parts[2]), path=parts[3])
        gt_index = _nearest_index(gt_timestamps, rgb.timestamp, max_difference=max_difference, used=None)
        if gt_index is None:
            continue
        associations.append(TumAssociation(rgb=rgb, depth=depth, groundtruth=groundtruth_entries[gt_index]))
    return associations


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


def _safe_extract(archive: tarfile.TarFile, out_dir: Path) -> None:
    out_root = out_dir.resolve()
    for member in archive.getmembers():
        target = (out_dir / member.name).resolve()
        if out_root not in target.parents and target != out_root:
            raise ValueError(f"unsafe archive member path: {member.name}")
    archive.extractall(out_dir)


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore[import-not-found]
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _find_sequence_root(root: Path) -> Path:
    if _has_sequence_files(root):
        return root
    for child in root.iterdir():
        if child.is_dir() and _has_sequence_files(child):
            return child
    raise FileNotFoundError(f"extracted archive did not contain TUM RGB-D sequence files under {root}")


def _has_sequence_files(root: Path) -> bool:
    return all((root / name).exists() for name in ("rgb.txt", "depth.txt", "groundtruth.txt"))


def _png_channels(color_type: int) -> int:
    if color_type == 0:
        return 1
    if color_type == 2:
        return 3
    if color_type == 6:
        return 4
    raise ValueError(f"unsupported PNG color type {color_type}")


def _unfilter_png_row(filter_type: int, scanline: bytearray, previous: bytearray, bpp: int) -> bytearray:
    recon = bytearray(len(scanline))
    for index, value in enumerate(scanline):
        left = recon[index - bpp] if index >= bpp else 0
        up = previous[index] if previous else 0
        up_left = previous[index - bpp] if previous and index >= bpp else 0
        if filter_type == 0:
            predictor = 0
        elif filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = up
        elif filter_type == 3:
            predictor = (left + up) // 2
        elif filter_type == 4:
            predictor = _paeth(left, up, up_left)
        else:
            raise ValueError(f"unsupported PNG filter type {filter_type}")
        recon[index] = (value + predictor) & 0xFF
    return recon


def _paeth(left: int, up: int, up_left: int) -> int:
    p = left + up - up_left
    pa = abs(p - left)
    pb = abs(p - up)
    pc = abs(p - up_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return up
    return up_left
