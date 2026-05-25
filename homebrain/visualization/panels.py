from __future__ import annotations

from pathlib import Path

import numpy as np


def normalize_gray(
    array: np.ndarray,
    *,
    strict_2d: bool = False,
    default_shape: tuple[int, int] = (32, 32),
) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32).squeeze()
    if values.ndim != 2:
        if strict_2d:
            raise ValueError(f"expected 2D review panel, got shape {values.shape}")
        return np.zeros(default_shape, dtype=np.uint8)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    minimum = float(np.percentile(finite, 1))
    maximum = float(np.percentile(finite, 99))
    if maximum <= minimum:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = (values - np.float32(minimum)) / np.float32(maximum - minimum)
    normalized = np.where(np.isfinite(normalized), normalized, np.float32(0.0))
    return np.clip(normalized * np.float32(255.0), 0, 255).astype(np.uint8)


def thumbnail_shape(panel: np.ndarray, max_side: int = 96, min_side: int = 16) -> tuple[int, int]:
    height, width = np.asarray(panel).shape[:2]
    scale = max_side / float(max(height, width, 1))
    out_h = max(1, int(round(height * scale)))
    out_w = max(1, int(round(width * scale)))
    if max(height, width) < min_side:
        scale = min_side / float(max(height, width, 1))
        out_h = max(1, int(round(height * scale)))
        out_w = max(1, int(round(width * scale)))
    return out_h, out_w


def resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(array)
    out_h, out_w = shape
    row_index = np.linspace(0, values.shape[0] - 1, out_h).round().astype(np.int64)
    col_index = np.linspace(0, values.shape[1] - 1, out_w).round().astype(np.int64)
    return values[row_index[:, None], col_index[None, :]]


def gray_rgb(panel: np.ndarray) -> np.ndarray:
    values = np.asarray(panel, dtype=np.uint8)
    return np.stack([values, values, values], axis=2)


def tint(panel: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    values = np.asarray(panel, dtype=np.float32) / np.float32(255.0)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    for channel, channel_value in enumerate(color):
        rgb[:, :, channel] = np.clip(values * np.float32(channel_value), 0, 255).astype(np.uint8)
    return rgb


def join_with_gap(items: list[np.ndarray], *, gap: int, axis: int) -> np.ndarray:
    if not items:
        raise ValueError("cannot join empty image list")
    if axis == 1:
        height = max(item.shape[0] for item in items)
        padded = [pad_to(item, height, item.shape[1]) for item in items]
        spacer = np.full((height, gap, 3), 255, dtype=np.uint8)
        pieces: list[np.ndarray] = []
        for index, item in enumerate(padded):
            if index:
                pieces.append(spacer)
            pieces.append(item)
        return np.concatenate(pieces, axis=1)
    if axis == 0:
        width = max(item.shape[1] for item in items)
        padded = [pad_to(item, item.shape[0], width) for item in items]
        spacer = np.full((gap, width, 3), 255, dtype=np.uint8)
        pieces = []
        for index, item in enumerate(padded):
            if index:
                pieces.append(spacer)
            pieces.append(item)
        return np.concatenate(pieces, axis=0)
    raise ValueError(f"unsupported join axis: {axis}")


def pad_to(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[: image.shape[0], : image.shape[1], :] = image
    return canvas


def write_ppm(path: str | Path, image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"PPM image must be HxWx3, got shape {image.shape}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    height, width, _channels = image.shape
    with target.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(np.asarray(image, dtype=np.uint8).tobytes(order="C"))


def write_pgm(path: str | Path, image: np.ndarray) -> None:
    if image.ndim != 2:
        raise ValueError(f"PGM image must be 2D, got shape {image.shape}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape
    with target.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        handle.write(np.asarray(image, dtype=np.uint8).tobytes(order="C"))
