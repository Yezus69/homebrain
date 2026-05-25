from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.messages.schema import FrameEvent, JsonDict

DEFAULT_MOGE2_MODEL_ID = "Ruicheng/moge-2-vits-normal"
MOGE_INSTALL_HINT = (
    "MoGe is not installed or importable. Install the official MoGe package/check out the "
    "official MoGe repository in this environment, or set HOMEBRAIN_MOGE_DIR so Python can "
    "import `moge.model.v2.MoGeModel`. Then provide a local checkpoint path, set "
    "HOMEBRAIN_MOGE_MODEL_ID, or set HOMEBRAIN_MOGE_ALLOW_DOWNLOAD=1 to allow the default "
    "MoGe-2 small model."
)


def run_scene_teacher(
    log_dir: str | Path,
    frames: list[FrameEvent],
    model_id: str | None,
    model_dir: str | Path | None,
    checkpoint: str | Path | None,
    device: str | None,
) -> JsonDict:
    """Run the official MoGe v2 interface and return SceneTeacher-compatible dicts."""
    log_root = Path(log_dir)
    model_dir_path = Path(model_dir) if model_dir is not None else _env_path("HOMEBRAIN_MOGE_DIR")
    if model_dir_path is not None and model_dir_path.exists() and str(model_dir_path) not in sys.path:
        sys.path.insert(0, str(model_dir_path))

    try:
        from moge.model.v2 import MoGeModel  # type: ignore
    except Exception as exc:  # noqa: BLE001 - setup error must be clear and actionable.
        raise RuntimeError(MOGE_INSTALL_HINT) from exc

    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001 - torch is only required for the real adapter.
        raise RuntimeError(
            "MoGe import succeeded, but torch is not importable. Install the torch build required by "
            "the selected MoGe package before running backend=real."
        ) from exc

    model_source, resolved_model_id = _resolve_model_source(model_id=model_id, checkpoint=checkpoint)
    try:
        model = MoGeModel.from_pretrained(model_source)
    except Exception as exc:  # noqa: BLE001 - preserve the true model/checkpoint failure.
        raise RuntimeError(f"MoGe model load failed for source {model_source!r}: {exc}") from exc

    resolved_device = device or ("cuda" if _torch_cuda_available(torch) else "cpu")
    try:
        if hasattr(model, "to"):
            model = model.to(resolved_device)
        if hasattr(model, "eval"):
            model.eval()
    except Exception as exc:  # noqa: BLE001 - device setup is an external runtime issue.
        raise RuntimeError(f"MoGe model device setup failed for device {resolved_device!r}: {exc}") from exc

    predictions: list[JsonDict] = []
    for frame in frames:
        rgb = _decode_frame_rgb(log_root=log_root, frame=frame)
        input_image = torch.as_tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        if hasattr(input_image, "to"):
            input_image = input_image.to(resolved_device)
        try:
            with torch.no_grad():
                output = model.infer(input_image)
        except Exception as exc:  # noqa: BLE001 - real inference failures should surface.
            raise RuntimeError(f"MoGe inference failed for frame_id={frame.frame_id}: {exc}") from exc
        predictions.append(
            _prediction_from_output(
                output=output,
                frame=frame,
                model_source=model_source,
                model_id=resolved_model_id,
                device=resolved_device,
            )
        )
    return {"frames": predictions, "windows": []}


def _resolve_model_source(*, model_id: str | None, checkpoint: str | Path | None) -> tuple[str, str]:
    checkpoint_path = Path(checkpoint) if checkpoint is not None else _env_path("HOMEBRAIN_MOGE_CHECKPOINT")
    if checkpoint_path is not None and checkpoint_path.exists():
        source = checkpoint_path.as_posix()
        return source, source

    env_model_id = os.environ.get("HOMEBRAIN_MOGE_MODEL_ID")
    if env_model_id:
        return env_model_id, env_model_id

    allow_default_download = os.environ.get("HOMEBRAIN_MOGE_ALLOW_DOWNLOAD") == "1"
    if allow_default_download:
        selected = model_id if model_id and model_id != "moge-local" else DEFAULT_MOGE2_MODEL_ID
        return selected, selected

    missing_checkpoint = checkpoint_path.as_posix() if checkpoint_path is not None else None
    raise RuntimeError(
        "MoGe is importable, but no allowed model source is configured. Provide an existing "
        f"checkpoint path via --checkpoint/HOMEBRAIN_MOGE_CHECKPOINT (current: {missing_checkpoint!r}), "
        "set HOMEBRAIN_MOGE_MODEL_ID for an operator-selected model id, or set "
        "HOMEBRAIN_MOGE_ALLOW_DOWNLOAD=1 to allow the default Ruicheng/moge-2-vits-normal download."
    )


def _prediction_from_output(
    *,
    output: Any,
    frame: FrameEvent,
    model_source: str,
    model_id: str,
    device: str,
) -> JsonDict:
    if not isinstance(output, dict):
        raise RuntimeError(f"MoGe infer returned {type(output)!r}; expected a dict with depth/points/mask")
    output_keys = sorted(str(key) for key in output.keys())
    depth = _to_numpy(output.get("depth"))
    points = _to_numpy(output.get("points"))
    intrinsics = _to_numpy(output.get("intrinsics"))
    mask = _to_numpy(output.get("mask"))
    if depth is None:
        raise RuntimeError(f"MoGe output is missing required key 'depth'; keys={output_keys}")
    if points is None:
        raise RuntimeError(f"MoGe output is missing required key 'points'; keys={output_keys}")
    if intrinsics is None:
        raise RuntimeError(f"MoGe output is missing required key 'intrinsics'; keys={output_keys}")
    if mask is None:
        raise RuntimeError(f"MoGe output is missing required key 'mask'; keys={output_keys}")

    depth_hw = _squeeze_hw(depth, key="depth").astype(np.float32)
    point_map = _normalize_points(points).astype(np.float32)
    validity_mask = (_squeeze_hw(mask, key="mask") > 0).astype(np.uint8)
    confidence = _extract_confidence(output)
    if confidence is None:
        finite_depth = np.isfinite(depth_hw) & (depth_hw > np.float32(0.0))
        finite_points = np.isfinite(point_map).all(axis=2)
        confidence = (finite_depth & finite_points & (validity_mask > 0)).astype(np.float32)
        confidence_source = "finite_depth_points_and_moge_mask"
    else:
        confidence = _squeeze_hw(confidence, key="confidence").astype(np.float32)
        confidence_source = "moge_output"

    return {
        "depth": depth_hw,
        "point_map": point_map,
        "intrinsics": np.asarray(intrinsics, dtype=np.float32).squeeze(),
        "confidence": confidence,
        "validity_mask": validity_mask,
        "dynamic_motion_mask": np.zeros(depth_hw.shape, dtype=np.uint8),
        "extra_metadata": {
            "backend": "real",
            "adapter": "homebrain.teachers.moge_official_adapter:run_scene_teacher",
            "model_id": model_id,
            "model_source": model_source,
            "output_keys": output_keys,
            "device": device,
            "scale_status": "metric",
            "scale_source": "moge2_metric_output",
            "confidence_source": confidence_source,
            "validity_mask_source": "moge_output_mask",
            "floor_traversable_mask_source": "absent_not_estimated_by_adapter",
            "obstacle_risk_mask_source": "absent_not_estimated_by_adapter",
            "dynamic_motion_mask_source": "unavailable_static_zero_placeholder",
            "placeholder_masks": {
                "dynamic_motion_mask": "zeros because MoGe is single-frame/static geometry; unavailable, not motion truth"
            },
            "source_data_ref": frame.data_ref,
        },
    }


def _extract_confidence(output: dict[Any, Any]) -> np.ndarray | None:
    for key in ("confidence", "conf", "confidence_map"):
        if key in output:
            return _to_numpy(output.get(key))
    return None


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _squeeze_hw(value: np.ndarray, *, key: str) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise RuntimeError(f"MoGe output {key!r} must be HxW, got shape {array.shape}")
    return array


def _normalize_points(value: np.ndarray) -> np.ndarray:
    points = np.asarray(value)
    while points.ndim > 3 and points.shape[0] == 1:
        points = points[0]
    if points.ndim == 3 and points.shape[-1] == 3:
        return points
    if points.ndim == 3 and points.shape[0] == 3:
        return np.moveaxis(points, 0, 2)
    raise RuntimeError(f"MoGe output 'points' must be HxWx3 or 3xHxW, got shape {points.shape}")


def _decode_frame_rgb(*, log_root: Path, frame: FrameEvent) -> np.ndarray:
    data_path = log_root / frame.data_ref
    if not data_path.exists():
        raise RuntimeError(f"frame data_ref does not exist: {data_path}")
    frame_format = frame.format.lower()
    if frame_format == "rgb8":
        raw = np.frombuffer(data_path.read_bytes(), dtype=np.uint8)
        expected = frame.height * frame.width * 3
        if raw.size != expected:
            raise RuntimeError(f"rgb8 frame {data_path} has {raw.size} bytes, expected {expected}")
        return raw.reshape((frame.height, frame.width, 3)).copy()
    if frame_format == "gray8":
        raw = np.frombuffer(data_path.read_bytes(), dtype=np.uint8)
        expected = frame.height * frame.width
        if raw.size != expected:
            raise RuntimeError(f"gray8 frame {data_path} has {raw.size} bytes, expected {expected}")
        gray = raw.reshape((frame.height, frame.width)).copy()
        return np.stack([gray, gray, gray], axis=2)

    suffix = data_path.suffix.lower()
    if suffix in {".ppm", ".pgm"}:
        return _decode_netpbm(data_path)
    if suffix in {".jpg", ".jpeg", ".png"} or frame_format in {"jpeg", "jpg", "png"}:
        return _decode_with_optional_image_io(data_path)
    if frame_format in {"ppm", "pgm"}:
        return _decode_netpbm(data_path)
    raise RuntimeError(f"unsupported frame encoding for MoGe adapter: format={frame.format!r} path={data_path}")


def _decode_netpbm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    stream = io.BytesIO(data)
    magic = _read_token(stream)
    if magic not in {b"P5", b"P6"}:
        raise RuntimeError(f"unsupported Netpbm magic in {path}: {magic!r}")
    width = int(_read_token(stream))
    height = int(_read_token(stream))
    max_value = int(_read_token(stream))
    if max_value <= 0 or max_value > 255:
        raise RuntimeError(f"only 8-bit PPM/PGM images are supported, got max value {max_value}")
    channels = 3 if magic == b"P6" else 1
    raw = np.frombuffer(stream.read(width * height * channels), dtype=np.uint8)
    expected = width * height * channels
    if raw.size != expected:
        raise RuntimeError(f"Netpbm payload in {path} has {raw.size} bytes, expected {expected}")
    if channels == 3:
        return raw.reshape((height, width, 3)).copy()
    gray = raw.reshape((height, width)).copy()
    return np.stack([gray, gray, gray], axis=2)


def _read_token(stream: io.BytesIO) -> bytes:
    token = bytearray()
    while True:
        char = stream.read(1)
        if not char:
            raise RuntimeError("unexpected EOF while reading Netpbm header")
        if char == b"#":
            while char not in {b"\n", b""}:
                char = stream.read(1)
            continue
        if char.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(char)


def _decode_with_optional_image_io(path: Path) -> np.ndarray:
    cv2_error: str | None = None
    try:
        import cv2  # type: ignore

        image = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cv2.imread returned None for {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        cv2_error = str(exc)

    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except ImportError as exc:
        raise RuntimeError(
            "Encoded JPG/PNG input requires opencv-python or Pillow in this environment. "
            f"Could not decode {path}."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        detail = f" cv2 error: {cv2_error}" if cv2_error else ""
        raise RuntimeError(f"Could not decode encoded image {path}: {exc}.{detail}") from exc


def _torch_cuda_available(torch_module: Any) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not hasattr(cuda, "is_available"):
        return False
    try:
        return bool(cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None
