from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from homebrain.brain.direct_bev_student_v0 import DIRECT_BEV_CHANNELS, load_checkpoint
from homebrain.data.spatial_dataset import load_example_npz, read_json
from homebrain.datasets.tum_rgbd import read_depth_png_m, read_png
from homebrain.train.train_direct_bev_student_v0 import RealRGBDRouteBEVDataset

VOXEL_VIEWER_SCHEMA_VERSION = "homebrain.bev_voxel_viewer.v0"
DEFAULT_CHECKPOINT = "artifacts/goal31_direct_bev_hazard_real_xyz_val_verified_600/checkpoint.pt"
DEFAULT_PACK = "artifacts/goal31_real_rgbd_hazard_pack_xyz_val_verified"
PREDICTION_CHANNELS: tuple[str, ...] = (*DIRECT_BEV_CHANNELS, "hazard", "dynamic_risk")
GT_CHANNELS: dict[str, str] = {
    "free": "target_current_bev_free",
    "occupied": "target_current_bev_occupied",
    "unknown": "target_current_bev_unknown",
    "traversable": "target_current_bev_traversable",
    "risky": "target_current_bev_risky",
    "hazard": "target_bev_hazard",
    "dynamic_risk": "target_dynamic_residual_risk",
}


def build_bev_voxel_viewer(
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    pack_dir: str | Path = DEFAULT_PACK,
    out_dir: str | Path,
    split: str = "val",
    max_examples: int = 32,
    max_scene_voxels: int = 12_000,
    device_name: str | None = "cpu",
) -> Path:
    output = Path(out_dir)
    data_dir = output / "data"
    frame_dir = output / "frames"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    pack_root = Path(pack_dir)
    manifest = read_json(pack_root / "manifest.json")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()
    dataset = RealRGBDRouteBEVDataset(pack_root, split=split, image_size=tuple(model.config.image_size), cache_in_memory=False)
    selected_indices = _selected_record_indices(dataset.records, max_examples=max_examples)
    examples = [
        _example_payload(
            dataset=dataset,
            index=index,
            pack_root=pack_root,
            frame_dir=frame_dir,
            model=model,
            device=device,
            max_scene_voxels=max_scene_voxels,
        )
        for index in selected_indices
    ]
    cell_size_m = float(manifest.get("meters_per_cell", 0.05) or 0.05)
    voxel_size_m = 0.01
    subdivisions = max(1, int(round(cell_size_m / voxel_size_m)))
    data = {
        "schema_version": VOXEL_VIEWER_SCHEMA_VERSION,
        "checkpoint": Path(checkpoint).as_posix(),
        "pack": pack_root.as_posix(),
        "split": split,
        "model_name": payload.get("model_name"),
        "metadata": {
            "replay_only": payload.get("metadata", {}).get("replay_only"),
            "not_executed": payload.get("metadata", {}).get("not_executed"),
            "control_safe": payload.get("metadata", {}).get("control_safe"),
            "raw_pwm_emitted": payload.get("metadata", {}).get("raw_pwm_emitted"),
            "hardware_validated": payload.get("metadata", {}).get("hardware_validated"),
            "no_teacher_fields_at_runtime": payload.get("metadata", {}).get("no_teacher_fields_at_runtime"),
        },
        "grid_shape": list(dataset.bev_shape),
        "cell_size_m": cell_size_m,
        "voxel_size_m": voxel_size_m,
        "camera_height_m": 0.35,
        "subdivisions_per_cell": subdivisions,
        "visible_extent_m": [dataset.bev_shape[1] * cell_size_m, dataset.bev_shape[0] * cell_size_m],
        "scene_voxel_frame": {
            "x": "left_m",
            "y": "up_m",
            "z": "forward_m",
            "origin": "rgbd_camera",
            "note": "3D scene voxels are reconstructed from RGB-D depth. Checkpoint outputs are BEV overlays, not volumetric predictions.",
        },
        "channels": list(PREDICTION_CHANNELS),
        "examples": examples,
    }
    _write_json_compact(data_dir / "viewer_data.json", data)
    _write_static_files(output)
    return output / "index.html"


def _selected_record_indices(records: list[dict[str, Any]], *, max_examples: int) -> list[int]:
    indices = list(range(len(records)))
    if max_examples <= 0 or len(indices) <= max_examples:
        return indices
    positives = [index for index, record in enumerate(records) if record.get("hazard_positive") is True]
    negatives = [index for index in indices if index not in set(positives)]
    return (positives + negatives)[:max_examples]


def _example_payload(
    *,
    dataset: RealRGBDRouteBEVDataset,
    index: int,
    pack_root: Path,
    frame_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    max_scene_voxels: int,
) -> dict[str, Any]:
    item = dataset[index]
    record = dataset.records[index]
    arrays = load_example_npz(pack_root / str(record["example_path"]))
    with torch.no_grad():
        outputs = model(
            item["rgb"][None, ...].to(device),
            depth=item["depth"][None, ...].to(device),
            sensor_mask=item["sensor_mask"][None, ...].to(device),
            pose_delta_prev=item["pose_delta_prev"][None, ...].to(device),
            previous_action=item["previous_action"][None, ...].to(device),
        )
    pred_bev = torch.sigmoid(outputs["bev_logits"])[0].detach().cpu().numpy().astype(np.float32)
    pred_dynamic = torch.sigmoid(outputs["dynamic_risk_logits"])[0, 0].detach().cpu().numpy().astype(np.float32)
    pred_hazard = torch.sigmoid(outputs["hazard_logits"])[0, 0].detach().cpu().numpy().astype(np.float32)
    pred = {name: _array_payload(pred_bev[channel_index]) for channel_index, name in enumerate(DIRECT_BEV_CHANNELS)}
    pred["hazard"] = _array_payload(pred_hazard)
    pred["dynamic_risk"] = _array_payload(pred_dynamic)
    gt = {
        channel: _array_payload(np.asarray(arrays.get(array_name, np.zeros(dataset.bev_shape, dtype=np.float32)), dtype=np.float32))
        for channel, array_name in GT_CHANNELS.items()
    }
    image_rel = _copy_rgb_preview(record, frame_dir, index=index)
    scene_voxels = _scene_voxels_from_rgbd(record, voxel_size_m=0.01, max_voxels=max_scene_voxels)
    return {
        "index": int(index),
        "route_id": str(record.get("route_id", "")),
        "split": str(record.get("split", "")),
        "frame_id": int(np.asarray(arrays.get("frame_index", np.asarray(index))).item()),
        "timestamp_ns": int(record.get("timestamp_ns", 0) or 0),
        "rgb_preview": image_rel,
        "hazard_positive": bool(record.get("hazard_positive") is True),
        "target_bev_hazard_positive_cells": int(record.get("target_bev_hazard_positive_cells", 0) or 0),
        "pred_positive_counts_at_0_5": {channel: _count_positive(pred[channel]) for channel in pred},
        "gt_positive_counts_at_0_5": {channel: _count_positive(gt[channel]) for channel in gt},
        "scene_voxels": scene_voxels,
        "pred": pred,
        "gt": gt,
    }


def _array_payload(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float32)
    return {
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "values": np.round(np.clip(values, 0.0, 1.0).reshape(-1), 4).tolist(),
    }


def _count_positive(payload: dict[str, Any]) -> int:
    return int(sum(1 for value in payload["values"] if float(value) >= 0.5))


def _copy_rgb_preview(record: dict[str, Any], frame_dir: Path, *, index: int) -> str | None:
    source_value = record.get("rgb_path")
    if not isinstance(source_value, str) or not source_value:
        return None
    source = Path(source_value)
    if not source.exists():
        return None
    suffix = source.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        return None
    target = frame_dir / f"example_{index:04d}{suffix}"
    shutil.copyfile(source, target)
    return target.relative_to(frame_dir.parent).as_posix()


def _scene_voxels_from_rgbd(record: dict[str, Any], *, voxel_size_m: float, max_voxels: int) -> dict[str, Any]:
    depth_path = record.get("depth_path")
    rgb_path = record.get("rgb_path")
    intrinsics = record.get("camera_intrinsics") if isinstance(record.get("camera_intrinsics"), dict) else {}
    if not isinstance(depth_path, str) or not isinstance(rgb_path, str):
        return _empty_scene_voxels("missing_rgb_or_depth_path")
    depth_source = Path(depth_path)
    rgb_source = Path(rgb_path)
    if not depth_source.exists() or not rgb_source.exists():
        return _empty_scene_voxels("rgb_or_depth_path_missing")
    scale = float(intrinsics.get("depth_scale", 5000.0) or 5000.0)
    depth = read_depth_png_m(depth_source, scale=scale)
    rgb = _read_rgb_image(rgb_source)
    if rgb.shape[:2] != depth.shape:
        rgb = _resize_rgb_nearest(rgb, depth.shape)
    fx = _finite_intrinsic(intrinsics.get("fx"), default=float(max(depth.shape)))
    fy = _finite_intrinsic(intrinsics.get("fy"), default=float(max(depth.shape)))
    cx = _finite_intrinsic(intrinsics.get("cx"), default=(depth.shape[1] - 1) / 2.0)
    cy = _finite_intrinsic(intrinsics.get("cy"), default=(depth.shape[0] - 1) / 2.0)
    valid = np.isfinite(depth) & (depth > np.float32(0.05)) & (depth <= np.float32(4.0))
    valid_count = int(np.count_nonzero(valid))
    if valid_count <= 0:
        return _empty_scene_voxels("no_valid_depth")
    target_samples = max(1, max_voxels * 3)
    stride = max(1, int(math.sqrt(valid_count / target_samples)))
    rows, cols = np.nonzero(valid[::stride, ::stride])
    rows = rows.astype(np.int64) * stride
    cols = cols.astype(np.int64) * stride
    if rows.size == 0:
        return _empty_scene_voxels("no_depth_after_sampling")
    z = depth[rows, cols].astype(np.float32)
    x_cam = ((cols.astype(np.float32) - np.float32(cx)) / np.float32(fx)) * z
    y_cam = ((rows.astype(np.float32) - np.float32(cy)) / np.float32(fy)) * z
    left = -x_cam
    up = -y_cam
    forward = z
    coords_cm = np.stack(
        [
            np.round(left / np.float32(voxel_size_m)),
            np.round(up / np.float32(voxel_size_m)),
            np.round(forward / np.float32(voxel_size_m)),
        ],
        axis=1,
    ).astype(np.int16)
    colors = rgb[rows, cols, :3].astype(np.uint8)
    unique: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for coord, color in zip(coords_cm, colors):
        key = (int(coord[0]), int(coord[1]), int(coord[2]))
        if key not in unique:
            unique[key] = (int(color[0]), int(color[1]), int(color[2]))
    items = sorted(unique.items(), key=lambda item: (item[0][2], item[0][0], item[0][1]))
    if len(items) > max_voxels:
        step = len(items) / float(max_voxels)
        items = [items[min(int(index * step), len(items) - 1)] for index in range(max_voxels)]
    flat_coords: list[int] = []
    flat_colors: list[int] = []
    for coord, color in items:
        flat_coords.extend(coord)
        flat_colors.extend(color)
    return {
        "voxel_size_m": float(voxel_size_m),
        "coordinate_quantization": "signed_centimeter_offsets_from_rgbd_camera",
        "count": int(len(items)),
        "source_valid_depth_pixel_count": valid_count,
        "sample_stride_px": int(stride),
        "coords_cm": flat_coords,
        "rgb": flat_colors,
    }


def _empty_scene_voxels(reason: str) -> dict[str, Any]:
    return {
        "voxel_size_m": 0.01,
        "coordinate_quantization": "signed_centimeter_offsets_from_rgbd_camera",
        "count": 0,
        "source_valid_depth_pixel_count": 0,
        "sample_stride_px": 0,
        "coords_cm": [],
        "rgb": [],
        "missing_reason": reason,
    }


def _read_rgb_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".ppm":
        return _read_ppm(path)
    try:
        from PIL import Image  # type: ignore[import-not-found]

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception:
        image = read_png(path)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        return np.asarray(image[:, :, :3], dtype=np.uint8)


def _read_ppm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        if handle.readline().strip() != b"P6":
            raise ValueError(f"unsupported PPM file: {path}")
        dims = handle.readline().strip()
        while dims.startswith(b"#"):
            dims = handle.readline().strip()
        width, height = [int(value) for value in dims.split()[:2]]
        max_value = int(handle.readline().strip())
        if max_value != 255:
            raise ValueError(f"unsupported PPM max value {max_value}")
        data = np.frombuffer(handle.read(width * height * 3), dtype=np.uint8)
    return data.reshape(height, width, 3)


def _resize_rgb_nearest(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out_h, out_w = shape
    rows = np.linspace(0, rgb.shape[0] - 1, out_h).round().astype(np.int64)
    cols = np.linspace(0, rgb.shape[1] - 1, out_w).round().astype(np.int64)
    return np.asarray(rgb, dtype=np.uint8)[rows[:, None], cols[None, :]]


def _finite_intrinsic(value: Any, *, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0.0:
        return float(value)
    return float(default)


def _write_json_compact(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, separators=(",", ":"))
        handle.write("\n")


def _write_static_files(output: Path) -> None:
    (output / "index.html").write_text(_INDEX_HTML, encoding="utf-8")
    (output / "style.css").write_text(_STYLE_CSS, encoding="utf-8")
    (output / "viewer.js").write_text(_VIEWER_JS, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a static Three.js 1 cm voxel BEV checkpoint-vs-GT viewer.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pack", default=DEFAULT_PACK)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--max-scene-voxels", type=int, default=12_000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    index = build_bev_voxel_viewer(
        checkpoint=args.checkpoint,
        pack_dir=args.pack,
        out_dir=args.out,
        split=args.split,
        max_examples=args.max_examples,
        max_scene_voxels=args.max_scene_voxels,
        device_name=args.device,
    )
    print(f"wrote BEV voxel viewer to {index.as_posix()}")
    return 0


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HomeBrain BEV Voxel Viewer</title>
  <link rel="stylesheet" href="./style.css">
</head>
<body>
  <div id="shell">
    <header id="toolbar">
      <div id="title">HomeBrain BEV Voxels</div>
      <select id="exampleSelect" aria-label="Example"></select>
      <select id="channelSelect" aria-label="Channel"></select>
      <label class="rangeLabel">Threshold <input id="threshold" type="range" min="0" max="1" step="0.01" value="0.5"></label>
      <output id="thresholdValue">0.50</output>
      <button id="playButton" type="button" title="Play examples">Play</button>
      <button id="resetButton" type="button" title="Reset camera">Reset</button>
    </header>
    <main id="workspace">
      <canvas id="scene"></canvas>
      <aside id="inspector">
        <img id="rgbPreview" alt="">
        <dl id="stats"></dl>
      </aside>
    </main>
  </div>
  <script type="module" src="./viewer.js"></script>
</body>
</html>
"""


_STYLE_CSS = """
html, body {
  height: 100%;
  margin: 0;
  background: #111418;
  color: #e6edf3;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
#shell {
  height: 100%;
  display: grid;
  grid-template-rows: 48px 1fr;
}
#toolbar {
  display: grid;
  grid-template-columns: minmax(160px, 1fr) minmax(230px, 1.5fr) 140px 220px 48px 72px 72px;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  border-bottom: 1px solid #2b333d;
  background: #171b21;
}
#title {
  font-size: 15px;
  font-weight: 650;
  white-space: nowrap;
}
select, button {
  min-height: 32px;
  border: 1px solid #3b4652;
  background: #20262e;
  color: #f2f5f8;
  border-radius: 6px;
  padding: 0 10px;
  font: inherit;
}
button:hover, select:hover { border-color: #6c7a89; }
.rangeLabel {
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  color: #bac4cf;
}
#threshold { width: 100%; }
#thresholdValue {
  color: #d8e1ea;
  font-variant-numeric: tabular-nums;
}
#workspace {
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 280px;
}
#scene {
  width: 100%;
  height: 100%;
  display: block;
}
#inspector {
  border-left: 1px solid #2b333d;
  background: #151a20;
  padding: 12px;
  overflow: auto;
}
#rgbPreview {
  width: 100%;
  max-height: 190px;
  object-fit: contain;
  background: #0b0e12;
  border: 1px solid #2b333d;
}
#stats {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px 12px;
  margin: 14px 0 0;
  font-size: 13px;
}
#stats dt { color: #93a2b2; }
#stats dd {
  margin: 0;
  color: #f2f5f8;
  font-variant-numeric: tabular-nums;
}
@media (max-width: 900px) {
  #shell { grid-template-rows: auto 1fr; }
  #toolbar {
    grid-template-columns: 1fr 1fr;
    align-items: stretch;
  }
  #workspace { grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr) 230px; }
  #inspector { border-left: 0; border-top: 1px solid #2b333d; }
}
"""


_VIEWER_JS = """
const data = await fetch('./data/viewer_data.json').then((response) => response.json());
let THREE;
try {
  THREE = await import('./vendor/three.module.js');
} catch (_error) {
  THREE = await import('https://unpkg.com/three@0.160.0/build/three.module.js');
}

const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x111418, 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 100);
const state = {
  exampleIndex: 0,
  channel: 'hazard',
  threshold: 0.5,
  playing: false,
  theta: Math.PI * 0.22,
  phi: Math.PI * 0.31,
  radius: 6.4,
  target: new THREE.Vector3(0, 0, 0),
  drag: null,
};

const voxelGroup = new THREE.Group();
scene.add(voxelGroup);
window.__homebrainVoxelViewer = { data, state, scene, voxelGroup };
scene.add(new THREE.HemisphereLight(0xf8fbff, 0x26313d, 1.4));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
keyLight.position.set(2.5, 4, 3);
scene.add(keyLight);

const gridGroup = new THREE.Group();
const minorGrid = new THREE.GridHelper(8, 800, 0x35404c, 0x222a33);
const majorGrid = new THREE.GridHelper(8, 80, 0x576675, 0x35404c);
minorGrid.position.y = -(data.camera_height_m || 0.35);
majorGrid.position.y = -(data.camera_height_m || 0.35);
gridGroup.add(minorGrid, majorGrid);
scene.add(gridGroup);

const exampleSelect = document.getElementById('exampleSelect');
const channelSelect = document.getElementById('channelSelect');
const threshold = document.getElementById('threshold');
const thresholdValue = document.getElementById('thresholdValue');
const playButton = document.getElementById('playButton');
const resetButton = document.getElementById('resetButton');
const rgbPreview = document.getElementById('rgbPreview');
const stats = document.getElementById('stats');

for (const [index, example] of data.examples.entries()) {
  const option = document.createElement('option');
  option.value = String(index);
  option.textContent = `${example.route_id} / frame ${example.frame_id}`;
  exampleSelect.appendChild(option);
}
for (const channel of [...data.channels, 'difference']) {
  const option = document.createElement('option');
  option.value = channel;
  option.textContent = channel === 'difference' ? 'hazard difference' : channel;
  channelSelect.appendChild(option);
}
channelSelect.value = state.channel;

exampleSelect.addEventListener('change', () => {
  state.exampleIndex = Number(exampleSelect.value);
  rebuildVoxels();
});
channelSelect.addEventListener('change', () => {
  state.channel = channelSelect.value;
  rebuildVoxels();
});
threshold.addEventListener('input', () => {
  state.threshold = Number(threshold.value);
  thresholdValue.textContent = state.threshold.toFixed(2);
  rebuildVoxels();
});
playButton.addEventListener('click', () => {
  state.playing = !state.playing;
  playButton.textContent = state.playing ? 'Pause' : 'Play';
});
resetButton.addEventListener('click', () => resetCamera());

let lastPlayTime = performance.now();
function animate(now) {
  requestAnimationFrame(animate);
  resize();
  if (state.playing && now - lastPlayTime > 900 && data.examples.length > 1) {
    state.exampleIndex = (state.exampleIndex + 1) % data.examples.length;
    exampleSelect.value = String(state.exampleIndex);
    rebuildVoxels();
    lastPlayTime = now;
  }
  updateCamera();
  renderer.render(scene, camera);
}

function resetCamera() {
  state.theta = Math.PI * 0.22;
  state.phi = Math.PI * 0.31;
  state.radius = 6.4;
  state.target.set(0, 0, 0);
}

function updateCamera() {
  const sinPhi = Math.sin(state.phi);
  camera.position.set(
    state.target.x + state.radius * sinPhi * Math.sin(state.theta),
    state.target.y + state.radius * Math.cos(state.phi),
    state.target.z + state.radius * sinPhi * Math.cos(state.theta),
  );
  camera.lookAt(state.target);
}

function resize() {
  const width = canvas.clientWidth || 1;
  const height = canvas.clientHeight || 1;
  if (canvas.width !== Math.floor(width * renderer.getPixelRatio()) ||
      canvas.height !== Math.floor(height * renderer.getPixelRatio())) {
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
}

canvas.addEventListener('pointerdown', (event) => {
  canvas.setPointerCapture(event.pointerId);
  state.drag = { x: event.clientX, y: event.clientY, theta: state.theta, phi: state.phi };
});
canvas.addEventListener('pointermove', (event) => {
  if (!state.drag) return;
  const dx = event.clientX - state.drag.x;
  const dy = event.clientY - state.drag.y;
  state.theta = state.drag.theta - dx * 0.008;
  state.phi = Math.min(Math.PI * 0.48, Math.max(0.18, state.drag.phi + dy * 0.006));
});
canvas.addEventListener('pointerup', () => { state.drag = null; });
canvas.addEventListener('wheel', (event) => {
  event.preventDefault();
  state.radius = Math.min(16, Math.max(1.2, state.radius * Math.exp(event.deltaY * 0.001)));
}, { passive: false });

function rebuildVoxels() {
  voxelGroup.clear();
  const example = data.examples[state.exampleIndex];
  const channel = state.channel;
  const cellSize = data.cell_size_m;
  const voxel = data.voxel_size_m;
  const sub = data.subdivisions_per_cell;
  const [rows, cols] = data.grid_shape;
  const extentX = cols * cellSize;
  const offset = extentX * 0.58 + 0.42;
  voxelGroup.add(buildSceneMesh(example, -offset));
  voxelGroup.add(buildSceneMesh(example, offset));
  if (channel === 'difference') {
    voxelGroup.add(buildSideMesh(example, 'gt', 'hazard', -offset, rows, cols, sub, voxel, differenceColor));
    voxelGroup.add(buildSideMesh(example, 'pred', 'hazard', offset, rows, cols, sub, voxel, differenceColor));
  } else {
    voxelGroup.add(buildSideMesh(example, 'gt', channel, -offset, rows, cols, sub, voxel, valueColor));
    voxelGroup.add(buildSideMesh(example, 'pred', channel, offset, rows, cols, sub, voxel, valueColor));
  }
  updateInspector(example);
}

function buildSceneMesh(example, sideOffset) {
  const sceneVoxels = example.scene_voxels || { count: 0, coords_cm: [] };
  const count = Math.max(0, sceneVoxels.count || 0);
  const geometry = new THREE.BoxGeometry(data.voxel_size_m, data.voxel_size_m, data.voxel_size_m);
  const material = new THREE.MeshBasicMaterial({
    color: 0x8a98a6,
    transparent: true,
    opacity: 0.28,
    depthWrite: false,
  });
  const mesh = new THREE.InstancedMesh(geometry, material, count);
  mesh.name = sideOffset < 0 ? 'gt_rgbd_3d_scene' : 'pred_rgbd_3d_scene';
  const matrix = new THREE.Matrix4();
  const coords = sceneVoxels.coords_cm || [];
  const maxForwardM = (data.grid_shape[0] || 64) * data.cell_size_m;
  for (let i = 0; i < count; i += 1) {
    const leftM = (coords[i * 3] || 0) * 0.01;
    const upM = (coords[i * 3 + 1] || 0) * 0.01;
    const forwardM = (coords[i * 3 + 2] || 0) * 0.01;
    matrix.makeTranslation(sideOffset + leftM, upM, forwardM - maxForwardM / 2);
    mesh.setMatrixAt(i, matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
  return mesh;
}

function buildSideMesh(example, source, channel, sideOffset, rows, cols, sub, voxel, colorFn) {
  const geometry = new THREE.BoxGeometry(voxel, voxel, voxel);
  const maxCount = rows * cols * sub * sub;
  const material = new THREE.MeshBasicMaterial({
    color: source === 'gt' ? 0xffa31a : 0x19a8ff,
  });
  const mesh = new THREE.InstancedMesh(geometry, material, maxCount);
  mesh.name = `${source}_${channel}`;
  const matrix = new THREE.Matrix4();
  const color = new THREE.Color();
  const gtValues = example.gt[channel]?.values || example.gt.hazard.values;
  const predValues = example.pred[channel]?.values || example.pred.hazard.values;
  const values = source === 'gt' ? gtValues : predValues;
  let count = 0;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const flat = row * cols + col;
      const value = values[flat] || 0;
      const gtValue = gtValues[flat] || 0;
      const predValue = predValues[flat] || 0;
      if (!shouldShow(value, gtValue, predValue)) continue;
      colorFn(color, value, gtValue, predValue, source);
      for (let ySub = 0; ySub < sub; ySub += 1) {
        for (let xSub = 0; xSub < sub; xSub += 1) {
          const x = sideOffset + (col * sub + xSub + 0.5) * voxel - (cols * sub * voxel) / 2;
          const z = ((rows - 1 - row) * sub + ySub + 0.5) * voxel - (rows * sub * voxel) / 2;
          matrix.makeTranslation(x, -(data.camera_height_m || 0.35) + voxel * 2.0, z);
          mesh.setMatrixAt(count, matrix);
          count += 1;
        }
      }
    }
  }
  mesh.count = count;
  mesh.instanceMatrix.needsUpdate = true;
  return mesh;
}

function shouldShow(value, gtValue, predValue) {
  if (state.channel === 'difference') {
    return gtValue >= 0.5 || predValue >= state.threshold;
  }
  return value >= state.threshold;
}

function valueColor(color, value, _gtValue, _predValue, source) {
  const t = Math.max(0, Math.min(1, value));
  if (source === 'gt') {
    color.setRGB(1.0, 0.42 + 0.42 * t, 0.08);
  } else {
    color.setRGB(0.08, 0.55 + 0.38 * t, 1.0);
  }
}

function differenceColor(color, _value, gtValue, predValue, source) {
  const gtOn = gtValue >= 0.5;
  const predOn = predValue >= state.threshold;
  if (gtOn && predOn) {
    color.setRGB(0.72, 1.0, 0.76);
  } else if (source === 'gt' && gtOn) {
    color.setRGB(1.0, 0.18, 0.12);
  } else if (source === 'pred' && predOn) {
    color.setRGB(0.82, 0.28, 1.0);
  } else {
    color.setRGB(0.18, 0.22, 0.28);
  }
}

function updateInspector(example) {
  if (example.rgb_preview) {
    rgbPreview.src = example.rgb_preview;
    rgbPreview.style.display = 'block';
  } else {
    rgbPreview.removeAttribute('src');
    rgbPreview.style.display = 'none';
  }
  const channel = state.channel === 'difference' ? 'hazard' : state.channel;
  const predCount = example.pred_positive_counts_at_0_5[channel] ?? 0;
  const gtCount = example.gt_positive_counts_at_0_5[channel] ?? 0;
  const cubeMultiplier = data.subdivisions_per_cell * data.subdivisions_per_cell;
  stats.replaceChildren(
    stat('route', example.route_id),
    stat('frame', example.frame_id),
    stat('voxel', `${Math.round(data.voxel_size_m * 100)} cm`),
    stat('cell', `${Math.round(data.cell_size_m * 100)} cm`),
    stat('channel', state.channel),
    stat('GT cells', gtCount),
    stat('pred cells', predCount),
    stat('3D scene voxels', example.scene_voxels?.count ?? 0),
    stat('GT cubes', gtCount * cubeMultiplier),
    stat('pred cubes', predCount * cubeMultiplier),
    stat('replay', data.metadata.replay_only === true ? 'true' : 'false'),
  );
}

function stat(label, value) {
  const dt = document.createElement('dt');
  dt.textContent = label;
  const dd = document.createElement('dd');
  dd.textContent = String(value);
  const fragment = document.createDocumentFragment();
  fragment.append(dt, dd);
  return fragment;
}

resetCamera();
rebuildVoxels();
requestAnimationFrame(animate);
"""


if __name__ == "__main__":
    raise SystemExit(main())
