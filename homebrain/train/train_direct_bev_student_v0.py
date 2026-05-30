from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from homebrain.brain.direct_bev_student_v0 import DirectBEVStudentV0, DirectBEVStudentV0Config, save_checkpoint
from homebrain.data.spatial_dataset import json_sha256, load_example_npz, read_json, write_json
from homebrain.datasets.tum_rgbd import read_depth_png_m, read_png
from homebrain.messages.schema import JsonDict

TARGET_NAMES: tuple[str, ...] = (
    "target_current_bev_free",
    "target_current_bev_occupied",
    "target_current_bev_unknown",
    "target_current_bev_traversable",
    "target_current_bev_risky",
)


class RealRGBDRouteBEVDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        pack_dir: str | Path,
        *,
        split: str | None = None,
        image_size: tuple[int, int] = (160, 96),
        cache_in_memory: bool = True,
    ) -> None:
        self.root = Path(pack_dir)
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest.get("control_safe") is not False:
            raise ValueError("RealRGBDRouteBEVPack must remain control_safe=false")
        records = self.manifest.get("examples")
        if not isinstance(records, list):
            raise ValueError("pack manifest examples must be a list")
        self.records = [record for record in records if isinstance(record, dict)]
        if split is not None:
            self.records = [record for record in self.records if str(record.get("split")) == split]
        if not self.records:
            raise ValueError(f"no RealRGBDRouteBEV examples found for split={split!r}")
        self.image_size = image_size
        self.cache_in_memory = bool(cache_in_memory)
        self._cache: dict[int, dict[str, Any]] = {}
        grid = self.manifest.get("grid_shape")
        if not isinstance(grid, list) or len(grid) != 2:
            raise ValueError("pack manifest must include grid_shape")
        self.bev_shape = (int(grid[0]), int(grid[1]))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.cache_in_memory and index in self._cache:
            return self._cache[index]
        record = self.records[index]
        arrays = load_example_npz(self.root / str(record["example_path"]))
        rgb = _load_rgb_chw(Path(str(record["rgb_path"])), image_size=self.image_size)
        depth = _load_depth_chw(record, image_size=self.image_size)
        targets = np.stack([np.asarray(arrays[name], dtype=np.float32) for name in TARGET_NAMES], axis=0)
        depth_target, depth_valid = _target_metric_depth(arrays, image_size=self.image_size)
        item = {
            "rgb": torch.from_numpy(rgb),
            "depth": torch.from_numpy(depth),
            "sensor_mask": torch.from_numpy(np.asarray(arrays["sensor_mask"], dtype=np.float32)),
            "pose_delta_prev": torch.from_numpy(np.asarray(arrays["pose_delta_prev"], dtype=np.float32)),
            "previous_action": torch.from_numpy(np.asarray(arrays["previous_action"], dtype=np.float32)),
            "targets": torch.from_numpy(targets),
            "uncertainty": torch.from_numpy(np.asarray(arrays["target_uncertainty_map"], dtype=np.float32)[None, ...]),
            "dynamic": torch.from_numpy(np.asarray(arrays["target_dynamic_residual_risk"], dtype=np.float32)[None, ...]),
            "metric_depth": torch.from_numpy(depth_target),
            "metric_depth_valid": torch.from_numpy(depth_valid),
            "dynamic_occupancy": torch.from_numpy(
                np.asarray(arrays.get("target_dynamic_occupancy", arrays["target_dynamic_residual_risk"]), dtype=np.float32)[None, ...]
            ),
            "bev_flow_xy": torch.from_numpy(
                np.asarray(arrays.get("target_bev_flow_xy", np.zeros((2, *self.bev_shape), dtype=np.float32)), dtype=np.float32)
            ),
            "bev_flow_valid": torch.from_numpy(
                np.asarray(arrays.get("target_bev_flow_valid_mask", np.zeros(self.bev_shape, dtype=np.float32)), dtype=np.float32)[None, ...]
            ),
            "pose_delta_next": torch.from_numpy(np.asarray(arrays.get("pose_delta_next_teacher_only", np.zeros((3,), dtype=np.float32)), dtype=np.float32)),
            "pose_delta_next_mask": torch.from_numpy(np.asarray(arrays.get("pose_delta_next_teacher_only_mask", np.zeros((1,), dtype=np.float32)), dtype=np.float32)),
            "route_id": str(record.get("route_id", "")),
            "split": str(record.get("split", "")),
        }
        if self.cache_in_memory:
            self._cache[index] = item
        return item


def train_direct_bev_student_v0(
    *,
    pack_dir: str | Path,
    out_dir: str | Path,
    device_name: str | None = None,
    batch_size: int = 32,
    max_steps: int = 20_000,
    amp: bool = False,
    val_every: int = 500,
    image_size: tuple[int, int] = (160, 96),
    hidden_channels: int = 32,
) -> Path:
    pack_root = Path(pack_dir)
    manifest = read_json(pack_root / "manifest.json")
    train_set = RealRGBDRouteBEVDataset(pack_root, split="train", image_size=image_size)
    val_set: RealRGBDRouteBEVDataset | None
    try:
        val_set = RealRGBDRouteBEVDataset(pack_root, split="val", image_size=image_size)
    except ValueError:
        val_set = None
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DirectBEVStudentV0(
        DirectBEVStudentV0Config(
            bev_shape=train_set.bev_shape,
            image_size=image_size,
            hidden_channels=hidden_channels,
            use_depth=True,
        )
    ).to(device)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=1.0e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    history: list[JsonDict] = []
    model.train()
    for step, batch in zip(range(1, max_steps + 1), cycle(loader)):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(
                batch["rgb"].to(device),
                depth=batch["depth"].to(device),
                sensor_mask=batch["sensor_mask"].to(device),
                pose_delta_prev=batch["pose_delta_prev"].to(device),
                previous_action=batch["previous_action"].to(device),
            )
            loss, parts = direct_bev_loss(
                outputs,
                targets=batch["targets"].to(device),
                uncertainty=batch["uncertainty"].to(device),
                dynamic=batch["dynamic"].to(device),
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % max(1, val_every) == 0 or step == max_steps:
            record = {"step": int(step), "train_loss": float(loss.detach().cpu()), **parts}
            if val_set is not None:
                record.update(_quick_validate(model, val_set, device=device, batch_size=batch_size))
            history.append(record)
            model.train()

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics: JsonDict = {"training_steps": int(max_steps), "history": history, "final": history[-1] if history else {}}
    metadata: JsonDict = {
        "source_dataset_name": manifest.get("dataset_name"),
        "train_route_ids": manifest.get("train_route_ids", []),
        "heldout_route_ids": manifest.get("heldout_route_ids", []),
        "pack_manifest_sha256": json_sha256({key: value for key, value in manifest.items() if key != "manifest_sha256"}),
        "model_config": model.config.to_dict(),
        "input_modality_flags": {"rgb": True, "depth": True, "pose_delta_prev": True, "previous_action": True},
        "no_future_labels_used": True,
        "no_teacher_fields_at_runtime": True,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
        "product_training_approved": False,
    }
    checkpoint_path = output / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, metadata=metadata, metrics=metrics)
    write_json(output / "training_metrics.json", {"metadata": metadata, "metrics": metrics}, pretty=True)
    return checkpoint_path


def direct_bev_loss(
    outputs: dict[str, torch.Tensor],
    *,
    targets: torch.Tensor,
    uncertainty: torch.Tensor,
    dynamic: torch.Tensor,
) -> tuple[torch.Tensor, JsonDict]:
    bev_logits = outputs["bev_logits"]
    bce = F.binary_cross_entropy_with_logits(bev_logits, targets)
    dice = _dice_loss(torch.sigmoid(bev_logits), targets)
    uncertainty_loss = F.binary_cross_entropy_with_logits(outputs["uncertainty_logits"], uncertainty)
    dynamic_logits = outputs["dynamic_risk_logits"]
    pos = dynamic.sum().detach().clamp_min(1.0)
    neg = (dynamic.numel() - dynamic.sum()).detach().clamp_min(1.0)
    pos_weight = torch.clamp(neg / pos, 1.0, 40.0).to(dynamic_logits.device)
    dynamic_loss = F.binary_cross_entropy_with_logits(dynamic_logits, dynamic, pos_weight=pos_weight)
    loss = bce + dice + 0.5 * uncertainty_loss + 1.5 * dynamic_loss
    return loss, {
        "bev_bce": float(bce.detach().cpu()),
        "bev_dice": float(dice.detach().cpu()),
        "uncertainty_loss": float(uncertainty_loss.detach().cpu()),
        "dynamic_loss": float(dynamic_loss.detach().cpu()),
    }


def _quick_validate(
    model: DirectBEVStudentV0,
    dataset: RealRGBDRouteBEVDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> JsonDict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            outputs = model(
                batch["rgb"].to(device),
                depth=batch["depth"].to(device),
                sensor_mask=batch["sensor_mask"].to(device),
                pose_delta_prev=batch["pose_delta_prev"].to(device),
                previous_action=batch["previous_action"].to(device),
            )
            loss, _parts = direct_bev_loss(
                outputs,
                targets=batch["targets"].to(device),
                uncertainty=batch["uncertainty"].to(device),
                dynamic=batch["dynamic"].to(device),
            )
            losses.append(float(loss.detach().cpu()))
            if index >= 3:
                break
    return {"val_loss": float(sum(losses) / max(len(losses), 1))}


def _dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = pred.flatten(2)
    target_f = target.flatten(2)
    intersection = (pred_f * target_f).sum(dim=2)
    denom = pred_f.sum(dim=2) + target_f.sum(dim=2)
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    return 1.0 - dice.mean()


def _load_rgb_chw(path: Path, *, image_size: tuple[int, int]) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".ppm":
        image = _read_ppm(path)
    else:
        try:
            from PIL import Image  # type: ignore[import-not-found]
            with Image.open(path) as pil:
                image = np.asarray(pil.convert("RGB"), dtype=np.uint8)
        except Exception:
            image = read_png(path)
            if image.ndim == 2:
                image = np.repeat(image[:, :, None], 3, axis=2)
            image = image[:, :, :3]
    resized = _resize_hwc(image.astype(np.float32) / np.float32(255.0), image_size)
    return np.transpose(resized, (2, 0, 1)).astype(np.float32)


def _load_depth_chw(record: JsonDict, *, image_size: tuple[int, int]) -> np.ndarray:
    depth_path = record.get("depth_path")
    if not isinstance(depth_path, str):
        width, height = image_size
        return np.zeros((1, height, width), dtype=np.float32)
    intrinsics = record.get("camera_intrinsics") if isinstance(record.get("camera_intrinsics"), dict) else {}
    scale = float(intrinsics.get("depth_scale", 5000.0))
    depth = read_depth_png_m(depth_path, scale=scale)
    return _resize_hwc(depth[:, :, None].astype(np.float32), image_size).transpose(2, 0, 1).astype(np.float32)


def _target_metric_depth(arrays: dict[str, np.ndarray], *, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if "target_metric_depth" not in arrays:
        width, height = image_size
        return np.zeros((1, height, width), dtype=np.float32), np.zeros((1, height, width), dtype=np.float32)
    depth = np.asarray(arrays["target_metric_depth"], dtype=np.float32)
    valid = np.asarray(arrays.get("target_metric_depth_valid_mask", np.isfinite(depth) & (depth > 0.0)), dtype=np.float32)
    depth_r = _resize_hwc(depth[:, :, None], image_size).transpose(2, 0, 1).astype(np.float32)
    valid_r = _resize_hwc(valid[:, :, None], image_size).transpose(2, 0, 1).astype(np.float32)
    return depth_r, (valid_r > 0.5).astype(np.float32)


def _resize_hwc(array: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    rows = np.linspace(0, array.shape[0] - 1, height).round().astype(np.int64)
    cols = np.linspace(0, array.shape[1] - 1, width).round().astype(np.int64)
    return np.asarray(array, dtype=np.float32)[rows[:, None], cols[None, :]]


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


def _parse_image_size(value: str) -> tuple[int, int]:
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("image size must be WIDTHxHEIGHT, for example 160x96")
    width, height = value.lower().split("x", 1)
    return (int(width), int(height))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train DirectBEVStudentV0 from a real RGB-D route BEV pack.")
    parser.add_argument("--pack", required=True, help="RealRGBDRouteBEVPack directory.")
    parser.add_argument("--out", required=True, help="Output checkpoint directory.")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision.")
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--image-size", type=_parse_image_size, default=(160, 96), help="Input resize as WIDTHxHEIGHT.")
    parser.add_argument("--hidden-channels", type=int, default=32)
    args = parser.parse_args(argv)
    train_direct_bev_student_v0(
        pack_dir=args.pack,
        out_dir=args.out,
        device_name=args.device,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        amp=args.amp,
        val_every=args.val_every,
        image_size=args.image_size,
        hidden_channels=args.hidden_channels,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
