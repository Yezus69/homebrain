from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from homebrain.brain.counterfactual_dynamic_bev_world_model_v0 import (
    COUNTERFACTUAL_DYNAMIC_BEV_WORLD_MODEL_V0_SOURCE,
    CounterfactualDynamicBEVWorldModelV0,
    CounterfactualDynamicBEVWorldModelV0Config,
    save_checkpoint,
)
from homebrain.data.spatial_dataset import load_example_npz, read_json, write_json


class DynamicBEVWorldDataset(Dataset[dict[str, Any]]):
    def __init__(self, pack_dir: str | Path, *, split: str | None = None, tiny_limit: int | None = None) -> None:
        self.root = Path(pack_dir)
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest.get("control_safe") is not False:
            raise ValueError("DynamicBEVWorldPackV0 must remain control_safe=false")
        records = [record for record in self.manifest.get("examples", []) if isinstance(record, dict)]
        if split is not None:
            records = [record for record in records if str(record.get("split", "")) == split]
        if tiny_limit is not None:
            records = records[: max(1, int(tiny_limit))]
        if not records:
            raise ValueError(f"no DynamicBEVWorldPackV0 examples found in {self.root} for split={split!r}")
        self.records = records
        grid_shape = self.manifest.get("grid_shape")
        if not isinstance(grid_shape, list) or len(grid_shape) != 2:
            first = load_example_npz(self.root / str(records[0]["example_path"]))
            grid_shape = list(np.asarray(first["bev_history"]).shape[-2:])
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.history_frames = int(self.manifest.get("history_frames", 6))
        self.horizons_s = tuple(float(value) for value in self.manifest.get("future_horizons_sec", (0.5, 1.0, 2.0, 3.0)))
        self.candidate_ids = tuple(str(value) for value in self.manifest.get("candidate_ids", []))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        example = load_example_npz(self.root / str(record["example_path"]))
        return {
            "bev_history": _tensor(example["bev_history"]),
            "pose_delta_history": _tensor(example["pose_delta_history"]),
            "previous_action_history": _tensor(example["previous_action_history"]),
            "sensor_mask_history": _tensor(example["sensor_mask_history"]),
            "candidate_trajectories": _tensor(example["candidate_trajectory_xytheta"]),
            "future_occupied": _tensor(example["future_occupied"]),
            "future_free": _tensor(example["future_free"]),
            "future_unknown": _tensor(example["future_unknown"]),
            "future_risky": _tensor(example["future_risky"]),
            "future_dynamic_risk": _tensor(example["future_dynamic_risk"]),
            "future_flow_xy": _tensor(example["future_flow_xy"]),
            "future_valid_mask": _tensor(example["future_valid_mask"]),
            "horizon_valid": _tensor(example["horizon_valid"]),
            "candidate_collision_risk": _tensor(example["candidate_collision_risk"]),
            "candidate_dynamic_risk": _tensor(example["candidate_dynamic_risk"]),
            "candidate_unknown_exposure": _tensor(example["candidate_unknown_exposure"]),
            "candidate_total_teacher_risk": _tensor(example["candidate_total_teacher_risk"]),
            "candidate_safe_label": _tensor(example["candidate_safe_label"]),
            "candidate_rank_target": _tensor(example["candidate_rank_target"]),
            "route_id": str(record.get("route_id", "")),
            "split": str(record.get("split", "")),
        }


def train_counterfactual_dynamic_bev_world_model_v0(
    *,
    pack: str | Path,
    out_dir: str | Path,
    device_name: str | None = None,
    batch_size: int = 32,
    max_steps: int = 20_000,
    learning_rate: float = 1.0e-3,
    hidden_channels: int = 32,
    amp: bool = False,
    seed: int = 30,
    command: str | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_dataset = DynamicBEVWorldDataset(pack, split="train")
    try:
        val_dataset = DynamicBEVWorldDataset(pack, split="val")
    except ValueError:
        val_dataset = train_dataset
    config = CounterfactualDynamicBEVWorldModelV0Config(
        bev_shape=train_dataset.grid_shape,
        history_frames=train_dataset.history_frames,
        hidden_channels=hidden_channels,
        future_horizons_sec=train_dataset.horizons_s,
        meters_per_cell=float(train_dataset.manifest.get("meters_per_cell", 0.05)),
        robot_radius_m=float(train_dataset.manifest.get("robot_radius_m", 0.18)),
    )
    model_base = CounterfactualDynamicBEVWorldModelV0(config).to(device)
    cuda_device_ids = _cuda_training_device_ids() if device.type == "cuda" else []
    data_parallel = bool(len(cuda_device_ids) > 1)
    model: torch.nn.Module = (
        torch.nn.DataParallel(model_base, device_ids=cuda_device_ids, output_device=cuda_device_ids[0])
        if data_parallel
        else model_base
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True, generator=_generator(seed))
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    last_losses: dict[str, float] = {}
    iterator = cycle(train_loader)
    model.train()
    for _step in range(max_steps):
        batch = _batch_to_device(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            losses = compute_counterfactual_dynamic_losses(model_forward(model, batch), batch)
        scaler.scale(losses["loss_total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        last_losses = {key: float(value.detach().cpu()) for key, value in losses.items()}
    if max_steps == 0:
        batch = _batch_to_device(next(iter(train_loader)), device)
        last_losses = {key: float(value.detach().cpu()) for key, value in compute_counterfactual_dynamic_losses(model_forward(model, batch), batch).items()}
    val_losses = evaluate_loss(model, val_loader, device=device)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    pack_sha = _file_hash(Path(pack) / "manifest.json")
    metadata = {
        "model_type": "CounterfactualDynamicBEVWorldModelV0",
        "pack": Path(pack).as_posix(),
        "pack_sha256": pack_sha,
        "train_route_ids": list(train_dataset.manifest.get("train_route_ids", [])),
        "val_route_ids": list(train_dataset.manifest.get("val_route_ids", [])),
        "future_horizons_sec": [float(value) for value in train_dataset.horizons_s],
        "candidate_count": len(train_dataset.candidate_ids),
        "no_future_labels_used_at_runtime": True,
        "no_teacher_fields_at_runtime": True,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
    }
    report = {
        "schema_version": "homebrain.counterfactual_dynamic_bev_world_model_v0.train_report.v0",
        "model": COUNTERFACTUAL_DYNAMIC_BEV_WORLD_MODEL_V0_SOURCE,
        "pack": Path(pack).as_posix(),
        "out_dir": output.as_posix(),
        "device": str(device),
        "cuda_device_count": int(torch.cuda.device_count()) if device.type == "cuda" else 0,
        "cuda_training_device_ids": cuda_device_ids,
        "cuda_training_device_names": [torch.cuda.get_device_name(index) for index in cuda_device_ids]
        if device.type == "cuda"
        else [],
        "data_parallel": data_parallel,
        "batch_size": int(batch_size),
        "max_steps": int(max_steps),
        "learning_rate": float(learning_rate),
        "amp": bool(amp),
        "train_example_count": len(train_dataset),
        "val_example_count": len(val_dataset),
        "loss_report": {**last_losses, **{f"val_{key}": value for key, value in val_losses.items()}},
        **metadata,
    }
    write_json(output / "train_loss_report.json", report, pretty=True)
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    save_checkpoint(output / "checkpoint.pt", raw_model.cpu(), metadata=metadata, metrics=report)
    return report


def model_forward(model: CounterfactualDynamicBEVWorldModelV0, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(
        batch["bev_history"],
        batch["pose_delta_history"],
        batch["candidate_trajectories"],
        previous_action_history=batch.get("previous_action_history"),
        sensor_mask=batch.get("sensor_mask_history"),
    )


def compute_counterfactual_dynamic_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    mask = batch["future_valid_mask"].to(dtype=outputs["future_occupied_logits"].dtype)
    loss_future_occupied = (
        _bce(outputs["future_occupied_logits"], batch["future_occupied"], mask)
        + 0.5 * _bce(outputs["future_free_logits"], batch["future_free"], mask)
        + 0.35 * _bce(outputs["future_unknown_logits"], batch["future_unknown"], mask)
        + 0.5 * _dice_loss(outputs["future_occupied_logits"], batch["future_occupied"], mask)
        + 0.25 * _bce(outputs["future_risky_logits"], batch["future_risky"], mask)
    )
    loss_future_dynamic = _bce(outputs["future_dynamic_risk_logits"], batch["future_dynamic_risk"], mask)
    dynamic_mask = mask.unsqueeze(2) * (batch["future_dynamic_risk"].unsqueeze(2) > 0.05).to(dtype=mask.dtype)
    loss_flow = _charbonnier(outputs["future_flow_xy"] - batch["future_flow_xy"], dynamic_mask)
    candidate_target = torch.maximum(
        batch["candidate_total_teacher_risk"],
        batch["candidate_collision_risk"].amax(dim=2),
    )
    candidate_hard_risk = (candidate_target >= 0.5).to(dtype=candidate_target.dtype)
    candidate_hard_dynamic = (batch["candidate_dynamic_risk"].amax(dim=2) >= 0.5).to(dtype=candidate_target.dtype)
    candidate_mask = torch.ones_like(candidate_target)
    loss_candidate_bce = _bce(outputs["candidate_risk_logits"], candidate_hard_risk, candidate_mask)
    loss_candidate_bce = loss_candidate_bce + 0.5 * _bce(
        outputs["candidate_dynamic_risk_logits"],
        candidate_hard_dynamic,
        candidate_mask,
    )
    loss_candidate_bce = loss_candidate_bce + 0.35 * _bce(
        outputs["candidate_unknown_exposure_logits"],
        batch["candidate_unknown_exposure"].amax(dim=2),
        candidate_mask,
    )
    loss_candidate_rank = _pairwise_ranking_loss(outputs["candidate_score"], batch["candidate_total_teacher_risk"])
    uncertainty_target = torch.maximum(batch["future_unknown"], batch["future_dynamic_risk"] * 0.5)
    loss_uncertainty = _bce(outputs["future_uncertainty_logits"], uncertainty_target, mask)
    loss_uncertainty = loss_uncertainty + _confident_wrong_free_penalty(outputs, batch, mask)
    loss_equivariance = _equivariance_proxy(outputs, batch, mask)
    loss_total = (
        loss_future_occupied
        + 0.7 * loss_future_dynamic
        + 0.2 * loss_flow
        + loss_candidate_bce
        + 0.5 * loss_candidate_rank
        + 0.25 * loss_uncertainty
        + 0.1 * loss_equivariance
    )
    return {
        "loss_future_occupied": loss_future_occupied,
        "loss_future_dynamic_risk": loss_future_dynamic,
        "loss_future_flow": loss_flow,
        "loss_candidate_bce": loss_candidate_bce,
        "loss_candidate_rank": loss_candidate_rank,
        "loss_uncertainty": loss_uncertainty,
        "loss_equivariance": loss_equivariance,
        "loss_total": loss_total,
    }


@torch.no_grad()
def evaluate_loss(
    model: CounterfactualDynamicBEVWorldModelV0,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = _batch_to_device(batch, device)
        losses = compute_counterfactual_dynamic_losses(model_forward(model, batch), batch)
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
        count += 1
    return {key: value / max(count, 1) for key, value in sums.items()}


def _bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    while mask.ndim < logits.ndim:
        mask = mask.unsqueeze(1)
    mask = torch.broadcast_to(mask, logits.shape)
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive = ((target >= 0.05).to(logits.dtype) * mask).sum()
    negative = ((target < 0.05).to(logits.dtype) * mask).sum()
    total = (positive + negative).clamp_min(1.0)
    pos_weight = (0.5 * total / positive.clamp_min(1.0)).clamp(max=20.0)
    neg_weight = (0.5 * total / negative.clamp_min(1.0)).clamp(max=20.0)
    weights = torch.where(target >= 0.05, pos_weight, neg_weight)
    return (raw * weights * mask).sum() / mask.sum().clamp_min(1.0)


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    target = target.to(device=logits.device, dtype=logits.dtype)
    intersection = (prob * target * mask).sum()
    denom = (prob * mask).sum() + (target * mask).sum()
    return 1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)


def _charbonnier(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=error.device, dtype=error.dtype)
    return (torch.sqrt(error * error + 1.0e-4) * mask).sum() / mask.sum().clamp_min(1.0)


def _pairwise_ranking_loss(score: torch.Tensor, target_risk: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for left in range(score.shape[1]):
        for right in range(left + 1, score.shape[1]):
            diff = target_risk[:, left] - target_risk[:, right]
            valid = torch.abs(diff) > 1.0e-4
            if not torch.any(valid):
                continue
            sign = torch.sign(diff[valid])
            pred = score[:, left][valid] - score[:, right][valid]
            losses.append(F.relu(0.1 - sign * pred).mean())
    if not losses:
        return score.sum() * 0.0
    return torch.stack(losses).mean()


def _confident_wrong_free_penalty(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    free_prob = torch.sigmoid(outputs["future_free_logits"])
    uncertainty = torch.sigmoid(outputs["future_uncertainty_logits"])
    occupied = batch["future_occupied"].to(device=free_prob.device, dtype=free_prob.dtype)
    wrong_confident_free = F.relu(free_prob - 0.4) * occupied * (1.0 - uncertainty)
    return (wrong_confident_free * mask).sum() / mask.sum().clamp_min(1.0)


def _equivariance_proxy(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    current_occupied = batch["bev_history"][:, -1, 1].unsqueeze(1)
    future_static = batch["future_dynamic_risk"] < 0.05
    pred = torch.sigmoid(outputs["future_occupied_logits"])
    target = current_occupied.expand_as(pred)
    static_mask = mask * future_static.to(dtype=mask.dtype)
    return (((pred - target) ** 2) * static_mask).sum() / static_mask.sum().clamp_min(1.0)


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value, dtype=np.float32).copy())


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _cuda_training_device_ids() -> list[int]:
    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    preferred = [index for index, name in enumerate(names) if "4090" in name]
    return preferred or list(range(torch.cuda.device_count()))


def _file_hash(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CounterfactualDynamicBEVWorldModelV0.")
    parser.add_argument("--pack", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=30)
    args = parser.parse_args(argv)
    report = train_counterfactual_dynamic_bev_world_model_v0(
        pack=args.pack,
        out_dir=args.out,
        device_name=args.device,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        hidden_channels=args.hidden_channels,
        amp=args.amp,
        seed=args.seed,
        command=" ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
    )
    print(json.dumps({"loss_report": report["loss_report"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
