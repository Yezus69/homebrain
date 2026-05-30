from __future__ import annotations

import argparse
from contextlib import nullcontext
from itertools import cycle
from pathlib import Path
from typing import Any
import hashlib
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from homebrain.brain.homebrain_net_v0 import (
    HOMEBRAIN_NET_V0_FORBIDDEN_RUNTIME_FIELDS,
    HomeBrainNetV0,
    HomeBrainNetV0Config,
    load_checkpoint as load_homebrain_checkpoint,
    save_checkpoint,
    warm_start_homebrain_net_v0,
)
from homebrain.data.spatial_dataset import read_json, write_json
from homebrain.train.train_counterfactual_dynamic_bev_world_model_v0 import (
    DynamicBEVWorldDataset,
    compute_counterfactual_dynamic_losses,
)
from homebrain.train.train_direct_bev_student_v0 import RealRGBDRouteBEVDataset

HOMEBRAIN_NET_V0_STAGES = ("perception", "memory", "world", "joint")


def train_homebrain_net_v0(
    *,
    stage: str,
    out_dir: str | Path,
    real_pack: str | Path | None = None,
    world_pack: str | Path | None = None,
    device_name: str | None = None,
    batch_size: int = 8,
    max_steps: int = 1000,
    learning_rate: float | None = None,
    image_size: tuple[int, int] = (160, 96),
    hidden_channels: int = 32,
    amp: bool = False,
    init_homebrain_net_v0: str | Path | None = None,
    warm_start_direct_bev: str | Path | None = None,
    warm_start_spatial_memory_v1: str | Path | None = None,
    warm_start_world_model: str | Path | None = None,
) -> dict[str, Any]:
    if stage not in HOMEBRAIN_NET_V0_STAGES:
        raise ValueError(f"stage must be one of {HOMEBRAIN_NET_V0_STAGES}")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    real_manifest = read_json(Path(real_pack) / "manifest.json") if real_pack is not None else None
    world_manifest = read_json(Path(world_pack) / "manifest.json") if world_pack is not None else None
    config = _config_for_stage(
        stage=stage,
        real_manifest=real_manifest,
        world_manifest=world_manifest,
        image_size=image_size,
        hidden_channels=hidden_channels,
    )
    model = HomeBrainNetV0(config)
    warm_start = _warm_start(
        model,
        init_homebrain=init_homebrain_net_v0,
        direct=warm_start_direct_bev,
        spatial=warm_start_spatial_memory_v1,
        world=warm_start_world_model,
    )
    model.to(device)
    _set_stage_trainable(model, stage)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate or _stage_lr(stage), weight_decay=1.0e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    loader = _loader_for_stage(stage, real_pack=real_pack, world_pack=world_pack, image_size=image_size, batch_size=batch_size)
    iterator = _cycle_loader(loader)
    last_losses: dict[str, float] = {}
    model.train()
    for _step in range(max_steps):
        batch = _batch_to_device(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        autocast_context = torch.autocast(device_type=device.type) if amp and device.type == "cuda" else nullcontext()
        with autocast_context:
            losses = _stage_losses(model, batch, stage)
        scaler.scale(losses["loss_total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        scaler.step(optimizer)
        scaler.update()
        last_losses = {key: float(value.detach().cpu()) for key, value in losses.items()}
    if max_steps == 0:
        batch = _batch_to_device(_first_batch(loader), device)
        last_losses = {key: float(value.detach().cpu()) for key, value in _stage_losses(model, batch, stage).items()}

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(stage=stage, real_pack=real_pack, world_pack=world_pack, real_manifest=real_manifest, world_manifest=world_manifest, config=config, warm_start=warm_start)
    report = {
        "schema_version": "homebrain.homebrain_net_v0.train_report.v0",
        "model_type": "HomeBrainNetV0",
        "stage": stage,
        "out_dir": output.as_posix(),
        "device": str(device),
        "batch_size": int(batch_size),
        "max_steps": int(max_steps),
        "amp": bool(amp),
        "loss_report": last_losses,
        **metadata,
    }
    write_json(output / "train_loss_report.json", report, pretty=True)
    save_checkpoint(output / "checkpoint.pt", model.cpu(), metadata=metadata, metrics=report)
    return report


def _config_for_stage(
    *,
    stage: str,
    real_manifest: dict[str, Any] | None,
    world_manifest: dict[str, Any] | None,
    image_size: tuple[int, int],
    hidden_channels: int,
) -> HomeBrainNetV0Config:
    manifest = world_manifest if stage == "world" and world_manifest is not None else real_manifest or world_manifest
    if manifest is None:
        raise ValueError("stage requires --real-pack or --world-pack with manifest.json")
    shape = manifest.get("grid_shape", [64, 64])
    horizons = tuple(float(v) for v in (world_manifest or {}).get("future_horizons_sec", (0.5, 1.0, 2.0, 3.0)))
    return HomeBrainNetV0Config(
        bev_shape=(int(shape[0]), int(shape[1])),
        image_size=image_size,
        hidden_channels=hidden_channels,
        history_frames=int((world_manifest or {}).get("history_frames", 6)),
        future_horizons_sec=horizons,
        meters_per_cell=float(manifest.get("meters_per_cell", 0.05)),
        robot_radius_m=float((world_manifest or {}).get("robot_radius_m", 0.18)),
    )


def _loader_for_stage(stage: str, *, real_pack: str | Path | None, world_pack: str | Path | None, image_size: tuple[int, int], batch_size: int) -> DataLoader:
    if stage == "joint" and real_pack is not None and world_pack is not None:
        real_dataset = RealRGBDRouteBEVDataset(real_pack, split="train", image_size=image_size)
        world_dataset = DynamicBEVWorldDataset(world_pack, split="train")
        return (
            DataLoader(real_dataset, batch_size=min(batch_size, len(real_dataset)), shuffle=True),
            DataLoader(world_dataset, batch_size=min(batch_size, len(world_dataset)), shuffle=True),
        )
    if stage == "world":
        if world_pack is None:
            raise ValueError("world stage requires --world-pack")
        dataset = DynamicBEVWorldDataset(world_pack, split="train")
    else:
        if real_pack is None:
            raise ValueError(f"{stage} stage requires --real-pack")
        dataset = RealRGBDRouteBEVDataset(real_pack, split="train", image_size=image_size)
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)


def _stage_losses(model: HomeBrainNetV0, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
    if stage == "joint" and "real" in batch and "world" in batch:
        real_losses = _stage_losses(model, batch["real"], "joint")
        world_losses = _stage_losses(model, batch["world"], "world")
        losses = {f"real_{key}": value for key, value in real_losses.items() if key != "loss_total"}
        losses.update({f"world_{key}": value for key, value in world_losses.items() if key != "loss_total"})
        losses["loss_total"] = real_losses["loss_total"] + world_losses["loss_total"]
        return losses
    if stage == "world":
        outputs = model.world_model(
            batch["bev_history"],
            batch["pose_delta_history"],
            batch["candidate_trajectories"],
            previous_action_history=batch.get("previous_action_history"),
            sensor_mask=batch.get("sensor_mask_history"),
        )
        return compute_counterfactual_dynamic_losses(outputs, batch)
    heads = {"pose", "bev_occupancy", "metric_depth", "dynamic"}
    if stage in {"memory", "joint"}:
        heads.add("memory")
    outputs = model(
        batch["rgb"],
        depth=batch["depth"],
        sensor_mask=batch["sensor_mask"],
        pose_delta_prev=batch["pose_delta_prev"],
        previous_action=batch["previous_action"],
        enabled_heads=heads,
    )
    return compute_homebrain_perception_losses(outputs, batch, use_memory=stage in {"memory", "joint"})


def compute_homebrain_perception_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], *, use_memory: bool = False) -> dict[str, torch.Tensor]:
    target = batch["targets"].to(dtype=outputs["bev_logits"].dtype)
    bev_logits = outputs["fused_memory_bev_logits"] if use_memory and "fused_memory_bev_logits" in outputs else outputs["bev_logits"]
    loss_occupancy = F.binary_cross_entropy_with_logits(bev_logits, target) + _dice_loss(torch.sigmoid(bev_logits), target)
    loss_uncertainty = F.binary_cross_entropy_with_logits(outputs["uncertainty_logits"], batch["uncertainty"].to(dtype=bev_logits.dtype))
    loss_dynamic = F.binary_cross_entropy_with_logits(outputs["dynamic_occupancy_logits"], batch["dynamic_occupancy"].to(dtype=bev_logits.dtype))
    depth_valid = batch["metric_depth_valid"].to(dtype=bev_logits.dtype)
    loss_depth = (torch.abs(outputs["metric_depth_m"] - batch["metric_depth"].to(dtype=bev_logits.dtype)) * depth_valid).sum() / depth_valid.sum().clamp_min(1.0)
    flow_mask = batch["bev_flow_valid"].to(dtype=bev_logits.dtype)
    loss_flow = (torch.sqrt((outputs["bev_flow_xy"] - batch["bev_flow_xy"].to(dtype=bev_logits.dtype)) ** 2 + 1.0e-4) * flow_mask).sum() / flow_mask.sum().clamp_min(1.0)
    pose_mask = batch["pose_delta_next_mask"].to(dtype=bev_logits.dtype)
    loss_pose = (((outputs["pose_delta"] - batch["pose_delta_next"].to(dtype=bev_logits.dtype)) ** 2).mean(dim=1) * pose_mask.view(-1)).sum() / pose_mask.sum().clamp_min(1.0)
    total = loss_occupancy + 0.5 * loss_uncertainty + loss_dynamic + 0.25 * loss_depth + 0.1 * loss_flow + 0.1 * loss_pose
    if use_memory and "fused_memory_bev_logits" in outputs:
        total = total + 0.5 * F.binary_cross_entropy_with_logits(outputs["fused_memory_bev_logits"], target)
    return {"loss_occupancy": loss_occupancy, "loss_uncertainty": loss_uncertainty, "loss_depth": loss_depth, "loss_dynamic": loss_dynamic, "loss_flow": loss_flow, "loss_pose": loss_pose, "loss_total": total}


def _cycle_loader(loader: Any) -> Any:
    if isinstance(loader, tuple):
        real_loader, world_loader = loader
        real_iter = cycle(real_loader)
        world_iter = cycle(world_loader)
        return ({"real": next(real_iter), "world": next(world_iter)} for _ in iter(int, 1))
    return cycle(loader)


def _first_batch(loader: Any) -> Any:
    if isinstance(loader, tuple):
        real_loader, world_loader = loader
        return {"real": next(iter(real_loader)), "world": next(iter(world_loader))}
    return next(iter(loader))


def _set_stage_trainable(model: HomeBrainNetV0, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = stage == "joint"
    prefixes = {
        "perception": ("perception.", "pose_head.", "metric_depth_head.", "bev_flow_head."),
        "memory": ("memory.",),
        "world": ("world_model.",),
    }.get(stage, ())
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad = True


def _metadata(
    *,
    stage: str,
    real_pack: str | Path | None,
    world_pack: str | Path | None,
    real_manifest: dict[str, Any] | None,
    world_manifest: dict[str, Any] | None,
    config: HomeBrainNetV0Config,
    warm_start: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest = world_manifest or real_manifest or {}
    return {
        "model_type": "HomeBrainNetV0",
        "stage": stage,
        "real_pack": Path(real_pack).as_posix() if real_pack is not None else None,
        "world_pack": Path(world_pack).as_posix() if world_pack is not None else None,
        "pack_sha256": _pack_sha(real_pack, world_pack),
        "train_route_ids": list(manifest.get("train_route_ids", [])),
        "val_route_ids": list(manifest.get("val_route_ids", manifest.get("heldout_route_ids", []))),
        "future_horizons_sec": [float(v) for v in config.future_horizons_sec],
        "model_config": config.to_dict(),
        "warm_start_summary": warm_start,
        "runtime_inputs": config.to_dict()["runtime_inputs"],
        "forbidden_runtime_fields": list(HOMEBRAIN_NET_V0_FORBIDDEN_RUNTIME_FIELDS),
        "no_future_labels_used_at_runtime": True,
        "no_teacher_fields_at_runtime": True,
        "no_route_ground_truth_at_runtime": True,
        "no_oracle_bev_at_runtime": True,
        "no_heavy_teacher_at_runtime": True,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
        "product_training_approved": False,
    }


def _warm_start(
    model: HomeBrainNetV0,
    *,
    init_homebrain: str | Path | None,
    direct: str | Path | None,
    spatial: str | Path | None,
    world: str | Path | None,
) -> dict[str, Any] | None:
    summary: dict[str, Any] = {}
    if init_homebrain is not None:
        summary["homebrain_net_v0"] = _copy_matching_homebrain_checkpoint(model, init_homebrain)
    checkpoints = {}
    if direct is not None:
        checkpoints["direct_bev"] = direct
    if spatial is not None:
        checkpoints["spatial_memory_v1"] = spatial
    if world is not None:
        checkpoints["world_model"] = world
    if checkpoints:
        summary["legacy_components"] = warm_start_homebrain_net_v0(model, checkpoints)
    if summary:
        copied_count = 0
        skipped_count = 0
        for item in summary.values():
            if isinstance(item, dict):
                copied_count += int(item.get("copied_count", 0))
                skipped_count += int(item.get("skipped_count", 0))
        summary["copied_count"] = copied_count
        summary["skipped_count"] = skipped_count
    return summary or None


def _copy_matching_homebrain_checkpoint(model: HomeBrainNetV0, checkpoint: str | Path) -> dict[str, Any]:
    source_model, payload = load_homebrain_checkpoint(checkpoint, map_location="cpu")
    source_state = source_model.state_dict()
    target_state = model.state_dict()
    copied: list[str] = []
    skipped: list[str] = []
    for key, value in source_state.items():
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape):
            target_state[key].copy_(value)
            copied.append(key)
        else:
            skipped.append(key)
    model.load_state_dict(target_state)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return {
        "checkpoint": Path(checkpoint).as_posix(),
        "source_stage": metadata.get("stage"),
        "copied_count": len(copied),
        "skipped_count": len(skipped),
        "copied_keys": copied[:40],
        "skipped_keys": skipped[:40],
    }


def _pack_sha(real_pack: str | Path | None, world_pack: str | Path | None) -> str | None:
    manifest = Path(world_pack or real_pack or "") / "manifest.json"
    return _file_hash(manifest) if manifest.exists() else None


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = pred.flatten(2)
    target_f = target.flatten(2)
    return 1.0 - ((2.0 * (pred_f * target_f).sum(dim=2) + 1.0) / (pred_f.sum(dim=2) + target_f.sum(dim=2) + 1.0)).mean()


def _stage_lr(stage: str) -> float:
    return 2.0e-4 if stage != "joint" else 5.0e-5


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = _batch_to_device(value, device)
        else:
            moved[key] = value
    return moved


def _parse_image_size(value: str) -> tuple[int, int]:
    width, height = value.lower().split("x", 1)
    return (int(width), int(height))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train HomeBrainNetV0 in staged Goal31 phases.")
    parser.add_argument("--stage", required=True, choices=HOMEBRAIN_NET_V0_STAGES)
    parser.add_argument("--real-pack", default=None)
    parser.add_argument("--world-pack", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--image-size", type=_parse_image_size, default=(160, 96))
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--init-homebrain-net-v0", default=None)
    parser.add_argument("--warm-start-direct-bev", default=None)
    parser.add_argument("--warm-start-spatial-memory-v1", default=None)
    parser.add_argument("--warm-start-world-model", default=None)
    args = parser.parse_args(argv)
    report = train_homebrain_net_v0(
        stage=args.stage,
        real_pack=args.real_pack,
        world_pack=args.world_pack,
        out_dir=args.out,
        device_name=args.device,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        hidden_channels=args.hidden_channels,
        amp=args.amp,
        init_homebrain_net_v0=args.init_homebrain_net_v0,
        warm_start_direct_bev=args.warm_start_direct_bev,
        warm_start_spatial_memory_v1=args.warm_start_spatial_memory_v1,
        warm_start_world_model=args.warm_start_world_model,
    )
    print(json.dumps({"stage": args.stage, "loss_report": report["loss_report"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
