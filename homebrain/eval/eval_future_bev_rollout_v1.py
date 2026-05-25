from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

from homebrain.brain.future_bev_rollout_v1 import load_checkpoint
from homebrain.data.spatial_dataset import write_json
from homebrain.train.future_bev_rollout_dataset import FutureBEVRolloutDataset
from homebrain.train.train_future_bev_rollout_v1 import evaluate_future_bev_rollout_v1


def eval_future_bev_rollout_v1(
    *,
    checkpoint: str | Path,
    rollout_pack: str | Path | None = None,
    out_path: str | Path,
    split: str | None = "val",
    batch_size: int = 8,
    device_name: str | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model.to(device)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    resolved_pack = rollout_pack or metadata.get("rollout_pack")
    if not isinstance(resolved_pack, (str, Path)):
        raise ValueError("supply --rollout-pack or train with rollout_pack metadata")
    dataset = FutureBEVRolloutDataset(resolved_pack, split=split)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    eval_metrics = evaluate_future_bev_rollout_v1(model, loader, device=device, dataset=dataset)
    train_metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    metrics: dict[str, Any] = {
        "schema_version": "homebrain.future_bev_rollout_v1_eval_metrics.v1",
        "checkpoint": Path(checkpoint).as_posix(),
        "rollout_pack": Path(resolved_pack).as_posix(),
        "split": split,
        "batch_size": int(batch_size),
        "device": str(device),
        "eval_command": command,
        "example_count": len(dataset),
        "train_loss_start": train_metrics.get("train_loss_start"),
        "train_loss_end": train_metrics.get("train_loss_end"),
        "loss_reduction_ratio": train_metrics.get("loss_reduction_ratio"),
        **eval_metrics,
        "representation_pretraining_only": True,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "product_training_approved": False,
        "raw_pwm_emitted": False,
    }
    write_json(out_path, metrics, pretty=True)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate replay-only Future BEV Rollout v1.")
    parser.add_argument("--checkpoint", required=True, help="Future BEV Rollout v1 checkpoint.")
    parser.add_argument("--rollout-pack", default=None, help="Input FutureBEVRolloutPack directory.")
    parser.add_argument("--out", required=True, help="Output metrics JSON path.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    metrics = eval_future_bev_rollout_v1(
        checkpoint=args.checkpoint,
        rollout_pack=args.rollout_pack,
        out_path=args.out,
        split=args.split,
        batch_size=args.batch_size,
        device_name=args.device,
        command=" ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
    )
    print(json.dumps({"metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
