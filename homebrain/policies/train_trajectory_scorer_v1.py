from __future__ import annotations

import argparse
import json
from pathlib import Path

from homebrain.data.spatial_dataset import read_json, write_json
from homebrain.policies.trajectory_scorer_net_v0 import _parse_repeated, train_trajectory_scorer_v0


TRAJECTORY_SCORER_V1_TRAIN_SCHEMA_VERSION = "homebrain.trajectory_scorer_v1_train_metrics.v0"


def train_trajectory_scorer_v1(
    *,
    action_pack: str | Path,
    out_dir: str | Path,
    source_names: set[str] | None = None,
    source_families: set[str] | None = None,
    max_steps: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1.0e-3,
    device_name: str | None = None,
    max_examples: int | None = None,
    seed: int = 14,
    bev_source: str = "oracle",
    modeld_dir: str | Path | None = None,
    class_balanced_loss: bool = True,
) -> dict:
    manifest = read_json(Path(action_pack) / "manifest.json")
    if manifest.get("schema_version") != "homebrain.action_label_pack.v5":
        raise ValueError("TrajectoryScorerNet v1 requires ActionLabelPack v5")
    metrics = train_trajectory_scorer_v0(
        action_pack=action_pack,
        out_dir=out_dir,
        source_names=source_names,
        source_families=source_families,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device_name=device_name,
        max_examples=max_examples,
        seed=seed,
        bev_source=bev_source,
        modeld_dir=modeld_dir,
        class_balanced_loss=class_balanced_loss,
    )
    metrics = dict(metrics)
    metrics.update(
        {
            "schema_version": TRAJECTORY_SCORER_V1_TRAIN_SCHEMA_VERSION,
            "model_name": "TrajectoryScorerNetV1",
            "architecture": "TrajectoryScorerNetV0",
            "label_source": "future_motion_behavior_cloning",
            "action_pack_version": 5,
            "not_synthetic_expert": True,
            "class_balanced_loss": bool(class_balanced_loss),
        }
    )
    write_json(Path(out_dir) / "train_metrics.json", metrics, pretty=True)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train replay-only TrajectoryScorerNet v1 from ActionLabelPack v5.")
    parser.add_argument("--action-pack", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--source-family", action="append", default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--bev-source", choices=("oracle", "model"), default="oracle")
    parser.add_argument("--modeld", default=None)
    parser.add_argument("--no-class-balanced-loss", action="store_true")
    args = parser.parse_args(argv)
    metrics = train_trajectory_scorer_v1(
        action_pack=args.action_pack,
        out_dir=args.out,
        source_names=_parse_repeated(args.source_name),
        source_families=_parse_repeated(args.source_family),
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device_name=args.device,
        max_examples=args.max_examples,
        seed=args.seed,
        bev_source=args.bev_source,
        modeld_dir=args.modeld,
        class_balanced_loss=not args.no_class_balanced_loss,
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
