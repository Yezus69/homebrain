from __future__ import annotations

import argparse
import json
from pathlib import Path

from homebrain.data.spatial_dataset import read_json, write_json
from homebrain.policies.trajectory_scorer_net_v0 import _parse_repeated, eval_trajectory_scorer_v0


TRAJECTORY_SCORER_V1_EVAL_SCHEMA_VERSION = "homebrain.trajectory_scorer_v1_eval_metrics.v0"


def eval_trajectory_scorer_v1(
    *,
    checkpoint: str | Path,
    action_pack: str | Path,
    out_path: str | Path,
    source_names: set[str] | None = None,
    source_families: set[str] | None = None,
    bev_source: str = "oracle",
    modeld_dir: str | Path | None = None,
    split: str | None = "val",
    device_name: str | None = None,
    max_examples: int | None = None,
    viz_dir: str | Path | None = None,
    max_viz_frames: int = 24,
    candidate_feature_mode: str | None = None,
) -> dict:
    manifest = read_json(Path(action_pack) / "manifest.json")
    if manifest.get("schema_version") != "homebrain.action_label_pack.v5":
        raise ValueError("TrajectoryScorerNet v1 eval requires ActionLabelPack v5")
    metrics = eval_trajectory_scorer_v0(
        checkpoint=checkpoint,
        action_pack=action_pack,
        out_path=out_path,
        source_names=source_names,
        source_families=source_families,
        bev_source=bev_source,
        modeld_dir=modeld_dir,
        split=split,
        device_name=device_name,
        max_examples=max_examples,
        viz_dir=viz_dir,
        max_viz_frames=max_viz_frames,
        candidate_feature_mode=candidate_feature_mode,
    )
    metrics = dict(metrics)
    metrics.update(
        {
            "schema_version": TRAJECTORY_SCORER_V1_EVAL_SCHEMA_VERSION,
            "model_name": "TrajectoryScorerNetV1",
            "architecture": "TrajectoryScorerNetV0",
            "label_source": "future_motion_behavior_cloning",
            "action_pack_version": 5,
            "not_synthetic_expert": True,
        }
    )
    write_json(out_path, metrics, pretty=True)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate replay-only TrajectoryScorerNet v1 on ActionLabelPack v5.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--action-pack", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--source-family", action="append", default=None)
    parser.add_argument(
        "--bev-source",
        choices=("oracle", "model", "v1_current_bev", "v1_memory_bev", "model_current", "model_memory"),
        default="oracle",
    )
    parser.add_argument("--modeld", default=None)
    parser.add_argument("--split", default="val", help="train, val, all, none, or empty string for all examples")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--viz-out", default=None)
    parser.add_argument("--max-viz-frames", type=int, default=24)
    parser.add_argument(
        "--candidate-feature-mode",
        choices=("checkpoint", "all", "all_features", "signed_ablation", "signed_feature_ablation"),
        default="checkpoint",
    )
    args = parser.parse_args(argv)
    metrics = eval_trajectory_scorer_v1(
        checkpoint=args.checkpoint,
        action_pack=args.action_pack,
        out_path=args.out,
        source_names=_parse_repeated(args.source_name),
        source_families=_parse_repeated(args.source_family),
        bev_source=args.bev_source,
        modeld_dir=args.modeld,
        split=None if args.split in {"", "all", "none"} else args.split,
        device_name=args.device,
        max_examples=args.max_examples,
        viz_dir=args.viz_out,
        max_viz_frames=args.max_viz_frames,
        candidate_feature_mode=None if args.candidate_feature_mode == "checkpoint" else args.candidate_feature_mode,
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
