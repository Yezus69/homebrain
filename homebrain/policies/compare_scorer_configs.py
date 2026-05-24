from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from homebrain.data.spatial_dataset import read_json, write_json
from homebrain.policies.bev_action_sanity import infer_policy_bev_source, load_bev_frames_for_action
from homebrain.policies.configurable_scorer import (
    TransparentScorerConfig,
    decisions_jsonl_records,
    evaluate_config,
    write_jsonl,
)
from homebrain.policies.run_trajectory_scorer import run_trajectory_scorer

COMPARISON_SCHEMA_VERSION = "homebrain.scorer_config_comparison.v0"


def compare_scorer_configs(
    *,
    sources: list[str | Path],
    out_dir: str | Path,
    config_path: str | Path,
) -> dict[str, object]:
    if not sources:
        raise ValueError("at least one --source is required")
    output = Path(out_dir)
    _clear_outputs(output)
    output.mkdir(parents=True, exist_ok=True)
    config = _load_selected_config(config_path)
    comparisons = []
    for source in sources:
        comparisons.append(_compare_one_source(Path(source), output=output, config=config))

    report: dict[str, object] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "sources": [Path(source).as_posix() for source in sources],
        "config_source": Path(config_path).as_posix(),
        "calibrated_config": config.to_dict(),
        "comparisons": comparisons,
        "reviewed_sources_remaining_stop_heavy": [
            row["source"]
            for row in comparisons
            if row.get("action_learning_role") == "geometry_only"
        ],
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
    }
    write_json(output / "scorer_config_comparison.json", report, pretty=True)
    return report


def _compare_one_source(source: Path, *, output: Path, config: TransparentScorerConfig) -> dict[str, object]:
    slug = _slug(source)
    source_dir = output / slug
    source_dir.mkdir(parents=True, exist_ok=True)
    frames, metadata = load_bev_frames_for_action(source, scratch_dir=source_dir / "_scratch")
    old_policy_dir = source_dir / "old_policy"
    run_trajectory_scorer(
        log_dir=source,
        out_dir=old_policy_dir,
        bev_source=infer_policy_bev_source(source),
    )
    old_eval = read_json(old_policy_dir / "trajectory_eval.json")
    calibrated_records = decisions_jsonl_records(frames, config=config)
    write_jsonl(source_dir / "calibrated_trajectory_decisions.jsonl", calibrated_records)
    calibrated_eval = evaluate_config(frames, config=config)
    calibrated_eval.update(
        {
            "schema_version": "homebrain.calibrated_trajectory_eval.v0",
            "source": source.as_posix(),
            "scorer_config": config.to_dict(),
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
        }
    )
    write_json(source_dir / "calibrated_trajectory_eval.json", calibrated_eval, pretty=True)
    action_supervision_ok_fraction = _action_supervision_ok_fraction(frames)
    calibrated_stop_fraction = float(calibrated_eval.get("stop_fraction", 0.0))
    old_stop_fraction = float(old_eval.get("stop_selected_fraction", 0.0))
    reviewed_or_model = not any(frame.source_family == "controlled_bev" for frame in frames)
    role = "action_candidate"
    reason = "calibrated_source_not_stop_heavy"
    if reviewed_or_model and (action_supervision_ok_fraction <= 0.0 or calibrated_stop_fraction >= 0.75):
        role = "geometry_only"
        reason = "not_robot_frame_action_truth_or_remains_stop_heavy"
    comparison = {
        "source": source.as_posix(),
        "source_metadata": metadata,
        "old_policy_dir": old_policy_dir.as_posix(),
        "calibrated_decisions": (source_dir / "calibrated_trajectory_decisions.jsonl").as_posix(),
        "calibrated_eval": (source_dir / "calibrated_trajectory_eval.json").as_posix(),
        "old_stop_fraction": old_stop_fraction,
        "calibrated_stop_fraction": calibrated_stop_fraction,
        "old_motion_fraction": 1.0 - old_stop_fraction,
        "calibrated_motion_fraction": 1.0 - calibrated_stop_fraction,
        "action_supervision_ok_fraction": action_supervision_ok_fraction,
        "action_learning_role": role,
        "action_learning_role_reason": reason,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
    }
    write_json(source_dir / "comparison.json", comparison, pretty=True)
    return comparison


def _action_supervision_ok_fraction(frames: list[object]) -> float:
    from homebrain.policies.bev_action_sanity import ActionSanityConfig, evaluate_loaded_frame

    if not frames:
        return 0.0
    config = ActionSanityConfig()
    count = 0
    for frame in frames:
        metric = evaluate_loaded_frame(frame, config=config)  # type: ignore[arg-type]
        count += int(bool(metric.get("action_supervision_ok")))
    return float(count / max(len(frames), 1))


def _load_selected_config(path: str | Path) -> TransparentScorerConfig:
    report = read_json(path)
    selected = report.get("selected_config")
    if not isinstance(selected, dict):
        selected_metrics = report.get("selected_metrics")
        if isinstance(selected_metrics, dict) and isinstance(selected_metrics.get("config"), dict):
            selected = selected_metrics["config"]
    if not isinstance(selected, dict):
        raise ValueError(f"sweep report does not contain selected_config: {path}")
    return TransparentScorerConfig.from_dict(selected)


def _slug(path: Path) -> str:
    text = path.name or "source"
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)[:80]


def _clear_outputs(output: Path) -> None:
    if not output.exists():
        return
    for name in ("scorer_config_comparison.json",):
        path = output / name
        if path.exists():
            path.unlink()
    for child in output.iterdir():
        if child.is_dir():
            shutil.rmtree(child)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare old hardcoded and calibrated transparent trajectory scorers.")
    parser.add_argument("--source", action="append", required=True, help="Input SpatialTrainPack or modeld output. May repeat.")
    parser.add_argument("--config", required=True, help="scorer_sweep.json containing selected_config.")
    parser.add_argument("--out", required=True, help="Output comparison directory.")
    args = parser.parse_args(argv)
    report = compare_scorer_configs(
        sources=[Path(source) for source in args.source],
        out_dir=args.out,
        config_path=args.config,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
