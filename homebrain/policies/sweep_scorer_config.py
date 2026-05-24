from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from homebrain.data.spatial_dataset import write_json
from homebrain.policies.bev_action_sanity import load_bev_frames_for_action
from homebrain.policies.configurable_scorer import (
    TransparentScorerConfig,
    default_sweep_configs,
    evaluate_config,
)

SWEEP_SCHEMA_VERSION = "homebrain.scorer_config_sweep.v0"


def sweep_scorer_config(
    *,
    sources: list[str | Path],
    out_dir: str | Path,
) -> dict[str, object]:
    if not sources:
        raise ValueError("at least one --source is required")
    output = Path(out_dir)
    _clear_outputs(output)
    output.mkdir(parents=True, exist_ok=True)

    frames = []
    source_metadata = []
    for source in sources:
        loaded, metadata = load_bev_frames_for_action(source, scratch_dir=output / "_scratch")
        frames.extend(loaded)
        source_metadata.append(metadata)
    if not frames:
        raise ValueError("no BEV frames found for scorer sweep")

    rows = [evaluate_config(frames, config=config) for config in default_sweep_configs()]
    eligible = [
        row
        for row in rows
        if bool(row.get("passes_controlled_open")) and bool(row.get("passes_controlled_blocked"))
    ]
    selected = min(
        eligible,
        key=lambda row: (
            float(row.get("collision_proxy_rate", 1.0)),
            float(row.get("stop_fraction", 1.0)),
            -float(row.get("reviewed_motion_rate", 0.0)),
            json.dumps(row.get("config", {}), sort_keys=True),
        ),
        default=None,
    )
    pareto = sorted(
        rows,
        key=lambda row: (
            not (bool(row.get("passes_controlled_open")) and bool(row.get("passes_controlled_blocked"))),
            float(row.get("collision_proxy_rate", 1.0)),
            float(row.get("stop_fraction", 1.0)),
            -float(row.get("reviewed_motion_rate", 0.0)),
            json.dumps(row.get("config", {}), sort_keys=True),
        ),
    )
    report: dict[str, object] = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "sources": [Path(source).as_posix() for source in sources],
        "source_metadata": source_metadata,
        "frame_count": len(frames),
        "config_count": len(rows),
        "eligible_config_count": len(eligible),
        "selected_config": selected.get("config") if isinstance(selected, dict) else None,
        "selected_metrics": selected,
        "pareto_table": pareto[:50],
        "all_results_path": "scorer_sweep_all_results.json",
        "pareto_table_path": "scorer_sweep_pareto.csv",
        "selection_policy": {
            "controlled_gates": {
                "controlled_open_motion_rate_min": 0.95,
                "blocked_map_stop_rate_min": 0.95,
            },
            "sort": [
                "collision_proxy_rate asc",
                "stop_fraction asc",
                "reviewed_motion_rate desc",
                "config json asc",
            ],
            "not_learned": True,
            "no_forced_motion": True,
        },
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
    }
    write_json(output / "scorer_sweep.json", report, pretty=True)
    write_json(output / "scorer_sweep_all_results.json", {"results": rows, "control_safe": False}, pretty=True)
    _write_csv(output / "scorer_sweep_pareto.csv", pareto)
    return report


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "rank",
        "controlled_open_motion_rate",
        "blocked_map_stop_rate",
        "reviewed_motion_rate",
        "collision_proxy_rate",
        "stop_fraction",
        "passes_controlled_open",
        "passes_controlled_blocked",
        "unknown_weight",
        "risk_weight",
        "obstacle_threshold",
        "footprint_radius_cells",
        "obstacle_inflation_cells",
        "stop_bias",
    ]
    lines = [",".join(headers)]
    for index, row in enumerate(rows, start=1):
        config = row.get("config") if isinstance(row.get("config"), dict) else {}
        values = [
            index,
            row.get("controlled_open_motion_rate", 0.0),
            row.get("blocked_map_stop_rate", 0.0),
            row.get("reviewed_motion_rate", 0.0),
            row.get("collision_proxy_rate", 0.0),
            row.get("stop_fraction", 0.0),
            row.get("passes_controlled_open", False),
            row.get("passes_controlled_blocked", False),
            config.get("unknown_weight", ""),
            config.get("risk_weight", ""),
            config.get("obstacle_threshold", ""),
            config.get("footprint_radius_cells", ""),
            config.get("obstacle_inflation_cells", ""),
            config.get("stop_bias", ""),
        ]
        lines.append(",".join(str(value) for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clear_outputs(output: Path) -> None:
    if output.exists():
        for name in ("scorer_sweep.json", "scorer_sweep_all_results.json", "scorer_sweep_pareto.csv"):
            path = output / name
            if path.exists():
                path.unlink()
        scratch = output / "_scratch"
        if scratch.exists():
            shutil.rmtree(scratch)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep transparent replay-only trajectory scorer knobs.")
    parser.add_argument("--source", action="append", required=True, help="Input BEV pack/modeld source. May repeat.")
    parser.add_argument("--out", required=True, help="Output sweep directory.")
    args = parser.parse_args(argv)
    report = sweep_scorer_config(sources=[Path(source) for source in args.source], out_dir=args.out)
    printable = {key: value for key, value in report.items() if key != "pareto_table"}
    print(json.dumps(printable, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
