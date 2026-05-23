from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from homebrain.data.spatial_dataset import mean, read_json, write_json
from homebrain.messages.schema import JsonDict
from homebrain.policies.run_trajectory_scorer import (
    BevDecisionInput,
    _load_decision_inputs,
    run_trajectory_scorer,
)
from homebrain.policies.trajectory_scorer import LocalBev

STOP_AUDIT_SCHEMA_VERSION = "homebrain.stop_heavy_audit.v0"
HISTOGRAM_BINS = (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1.000001)
OCCUPIED_HIT_THRESHOLD = 0.35
UNKNOWN_HIT_THRESHOLD = 0.5
UNKNOWN_BLOCK_THRESHOLD = 0.95


def audit_stop_heavy(
    *,
    policy_dir: str | Path,
    out_dir: str | Path,
    labels: str | Path | None = None,
    modeld: str | Path | None = None,
    max_contact_frames: int = 24,
) -> dict[str, Any]:
    policy_root = Path(policy_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    _clear_audit_outputs(output)

    decisions = _read_decisions(policy_root)
    manifest = _read_optional_json(policy_root / "trajectory_policy_manifest.json")
    eval_metrics = _read_optional_json(policy_root / "trajectory_eval.json")
    records, bev_load_error = _load_audit_records(policy_root=policy_root, manifest=manifest, modeld=modeld, labels=labels)
    records_by_key = _records_by_key(records)

    policy_metrics = _policy_metrics(decisions)
    footprint_metrics = _footprint_metrics(decisions, records_by_key)
    channel_histograms = _bev_channel_histograms(records)
    likely_root_causes = _likely_root_causes(
        stop_fraction=float(policy_metrics["stop_selected_fraction"]),
        risky_fraction=float(policy_metrics["risky_candidate_fraction"]),
        uncertainty_penalty=float(policy_metrics["uncertainty_penalty_mean"]),
        coverage_gain=float(policy_metrics["coverage_gain_mean"]),
        footprint_metrics=footprint_metrics,
        channel_histograms=channel_histograms,
        bev_load_error=bev_load_error,
    )

    comparison = None
    if labels is not None:
        comparison = _write_label_policy_comparison(
            policy_root=policy_root,
            labels=Path(labels),
            output=output,
            model_decisions=decisions,
        )

    contact_sheet_path = output / "worst_frame_contact_sheet.ppm"
    _write_worst_frame_contact_sheet(
        path=contact_sheet_path,
        decisions=decisions,
        records_by_key=records_by_key,
        max_frames=max_contact_frames,
    )

    audit: dict[str, Any] = {
        "schema_version": STOP_AUDIT_SCHEMA_VERSION,
        "policy_dir": policy_root.as_posix(),
        "modeld_dir": Path(modeld).as_posix() if modeld is not None else None,
        "labels": Path(labels).as_posix() if labels is not None else None,
        "policy_manifest": manifest,
        "policy_eval": eval_metrics,
        "frame_count": len(decisions),
        "bev_frame_count": len(records),
        "bev_load_error": bev_load_error,
        "stop_selected_fraction": policy_metrics["stop_selected_fraction"],
        "selected_motion_fraction": policy_metrics["selected_motion_fraction"],
        "risky_candidate_fraction": policy_metrics["risky_candidate_fraction"],
        "occupied_hit_rate": footprint_metrics["occupied_hit_rate"],
        "unknown_hit_rate": footprint_metrics["unknown_hit_rate"],
        "uncertainty_penalty_mean": policy_metrics["uncertainty_penalty_mean"],
        "coverage_gain_mean": policy_metrics["coverage_gain_mean"],
        "candidate_footprint_block_rate": footprint_metrics["candidate_footprint_block_rate"],
        "selected_occupied_hit_rate": footprint_metrics["selected_occupied_hit_rate"],
        "selected_unknown_hit_rate": footprint_metrics["selected_unknown_hit_rate"],
        "selected_motion_occupied_hit_rate": footprint_metrics["selected_motion_occupied_hit_rate"],
        "all_candidate_uncertainty_penalty_mean": policy_metrics["all_candidate_uncertainty_penalty_mean"],
        "all_candidate_coverage_gain_mean": policy_metrics["all_candidate_coverage_gain_mean"],
        "bev_channel_histograms": channel_histograms,
        "likely_root_causes": likely_root_causes,
        "label_bev_policy_comparison": comparison,
        "worst_frame_contact_sheet": contact_sheet_path.name,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
    }
    write_json(output / "stop_heavy_audit.json", audit, pretty=True)
    return audit


def _read_decisions(policy_root: Path) -> list[JsonDict]:
    path = policy_root / "trajectory_decisions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing policy decisions: {path}")
    decisions: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            if not isinstance(data, dict):
                raise ValueError(f"decision line must be a JSON object: {path}")
            decisions.append(data)
    return decisions


def _read_optional_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    return read_json(path)


def _load_audit_records(
    *,
    policy_root: Path,
    manifest: JsonDict | None,
    modeld: str | Path | None,
    labels: str | Path | None,
) -> tuple[list[BevDecisionInput], str | None]:
    source_root: Path | None = None
    bev_source = "model"
    if modeld is not None:
        source_root = Path(modeld)
        bev_source = "model"
    elif manifest is not None and isinstance(manifest.get("source"), str):
        source_root = Path(str(manifest["source"]))
        bev_source = str(manifest.get("bev_source", "model"))
    elif labels is not None:
        source_root = Path(labels)
        bev_source = "labels"
    else:
        source_root = policy_root

    try:
        records, _metadata = _load_decision_inputs(
            source_root,
            policy_root,
            checkpoint=None,
            features=None,
            bev_source=bev_source,
            device_name=None,
        )
        return records, None
    except Exception as exc:  # noqa: BLE001 - audit should still report serialized score issues.
        return [], str(exc)


def _records_by_key(records: list[BevDecisionInput]) -> dict[tuple[str, str, int], BevDecisionInput]:
    return {
        (record.sequence_id, record.camera_id, int(record.frame_id)): record
        for record in records
    }


def _policy_metrics(decisions: list[JsonDict]) -> dict[str, float]:
    selected_ids = [str(decision.get("selected_candidate_id", "")) for decision in decisions]
    selected_scores = [
        score for score in (_selected_score(decision) for decision in decisions) if isinstance(score, dict)
    ]
    all_scores = [
        candidate.get("score")
        for decision in decisions
        for candidate in _decision_candidates(decision)
        if isinstance(candidate.get("score"), dict)
    ]
    stop_count = sum(1 for value in selected_ids if value == "stop")
    risky_fractions = [
        float(decision.get("risky_candidate_fraction", 0.0))
        for decision in decisions
        if isinstance(decision.get("risky_candidate_fraction"), (int, float))
    ]
    return {
        "stop_selected_fraction": float(stop_count / max(len(decisions), 1)),
        "selected_motion_fraction": float((len(decisions) - stop_count) / max(len(decisions), 1)),
        "risky_candidate_fraction": mean(risky_fractions),
        "uncertainty_penalty_mean": mean([float(score.get("uncertainty_penalty", 0.0)) for score in selected_scores]),
        "coverage_gain_mean": mean([float(score.get("coverage_gain_proxy", 0.0)) for score in selected_scores]),
        "all_candidate_uncertainty_penalty_mean": mean(
            [float(score.get("uncertainty_penalty", 0.0)) for score in all_scores if isinstance(score, dict)]
        ),
        "all_candidate_coverage_gain_mean": mean(
            [float(score.get("coverage_gain_proxy", 0.0)) for score in all_scores if isinstance(score, dict)]
        ),
    }


def _footprint_metrics(
    decisions: list[JsonDict],
    records_by_key: dict[tuple[str, str, int], BevDecisionInput],
) -> dict[str, float]:
    occupied_hits = 0
    unknown_hits = 0
    blocked = 0
    total = 0
    selected_occupied_hits = 0
    selected_unknown_hits = 0
    selected_total = 0
    selected_motion_occupied_hits = 0
    selected_motion_total = 0

    for decision in decisions:
        record = _matching_record(decision, records_by_key)
        if record is None:
            continue
        occupied = _occupied(record.bev)
        unknown = _as_probability(record.bev.unknown)
        for candidate in _decision_candidates(decision):
            cells = _candidate_cells(candidate)
            if not cells:
                occ_hit = True
                unknown_hit = True
            else:
                occ_values = _cell_values(occupied, cells, default=1.0)
                unknown_values = _cell_values(unknown, cells, default=1.0)
                occ_hit = bool(np.max(occ_values) >= OCCUPIED_HIT_THRESHOLD)
                unknown_hit = bool(np.max(unknown_values) >= UNKNOWN_HIT_THRESHOLD)
            candidate_id = str(candidate.get("id", ""))
            if candidate_id != "stop":
                total += 1
                occupied_hits += int(occ_hit)
                unknown_hits += int(unknown_hit)
                blocked += int(occ_hit or unknown_hit)
            if candidate_id == decision.get("selected_candidate_id"):
                selected_total += 1
                selected_occupied_hits += int(occ_hit)
                selected_unknown_hits += int(unknown_hit)
                if candidate_id != "stop":
                    selected_motion_total += 1
                    selected_motion_occupied_hits += int(occ_hit)

    return {
        "occupied_hit_rate": float(occupied_hits / max(total, 1)),
        "unknown_hit_rate": float(unknown_hits / max(total, 1)),
        "candidate_footprint_block_rate": float(blocked / max(total, 1)),
        "selected_occupied_hit_rate": float(selected_occupied_hits / max(selected_total, 1)),
        "selected_unknown_hit_rate": float(selected_unknown_hits / max(selected_total, 1)),
        "selected_motion_occupied_hit_rate": float(selected_motion_occupied_hits / max(selected_motion_total, 1)),
    }


def _bev_channel_histograms(records: list[BevDecisionInput]) -> dict[str, JsonDict]:
    channels: dict[str, list[np.ndarray]] = {
        "free": [],
        "occupied": [],
        "unknown": [],
        "traversable": [],
        "risky": [],
        "confidence": [],
        "uncertainty": [],
    }
    for record in records:
        bev = record.bev
        channels["free"].append(_as_probability(bev.free).reshape(-1))
        channels["occupied"].append(_as_probability(bev.occupied).reshape(-1))
        channels["unknown"].append(_as_probability(bev.unknown).reshape(-1))
        if bev.traversable is not None:
            channels["traversable"].append(_as_probability(bev.traversable).reshape(-1))
        if bev.risky is not None:
            channels["risky"].append(_as_probability(bev.risky).reshape(-1))
        if bev.confidence is not None:
            channels["confidence"].append(_as_probability(bev.confidence).reshape(-1))
        if bev.uncertainty is not None:
            channels["uncertainty"].append(_as_probability(bev.uncertainty).reshape(-1))

    histograms: dict[str, JsonDict] = {}
    for name, values in channels.items():
        if not values:
            continue
        merged = np.concatenate(values).astype(np.float32)
        counts, bins = np.histogram(merged, bins=np.asarray(HISTOGRAM_BINS, dtype=np.float32))
        histograms[name] = {
            "mean": float(np.mean(merged)),
            "min": float(np.min(merged)),
            "max": float(np.max(merged)),
            "nonzero_fraction": float(np.count_nonzero(merged > np.float32(0.0)) / max(merged.size, 1)),
            "bins": [round(float(value), 6) for value in bins.tolist()],
            "counts": [int(value) for value in counts.tolist()],
        }
    return histograms


def _likely_root_causes(
    *,
    stop_fraction: float,
    risky_fraction: float,
    uncertainty_penalty: float,
    coverage_gain: float,
    footprint_metrics: dict[str, float],
    channel_histograms: dict[str, JsonDict],
    bev_load_error: str | None,
) -> list[str]:
    causes: list[str] = []
    if bev_load_error:
        causes.append("bev_channels_unavailable_audit_fell_back_to_serialized_policy_scores")
    if stop_fraction >= 0.80 and risky_fraction >= 0.80:
        causes.append("most_motion_candidates_are_marked_risky_so_stop_bonus_dominates")
    if footprint_metrics.get("occupied_hit_rate", 0.0) >= 0.80:
        causes.append("candidate_footprints_frequently_overlap_occupied_or_risky_cells")
    if footprint_metrics.get("unknown_hit_rate", 0.0) >= 0.80:
        causes.append("candidate_footprints_frequently_overlap_unknown_cells")
    if footprint_metrics.get("candidate_footprint_block_rate", 0.0) >= 0.80:
        causes.append("motion_candidate_footprints_are_mostly_blocked_by_occupied_or_unknown_bev")
    if uncertainty_penalty >= 0.50:
        causes.append("selected_candidates_have_high_uncertainty_penalty")
    if coverage_gain <= 0.25:
        causes.append("selected_candidates_have_little_or_no_coverage_gain")
    if _channel_mean(channel_histograms, "free") <= 0.05 and _channel_mean(channel_histograms, "traversable") <= 0.05:
        causes.append("free_or_traversable_probability_is_sparse")
    if _channel_mean(channel_histograms, "occupied") >= 0.30 or _channel_mean(channel_histograms, "risky") >= 0.30:
        causes.append("occupied_or_risky_probability_is_dense")
    if _channel_mean(channel_histograms, "unknown") >= 0.70:
        causes.append("bev_is_dominated_by_unknown_cells")
    if not causes:
        causes.append("no_single_dominant_stop_cause_detected_from_available_replay_artifacts")
    return causes


def _write_label_policy_comparison(
    *,
    policy_root: Path,
    labels: Path,
    output: Path,
    model_decisions: list[JsonDict],
) -> dict[str, Any]:
    label_policy_dir = output / "label_bev_policy"
    run_trajectory_scorer(log_dir=labels, out_dir=label_policy_dir, bev_source="labels")
    label_decisions = _read_decisions(label_policy_dir)
    comparison = _compare_decisions(model_decisions, label_decisions)
    comparison.update(
        {
            "schema_version": "homebrain.model_label_policy_comparison.v0",
            "model_policy_dir": policy_root.as_posix(),
            "label_pack": labels.as_posix(),
            "label_policy_dir": label_policy_dir.as_posix(),
            "replay_only": True,
            "not_executed": True,
            "control_safe": False,
        }
    )
    write_json(output / "model_vs_label_policy_comparison.json", comparison, pretty=True)
    return comparison


def _compare_decisions(model_decisions: list[JsonDict], label_decisions: list[JsonDict]) -> dict[str, Any]:
    model_by_key = _decisions_by_key(model_decisions)
    label_by_key = _decisions_by_key(label_decisions)
    overlap_keys = sorted(set(model_by_key) & set(label_by_key))
    if not overlap_keys:
        count = min(len(model_decisions), len(label_decisions))
        pairs = [(model_decisions[index], label_decisions[index]) for index in range(count)]
    else:
        pairs = [(model_by_key[key], label_by_key[key]) for key in overlap_keys]
    model_ids = [str(pair[0].get("selected_candidate_id", "")) for pair in pairs]
    label_ids = [str(pair[1].get("selected_candidate_id", "")) for pair in pairs]
    agreement = sum(1 for model_id, label_id in zip(model_ids, label_ids) if model_id == label_id)
    model_stop = sum(1 for value in model_ids if value == "stop")
    label_stop = sum(1 for value in label_ids if value == "stop")
    model_stop_label_motion = sum(1 for model_id, label_id in zip(model_ids, label_ids) if model_id == "stop" and label_id != "stop")
    model_motion_label_stop = sum(1 for model_id, label_id in zip(model_ids, label_ids) if model_id != "stop" and label_id == "stop")
    return {
        "model_frame_count": len(model_decisions),
        "label_frame_count": len(label_decisions),
        "overlap_count": len(pairs),
        "agreement_fraction": float(agreement / max(len(pairs), 1)),
        "model_stop_selected_fraction": float(model_stop / max(len(pairs), 1)),
        "model_selected_motion_fraction": float((len(pairs) - model_stop) / max(len(pairs), 1)),
        "label_stop_selected_fraction": float(label_stop / max(len(pairs), 1)),
        "label_selected_motion_fraction": float((len(pairs) - label_stop) / max(len(pairs), 1)),
        "model_stop_label_motion_fraction": float(model_stop_label_motion / max(len(pairs), 1)),
        "model_motion_label_stop_fraction": float(model_motion_label_stop / max(len(pairs), 1)),
        "model_selected_distribution": _distribution(model_ids),
        "label_selected_distribution": _distribution(label_ids),
    }


def _write_worst_frame_contact_sheet(
    *,
    path: Path,
    decisions: list[JsonDict],
    records_by_key: dict[tuple[str, str, int], BevDecisionInput],
    max_frames: int,
) -> None:
    ranked = sorted(decisions, key=_decision_severity, reverse=True)[:max_frames]
    tiles: list[np.ndarray] = []
    for decision in ranked:
        record = _matching_record(decision, records_by_key)
        if record is None:
            continue
        tiles.append(_audit_tile(record.bev, decision))
    _write_contact_sheet(path, tiles)


def _audit_tile(bev: LocalBev, decision: JsonDict) -> np.ndarray:
    free = _as_probability(bev.free)
    occupied = _occupied(bev)
    unknown = _as_probability(bev.unknown)
    uncertainty = _as_probability(bev.uncertainty) if bev.uncertainty is not None else unknown
    rgb = np.stack([np.maximum(occupied, uncertainty * 0.25), free, np.maximum(unknown * 0.7, uncertainty)], axis=2)
    tile = (np.clip(rgb, 0.0, 1.0) * np.float32(170.0)).round().astype(np.uint8)
    scale = 6 if min(bev.shape) >= 16 else 16
    tile = np.repeat(np.repeat(tile, scale, axis=0), scale, axis=1)
    selected_id = str(decision.get("selected_candidate_id", ""))
    for candidate in _decision_candidates(decision):
        score = candidate.get("score") if isinstance(candidate.get("score"), dict) else {}
        risky = bool(score.get("risky", False)) if isinstance(score, dict) else False
        color = np.asarray([200, 70, 70] if risky else [130, 130, 130], dtype=np.uint8)
        if str(candidate.get("id", "")) == selected_id:
            color = np.asarray([255, 230, 40], dtype=np.uint8)
        for row, col in _candidate_cells(candidate):
            _paint_cell(tile, row, col, scale, color)
    return tile


def _write_contact_sheet(path: Path, tiles: list[np.ndarray]) -> None:
    if not tiles:
        path.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
        return
    tile_h, tile_w, _ = tiles[0].shape
    cols = min(6, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // cols
        col = index % cols
        sheet[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    path.write_bytes(f"P6\n{sheet.shape[1]} {sheet.shape[0]}\n255\n".encode("ascii") + sheet.tobytes())


def _paint_cell(tile: np.ndarray, row: int, col: int, scale: int, color: np.ndarray) -> None:
    row_start = row * scale
    col_start = col * scale
    if row_start < 0 or col_start < 0 or row_start >= tile.shape[0] or col_start >= tile.shape[1]:
        return
    row_end = min(row_start + scale, tile.shape[0])
    col_end = min(col_start + scale, tile.shape[1])
    patch = tile[row_start:row_end, col_start:col_end]
    tile[row_start:row_end, col_start:col_end] = ((patch.astype(np.uint16) + color.astype(np.uint16)) // 2).astype(np.uint8)


def _decision_severity(decision: JsonDict) -> tuple[float, float, float, float]:
    selected = _selected_score(decision) or {}
    return (
        1.0 if decision.get("selected_candidate_id") == "stop" else 0.0,
        float(decision.get("risky_candidate_fraction", 0.0)),
        float(selected.get("risk_score", 0.0)) if isinstance(selected, dict) else 0.0,
        float(selected.get("unknown_penalty", 0.0)) if isinstance(selected, dict) else 0.0,
    )


def _matching_record(
    decision: JsonDict,
    records_by_key: dict[tuple[str, str, int], BevDecisionInput],
) -> BevDecisionInput | None:
    key = (
        str(decision.get("sequence_id", "")),
        str(decision.get("camera_id", "")),
        int(decision.get("frame_id", -1)),
    )
    return records_by_key.get(key)


def _decision_candidates(decision: JsonDict) -> list[JsonDict]:
    candidates = decision.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def _selected_score(decision: JsonDict) -> JsonDict | None:
    score = decision.get("selected_score")
    if isinstance(score, dict):
        return score
    selected_id = decision.get("selected_candidate_id")
    for candidate in _decision_candidates(decision):
        if candidate.get("id") == selected_id and isinstance(candidate.get("score"), dict):
            return candidate["score"]
    return None


def _candidate_cells(candidate: JsonDict) -> tuple[tuple[int, int], ...]:
    cells = candidate.get("footprint_cells")
    if not isinstance(cells, list):
        return ()
    parsed: list[tuple[int, int]] = []
    for cell in cells:
        if isinstance(cell, list) and len(cell) == 2:
            parsed.append((int(cell[0]), int(cell[1])))
    return tuple(parsed)


def _decisions_by_key(decisions: list[JsonDict]) -> dict[tuple[str, str, int], JsonDict]:
    return {
        (
            str(decision.get("sequence_id", "")),
            str(decision.get("camera_id", "")),
            int(decision.get("frame_id", -1)),
        ): decision
        for decision in decisions
    }


def _distribution(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _occupied(bev: LocalBev) -> np.ndarray:
    return np.maximum(_as_probability(bev.occupied), _as_probability(bev.risky) if bev.risky is not None else 0.0)


def _cell_values(array: np.ndarray, cells: tuple[tuple[int, int], ...], *, default: float) -> np.ndarray:
    if not cells:
        return np.asarray([default], dtype=np.float32)
    values: list[float] = []
    for row, col in cells:
        if 0 <= row < array.shape[0] and 0 <= col < array.shape[1]:
            values.append(float(array[row, col]))
        else:
            values.append(default)
    if not values:
        values.append(default)
    return np.asarray(values, dtype=np.float32)


def _as_probability(array: np.ndarray | float | None) -> np.ndarray:
    if array is None:
        return np.asarray(0.0, dtype=np.float32)
    return np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)


def _channel_mean(channel_histograms: dict[str, JsonDict], name: str) -> float:
    stats = channel_histograms.get(name)
    if not isinstance(stats, dict):
        return 0.0
    value = stats.get("mean")
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _clear_audit_outputs(output: Path) -> None:
    for name in (
        "stop_heavy_audit.json",
        "worst_frame_contact_sheet.ppm",
        "model_vs_label_policy_comparison.json",
    ):
        path = output / name
        if path.exists():
            path.unlink()
    label_policy = output / "label_bev_policy"
    if label_policy.exists():
        shutil.rmtree(label_policy)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit stop-heavy replay-only trajectory policy artifacts.")
    parser.add_argument("--policy", required=True, help="Policy artifact directory.")
    parser.add_argument("--out", required=True, help="Output audit directory.")
    parser.add_argument("--labels", default=None, help="Optional SpatialTrainPack or BEV label pack for comparison.")
    parser.add_argument("--modeld", default=None, help="Optional modeld output directory for BEV channel audit.")
    parser.add_argument("--max-contact-frames", type=int, default=24)
    args = parser.parse_args(argv)
    metrics = audit_stop_heavy(
        policy_dir=args.policy,
        out_dir=args.out,
        labels=args.labels,
        modeld=args.modeld,
        max_contact_frames=args.max_contact_frames,
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
