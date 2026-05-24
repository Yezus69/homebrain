from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from homebrain.data.spatial_dataset import (
    DETERMINISTIC_CREATED_AT_UTC,
    scalar_bool,
    scalar_float,
    scalar_str,
    write_deterministic_npz,
    write_json,
)
from homebrain.messages.schema import JsonDict
from homebrain.policies.build_action_label_pack import (
    ACTION_LABEL_EXAMPLE_SCHEMA_VERSION,
    ACTION_LABEL_PACK_SCHEMA_VERSION_V5,
    FutureMotionLabel,
    PreparedActionExample,
    SourceFrame,
    _clear_generated_outputs,
    _distribution,
    _example_arrays,
    _labels_with_selected_candidate,
    _load_spatial_pack_frames,
    _meters_per_cell,
    _robot_radius_m,
    _scoring_description,
    expert_labels_for_frame,
)
from homebrain.policies.candidate_trajectories import candidates_hash, generate_default_candidates
from homebrain.policies.future_motion_action_labels import (
    CandidateMatchDistance,
    FutureMotionActionLabel,
    TimedPose2D,
    match_future_motion_to_candidate,
    pose_sample_from_base_pose,
)
from homebrain.teachers.artifacts import file_sha256, relative_to_root


LABEL_SOURCE = "future_motion_behavior_cloning"


def build_action_label_pack_v5(
    *,
    sources: list[str | Path],
    out_dir: str | Path,
    max_examples: int | None = None,
    horizon_s: float | None = None,
    bc_confidence_floor: float = 0.05,
) -> Path:
    if not sources:
        raise ValueError("at least one source is required")
    output = Path(out_dir)
    examples_dir = output / "examples"
    _clear_generated_outputs(output)
    examples_dir.mkdir(parents=True, exist_ok=True)

    source_roots = [Path(source) for source in sources]
    all_frames: list[SourceFrame] = []
    source_manifests: list[JsonDict] = []
    for source_root in source_roots:
        frames, manifest = _load_spatial_pack_frames(source_root)
        all_frames.extend(frames)
        source_manifests.append(manifest)
    if not all_frames:
        raise ValueError("no BEV frames found for action labeling")

    first_shape = all_frames[0].bev.shape
    meters_per_cell = _meters_per_cell(source_manifests[0], first_shape)
    robot_radius_m = _robot_radius_m(source_manifests[0])
    candidates = generate_default_candidates(
        grid_shape=first_shape,
        meters_per_cell=meters_per_cell,
        robot_radius_m=robot_radius_m,
    )
    candidate_ids = [candidate.id for candidate in candidates]
    sequences = _pose_sequences_by_key(all_frames)

    prepared: list[tuple[PreparedActionExample, FutureMotionActionLabel]] = []
    excluded_frames: list[JsonDict] = []
    for frame in all_frames:
        if frame.bev.shape != first_shape:
            raise ValueError(f"all ActionLabelPack v5 examples must share one grid shape; got {frame.bev.shape}")
        coverage_labels = expert_labels_for_frame(bev=frame.bev, candidates=candidates)
        bc_label = match_future_motion_to_candidate(
            sequence=sequences.get(_sequence_key(frame), []),
            current_frame_id=frame.frame_id,
            current_timestamp_ns=frame.timestamp_ns,
            candidates=candidates,
            horizon_s=horizon_s,
        )
        reasons = _exclusion_reasons(frame, bc_label)
        if reasons:
            excluded_frames.append(_excluded_record(frame, bc_label, reasons))
            continue
        training_labels = _labels_with_selected_candidate(coverage_labels, bc_label.candidate_id)
        future_labels = (_future_label_from_bc(bc_label),)
        prepared.append(
            (
                PreparedActionExample(
                    frame=frame,
                    training_labels=training_labels,
                    coverage_labels=coverage_labels,
                    future_labels=future_labels,
                    label_source_type=LABEL_SOURCE,
                ),
                bc_label,
            )
        )

    if max_examples is not None:
        prepared = prepared[:max_examples]
    if not prepared:
        raise ValueError("no valid behavior-cloning action labels found")

    examples: list[JsonDict] = []
    for index, (item, bc_label) in enumerate(prepared):
        frame = item.frame
        selected = _selected_candidate_id(item.training_labels)
        example_path = examples_dir / f"action_v5_{index:06d}.npz"
        arrays = _example_arrays(
            frame,
            item.training_labels,
            coverage_labels=item.coverage_labels,
            future_labels=item.future_labels,
            label_source_type=item.label_source_type,
            pack_version=5,
        )
        arrays.update(_bc_label_arrays(bc_label))
        write_deterministic_npz(example_path, arrays)
        examples.append(
            {
                "example_path": relative_to_root(example_path, output),
                "example_sha256": file_sha256(example_path),
                "sequence_id": frame.sequence_id,
                "camera_id": frame.camera_id,
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "source_name": frame.source_name,
                "source_family": frame.source_family,
                "source_ref": frame.source_ref,
                "supervision_grade": frame.supervision_grade,
                "scenario_name": frame.scenario_name,
                "selected_candidate_id": selected,
                "selected_by_expert_count": 1,
                "candidate_count": len(item.training_labels),
                "label_source_type": LABEL_SOURCE,
                "coverage_expert_selected_candidate_id": _selected_candidate_id(item.coverage_labels),
                "future_motion_primary_candidate_id": bc_label.candidate_id,
                "future_motion_primary_match_error": bc_label.best_distance,
                "bc_label_valid": True,
                "bc_label_confidence": round(float(bc_label.bc_label_confidence), 6),
                "bc_label_margin": round(float(bc_label.margin), 6),
                "source_weight": round(float(frame.source_weight), 6),
                "action_supervision_ok": bool(frame.action_supervision_ok),
                "robot_frame_truth": bool(frame.action_sanity.get("robot_frame_truth", False)),
                "origin_frame_status": frame.action_sanity.get("origin_frame_status"),
                "replay_only": True,
                "not_executed": True,
                "control_safe": False,
                "product_training_approved": False,
            }
        )

    selected_ids = [str(record["selected_candidate_id"]) for record in examples]
    source_ids = [str(record["source_name"]) for record in examples]
    confidence_values = [float(record["bc_label_confidence"]) for record in examples]
    synthetic_agreements = [
        str(record["selected_candidate_id"]) == str(record["coverage_expert_selected_candidate_id"])
        for record in examples
    ]
    manifest: JsonDict = {
        "schema_version": ACTION_LABEL_PACK_SCHEMA_VERSION_V5,
        "example_schema_version": ACTION_LABEL_EXAMPLE_SCHEMA_VERSION,
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "package_type": "ActionLabelPack",
        "version": 5,
        "pack_version": 5,
        "label_source": LABEL_SOURCE,
        "not_synthetic_expert": True,
        "source_dirs": [source.as_posix() for source in source_roots],
        "source_manifest_sha256": [
            file_sha256(source / "manifest.json") if (source / "manifest.json").exists() else "missing"
            for source in source_roots
        ],
        "example_count": len(examples),
        "candidate_count": len(candidates),
        "candidate_ids": candidate_ids,
        "candidate_hash": candidates_hash(candidates),
        "grid_shape": [first_shape[0], first_shape[1]],
        "meters_per_cell": round(float(meters_per_cell), 6),
        "robot_radius_m": round(float(robot_radius_m), 6),
        "source_distribution": _distribution(source_ids),
        "selected_distribution": _distribution(selected_ids),
        "label_source_distribution": {LABEL_SOURCE: len(examples)},
        "source_frame_counts": _source_frame_counts(all_frames, examples, excluded_frames),
        "excluded_frames": excluded_frames,
        "excluded_frame_count": len(excluded_frames),
        "excluded_reason_distribution": _excluded_reason_distribution(excluded_frames),
        "action_sanity_filter": {
            "enabled": True,
            "schema_version": "homebrain.bev_action_sanity.v0",
            "included_by_source": _distribution(source_ids),
            "excluded_by_source": _distribution([str(item["source_name"]) for item in excluded_frames]),
            "excluded_frame_count": len(excluded_frames),
            "policy": "v5 includes only bc_label_valid, action_supervision_ok, robot_frame_truth frames",
        },
        "bc_labeling": {
            "source": LABEL_SOURCE,
            "horizon_s": round(float(horizon_s), 6) if horizon_s is not None else "candidate_max_duration",
            "confidence_floor_for_qa": float(bc_confidence_floor),
            "confidence_mean": float(sum(confidence_values) / max(len(confidence_values), 1)),
            "top3_recorded": True,
            "invalid_reasons": [
                "future_horizon_truncated",
                "pose_missing",
                "stationary_below_threshold",
            ],
        },
        "synthetic_oracle_overlap": {
            "basis": "coverage_expert_selected_candidate_id",
            "overlap_count": len(synthetic_agreements),
            "agreement_fraction": float(sum(1 for value in synthetic_agreements if value) / max(len(synthetic_agreements), 1)),
        },
        "scoring": _scoring_description(),
        "supervision_type": "candidate_id_from_dataset_future_motion_behavior_cloning",
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "product_training_approved": False,
        "raw_pwm_emitted": False,
        "control_safe_claim": False,
        "examples": examples,
    }
    write_json(output / "manifest.json", manifest, pretty=True)
    return output / "manifest.json"


def _pose_sequences_by_key(frames: list[SourceFrame]) -> dict[tuple[str, str, str], list[TimedPose2D]]:
    grouped: dict[tuple[str, str, str], list[TimedPose2D]] = {}
    for frame in frames:
        sample = pose_sample_from_base_pose(
            frame.base_pose,
            timestamp_ns=frame.timestamp_ns,
            frame_id=frame.frame_id,
        )
        if sample is not None:
            grouped.setdefault(_sequence_key(frame), []).append(sample)
    return {key: sorted(values, key=lambda item: item.timestamp_ns) for key, values in grouped.items()}


def _sequence_key(frame: SourceFrame) -> tuple[str, str, str]:
    return (frame.source_name, frame.sequence_id, frame.camera_id)


def _exclusion_reasons(frame: SourceFrame, bc_label: FutureMotionActionLabel) -> list[str]:
    reasons: list[str] = []
    if not bc_label.bc_label_valid:
        reasons.append(str(bc_label.invalid_reason or "bc_label_invalid"))
    if not frame.action_supervision_ok:
        reasons.append("action_supervision_not_ok")
    if not bool(frame.action_sanity.get("robot_frame_truth", False)):
        reasons.append("robot_frame_truth_false")
    return reasons


def _excluded_record(
    frame: SourceFrame,
    bc_label: FutureMotionActionLabel,
    reasons: list[str],
) -> JsonDict:
    return {
        "sequence_id": frame.sequence_id,
        "camera_id": frame.camera_id,
        "frame_id": frame.frame_id,
        "timestamp_ns": frame.timestamp_ns,
        "source_name": frame.source_name,
        "source_family": frame.source_family,
        "source_ref": frame.source_ref,
        "scenario_name": frame.scenario_name,
        "action_supervision_ok": bool(frame.action_supervision_ok),
        "robot_frame_truth": bool(frame.action_sanity.get("robot_frame_truth", False)),
        "bc_label_valid": bool(bc_label.bc_label_valid),
        "bc_label_invalid_reason": bc_label.invalid_reason,
        "exclude_reasons": list(reasons),
    }


def _future_label_from_bc(label: FutureMotionActionLabel) -> FutureMotionLabel:
    return FutureMotionLabel(
        horizon_s=float(label.horizon_s),
        best_candidate_id=label.candidate_id if label.bc_label_valid else "missing",
        match_error=float(label.best_distance),
        target_frame_id=int(label.target_frame_id),
        dx=float(label.dx_m),
        dy=float(label.dy_m),
        dyaw=float(label.dyaw_rad),
        mask=1.0 if label.bc_label_valid else 0.0,
    )


def _bc_label_arrays(label: FutureMotionActionLabel) -> dict[str, np.ndarray]:
    top3 = list(label.top3)
    while len(top3) < 3:
        top3.append(
            CandidateMatchDistance(
                candidate_id="missing",
                total_distance=float("inf"),
                position_l2_m=float("inf"),
                yaw_l1_rad=float("inf"),
            )
        )
    return {
        "bc_label_valid": scalar_bool(label.bc_label_valid),
        "bc_label_invalid_reason": scalar_str(label.invalid_reason or ""),
        "bc_label_candidate_id": scalar_str(label.candidate_id),
        "bc_label_confidence": scalar_float(label.bc_label_confidence),
        "bc_label_margin": scalar_float(label.margin),
        "bc_label_best_distance": scalar_float(label.best_distance),
        "bc_label_second_best_distance": scalar_float(label.second_best_distance),
        "bc_top3_candidate_ids": np.asarray([item.candidate_id for item in top3[:3]]),
        "bc_top3_total_distance": np.asarray([item.total_distance for item in top3[:3]], dtype=np.float32),
        "bc_top3_position_l2_m": np.asarray([item.position_l2_m for item in top3[:3]], dtype=np.float32),
        "bc_top3_yaw_l1_rad": np.asarray([item.yaw_l1_rad for item in top3[:3]], dtype=np.float32),
        "not_synthetic_expert": scalar_bool(True),
    }


def _selected_candidate_id(labels: object) -> str:
    selected = [label.candidate_id for label in labels if label.selected_by_expert]  # type: ignore[attr-defined]
    if len(selected) != 1:
        raise ValueError("expected exactly one selected expert label")
    return str(selected[0])


def _source_frame_counts(
    all_frames: list[SourceFrame],
    included_examples: list[JsonDict],
    excluded_frames: list[JsonDict],
) -> JsonDict:
    counts: dict[str, JsonDict] = {}
    for frame in all_frames:
        record = counts.setdefault(frame.source_name, {"included": 0, "excluded": 0, "total": 0})
        record["total"] = int(record["total"]) + 1
    for record in included_examples:
        source = str(record["source_name"])
        counts.setdefault(source, {"included": 0, "excluded": 0, "total": 0})
        counts[source]["included"] = int(counts[source]["included"]) + 1
    for record in excluded_frames:
        source = str(record["source_name"])
        counts.setdefault(source, {"included": 0, "excluded": 0, "total": 0})
        counts[source]["excluded"] = int(counts[source]["excluded"]) + 1
    return dict(sorted(counts.items()))


def _excluded_reason_distribution(excluded_frames: list[JsonDict]) -> JsonDict:
    reasons: list[str] = []
    for record in excluded_frames:
        raw = record.get("exclude_reasons")
        if isinstance(raw, list):
            reasons.extend(str(item) for item in raw)
    return _distribution(reasons)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build ActionLabelPack v5 from dataset future-motion behavior cloning.")
    parser.add_argument("--source", action="append", required=True, help="Input robot-frame SpatialTrainPack.")
    parser.add_argument("--out", required=True, help="Output ActionLabelPack v5 directory.")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--horizon-s", type=float, default=None)
    parser.add_argument("--bc-confidence-floor", type=float, default=0.05)
    args = parser.parse_args(argv)
    manifest = build_action_label_pack_v5(
        sources=[Path(source) for source in args.source],
        out_dir=args.out,
        max_examples=args.max_examples,
        horizon_s=args.horizon_s,
        bc_confidence_floor=args.bc_confidence_floor,
    )
    print(json.dumps({"manifest": manifest.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
