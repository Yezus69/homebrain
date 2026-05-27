from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from homebrain.data.spatial_dataset import read_json, write_json
from homebrain.eval.eval_direct_bev_student_v0 import runtime_field_leakage_passed
from homebrain.brain.modeld import SceneState
from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.policies.runtime_decision import decide_trajectory, load_runtime_future_rollout_scorer
from homebrain.policies.trajectory_scorer import CoverageMemory, LocalBev, score_trajectories
from homebrain.runtime.direct_bev_runtime import load_runtime_direct_bev_student, predict_local_bev_from_rgbd
from homebrain.train.train_direct_bev_student_v0 import RealRGBDRouteBEVDataset


def eval_real_rgbd_brain_slice(
    *,
    checkpoint: str | Path,
    pack_dir: str | Path,
    out_path: str | Path,
    split: str = "val",
    future_rollout_checkpoint: str | Path | None = None,
    device_name: str | None = None,
) -> Path:
    manifest = read_json(Path(pack_dir) / "manifest.json")
    runtime = load_runtime_direct_bev_student(checkpoint, device=device_name)
    dataset = RealRGBDRouteBEVDataset(pack_dir, split=split, image_size=tuple(runtime.model.config.image_size))
    grid_shape = tuple(int(value) for value in manifest.get("grid_shape", runtime.model.config.bev_shape))
    meters_per_cell = float(manifest.get("meters_per_cell", 0.05))
    candidates = generate_default_candidates(grid_shape=grid_shape, meters_per_cell=meters_per_cell, robot_radius_m=0.18)
    coverage = CoverageMemory(grid_shape, meters_per_cell=meters_per_cell)
    scene_state = SceneState.create(
        local_shape=grid_shape,
        meters_per_cell=meters_per_cell,
        pose_estimate=(0.0, 0.0, 0.0),
    )
    pose_estimate = (0.0, 0.0, 0.0)
    future = (
        load_runtime_future_rollout_scorer(future_rollout_checkpoint, device=runtime.device)
        if future_rollout_checkpoint is not None
        else None
    )
    selected = Counter()
    changed_cases: list[dict[str, object]] = []
    teacher_agree = 0
    high_risk_selected = 0
    risk_changed = 0
    for index in range(len(dataset)):
        item = dataset[index]
        prediction = predict_local_bev_from_rgbd(
            runtime,
            item["rgb"],
            depth=item["depth"],
            pose_delta_prev=item["pose_delta_prev"].numpy(),
            previous_action=item["previous_action"].numpy(),
            sensor_mask=item["sensor_mask"].numpy(),
        )
        pose_delta = tuple(float(value) for value in item["pose_delta_prev"].numpy())
        pose_estimate = _integrate_pose(pose_estimate, pose_delta)
        predicted_stack = _stack_local_bev(prediction["local_bev"])
        scene_context = scene_state.update_from_local(
            pose_estimate=pose_estimate,
            current_bev=predicted_stack,
            memory_bev=predicted_stack,
            uncertainty=prediction["local_bev"].uncertainty
            if prediction["local_bev"].uncertainty is not None
            else prediction["local_bev"].unknown,
        )
        policy_bev = LocalBev(
            free=scene_context[0],
            occupied=scene_context[1],
            unknown=scene_context[2],
            traversable=scene_context[3],
            risky=scene_context[4],
            uncertainty=prediction["local_bev"].uncertainty,
            confidence=prediction["local_bev"].confidence,
            source="direct_bev_student_v0_scene_state",
        )
        decision = decide_trajectory(
            bev=policy_bev,
            candidates=candidates,
            coverage_memory=coverage,
            pose_delta=pose_delta,
            future_rollout_scorer=future,
            sensor_mask=item["sensor_mask"].numpy(),
            memory_bev=scene_context,
            bev_history=scene_state.history_for_rollout(int(future.model.config.history_steps)) if future is not None else None,
            uncertainty_map=policy_bev.uncertainty,
            policy_bev_source="direct_bev_student_v0_scene_state",
        )
        scene_state.record_decision(
            selected_trajectory_id=decision.selected_candidate_id,
            local_bev_ref=None,
            policy_bev_source="direct_bev_student_v0_scene_state",
            candidates=candidates,
            future_arrays=decision.artifact_arrays,
            step_index=index,
        )
        selected[decision.selected_candidate_id] += 1
        teacher_bev = _teacher_local_bev(item)
        teacher_decision = score_trajectories(bev=teacher_bev, candidates=candidates)
        if teacher_decision.selected_candidate_id == decision.selected_candidate_id:
            teacher_agree += 1
        selected_risk = _candidate_risk(teacher_bev, candidates[decision.selected_candidate_index], item["dynamic"].numpy()[0])
        if selected_risk >= 0.5:
            high_risk_selected += 1
        no_risk_bev = LocalBev(
            free=policy_bev.free,
            occupied=policy_bev.occupied,
            unknown=policy_bev.unknown,
            traversable=policy_bev.traversable,
            risky=np.zeros_like(prediction["local_bev"].occupied, dtype=np.float32),
            uncertainty=policy_bev.uncertainty,
            source="direct_bev_student_v0_no_risk_ablation",
        )
        no_risk_decision = score_trajectories(bev=no_risk_bev, candidates=candidates)
        if no_risk_decision.selected_candidate_id != decision.selected_candidate_id:
            risk_changed += 1
            changed_cases.append(
                {
                    "index": int(index),
                    "route_id": str(item["route_id"]),
                    "without_direct_risk": no_risk_decision.selected_candidate_id,
                    "with_direct_risk": decision.selected_candidate_id,
                }
            )

    count = max(len(dataset), 1)
    report = {
        "schema_version": "homebrain.eval_real_rgbd_brain_slice.v0",
        "checkpoint": Path(checkpoint).as_posix(),
        "pack": Path(pack_dir).as_posix(),
        "split": split,
        "selected_candidate_id_distribution": dict(sorted(selected.items())),
        "stop_rate": float(selected.get("stop", 0) / count),
        "high_risk_selected_rate_using_teacher_labels_eval_only": float(high_risk_selected / count),
        "agreement_with_teacher_bev_transparent_scorer": float(teacher_agree / count),
        "direct_bev_risk_changed_candidate_choice_count": int(risk_changed),
        "direct_bev_risk_changed_candidate_choice_cases": changed_cases[:50],
        "no_runtime_leakage": runtime_field_leakage_passed(runtime.metadata),
        "scene_state_online": True,
        "scene_context_consumed_by_policy": True,
        "scene_state_debug": scene_state.to_debug(),
        "teacher_labels_used_after_runtime_decision_only": True,
        "replay_only": True,
        "not_executed": True,
        "control_safe": False,
        "raw_pwm_emitted": False,
        "hardware_validated": False,
    }
    write_json(out_path, report, pretty=True)
    return Path(out_path)


def _teacher_local_bev(item: dict[str, object]) -> LocalBev:
    targets = item["targets"].numpy().astype(np.float32)  # type: ignore[index,union-attr]
    uncertainty = item["uncertainty"].numpy()[0].astype(np.float32)  # type: ignore[index,union-attr]
    dynamic = item["dynamic"].numpy()[0].astype(np.float32)  # type: ignore[index,union-attr]
    return LocalBev(
        free=targets[0],
        occupied=targets[1],
        unknown=targets[2],
        traversable=targets[3],
        risky=np.maximum(targets[4], dynamic).astype(np.float32),
        uncertainty=uncertainty,
        confidence=(1.0 - uncertainty).astype(np.float32),
        source="teacher_bev_eval_only_after_decision",
    )


def _candidate_risk(bev: LocalBev, candidate: object, dynamic: np.ndarray) -> float:
    risky = np.maximum(bev.risky if bev.risky is not None else bev.occupied, dynamic)
    values = []
    for row, col in getattr(candidate, "footprint_cells", ()):
        if 0 <= row < risky.shape[0] and 0 <= col < risky.shape[1]:
            values.append(float(risky[row, col]))
    return float(max(values)) if values else 1.0


def _stack_local_bev(bev: LocalBev) -> np.ndarray:
    return np.stack(
        [
            bev.free,
            bev.occupied,
            bev.unknown,
            bev.traversable if bev.traversable is not None else bev.free,
            bev.risky if bev.risky is not None else bev.occupied,
        ],
        axis=0,
    ).astype(np.float32)


def _integrate_pose(
    pose: tuple[float, float, float],
    delta: tuple[float, float, float],
) -> tuple[float, float, float]:
    x_m, y_m, yaw = pose
    dx, dy, dyaw = delta
    x_m += float(np.cos(yaw) * dx - np.sin(yaw) * dy)
    y_m += float(np.sin(yaw) * dx + np.cos(yaw) * dy)
    yaw = float(yaw + dyaw)
    while yaw > np.pi:
        yaw -= 2.0 * float(np.pi)
    while yaw < -np.pi:
        yaw += 2.0 * float(np.pi)
    return (float(x_m), float(y_m), float(yaw))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a replay-only real RGB-D brain slice: DirectBEVStudentV0 -> LocalBev -> candidate decision."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    parser.add_argument("--future-rollout-checkpoint", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    eval_real_rgbd_brain_slice(
        checkpoint=args.checkpoint,
        pack_dir=args.pack,
        split=args.split,
        out_path=args.out,
        future_rollout_checkpoint=args.future_rollout_checkpoint,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
