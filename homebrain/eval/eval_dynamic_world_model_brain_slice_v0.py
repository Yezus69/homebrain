from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from homebrain.brain.modeld import SceneState
from homebrain.data.spatial_dataset import write_deterministic_npz, write_json
from homebrain.policies.candidate_trajectories import generate_default_candidates
from homebrain.policies.trajectory_scorer import CoverageMemory, LocalBev, score_trajectories
from homebrain.runtime.counterfactual_world_model_scorer import (
    RUNTIME_SAFETY_DEBUG,
    load_counterfactual_world_model,
    score_candidates_with_world_model,
)
from homebrain.train.train_counterfactual_dynamic_bev_world_model_v0 import DynamicBEVWorldDataset


def eval_dynamic_world_model_brain_slice_v0(
    *,
    checkpoint: str | Path,
    pack: str | Path,
    out_path: str | Path,
    split: str = "val",
    device_name: str | None = None,
) -> Path:
    runtime = load_counterfactual_world_model(checkpoint, device=device_name)
    dataset = DynamicBEVWorldDataset(pack, split=split)
    candidates = generate_default_candidates(
        grid_shape=dataset.grid_shape,
        meters_per_cell=float(dataset.manifest.get("meters_per_cell", 0.05)),
        robot_radius_m=float(dataset.manifest.get("robot_radius_m", 0.18)),
    )
    scene_state = SceneState.create(
        local_shape=dataset.grid_shape,
        meters_per_cell=float(dataset.manifest.get("meters_per_cell", 0.05)),
        pose_estimate=(0.0, 0.0, 0.0),
    )
    coverage = CoverageMemory(dataset.grid_shape, meters_per_cell=float(dataset.manifest.get("meters_per_cell", 0.05)))
    selected = Counter()
    old_selected = Counter()
    high_risk_selected = 0
    unknown_exposure_sum = 0.0
    changed = 0
    prevented: list[dict[str, object]] = []
    worse: list[dict[str, object]] = []
    selected_indices: list[int] = []
    old_indices: list[int] = []
    pose = (0.0, 0.0, 0.0)
    for index in range(len(dataset)):
        sample = dataset[index]
        history = sample["bev_history"].numpy().astype(np.float32)
        current = history[-1]
        local = _local_bev(current)
        pose_delta = tuple(float(value) for value in sample["pose_delta_history"][-1].numpy()[:3])
        pose = _integrate_pose(pose, pose_delta)
        scene_context = scene_state.update_from_local(
            pose_estimate=pose,
            current_bev=current[:5],
            memory_bev=current[:5],
            uncertainty=current[6],
        )
        world = score_candidates_with_world_model(
            runtime,
            local_bev_history=history,
            pose_delta_history=sample["pose_delta_history"].numpy(),
            previous_action_history=sample["previous_action_history"].numpy(),
            sensor_mask=sample["sensor_mask_history"].numpy(),
            candidates=candidates,
            scene_state=scene_state,
        )
        old_decision = score_trajectories(bev=_local_bev(np.concatenate([scene_context, current[5:7]], axis=0)), candidates=candidates, coverage_memory=coverage)
        selected_id = str(world["selected_candidate_id"])
        selected_idx = int(world["selected_candidate_index"])
        old_idx = _candidate_index(candidates, old_decision.selected_candidate_id)
        selected[selected_id] += 1
        old_selected[old_decision.selected_candidate_id] += 1
        selected_indices.append(selected_idx)
        old_indices.append(old_idx)
        risk = sample["candidate_total_teacher_risk"].numpy().astype(np.float32)
        unknown = sample["candidate_unknown_exposure"].numpy().astype(np.float32).max(axis=1)
        if float(risk[selected_idx]) >= 0.5:
            high_risk_selected += 1
        unknown_exposure_sum += float(unknown[selected_idx])
        if selected_idx != old_idx:
            changed += 1
            case = {
                "index": int(index),
                "old_candidate_id": old_decision.selected_candidate_id,
                "world_model_candidate_id": selected_id,
                "old_teacher_risk": round(float(risk[old_idx]), 6),
                "world_teacher_risk": round(float(risk[selected_idx]), 6),
            }
            if float(risk[old_idx]) >= 0.5 and float(risk[selected_idx]) < 0.5:
                prevented.append(case)
            if float(risk[old_idx]) < 0.5 and float(risk[selected_idx]) >= 0.5:
                worse.append(case)
        scene_state.record_decision(
            selected_trajectory_id=selected_id,
            local_bev_ref=None,
            policy_bev_source="counterfactual_dynamic_world_model_scene_state",
            candidates=candidates,
            future_arrays={},
            step_index=index,
        )
    count = max(len(dataset), 1)
    artifact_path = Path(out_path).with_suffix(".selected_candidates.npz")
    write_deterministic_npz(
        artifact_path,
        {
            "selected_candidate_index": np.asarray(selected_indices, dtype=np.int64),
            "old_selected_candidate_index": np.asarray(old_indices, dtype=np.int64),
            "replay_only": np.asarray([True], dtype=np.bool_),
            "not_executed": np.asarray([True], dtype=np.bool_),
            "control_safe": np.asarray([False], dtype=np.bool_),
            "raw_pwm_emitted": np.asarray([False], dtype=np.bool_),
            "hardware_validated": np.asarray([False], dtype=np.bool_),
        },
    )
    report = {
        "schema_version": "homebrain.eval_dynamic_world_model_brain_slice_v0.v0",
        "checkpoint": Path(checkpoint).as_posix(),
        "pack": Path(pack).as_posix(),
        "split": split,
        "selected_candidate_id_distribution": dict(sorted(selected.items())),
        "old_scorer_selected_candidate_id_distribution": dict(sorted(old_selected.items())),
        "stop_rate": float(selected.get("stop", 0) / count),
        "high_risk_selected_rate": float(high_risk_selected / count),
        "unknown_exposure_selected_mean": float(unknown_exposure_sum / count),
        "candidate_change_rate_vs_old_scorer": float(changed / count),
        "cases_where_world_model_prevented_risky_selection": prevented[:50],
        "cases_where_world_model_made_selection_worse": worse[:50],
        "selected_candidate_artifact": artifact_path.as_posix(),
        "scene_state_online": True,
        "scene_state_debug": scene_state.to_debug(),
        "no_runtime_leakage": True,
        **RUNTIME_SAFETY_DEBUG,
    }
    write_json(out_path, report, pretty=True)
    return Path(out_path)


def _local_bev(stack: np.ndarray) -> LocalBev:
    return LocalBev(
        free=stack[0],
        occupied=stack[1],
        unknown=stack[2],
        traversable=stack[3],
        risky=np.maximum(stack[4], stack[5]) if stack.shape[0] > 5 else stack[4],
        uncertainty=stack[6] if stack.shape[0] > 6 else stack[2],
        confidence=1.0 - (stack[6] if stack.shape[0] > 6 else stack[2]),
        source="counterfactual_dynamic_world_model_brain_slice",
    )


def _candidate_index(candidates: list[object], candidate_id: str) -> int:
    for index, candidate in enumerate(candidates):
        if str(getattr(candidate, "id", "")) == candidate_id:
            return index
    return 0


def _integrate_pose(pose: tuple[float, float, float], delta: tuple[float, float, float]) -> tuple[float, float, float]:
    x_m, y_m, yaw = pose
    dx, dy, dyaw = delta
    x_m += float(np.cos(yaw) * dx - np.sin(yaw) * dy)
    y_m += float(np.sin(yaw) * dx + np.cos(yaw) * dy)
    yaw = float(yaw + dyaw)
    return (float(x_m), float(y_m), float(yaw))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run replay-only SceneState -> Counterfactual World Model brain slice.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    eval_dynamic_world_model_brain_slice_v0(
        checkpoint=args.checkpoint,
        pack=args.pack,
        split=args.split,
        out_path=args.out,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
