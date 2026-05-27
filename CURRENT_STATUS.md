This file is intentionally small and replaceable.

Do not append long history here. After each meaningful PR or milestone, replace this file with the latest truthful state. Git history is the history.

## Current state

HomeBrain is a software-only, replay-first, mostly-neural robot-brain stack for a future low-speed indoor vacuum/mop robot.

It currently has:

- deterministic route/event/replay infrastructure,
- public robot/video data ingestion paths,
- offline open-weight teacher artifact paths,
- BEV/spatial pack builders,
- online `Brain.step(...)` style runtime replay,
- online `SceneState` and BEV memory,
- direct RGB-D runtime student feature path,
- fixed candidate trajectories,
- FutureBEV-based replay candidate scoring,
- route-heldout replay reports and visual artifacts.

It does not control real hardware.

## Latest accepted baseline

Latest accepted baseline:

```text
Goal28 caveat repair
```

Accepted evidence:

```text
accepted_goal28_scene_brain=true
accepted_goal28_caveat_repair=true
runtime_routes=3
runtime_scenes=corridor,office,market
runtime_policy_source=future_rollout
future_rollout_selection_mode=safe_argmin
student_feature_source=direct_rgbd
runtime_feature_source=direct_rgbd
v1_policy_bev_source=scene
pose_warp_source=odom_plus_visual_correction
teacher_runtime_dependency=false
future_or_groundtruth_runtime_dependency=false
route_pose_leakage_ablation_fraction=0.0
raw_pwm_emitted=false
control_safe=false
```

Primary artifact:

```text
runs/goal28_caveat_repair_three_scene_market_v1/milestone_report.json
```

Goal28 is replay-only public-data evidence. It is not product deployment and not hardware validation.

## Current blockers

HomeBrain is still blocked by:

- no real robot hardware loop,
- no synchronized owned RGB/IR/IMU/wheel/command logs,
- no real controller/watchdog/recovery/docking/safety gate,
- weak dynamic-risk supervision,
- weak traversability/free-space labels,
- limited public-route diversity,
- narrow route-heldout robustness,
- no policy proven good enough for physical deployment.

## Current next target

Next target:

```text
Goal29: action-conditioned dynamic-risk world model and replay hardware-safety envelope
```

Goal29 should preserve Goal28’s online `Brain.step(...)` and `SceneState` path while improving at least one of:

- dynamic-risk supervision,
- traversability/free-space labels,
- action-conditioned future BEV/risk prediction,
- learned candidate trajectory scoring,
- route-heldout robustness,
- replay watchdog/stop/recovery behavior,
- direct runtime student quality.

Goal29 should not be another wrapper milestone. It should improve measured robot-brain behavior under the gates in `EVALS.md`.

## Required safety status

Current outputs remain replay artifacts only:

```text
replay_only=true
not_executed=true
control_safe=false
raw_pwm_emitted=false
hardware_validated=false
```

Do not claim real robot readiness until there are physical robot logs, watchdogs, recovery behavior, safety tests, and maintainer approval.