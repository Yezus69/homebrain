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
- action-conditioned FutureBEV/future-risk replay candidate scoring,
- deterministic Goal29 dynamic-risk fixture pack and report path,
- replay-only safety envelope stop/recovery reasons,
- route-heldout replay reports and visual artifacts.

It does not control real hardware.

## Latest accepted baseline

Latest accepted baseline:

```text
Goal29 action-conditioned fixture slice
```

Accepted evidence:

```text
accepted_goal29_action_conditioned_fixture=true
fixtures=moving_obstacle_crossing_path,static_obstacle_in_known_free_space,unknown_corridor,high_uncertainty_near_footprint,future_unsafe_candidate
gates_improved=Gate F,Gate H
future_risk_auc_or_proxy=0.8235512979706741
candidate_risk_ranking_accuracy=0.7371428571428572
candidate_risk_ranking_accuracy_baseline_delta=0.13714285714285723
learned_selection_oracle_match_fraction=1.0
learned_selection_unsafe_selected_rate=0.0
dominant_action_fraction=0.6
selected_candidate_entropy=0.9502705392332347
route_heldout_goal29_available=false
raw_pwm_emitted=false
control_safe=false
hardware_validated=false
```

Primary artifact:

```text
runs/goal29_action_conditioned_fixture_v0/goal29_report.json
```

Goal29 is replay-only deterministic fixture evidence. It is not route-heldout success, product deployment, or hardware validation.

## Current blockers

HomeBrain is still blocked by:

- no real robot hardware loop,
- no synchronized owned RGB/IR/IMU/wheel/command logs,
- no real controller/watchdog/recovery/docking/safety gate,
- dynamic-risk supervision only proven on deterministic fixtures,
- weak traversability/free-space labels,
- limited public-route diversity,
- narrow route-heldout robustness,
- no policy proven good enough for physical deployment.

## Current next target

Next target:

```text
Route-heldout Goal29 hardening after the accepted deterministic fixture slice
```

The next step should preserve the accepted Goal29 fixture path while improving at least one of:

- route-heldout dynamic-risk supervision,
- traversability/free-space labels,
- action-conditioned future BEV/risk prediction,
- learned candidate trajectory scoring,
- route-heldout robustness,
- replay watchdog/stop/recovery behavior,
- direct runtime student quality.

Do not count deterministic fixtures as route-heldout robustness. The accepted fixture report improves Gate F/H, but Gate E still needs stronger copy-forward-beating evidence on route or richer dynamic-risk labels.

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
