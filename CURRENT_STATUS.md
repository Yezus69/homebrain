# CURRENT_STATUS.md

Keep this file short. It should explain where the repo is now, not preserve old
command history.

## Current State

HomeBrain is a deterministic replay/eval and offline-teacher stack for a
software-only floor-cleaning robot brain POC.

Implemented:

- typed event schemas and deterministic segment logs;
- image/video and public RGB-D route ingestion;
- replay/model/eval CLIs;
- teacher artifacts for DINO, DA3, Depth Pro-style depth, MoGe/VGGT-style scene
  geometry, plus fake backends for tests;
- geometry-to-BEV and SpatialTrainPack builders;
- SpatialMemoryNet v0/v1 with explicit local BEV memory;
- fixed candidate trajectories, transparent scorer, learned scorer, future
  motion action labels, and closed-loop replay reports.
- Future BEV Rollout v1 pack builder, dataset, model, train/eval CLIs, and
  replay-only runtime candidate scoring option.

Not implemented:

- real robot runtime process;
- synchronized owned RGB/IR/IMU/wheel/command logs;
- hardware controller, watchdog, recovery, docking, or physical safety gate;
- robust dynamic-risk labels;
- strong free-space traversability labels;
- global home-scale coverage memory;
- a policy good enough to drive a real cleaning robot.

## Current Policy

Open-weight models and public datasets such as OpenLORIS are allowed for local
POC training/eval when provenance is recorded. This no longer blocks useful
experiments. It still does not imply product approval, redistribution rights, or
control safety.

Replay/control safety remains:

```text
replay_only=true
not_executed=true
control_safe=false
raw_pwm_emitted=false
```

## Latest Known Model/Policy Result

Goal 22A best checkpoint:

```text
runs/goal22a_overnight_runtime_scorer_repair/train/model_current_all_features_seed21_pair_constrained_search2/checkpoint.pt
```

Observed result: non-collapsed closed-loop replay on cafe/office/corridor, but
still narrow. It mainly chooses `straight_medium` and `arc_right_small`.

Useful metrics from the last accepted report:

```text
learned_logs_collapsed = 0/6
max_dominant_fraction = 0.73053152039555
mean_entropy = 0.9406909534104936
weighted_v5_agreement = 0.25760193503800966
cmd_vel_non_null_count = 0
```

## Last Completed Goal

Future BEV Rollout v1: self-supervised future local spatial-state and
candidate-outcome pretraining from SpatialTrainPack-style RGB-D/pose route data.

## Future BEV Rollout v1 Proof

Objective attempted: add a real replay-only future BEV rollout block that warps
future BEV labels into the current robot frame with route pose, derives
candidate collision/unknown/new-area/progress labels, trains a small rollout
model, evaluates future/candidate metrics, and can shadow-score runtime replay
candidates without emitting commands.

Files changed: added `homebrain/train/future_bev_rollout_dataset.py`,
`homebrain/train/build_future_bev_rollout_pack.py`,
`homebrain/brain/future_bev_rollout_v1.py`,
`homebrain/train/train_future_bev_rollout_v1.py`,
`homebrain/eval/eval_future_bev_rollout_v1.py`, and
`tests/test_future_bev_rollout_v1.py`; updated `modeld`, `replayd`,
runtime trajectory decisions, checkpoint constants, and this status file.

Commands run: required docs were read; repo files and candidate/replay modules
were inspected with `rg`/PowerShell; `python -m py_compile` on new/modified
rollout, runtime, modeld, replayd, and test modules; `python -m pytest
tests\test_future_bev_rollout_v1.py -q`; `python -m pytest
tests\test_goal11a_trajectory_scorer_v0.py tests\test_goal12a_spatial_memory_v1.py -q`;
full `python -m pytest -q`; `git diff --check`; `git status --short`.

Pass/fail results: py_compile passed; focused Future BEV Rollout tests passed
with `5 passed`; related replay/modeld scorer tests passed with `13 passed`;
full pytest passed with `125 passed in 185.34s`; `git diff --check` passed.
PowerShell tool output still appends a non-project `-Command` warning after
commands.

Artifacts created: no persistent real-data artifact was created. The new CLIs
write deterministic `future_bev_rollout_pack` manifests/examples, checkpoints,
and eval metrics when run on local SpatialTrainPack/public route data; tests use
temporary tiny fixtures only.

What this enables next: train on available OpenLORIS/TUM-style public indoor
routes to pretrain "what will happen if I take this candidate" before hardware,
then fine-tune the same future/candidate heads on real robot logs later.

Remaining risks: current proof is tiny-fixture smoke plus unit coverage, not a
real public-route result yet; labels remain weak, replay-only, and pose/BEV
quality-bound; candidate outcomes are not control safety evidence and runtime
still reports `cmd_vel=None`, `replay_only=true`, `not_executed=true`,
`control_safe=false`.

Recommended next goal: build a real FutureBEVRolloutPack from local public route
packs, train/evaluate route-held-out metrics, and compare shadow candidate
diversity against the current learned trajectory scorer.
