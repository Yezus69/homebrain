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

## Latest Accepted Runtime Policy Result

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

Goal23A: real-route Future BEV Rollout v1 training, diagnosis, and repair.

## Goal23A Result

Objective attempted: make Future BEV Rollout v1 produce a meaningful route-held-out
result on existing OpenLORIS public route packs without adding a new architecture
or promoting unsafe policy behavior.

Files changed: hardened `homebrain/train/build_future_bev_rollout_pack.py`,
`homebrain/train/future_bev_rollout_dataset.py`,
`homebrain/brain/future_bev_rollout_v1.py`,
`homebrain/train/train_future_bev_rollout_v1.py`,
`homebrain/eval/eval_future_bev_rollout_v1.py`,
`homebrain/policies/runtime_decision.py`, checkpoint constants, and
`tests/test_future_bev_rollout_v1.py`; updated this status file.

Commands run: required docs were read; built a pre-repair source-order pack;
built repaired route-balanced all-route, cafe+office train, and held-out
corridor packs using DINO features; trained cafe+office and all-route rollout
checkpoints; evaluated cafe+office val, held-out corridor `--split all`, and
all-route val; `python -m py_compile` on touched rollout/runtime modules;
`python -m pytest tests\test_future_bev_rollout_v1.py -q`; related trajectory,
memory, closed-loop, collapse, and runtime scorer tests; full `python -m pytest
-q`.

Pass/fail results: focused rollout tests passed with `7 passed`; related tests
passed with `23 passed`; full pytest passed with `127 passed in 193.85s`.
PowerShell tool output still appends a non-project `-Command` warning after
commands.

Artifacts created:

```text
runs/goal23a_future_bev_rollout_route_heldout/GOAL23A_REPORT.md
runs/goal23a_future_bev_rollout_route_heldout/after_route_balanced_pack
runs/goal23a_future_bev_rollout_route_heldout/train_cafe_office_seed23/checkpoint.pt
runs/goal23a_future_bev_rollout_route_heldout/eval_heldout_corridor_all.json
runs/goal23a_future_bev_rollout_route_heldout/train_all_routes_seed23/checkpoint.pt
runs/goal23a_future_bev_rollout_route_heldout/eval_all_routes_val.json
```

Held-out corridor result: future free/occupied/unknown IoU/proxy
`0.2708681769803862 / 0.37841249065071053 / 0.9333786353592012`, collision AUROC
`0.7643962934968226`, action entropy `0.6904558764306521`, dominant action
fraction `0.5366666666666666`, selected distribution
`{"rotate_left": 161, "rotate_right": 139}`. It beats simple scalar/BEV
baselines but does not beat Goal22A as a useful replay policy.

Blunt outcome: do not integrate the Future BEV Rollout scorer as the default
runtime scorer yet. Keep it as a trained checkpoint plus diagnosis. It repaired
the data/label/eval path and produced nonzero route-held-out rollout metrics,
but replay decisions are rotate-heavy and candidate oracle agreement is low.

Remaining risks: labels remain weak, replay-only, and pose/BEV quality-bound;
candidate outcomes are not control safety evidence; runtime still reports
`cmd_vel=None`, `replay_only=true`, `not_executed=true`, `control_safe=false`.

Recommended next goal: repair candidate score alignment within the existing
rollout heads, using a stable ranking/objective that does not collapse before
running replay comparison against Goal22A.
