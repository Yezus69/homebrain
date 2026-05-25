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

Goal25: real public OpenLORIS route-heldout replay-as-live milestone.

## Goal25 Result

Objective: build one reproducible command that stages three OpenLORIS routes
from different scenes, imports robot-frame RGB-D, builds BEV labels, QA-checks
packs, extracts real DINO features, trains SpatialMemoryNetV1 leave-one-route-out
folds on the two RTX 4090s, rejects degenerate FutureBEV action-label groups,
trains FutureBEVRolloutV1, and evaluates heldout corridor replay without future
labels or route-pose leakage.

Command:

```text
python -m homebrain.tools.run_openloris_route_heldout_milestone --out runs\goal25_openloris_route_heldout_milestone --sequences cafe1-1_2,corridor1-1,office1-1_7 --heldout-sequence corridor1-1 --max-frames 96 --spatial-steps 30 --future-steps 30 --runtime-max-frames 48 --max-spatial-folds 2
```

Artifacts:

```text
runs/goal25_openloris_route_heldout_milestone/milestone_report.json
runs/goal25_openloris_route_heldout_milestone/train/spatial_v1_except_corridor1_1/checkpoint.pt
runs/goal25_openloris_route_heldout_milestone/train/spatial_v1_except_cafe1_1_2/checkpoint.pt
runs/goal25_openloris_route_heldout_milestone/train/future_bev_rollout_except_corridor1_1/checkpoint.pt
runs/goal25_openloris_route_heldout_milestone/runtime/heldout_corridor1_1_dino/runtime_report.json
runs/goal25_openloris_route_heldout_milestone/runtime/heldout_corridor1_1_direct_rgbd_slice/runtime_report.json
runs/goal25_openloris_route_heldout_milestone/contact_sheets/heldout_corridor1_1_dino_failure_contact_sheet.ppm
```

Key metrics: heldout corridor SpatialMemoryNetV1 eval current/fused IoU proxy
`0.18941278009276305 / 0.7033660123745601`, memory warp valid fraction `0.75`.
FutureBEV heldout corridor eval future unknown IoU proxy `0.9318639982272857`,
new-area ranking quality `0.6628571428571429`, but action entropy `0.0` and
dominant action fraction `1.0`. Runtime replay produced 96 decisions, p50/p95
latency `28.2573 / 31.700175 ms`, unsafe selected rate `0.0`, pose warp valid
fraction `0.9895833333333334`, and route-pose leakage fraction `0.0`. The
direct RGB-D runtime slice produced 48 decisions without precomputed DINO at
runtime and p95 latency `56.00589 ms`.

Failures and risks: all three route packs remain low-quality/quarantined.
Corridor and office robot-frame BEV QA fail low observed-cell density
(`observed_cell_ratio_mean_below_0.05`). FutureBEV labels passed only after
explicitly rejecting 108 train-pack examples and 12 heldout examples from
degenerate route/split groups. Future rollout and runtime selected-candidate
audits show collapse (`entropy=0.0`, dominant fraction `1.0`). This is not
control-safe and not product-approved.

Verification:

```text
python -m py_compile homebrain\brain\modeld.py homebrain\datasets\openloris_scene.py homebrain\datasets\openloris_to_route.py homebrain\eval\closed_loop_replay_report.py homebrain\runtime\replay_openloris_brain.py homebrain\train\build_future_bev_rollout_pack.py homebrain\tools\run_openloris_route_heldout_milestone.py
python -m pytest tests\test_goal12a_spatial_memory_v1.py tests\test_future_bev_rollout_v1.py tests\test_goal20a_closed_loop_replay.py -q
git diff --check
python -m pytest -q
```

Pass/fail: focused tests passed with `22 passed`; full pytest passed with
`128 passed in 188.08s`. `git diff --check` passed, with only line-ending
warnings from Git on Windows. The shell tool still appends a non-project
PowerShell `-Command` warning after commands.

Recommended next step: collect or select routes with higher observed BEV
density and train a non-collapsed FutureBEV/action selector before any hardware
claim.
