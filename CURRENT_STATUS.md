# CURRENT_STATUS.md

Keep this file short. It should explain where the repo is now, not preserve old
command history.

## Current State

HomeBrain is a deterministic replay/eval, offline-teacher, and early runtime
stack for a software-only but production-directed floor-cleaning robot brain
POC.

Implemented:

- typed event schemas and deterministic segment logs;
- image/video and public RGB-D route ingestion;
- replay/model/eval CLIs;
- teacher artifacts for DINO, DA3, Depth Pro-style depth, MoGe/VGGT-style scene
  geometry, plus fake backends for tests;
- geometry-to-BEV and SpatialTrainPack builders;
- SpatialMemoryNet v0/v1 with explicit local BEV memory;
- online-style `Brain.step(...)` for SpatialMemoryNetV1 replay ticks, owning
  persistent memory, coverage memory, pose estimate, and recent action history;
- fixed candidate trajectories, transparent scorer, learned scorer, future
  motion action labels, and closed-loop replay reports;
- Future BEV Rollout v1 pack builder, dataset, model, train/eval CLIs, and
  replay-only runtime candidate scoring option;
- scene-level runtime memory artifacts with fused scene BEV, current local BEV,
  pose trace, uncertainty/seen maps, selected trajectory overlays, and predicted
  future overlays.

Not implemented:

- real robot runtime process;
- synchronized owned RGB/IR/IMU/wheel/command logs;
- hardware controller, watchdog, recovery, docking, or physical safety gate;
- robust dynamic-risk labels;
- strong free-space traversability labels;
- global home-scale coverage memory;
- non-collapsed route-heldout runtime policy across multiple real routes;
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

Goal26: real public OpenLORIS scene-level `Brain.step(...)` runtime milestone.

## Goal26 Result

Objective: connect the existing OpenLORIS ingestion, deterministic replay,
SpatialMemoryNetV1 memory, odom-based online pose estimate, physical BEV
geometry, FutureBEV candidate prediction, guided candidate selection, bounded
`cmd_vel` proposals, latency/leakage/collapse reports, and scene-level memory
visuals into one real-data replay-as-live path.

Command:

```text
python -m homebrain.tools.run_openloris_route_heldout_milestone --out runs\goal26_scene_runtime_brain_milestone --sequences cafe1-1_2,corridor1-1,office1-1_7 --heldout-sequence corridor1-1 --max-frames 96 --spatial-steps 30 --future-steps 30 --runtime-max-frames 96 --runtime-feature-source direct_rgbd --max-spatial-folds 2 --future-rollout-selection-mode guided_transparent
```

Artifacts:

```text
runs/goal26_scene_runtime_brain_milestone/milestone_report.json
runs/goal26_scene_runtime_brain_milestone/runtime/heldout_corridor1_1_direct_rgbd/runtime_report.json
runs/goal26_scene_runtime_brain_milestone/runtime/heldout_corridor1_1_direct_rgbd/scene_memory/corridor1_1_scene_memory.npz
runs/goal26_scene_runtime_brain_milestone/runtime/heldout_corridor1_1_direct_rgbd/scene_memory/corridor1_1_scene_memory.ppm
runs/goal26_scene_runtime_brain_milestone/train/spatial_v1_except_corridor1_1/checkpoint.pt
runs/goal26_scene_runtime_brain_milestone/train/future_bev_rollout_except_corridor1_1/checkpoint.pt
```

Key metrics: route count `3`, heldout `corridor1-1`, runtime API steps `96`,
current/fused BEV IoU proxy `0.18949444219026537 / 0.7033406457730702`, future
unknown IoU proxy `0.9318639982272857`, selected action entropy `1.0`, dominant
action fraction `0.5`, unsafe selected rate `0.0`, bounded cmd_vel proposals
`96`, p50/p95 step latency `53.9916 / 58.851275 ms`, teacher runtime dependency
`false`, future/ground-truth runtime dependency `false`, route-pose leakage
fraction `0.0`, raw PWM `false`, control safe `false`.

Scene-memory metrics: observed cell ratio `0.15047021943573669`; unknown
reduction vs current `-0.31632888317108154` (regressed, not a win); pose metric
is `scene_pose_trace_vs_pose_source_relative_trace_eval_only` using
`runtime_odometry_proxy`, not independent ground truth.

Files changed:

```text
homebrain/brain/future_bev_rollout_v1.py
homebrain/brain/modeld.py
homebrain/policies/runtime_decision.py
homebrain/runtime/replay_openloris_brain.py
homebrain/tools/run_openloris_route_heldout_milestone.py
tests/test_goal20a_closed_loop_replay.py
CURRENT_STATUS.md
BLOCKERS.md
```

Verification so far:

```text
python -m py_compile homebrain\brain\modeld.py homebrain\policies\runtime_decision.py homebrain\brain\future_bev_rollout_v1.py homebrain\runtime\replay_openloris_brain.py homebrain\tools\run_openloris_route_heldout_milestone.py
python -m pytest tests\test_goal20a_closed_loop_replay.py tests\test_goal12a_spatial_memory_v1.py tests\test_future_bev_rollout_v1.py -q
git diff --check
python -m pytest -q
```

Pass/fail: focused tests passed with `22 passed`; full pytest passed with
`128 passed in 197.64s`; `git diff --check` passed with only Git line-ending
warnings on Windows.

Risks: action diversity is produced by a transparent/FutureBEV guided selector
with a temporal diversity prior because raw FutureBEV argmin was previously
collapsed. Traversability/free-space labels remain weak. Scene memory is useful
as an artifact and API proof, but not yet home-scale, independent-SLAM-grade, or
control-safe.

Recommended next step: improve the direct RGB-D student quality and BEV
observed-cell density, then replace the hand-weighted temporal diversity prior
with a learned non-collapsed scorer trained/evaluated route-heldout on more
real routes.

## Active Next Goal

Goal27 must implement the original scene-level robot-brain request, not repeat
Goal26 plumbing. A successful run must train/evaluate a real direct RGB-D or
RGB/IR student path, improve measured scene memory and physical BEV geometry,
predict nonzero future free/occupied/unknown state for candidate actions, and
select bounded trajectories with learned non-collapsed scoring.

Goal27 must not count these as acceptance:

```text
guided_transparent
temporal diversity prior
hand-weighted action alternation
patch-stat direct RGB-D adapter only
odometry-vs-odometry pose metric as independent localization
scene-memory artifact existence without unknown reduction
```

Minimum accepted report flags/metrics:

```text
accepted_policy_uses_guided_transparent=false
accepted_policy_uses_handcrafted_diversity_prior=false
teacher_runtime_dependency=false
future_or_groundtruth_runtime_dependency=false
route_pose_leakage_ablation_fraction=0.0
action_entropy>0.0
dominant_action_fraction<1.0
unknown_reduction_vs_current>0.0
coverage_memory_cells_seen>0
future_free_iou_or_proxy>0.0
future_occupied_iou_or_proxy>0.0
latency_step_p95_ms<=100.0
```

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

Goal25 next step is superseded by Goal26 and the active Goal27 requirements
above. Do not repeat Goal25/Goal26 connector work as success; improve learned
direct runtime, scene memory, future free/occupied prediction, and policy
selection quality.

## Documentation Alignment Update

Objective: align repo instructions and README docs toward meaningful
production-directed robot-brain work: connected runtime modules, real public
robot data, no synthetic/mock milestones, no teacher leakage at control time,
and non-collapsed route-heldout policy progress.

Files changed:

```text
README.md
AGENTS.md
PROJECT_BRIEF.md
ARCHITECTURE.md
EVALS.md
BLOCKERS.md
DECISIONS.md
CURRENT_STATUS.md
```

Verification: docs-only change; `git diff --check` passed with only Git
line-ending warnings on Windows. No pytest run was required because no code
changed.
