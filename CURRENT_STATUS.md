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

Goal24A: real OpenLORIS ingestion, robot-frame BEV QA, and replay-as-live
runtime decisions with bounded `cmd_vel` proposals.

## Goal24A Result

Objective attempted: move the existing OpenLORIS, robot RGB-D BEV, spatial
memory, trajectory scoring, runtime decision, and closed-loop report paths
toward a deployable replay-only robot-frame brain without using fake data as
main evidence.

Files changed: `homebrain/datasets/openloris_scene.py`,
`homebrain/datasets/openloris_to_route.py`, `homebrain/geometry/qa_robot_frame_bev.py`,
`homebrain/brain/modeld.py`, `homebrain/policies/runtime_decision.py`,
`homebrain/eval/closed_loop_replay_report.py`, `homebrain/runtime/replay_openloris_brain.py`,
`homebrain/runtime/__init__.py`, and focused tests.

Commands run:

```text
nvidia-smi
python -m homebrain.datasets.openloris_to_route --source data\public\openloris_scene\cafe1-1_2 --out runs\goal24_openloris_realtime_stack\route_cafe1_1_2_short --max-frames 80
python -m homebrain.geometry.robot_rgbd_to_bev --log runs\goal24_openloris_realtime_stack\route_cafe1_1_2_short --out runs\goal24_openloris_realtime_stack\route_cafe1_1_2_short\geometry\robot_rgbd_bev
python -m homebrain.geometry.validate_bev --bev runs\goal24_openloris_realtime_stack\route_cafe1_1_2_short\geometry\robot_rgbd_bev --out runs\goal24_openloris_realtime_stack\bev_validate_cafe1_1_2_short.json
python -m homebrain.geometry.visualize_bev --bev runs\goal24_openloris_realtime_stack\route_cafe1_1_2_short\geometry\robot_rgbd_bev --out runs\goal24_openloris_realtime_stack\bev_visuals_cafe1_1_2_short
python -m homebrain.data.pack_spatial_dataset --log runs\goal24_openloris_realtime_stack\route_cafe1_1_2_short --bev runs\goal24_openloris_realtime_stack\route_cafe1_1_2_short\geometry\robot_rgbd_bev --out runs\goal24_openloris_realtime_stack\spatial_pack_cafe1_1_2_short
python -m homebrain.data.qa_spatial_dataset --dataset runs\goal24_openloris_realtime_stack\spatial_pack_cafe1_1_2_short --out runs\goal24_openloris_realtime_stack\spatial_qa_cafe1_1_2_short.json
python -m homebrain.geometry.qa_robot_frame_bev --bev runs\goal24_openloris_realtime_stack\route_cafe1_1_2_short\geometry\robot_rgbd_bev --spatial-pack runs\goal24_openloris_realtime_stack\spatial_pack_cafe1_1_2_short --out runs\goal24_openloris_realtime_stack\robot_frame_bev_qa_cafe1_1_2_short.json
python -m homebrain.runtime.replay_openloris_brain --log runs\goal11b_nightly\routes\openloris_cafe1_1_2_route --out runs\goal24_openloris_realtime_stack\runtime_cafe1_1_2_with_baseline --checkpoint runs\goal12b_spatial_memory_v1\v1_window4_route_pose_warm_start\checkpoint.pt --features runs\goal11b_nightly\routes\openloris_cafe1_1_2_route\teacher_artifacts\dino --trajectory-scorer-checkpoint runs\goal22a_overnight_runtime_scorer_repair\train\model_current_all_features_seed21_pair_constrained_search2\checkpoint.pt --device cuda:0
python -m py_compile homebrain\datasets\openloris_scene.py homebrain\datasets\openloris_to_route.py homebrain\geometry\qa_robot_frame_bev.py homebrain\policies\runtime_decision.py homebrain\brain\modeld.py homebrain\eval\closed_loop_replay_report.py homebrain\runtime\__init__.py homebrain\runtime\replay_openloris_brain.py
python -m pytest tests\test_goal10a_robot_frame_bridge.py tests\test_goal20a_closed_loop_replay.py -q
git diff --check
python -m pytest -q
```

Pass/fail results: real OpenLORIS import, BEV generation, visualization, spatial
pack, robot-frame BEV QA, and replay-as-live runtime all completed. Focused
tests passed with `9 passed`; full pytest passed with `128 passed in 190.89s`.
PowerShell tool output still appends a non-project `-Command` warning after
commands.

Artifacts created:

```text
runs/goal24_openloris_realtime_stack/route_cafe1_1_2_short
runs/goal24_openloris_realtime_stack/route_cafe1_1_2_short/geometry/robot_rgbd_bev
runs/goal24_openloris_realtime_stack/bev_validate_cafe1_1_2_short.json
runs/goal24_openloris_realtime_stack/bev_visuals_cafe1_1_2_short/visualization_manifest.json
runs/goal24_openloris_realtime_stack/spatial_pack_cafe1_1_2_short
runs/goal24_openloris_realtime_stack/spatial_qa_cafe1_1_2_short.json
runs/goal24_openloris_realtime_stack/robot_frame_bev_qa_cafe1_1_2_short.json
runs/goal24_openloris_realtime_stack/runtime_cafe1_1_2_with_baseline/runtime_report.json
runs/goal24_openloris_realtime_stack/runtime_cafe1_1_2_with_baseline/runtime_report.md
```

Key real-data results: the new importer accepts measured OpenLORIS
`trans_matrix.yaml` camera-to-base extrinsics, writes per-artifact SHA-256
checksums, and rejects missing depth/calibration by default. The 80-frame real
cafe slice imported with `robot_frame_truth=true` and `extrinsics_source=openloris_trans_matrix_yaml`.
Robot-frame BEV QA passed with 80/80 calibrated frames, no pose jumps, mean
depth valid ratio `0.9491927083333334`, mean observed cell ratio
`0.0855224609375`, and zero route split leakage. Spatial dataset QA was
structurally trainable but quarantined for low confidence, so it is not counted
as a clean training-quality result.

Runtime result: the OpenLORIS cafe route replay produced 1200 online spatial
memory decisions and 1200 bounded `cmd_vel` proposals, with `cmd_vel` still not
executed. On `cuda:0` RTX 4090, p50/p95 end-to-end latency was
`41.557 / 54.96671499999998 ms`; p95 model and decision latency were
`8.9844 / 11.482949999999999 ms`; 10 Hz pass fraction was
`0.9808333333333333`. No raw PWM was emitted.

Transparent-baseline comparison: primary learned-scorer decisions matched 1200
baseline frames and differed on 1197 (`0.9975`). The learned scorer did not
collapse (`selected_candidate_unique_count=2`, entropy `0.9934800107379318`,
dominant fraction `0.5475`, stop fraction `0.0`). It improved route-progress
proxy by `0.08370833333333333` and coverage-gain proxy by
`0.10916666666666686`, but selected higher transparent risk and unknown
penalties than the heuristic baseline (`+0.011150962500000002` risk,
`+0.35400405250000005` unknown). This is replay evidence only, not a safety
claim.

Remaining risks: the strongest runtime policy is still narrow and not product
safe; the new short-route spatial pack is low-confidence; route replay predates
hardware and dynamic-obstacle safety; public data artifacts remain
replay-only/not-executed/control-safe-false unless proven otherwise.

Recommended next goal: expand the same real OpenLORIS measured-calibration
pipeline across office/corridor routes, rebuild route-level train/val/test
packs, and retrain the trajectory scorer with explicit gates for risk/unknown
penalty so the progress gain does not come by accepting worse transparent risk.
