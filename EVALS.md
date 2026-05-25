# EVALS.md - Active Gates

HomeBrain should prove claims with deterministic artifacts, not demos.

## Always Run

- Focused tests for the touched module.
- `python -m pytest -q` before claiming a broad cleanup or behavior change.
- `git diff --check` before finalizing edits.

## Gate 0: Log, Replay, Eval

Minimum proof:

- typed events serialize deterministically;
- segment logs round-trip;
- replay order is deterministic;
- eval writes JSON metrics;
- brain outputs remain inspectable.

Core metrics:

```text
event_count
frame_count
dropped_frame_count
event_ordering_error_count
replay_determinism_pass
brain_output_count
eval_runtime_sec
```

## Gate 1: Offline Teachers

Teachers may be real open-weight models or explicit fakes. They must record
provenance and cannot silently become runtime/control dependencies.

Core metrics:

```text
teacher_artifact_count
mock
synthetic
real_perception
depth_valid_ratio
confidence_valid_ratio
pose_valid_ratio
scale_status
missing_artifact_count
artifact_shape_error_count
```

Allowed POC use:

- Open-weight models and public datasets are allowed for local training/eval
  when provenance and use limits are recorded.
- Fake or synthetic teacher outputs remain review/test only.
- POC approval does not imply `control_safe=true`, product training approval, or
  redistribution permission.

## Gate 2: Geometry and Spatial Packs

Minimum proof:

- route metadata does not invent missing sensors;
- calibration, scale, and frame convention are explicit;
- BEV/spatial packs are deterministic;
- weak labels are marked weak.

Core metrics:

```text
free_ratio_mean
obstacle_ratio_mean
unknown_ratio_mean
confidence_mean
robot_frame_truth
action_supervision_ok
pose_label_count
weak_label
```

## Gate 3: Spatial Memory

Minimum proof:

- SpatialMemoryNet can run in replay/modeld;
- current BEV, memory BEV, pose, uncertainty, and debug artifacts are emitted;
- memory update masks and pose-warp behavior are explicit;
- route-out or scene-out checks are used when data allows.

Core metrics:

```text
current_bev_iou_or_proxy
fused_memory_bev_iou_or_proxy
unknown_reduction_vs_current
temporal_reprojection_consistency_iou
pose_delta_rmse
memory_warp_valid_fraction
update_mask_coverage_mean
memory_overwrite_fraction
memory_benefit_pass
```

## Gate 4: Trajectory Scoring

Minimum proof:

- candidates are generated deterministically;
- policy selects candidate trajectories, not raw PWM;
- scorer/debug outputs explain risk, unknown, coverage, smoothness, and
  uncertainty;
- closed-loop replay reports action distribution and collapse risk.

Core metrics:

```text
candidate_count
selected_candidate_id
action_entropy
dominant_action_fraction
weighted_v5_agreement
stop_selected_fraction
unsafe_selected_rate
coverage_memory_cells_seen
cmd_vel_non_null_count
```

Current safety flags for replay artifacts should remain:

```text
replay_only=true
not_executed=true
control_safe=false
raw_pwm_emitted=false
```

`product_training_approved=false` is still expected for public/teacher-derived
artifacts, but it must not block local POC experiments.

## Gate 5: Real Route-Heldout Replay

Minimum proof:

- use public or owned real robot data only for milestone metrics;
- split by route, not frame;
- build robot-frame RGB-D BEV labels and QA them;
- extract real teacher features only for training;
- reject degenerate FutureBEV action labels or groups explicitly;
- replay as online runtime: no future frames, no future labels, no raw PWM, and
  no route-pose leakage unless the report is marked as an ablation.

Core metrics:

```text
route_count
heldout_sequence
robot_frame_bev_qa_pass
current_bev_iou_or_proxy
fused_memory_bev_iou_or_proxy
future_unknown_iou_or_proxy
candidate_new_area_gain_ranking_quality
selected_candidate_entropy
dominant_action_fraction
unsafe_selected_rate
latency_end_to_end_p50_ms
latency_end_to_end_p95_ms
pose_warp_valid_fraction
route_pose_leakage_ablation_fraction
cmd_vel_proposal_count
raw_pwm_emitted
```

Current accepted route-heldout report:

```text
runs/goal25_openloris_route_heldout_milestone/milestone_report.json
```

The current report passes artifact creation and no-leakage gates, but fails
policy quality because FutureBEV/runtime action selection collapsed to one
candidate.
