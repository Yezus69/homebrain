# EVALS.md - Active Gates

HomeBrain should prove claims with deterministic artifacts, not demos.

## Always Run

- Focused tests for the touched module.
- `python -m pytest -q` before claiming a broad cleanup or behavior change.
- `git diff --check` before finalizing edits.
- For broad runtime/training work, produce or update a real-data replay report
  when feasible. If not feasible, state exactly why.

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
runs/goal26_scene_runtime_brain_milestone/milestone_report.json
```

The current report is a connector baseline, not the target robot brain. It
passes artifact creation, replay-only, no-raw-PWM, and no-leakage checks, but
it must not be treated as learned policy success because accepted action
diversity used `guided_transparent` and raw FutureBEV action selection still
collapsed.

## Gate 6: Deployable Runtime Integration

Minimum proof:

- an online-style `Brain.step(...)` or equivalent API owns persistent memory
  across sensor ticks;
- the control tick consumes only allowed online inputs;
- teacher outputs, future labels, route ground truth, and oracle BEV are absent
  from runtime unless the report marks an ablation;
- outputs are bounded candidate trajectories or `cmd_vel`, never raw PWM;
- stop/recovery/uncertainty reasons are explicit;
- route-heldout replay writes latency, leakage, action-collapse, and unsafe
  selection metrics;
- failure cases are inspectable through JSON and small visual/contact-sheet
  artifacts.
- accepted action diversity is produced by learned scoring or learned FutureBEV
  outputs, not `guided_transparent`, temporal diversity priors, randomization,
  or hand-authored alternation;
- scene memory improves measured physical geometry, not only artifact presence.

Core metrics:

```text
runtime_api_step_count
teacher_runtime_dependency
future_or_groundtruth_runtime_dependency
latency_step_p50_ms
latency_step_p95_ms
memory_update_latency_p95_ms
action_entropy
dominant_action_fraction
unsafe_selected_rate
stop_selected_fraction
recovery_selected_fraction
route_pose_leakage_ablation_fraction
cmd_vel_proposal_count
raw_pwm_emitted
accepted_policy_uses_guided_transparent
accepted_policy_uses_handcrafted_diversity_prior
unknown_reduction_vs_current
coverage_memory_cells_seen
future_free_iou_or_proxy
future_occupied_iou_or_proxy
pose_metric_source
pose_metric_independent_groundtruth
```

Hard fail conditions for an accepted Goal27-style runtime:

```text
teacher_runtime_dependency=true
future_or_groundtruth_runtime_dependency=true
route_pose_leakage_ablation_fraction>0.0
raw_pwm_emitted=true
accepted_policy_uses_guided_transparent=true
accepted_policy_uses_handcrafted_diversity_prior=true
action_entropy<=0.0
dominant_action_fraction>=1.0
unknown_reduction_vs_current<=0.0
coverage_memory_cells_seen<=0
future_free_iou_or_proxy<=0.0
future_occupied_iou_or_proxy<=0.0
latency_step_p95_ms>100.0
```

If a metric cannot be computed because labels are missing, the report must mark
the run blocked or use a clearly named proxy. It must not silently pass.

Product claim rule:

```text
control_safe=false
product_training_approved=false
hardware_validated=false
```

These stay false until owned robot data, hardware safety review, and physical
test evidence exist. The goal before hardware is production-directed software,
not a product-safety claim.
