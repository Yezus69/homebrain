HomeBrain progress is judged by gates, not by scaffolding.

A change is not successful because code was added. A change is successful only when it improves a measured gate, produces deterministic artifacts, compares against a relevant baseline when behavior changes, and keeps replay-only safety constraints intact.

This file is the acceptance judge for model, runtime, training, eval, and milestone work.

## Required safety invariants

These must remain true until the maintainer explicitly starts a real hardware-safety phase:

```text
replay_only=true
not_executed=true
control_safe=false
raw_pwm_emitted=false
hardware_validated=false
```

Runtime must not use:

- future labels,
- future frames,
- teacher-only artifacts,
- route ground truth,
- oracle BEV,
- heldout answers,
- hand-authored diversity tricks counted as learned policy success.

If a run uses any of those as an ablation, the report must mark it as an ablation and it must not count as accepted robot-brain progress.

## Always run

For any nontrivial code change:

```text
git diff --check
```

Run focused tests for the touched module.

For broad runtime, training, eval, or milestone work, also run:

```text
python -m pytest -q
```

If full tests are not feasible, state exactly why and run the strongest focused subset possible.

## Gate A: Replay/data correctness

Purpose: prove the route, event, segment, artifact, and replay spine is deterministic and honest.

Minimum proof:

- typed events serialize deterministically,
- segment logs round-trip,
- replay order is deterministic,
- timestamps are valid and monotonic where required,
- sensor masks are explicit,
- missing sensors are marked missing,
- unknown scale is marked unknown,
- bad calibration is rejected or warned on,
- eval writes machine-readable JSON metrics.

Useful metrics:

```text
event_count
frame_count
dropped_frame_count
event_ordering_error_count
timestamp_monotonicity_pass
segment_roundtrip_pass
replay_determinism_pass
missing_sensor_rate
unknown_scale_route_count
route_qa_warning_count
brain_output_count
eval_runtime_sec
```

Hard fail examples:

```text
invented_sensor=true
silent_missing_calibration=true
nondeterministic_replay=true
```

## Gate B: Teacher/provenance correctness

Purpose: use open-source/open-weight teachers without pretending weak outputs are ground truth.

Minimum proof:

- teacher artifacts include model name, backend, version/checkpoint when available,
- fake/mock teachers are visibly fake,
- weak labels are marked weak,
- license/provenance metadata exists,
- artifact shapes are checked,
- invalid artifacts are rejected or warned on,
- runtime student does not secretly depend on offline teacher outputs.

Useful metrics:

```text
teacher_artifact_count
teacher_backend
teacher_model_name
mock_artifact_count
synthetic_artifact_count
real_perception_artifact_count
weak_label_count
missing_artifact_count
artifact_shape_error_count
teacher_confidence_mean
teacher_confidence_valid_ratio
depth_valid_ratio
pose_valid_ratio
scale_status
runtime_teacher_dependency
```

Hard fail examples:

```text
fake_teacher_marked_real=true
weak_label_marked_ground_truth=true
teacher_runtime_dependency=true
missing_provenance=true
```

## Gate C: BEV label quality

Purpose: produce useful robot-frame free, occupied, unknown, traversability, and risk supervision.

Minimum proof:

- camera intrinsics are checked,
- camera-to-base extrinsics are checked,
- depth validity is handled,
- pose alignment is checked when pose exists,
- unknown space is represented explicitly,
- invalid or missing depth does not become fake free space,
- dynamic/risk labels are separated from static occupancy when available,
- projection tests cover valid and invalid inputs.

Useful metrics:

```text
free_ratio_mean
occupied_ratio_mean
unknown_ratio_mean
risk_ratio_mean
valid_depth_fraction
bev_coverage_fraction
robot_frame_truth
camera_to_base_available
pose_label_count
pose_alignment_error_or_proxy
weak_label
future_free_nonzero_fraction
future_occupied_nonzero_fraction
```

Hard fail examples:

```text
missing_extrinsics_silently_accepted=true
invalid_depth_marked_free=true
oracle_bev_runtime_dependency=true
```

## Gate D: Online SceneState/memory quality

Purpose: maintain online robot-frame scene memory without cheating.

Minimum proof:

- SceneState updates online only,
- memory uses past/current data only,
- pose deltas are explicit,
- uncertainty is updated,
- stale observations decay or are marked uncertain,
- coverage/seen maps are updated,
- memory artifacts are inspectable,
- tests prove no future leakage.

Useful metrics:

```text
scene_state_step_count
memory_coverage_cells_seen
unknown_reduction_vs_current
fused_memory_bev_iou_or_proxy
current_bev_iou_or_proxy
temporal_reprojection_consistency_iou
pose_delta_rmse_or_proxy
pose_warp_valid_fraction
update_mask_coverage_mean
memory_overwrite_fraction
uncertainty_mean
memory_benefit_pass
```

Hard fail examples:

```text
future_or_groundtruth_runtime_dependency=true
route_pose_leakage_ablation_fraction>0.0
scene_memory_not_used_by_policy=true
```

## Gate E: Future BEV/risk prediction

Purpose: predict what space will become free, occupied, unknown, or risky over short horizons.

Minimum proof:

- horizon-specific future targets exist,
- valid horizon masks exist,
- prediction is compared against current-BEV-copy-forward,
- uncertainty output exists,
- dynamic-obstacle fixture or real route evidence exists when changing dynamic behavior,
- runtime prediction uses only online-available inputs,
- no future-frame leakage at runtime.

Required baselines when this gate changes:

```text
current_bev_copy_forward
previous_accepted_checkpoint
```

Useful metrics:

```text
future_horizon_count
future_free_iou_or_proxy
future_occupied_iou_or_proxy
future_unknown_iou_or_proxy
future_risk_auc_or_proxy
future_risk_ap_or_proxy
copy_forward_free_iou_or_proxy
copy_forward_occupied_iou_or_proxy
copy_forward_unknown_iou_or_proxy
improvement_vs_copy_forward
uncertainty_error_correlation
valid_horizon_fraction
```

Hard fail examples:

```text
future_frame_runtime_dependency=true
future_label_runtime_dependency=true
future_free_iou_or_proxy<=0.0
future_occupied_iou_or_proxy<=0.0
```

## Gate F: Learned candidate scoring

Purpose: choose safer bounded local trajectories using scene memory and predicted future risk.

Minimum proof:

- finite candidate trajectory set exists,
- stop candidate is always available,
- candidate footprints are checked against BEV/risk,
- learned candidate scores are compared to a transparent scorer,
- future risk can affect candidate scores,
- behavior does not collapse to one action unless the scene truly requires it,
- selection is not driven by hand-authored diversity priors,
- tests include a case where a future obstacle changes the selected candidate.

Required baselines when this gate changes:

```text
stop_only
transparent_scorer
unknown_is_dangerous_scorer
previous_accepted_checkpoint
```

Useful metrics:

```text
candidate_count
selected_candidate_id
selected_candidate_entropy
dominant_action_fraction
stop_selected_fraction
unsafe_selected_rate
collision_risk_ranking_accuracy
future_collision_risk_ranking_accuracy
unknown_exposure_mean
candidate_new_area_gain_ranking_quality
coverage_gain_mean
progress_score_mean
accepted_policy_uses_guided_transparent
accepted_policy_uses_handcrafted_diversity_prior
```

Hard fail examples:

```text
raw_pwm_emitted=true
accepted_policy_uses_guided_transparent=true
accepted_policy_uses_handcrafted_diversity_prior=true
action_entropy<=0.0
dominant_action_fraction>=1.0
unsafe_selected_rate>0.0
```

## Gate G: Route-heldout robustness

Purpose: avoid overfitting to one route, one scene, one public-data quirk, or one synthetic fixture.

Minimum proof:

- split by route or scene, not by nearby frames,
- heldout routes are not used for training labels,
- metrics are reported per route and aggregate,
- degenerate routes are rejected instead of counted as success,
- caveats are listed clearly,
- train/heldout gap is visible.

Useful metrics:

```text
route_count
scene_count
train_route_count
heldout_route_count
heldout_sequence
runtime_heldout_sequences
per_route_metric_json_count
worst_route_score
route_rejection_count
route_rejection_reasons
train_heldout_gap
action_distribution_by_route
latency_end_to_end_p50_ms
latency_end_to_end_p95_ms
```

Hard fail examples:

```text
frame_split_leakage=true
heldout_route_used_for_training=true
degenerate_route_counted_success=true
route_pose_leakage_ablation_fraction>0.0
```

## Gate H: Hardware-readiness replay safety

Purpose: move toward real robot deployment without pretending current code is hardware-safe.

This gate is replay-only. It is not permission to drive hardware.

Minimum proof:

- proposed `cmd_vel` remains replay-only,
- command envelope is bounded,
- stale-sensor stop exists,
- high-uncertainty stop exists,
- high-risk stop exists,
- watchdog state exists in replay,
- recovery proposal exists but is not executed,
- every stop/recovery has an explicit reason,
- no raw PWM exists,
- no hardware transport exists,
- no hardware-safe claim exists.

Useful metrics:

```text
cmd_vel_proposal_count
cmd_vel_non_null_count
cmd_vel_executed_count
command_envelope_violation_count
watchdog_state_count
stale_sensor_stop_count
high_uncertainty_stop_count
high_risk_stop_count
recovery_selected_fraction
recovery_executed_count
stop_reason_counts
unsafe_proposal_rejection_rate
stale_sensor_detection_latency_ms
raw_pwm_emitted
hardware_transport_enabled
hardware_validated
control_safe
```

Hard fail examples:

```text
cmd_vel_executed_count>0
raw_pwm_emitted=true
hardware_transport_enabled=true
hardware_validated=true
control_safe=true
```

## Required failure-mode fixtures

When touching prediction, memory, policy, runtime decision, or safety logic, prefer deterministic fixtures for specific failures:

- moving obstacle crossing the robot path,
- static obstacle in known free space,
- unknown corridor,
- stale sensor input,
- bad or missing calibration,
- pose jump,
- narrow passage,
- dynamic object that was seen and then occluded,
- route with weak or missing depth,
- candidate action that looks safe now but unsafe in the future,
- action collapse to a single candidate,
- high uncertainty near the robot footprint.

A failure mode is useful only if a test, metric, or replay artifact can fail when behavior regresses.

## Baseline rule

When changing learned prediction, memory, scoring, or runtime decision behavior, compare against relevant baselines.

Common baselines:

```text
stop_only
current_bev_copy_forward
transparent_scorer
unknown_is_dangerous_scorer
previous_accepted_checkpoint
random_candidate_sanity_check
```

A learned model is not progress unless it beats at least one meaningful baseline on a relevant gate without violating safety invariants.

## Acceptance rule

A milestone or PR is accepted only if it states:

- which gates improved,
- which baselines were compared,
- which routes or fixtures were used,
- which artifacts prove the run happened,
- which tests passed,
- which caveats remain,
- whether all replay/hardware safety invariants stayed conservative.

No milestone may claim real robot readiness without physical robot logs, watchdogs, recovery behavior, safety tests, and maintainer approval.

## Report shape

For broad model/runtime/eval work, the report artifact should include at least:

```json
{
  "accepted": false,
  "gates_improved": [],
  "routes": [],
  "fixtures": [],
  "baselines": [],
  "metrics": {},
  "artifacts": [],
  "tests": [],
  "hard_failures": [],
  "caveats": [],
  "safety": {
    "replay_only": true,
    "not_executed": true,
    "control_safe": false,
    "raw_pwm_emitted": false,
    "hardware_validated": false
  }
}
```