# CURRENT_STATUS.md

Codex must update this file at the end of every goal. Historical goal logs older
than Goal 12 are archived under `docs/status_archive/` and are not part of the
default read-list.

## Current objective

Goal 15B completed: add a real owned-route scene-teacher signal gate. The MoGe
SceneTeacherPack v0 wrapper and audit tool can now distinguish fake review-only
geometry from real, non-mock, truthful owned-route geometry-pretrain candidates
without training SpatialMemoryNet, TrajectoryScorerNet, or any policy.

## Last completed goal

Goal 15B: real owned-route scene-teacher signal gate. Fake MoGe backend tests and
tiny owned-route audit verification pass; real MoGe/VGGT remain optional local
setup items, not required test dependencies.

## Current implementation status

- `homebrain` package contains deterministic log/replay/eval, teacher artifacts,
  image/video ingest, geometry-to-BEV, SpatialTrainPack, spatial v0/v1 training,
  trajectory scoring, and policy audit tools.
- Runtime/control constraints remain intact: no ROS/Nav2/Isaac/Habitat/sim was
  added, no raw PWM is emitted, and current policy artifacts keep
  `replay_only=true`, `not_executed=true`, `control_safe=false`, and
  `product_training_approved=false`.
- SpatialMemoryNet v1 has explicit persistent BEV memory, route-pose/odom warp
  defaults, conservative update masks, hard validation, and true route-out folds.
  Goal 12C showed useful memory in spatial eval, but not control-safe behavior.
- The old synthetic coverage/risk action oracle remains callable for ablation in
  v3/v4 builders, but Goal 13B showed it collapsed and disagreed with real future
  motion.
- Goal 14 added `homebrain.policies.future_motion_action_labels` and
  `build_action_label_pack_v5`; v5 labels are matched by resampled relative
  future trajectories and invalid frames are explicitly labeled
  `future_horizon_truncated`, `pose_missing`, or `stationary_below_threshold`.
- Goal 14 added TrajectoryScorerNet v1 train/eval entry points that reuse the v0
  architecture/input plumbing with v5 labels and class-balanced loss.
- Goal 14 updated Goal 13A shadow eval with `--scorer-checkpoint`; in scorer mode
  `oracle_bev` is the v5 behavior-cloning label oracle, while v0/v1 current and
  memory BEVs are scored by the learned checkpoint.
- Goal 15A added `SceneTeacherPack` v0 as a separate teacher artifact family for
  scene/geometry supervision. The fake VGGT-style backend writes deterministic
  depth, point maps, intrinsics, extrinsics, point tracks, confidence/validity
  masks, and floor/risk/dynamic placeholders for tests.
- Goal 15A added scene-teacher QA and a review-only scene-teacher-to-BEV
  conversion stub. The converter preserves `weak_label=true`,
  `robot_frame_truth=false`, `action_supervision_ok=false`, `replay_only=true`,
  `not_executed=true`, `control_safe=false`, and
  `product_training_approved=false`.
- Goal 15B added `homebrain.teachers.moge_scene_teacher` with fake and real
  backends following SceneTeacherPack v0. The real backend requires local
  operator-supplied MoGe assets or `HOMEBRAIN_MOGE_ADAPTER=module:function`;
  HomeBrain still does not clone or download teacher repos/checkpoints during
  runs.
- Goal 15B added `homebrain.tools.audit_scene_teacher_signal`, which combines
  route metadata truth, SceneTeacherPack QA, mask-source distribution, scale
  status, robot-frame/action-supervision claims, and `next_allowed_use`.
- Owned image/video route metadata can now explicitly record
  `owned_or_license_approved`; the audit blocks missing approval, invented
  IMU/odom/command streams, and robot-frame truth claims without measured
  camera-to-base plus base pose/odom.
- Active blockers are tracked in `BLOCKERS.md`; resolved old blockers are archived
  under `docs/blockers_archive/`.

## Goal completion log

### 026 - Goal 15B owned-route scene-teacher signal gate

Objective attempted: create the smallest path that can tell whether real
foundation geometry teachers produce useful spatial supervision on owned indoor
video, without training SpatialMemoryNet, TrajectoryScorerNet, or any policy.

Files changed: added `homebrain/teachers/moge_scene_teacher.py`,
`homebrain/tools/audit_scene_teacher_signal.py`, and
`tests/test_goal15b_scene_teacher_signal.py`; updated
`homebrain/teachers/run_scene_teacher.py`, `homebrain/teachers/__init__.py`,
`homebrain/ingest/image_sequence.py`, `CURRENT_STATUS.md`, `EVALS.md`,
`BLOCKERS.md`, and `LICENSE_AUDIT.md`.

Commands run: required context reads; `python -m py_compile
homebrain\teachers\moge_scene_teacher.py
homebrain\teachers\run_scene_teacher.py
homebrain\tools\audit_scene_teacher_signal.py
homebrain\ingest\image_sequence.py tests\test_goal15b_scene_teacher_signal.py`;
targeted tests `python -m pytest tests\test_goal15b_scene_teacher_signal.py
tests\test_goal15a_scene_teacher.py tests\test_ingest_image_sequence.py -q`;
tiny fixture frame generation under
`runs\goal15b_scene_teacher_signal_fake_input`; owned fixture import; fake MoGe
scene-teacher run; scene-teacher QA; scene-teacher signal audit; optional real
MoGe and VGGT availability checks; full `python -m pytest -q`.

Pass/fail results: py_compile passed. Targeted tests passed with `11 passed`.
Full pytest passed with `93 passed`. Fake MoGe SceneTeacherPack and the
owned-route signal audit ran end to end. Optional real MoGe and VGGT checks
returned clear unavailable setup messages because no local teacher checkout,
checkpoint, or adapter environment variables are configured; this is recorded as
optional setup, not a fake-backend gate failure.

Artifacts created: `runs/goal15b_scene_teacher_signal_fake_input/`,
`runs/goal15b_scene_teacher_signal_fake_route/`,
`runs/goal15b_scene_teacher_signal_fake_route/teacher_artifacts/moge_scene_v0_fake/`,
`runs/goal15b_scene_teacher_signal_fake_route/teacher_artifacts/moge_scene_v0_fake_qa.json`,
`runs/goal15b_scene_teacher_signal_fake_route/teacher_artifacts/moge_scene_v0_fake_signal_audit.json`,
and
`runs/goal15b_scene_teacher_signal_fake_route/teacher_artifacts/moge_scene_v0_fake_signal_audit.md`.

Metrics observed: fake MoGe QA reported `frame_count=4`,
`missing_artifact_count=0`, `artifact_shape_error_count=0`,
`depth_valid_ratio=1.0`, `confidence_valid_ratio=1.0`,
`pose_valid_ratio=1.0`, `temporal_geometry_consistency=0.9944600196821349`,
`scale_status=synthetic_metric_test_scale`, `control_safe=false`, and
`promotable_to_spatial_pack=false` with quarantine reasons
`mock_or_synthetic_teacher` and `relative_or_unknown_scale`. Signal audit
reported `next_allowed_use=review_only`, `hard_blockers=[]`,
`route_metadata_sensor_truth.truth_pass=true`,
`owned_or_license_approved=true`, `robot_frame_truth=false`,
`action_supervision_ok=false`, `promotable_to_spatial_pack=false`, and mask
sources from teacher output for confidence, validity, floor, obstacle, and
dynamic masks.

Blockers/risks: fake artifacts remain mock/synthetic and never promotable. Real
MoGe/VGGT signal on owned inbox video remains optional setup blocked because this
workspace lacks `external/moge`, `external/vggt`, local checkpoints, and
`HOMEBRAIN_MOGE_*`/`HOMEBRAIN_VGGT_*` adapter environment variables. Scene
teacher outputs are still offline geometry review artifacts, not robot-frame
action truth, not control safe, and not product-training approved.

Recommended next goal: install one real local MoGe or VGGT teacher adapter plus
checkpoint outside HomeBrain's run path, re-import `data/inbox/room_walk_001`
with explicit `--owned-or-license-approved`, run the real backend, and use
`audit_scene_teacher_signal` to decide whether the result stays `review_only` or
becomes a `geometry_pretrain_candidate`.

### 025 - Goal 15A foundation scene-teacher stack

Objective attempted: build a foundation-model teacher layer that turns raw indoor
video route logs into richer spatial-memory supervision artifacts, without
training a new action scorer, training a new spatial model, emitting `cmd_vel`, or
claiming control safety.

Files changed: added `homebrain/teachers/scene_teacher.py`,
`homebrain/teachers/vggt_scene_teacher.py`,
`homebrain/teachers/run_scene_teacher.py`,
`homebrain/teachers/qa_scene_teacher.py`,
`homebrain/geometry/scene_teacher_to_bev.py`, and
`tests/test_goal15a_scene_teacher.py`; updated `homebrain/teachers/__init__.py`,
`CURRENT_STATUS.md`, `EVALS.md`, `BLOCKERS.md`, and `LICENSE_AUDIT.md`.

Commands run: required context reads; `python -m py_compile
homebrain\teachers\scene_teacher.py homebrain\teachers\vggt_scene_teacher.py
homebrain\teachers\run_scene_teacher.py homebrain\teachers\qa_scene_teacher.py
homebrain\geometry\scene_teacher_to_bev.py
tests\test_goal15a_scene_teacher.py`; targeted tests `python -m pytest
tests\test_goal15a_scene_teacher.py -q`; fake route generation; fake scene
teacher run; scene-teacher QA; scene-teacher-to-BEV conversion; BEV validation;
optional real VGGT availability check; full `python -m pytest -q`.

Pass/fail results: py_compile passed. Targeted tests passed with `4 passed`.
Full pytest passed with `90 passed`. Fake scene-teacher route ran end to end.
Optional real VGGT returned a clear unavailable setup message because no local
`external/vggt` checkout/checkpoint is configured; this is recorded as optional
setup, not a goal failure.

Artifacts created: `runs/goal15a_scene_teacher_fake_route/`,
`runs/goal15a_scene_teacher_fake_route/teacher_artifacts/scene_v0/`,
`runs/goal15a_scene_teacher_fake_route/teacher_artifacts/scene_v0_qa.json`,
`runs/goal15a_scene_teacher_fake_route/geometry/scene_teacher_bev/`, and
`runs/goal15a_scene_teacher_fake_route/geometry/scene_teacher_bev_qa.json`.

Metrics observed: scene-teacher QA reported `frame_count=6`,
`missing_artifact_count=0`, `artifact_shape_error_count=0`,
`depth_valid_ratio=1.0`, `pose_valid_ratio=1.0`, `track_valid_ratio=1.0`,
`temporal_geometry_consistency=0.9835878353227269`,
`scale_status=synthetic_metric_test_scale`, `control_safe=false`, and
`promotable_to_spatial_pack=false` with quarantine reasons
`mock_or_synthetic_teacher` and `relative_or_unknown_scale`. Review BEV QA
reported `bev_frame_count=6`, zero missing/shape/nan errors,
`free_ratio_mean=0.5`, `obstacle_ratio_mean=0.25`,
`unknown_ratio_mean=0.25`, `confidence_mean=0.8550000190734863`,
`temporal_jitter_mean=0.0`, `weak_label=true`, and `control_safe=false`.

Blockers/risks: real VGGT integration is not configured and requires a local
checkout/checkpoint plus an operator-supplied adapter; HomeBrain still does not
download weights automatically. Scene-teacher BEV outputs are weak review
geometry only, not robot-frame action truth. All artifacts remain
`replay_only=true`, `not_executed=true`, `control_safe=false`,
`product_training_approved=false`, and no `cmd_vel` or raw PWM was emitted.

Recommended next goal: integrate a real local VGGT/MoGe/SAM2 teacher stack for
owned or approved indoor video, or collect a minimal owned route log with
calibrated camera/IMU/odometry before returning to policy training.

### 024 - Goal 14 future-motion behavior-cloning action labels

Objective attempted: build behavior-cloning action labels from dataset future
motion, replace the collapsed synthetic-oracle label source for the new v5 path,
prove non-collapsed labels are reachable on current data, train/evaluate a v5
scorer, rerun Goal 13A with the v5 scorer, reuse the Goal 13B audit, and trim
status/blocker archives.

Files changed: added `homebrain/policies/future_motion_action_labels.py`,
`homebrain/policies/build_action_label_pack_v5.py`,
`homebrain/policies/qa_action_label_pack_v5.py`,
`homebrain/policies/train_trajectory_scorer_v1.py`,
`homebrain/policies/eval_trajectory_scorer_v1.py`, and
`tests/test_goal14_behavior_cloning_labels.py`; updated
`homebrain/policies/build_action_label_pack.py`,
`homebrain/policies/qa_action_label_pack.py`,
`homebrain/policies/trajectory_scorer_net_v0.py`,
`homebrain/tools/goal13a_memory_policy_shadow_eval.py`, `AGENTS.md`,
`ARCHITECTURE.md`, `EVALS.md`, `BLOCKERS.md`, and `CURRENT_STATUS.md`; added
archives under `docs/status_archive/` and `docs/blockers_archive/`.

Commands run: required context reads; py_compile for new/modified Goal 14 modules;
targeted tests `python -m pytest tests\test_goal14_behavior_cloning_labels.py
tests\test_goal13a_memory_policy_shadow_eval.py
tests\test_goal11a_trajectory_scorer_v0.py -q`; real v5 build/QA; v1 scorer
train/eval; Goal 13A v5 shadow eval; Goal 13B audit on v5 decisions.

Pass/fail results: targeted tests passed with `11 passed`. ActionLabelPack v5 QA
passed the non-collapse gate. TrajectoryScorerNet v1 eval wrote metrics and kept
all safety flags false. Goal 13A v5 shadow eval intentionally failed the memory
action-benefit gate; this is now a data/model bottleneck finding, not a label
collapse.

Artifacts created: `runs/goal14_action_label_pack_v5/`,
`runs/goal14_action_label_pack_v5_qa.json`,
`runs/goal14_trajectory_scorer_v1/checkpoint.pt`,
`runs/goal14_trajectory_scorer_v1_eval.json`,
`runs/goal14_trajectory_scorer_v1_eval_predictions.jsonl`,
`runs/goal14_trajectory_scorer_v1_viz/`,
`runs/goal14_memory_policy_shadow_eval_v5_report.json`,
`runs/goal14_memory_policy_shadow_eval_v5_report.md`,
`runs/goal14_memory_policy_shadow_eval_v5_decisions.jsonl`,
`runs/goal14_memory_policy_shadow_eval_v5_worst.ppm`,
`runs/goal14_policy_collapse_audit_v5.json`,
`runs/goal14_policy_collapse_audit_v5.md`, and
`runs/goal14_policy_collapse_worst_v5.ppm`.

Metrics observed: v5 QA reported `example_count=2894`,
`action_entropy=2.3334583564283053`, `dominant_action_fraction=0.4644091223220456`,
`bc_label_confidence_mean=0.25870618115853394`, excluded frames `315`
(`future_horizon_truncated=216`, `stationary_below_threshold=99`), and
v5/synthetic-oracle agreement `0.0`. V1 scorer val reported
`top1_action_agreement=0.49568221070811747`, `beats_random=true`,
`agreement_with_synthetic_oracle_label=0.0`, `distribution_collapse_flag=false`,
and selected distribution `straight_medium=416`, `stop=79`,
`arc_right_medium=61`, `straight_short=23`. Goal 13A v5 shadow eval reported
normal memory action changed fraction `0.0`, current and memory future-motion
agreement both `0.18764302059496568`, and `memory_action_benefit_pass=false`.
The reused audit reported `oracle_labels_collapsed=false` and primary root cause
`memory_delta_too_small_for_action`, with `model_bev_collapsed=true`.

Blockers/risks: OpenLORIS remains local PoC only and not product-training
approved. The learned v5 scorer still collapses on model-BEV shadow decisions,
even though labels are non-collapsed. All artifacts remain replay/eval only,
not executed, not control-safe, and no `cmd_vel` or raw PWM was emitted.

Recommended next goal: option (b). Memory still does not help on non-collapsed
labels, so current data/model signal is the bottleneck. Widen robot-frame data
before further memory work or policy claims: add more diverse owned/public
robot-frame logs, rebuild v5 labels, and rerun scorer plus route-out/scene-out
shadow eval.

### 023 - Goal 13B replay-only Goal 13A collapse diagnosis

Summary: diagnosed the old Goal 13A collapse as `oracle_labels_collapsed` with
tie/order dominance; this is resolved only for the new v5 path. Artifacts:
`runs/goal13b_policy_collapse_audit.json`, `.md`, and `.ppm`.

### 022 - Goal 13A SpatialMemoryV1 memory-to-trajectory shadow evaluation

Summary: transparent scorer memory-action gate failed before v5 labels. This
remains useful as the old synthetic-oracle baseline. Artifacts:
`runs/goal13a_memory_policy_shadow_eval_report.json`, `.md`, decisions JSONL,
and contact sheet.

### 021 - Goal 12C OpenLORIS PoC policy and SpatialMemoryV1 hard validation

Summary: hard v1 spatial-memory validation passed for replay/eval representation
pretraining, including deployment-style, hidden-cell, occlusion, pose ablation,
and true leave-one-route-out checks. Artifacts:
`runs/goal12c_spatial_memory_v1_hard_validation_report.json` and `.md`.

### 020 - Goal 12B SpatialMemoryNetV1 parity and memory-sanity repair

Summary: repaired v1 current-BEV parity and showed route-pose memory benefit in
spatial eval. Artifacts: `runs/goal12b_spatial_memory_v1_parity_report.json`
and `.md`.

### 019 - Goal 12A SpatialMemoryNetV1 temporal egocentric memory

Summary: added the first explicit v1 temporal-memory infrastructure. The tiny
PoC failed the memory-benefit gate against the stronger v0 baseline, which led
to Goal 12B.
