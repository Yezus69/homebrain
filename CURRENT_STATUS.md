# CURRENT_STATUS.md

Codex must update this file at the end of every goal. Historical goal logs older
than Goal 12 are archived under `docs/status_archive/` and are not part of the
default read-list.

## Current objective

Goal 16A completed: the repo now has one active production-directed owned-frame
geometry probe command. It imports owned frames, attempts the requested real or
fake scene teacher, conditionally runs QA/audit/visual review only when upstream
artifacts exist, and writes `result.json` plus `result.md`.

## Last completed goal

Goal 16A: contract the repo around one production geometry-probe path. The real
owned MoGe probe currently answers `BLOCKED_MISSING_TEACHER_SETUP` because this
workspace still lacks local MoGe assets or `HOMEBRAIN_MOGE_ADAPTER`.

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
- Goal 15C split scene-teacher signal audit outcomes into `review_only`,
  `single_frame_geometry_pretrain_candidate`,
  `temporal_memory_pretrain_candidate`, and `blocked`. Single-frame geometry
  pretraining no longer requires temporal extrinsics; temporal-memory promotion
  requires real teacher extrinsics or route pose/odometry evidence.
- Goal 15C imported `data/inbox/room_walk_001/frames` to
  `runs/goal15c_room_walk_001_route` with `owned_or_license_approved=true`, but
  real MoGe did not run because local MoGe assets/adapters are missing.
- Goal 16A added `homebrain.tools.run_owned_geometry_probe` as the single active
  operator command for the current production geometry path. It wraps
  image-sequence ingest, scene teacher run, scene-teacher QA, signal audit,
  visual review, and final result reporting with explicit skip behavior when
  upstream artifacts do not exist.
- Policy, scorer, and memory-action goals are frozen until a real MoGe or VGGT
  SceneTeacherPack exists on owned or license-approved frames and passes signal
  audit beyond review-only.
- Owned image/video route metadata can now explicitly record
  `owned_or_license_approved`; the audit blocks missing approval, invented
  IMU/odom/command streams, and robot-frame truth claims without measured
  camera-to-base plus base pose/odom.
- Active blockers are tracked first in `BLOCKERS.md`; older non-active blockers
  are separated below them for history.

## Goal completion log

### 028 - Goal 16A single owned geometry production probe

Objective attempted: stop feature sprawl by adding one production-directed
vertical command that answers whether HomeBrain can produce a real spatial
supervision artifact from owned indoor frames today, without training any model.

Files changed: added `homebrain/tools/run_owned_geometry_probe.py`; updated
`tests/test_goal15b_scene_teacher_signal.py`, `EVALS.md`, `BLOCKERS.md`, and
`CURRENT_STATUS.md`.

Commands run: required context reads; `python -m py_compile
homebrain\tools\run_owned_geometry_probe.py`; production probe `python -m
homebrain.tools.run_owned_geometry_probe --frames
data\inbox\room_walk_001\frames --out runs\goal16a_owned_geometry_probe_moge_real
--camera front_rgb --fps 10 --teacher moge --backend real
--owned-or-license-approved`; targeted tests `python -m pytest
tests\test_goal15b_scene_teacher_signal.py -q`; full tests `python -m pytest -q`.

Pass/fail results: py_compile passed. The production probe passed as a command
and wrote a final result with `status=BLOCKED_MISSING_TEACHER_SETUP`; this is the
correct answer for this workspace because real MoGe setup is absent. Targeted
tests passed with `7 passed`. Full pytest passed with `97 passed`.

Artifacts created: `runs/goal16a_owned_geometry_probe_moge_real/result.json`,
`runs/goal16a_owned_geometry_probe_moge_real/result.md`, and
`runs/goal16a_owned_geometry_probe_moge_real/route/` with copied approved image
frames and route metadata. No real MoGe SceneTeacherPack, QA JSON, signal audit,
or visual review artifact was created.

Metrics observed: the probe imported `350` frames with `0` image load errors and
`owned_or_license_approved=true`. The scene teacher run was attempted with
`teacher=moge` and `backend=real`, then stopped before artifacts. Exact missing
setup fields in `result.json`: `HOMEBRAIN_MOGE_ADAPTER`,
`external/moge`, and `external/moge/checkpoints/moge.pt`.

Blockers/risks: active path is now the single probe command, but real teacher
signal is still unavailable in this workspace. Fake backend tests produce only
`REVIEW_ONLY_NOT_TRAINABLE`. No model was trained, no new teacher type, policy,
scorer, dataset, SAM2, ROS, Nav2, Isaac, Habitat, sim, `cmd_vel`, or raw PWM was
added.

Recommended next goal: install or point to a local MoGe checkout/checkpoint or a
HomeBrain-compatible `HOMEBRAIN_MOGE_ADAPTER=module:function`, then rerun the
same Goal 16A command and inspect the resulting QA, signal audit, and visual
review artifacts if real teacher artifacts are produced.

### 027 - Goal 15C real MoGe owned-route setup check

Objective attempted: run real MoGe on owned indoor frames and decide whether the
teacher path is worth continuing. The goal stopped at the required setup blocker
because the owned route exists but real MoGe assets do not.

Files changed: updated `homebrain/tools/audit_scene_teacher_signal.py`,
`tests/test_goal15b_scene_teacher_signal.py`, `CURRENT_STATUS.md`, `EVALS.md`,
`BLOCKERS.md`, and `LICENSE_AUDIT.md`.

Commands run: required context reads; `python -m
homebrain.ingest.image_sequence --frames data\inbox\room_walk_001\frames --out
runs\goal15c_room_walk_001_route --camera front_rgb --fps 10
--owned-or-license-approved`; `python -m
homebrain.teachers.run_scene_teacher --teacher moge --backend real --log
runs\goal15c_room_walk_001_route --out
runs\goal15c_room_walk_001_route\teacher_artifacts\moge_scene_v0_real`; local
checks for `external\moge`, `external\models`, and `HOMEBRAIN_MOGE*`;
`python -m py_compile homebrain\tools\audit_scene_teacher_signal.py
tests\test_goal15b_scene_teacher_signal.py`; targeted tests `python -m pytest
tests\test_goal15b_scene_teacher_signal.py -q`; full `python -m pytest -q`.

Pass/fail results: owned-route import passed with `350` frames and `0` image load
errors. Real MoGe did not run; the CLI reported that it requires a local
`external/moge` checkout via `--model-dir` or `HOMEBRAIN_MOGE_DIR`, or
`HOMEBRAIN_MOGE_ADAPTER=module:function`, and that HomeBrain does not clone
repositories during teacher runs. No real SceneTeacherPack exists, so
scene-teacher QA, signal audit, and visual review were not run. Py_compile
passed. Targeted tests passed with `5 passed`. Full pytest passed with
`95 passed`.

Artifacts created: `runs/goal15c_room_walk_001_route/` with copied approved
image frames and route metadata. No real MoGe teacher artifact, QA JSON, audit
JSON/Markdown, or visual contact sheet was created.

Metrics observed: imported route metadata reports `frame_count=350`,
`imported_frame_count=350`, `width=1920`, `height=1080`,
`owned_or_license_approved=true`, `has_imu=false`,
`has_wheel_odometry=false`, `has_commands=false`, `has_intrinsics=false`, and
`scale_source=unknown_image_only`. The failed real MoGe output produced no depth,
confidence, floor, obstacle, scale, or temporal metrics.

Blockers/risks: the real teacher path remains undecidable in this workspace. The
missing setup is exact: no `external/moge`, no MoGe checkpoint under
`external/models`, and no `HOMEBRAIN_MOGE_ADAPTER`, `HOMEBRAIN_MOGE_DIR`, or
`HOMEBRAIN_MOGE_CHECKPOINT` environment variable. No model was trained, no SAM2,
ROS, Nav2, Isaac, Habitat, sim, `cmd_vel`, or raw PWM was added.

Recommended next goal: install or point to a local MoGe checkout/checkpoint or a
HomeBrain-compatible `HOMEBRAIN_MOGE_ADAPTER=module:function`, rerun the same
owned route with the real backend, then produce SceneTeacherPack QA,
`audit_scene_teacher_signal`, and an RGB/depth/confidence/floor/obstacle visual
review artifact before deciding whether the teacher path is worth continuing.

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
