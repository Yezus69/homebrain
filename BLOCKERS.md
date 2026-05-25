# BLOCKERS.md

## Active Blockers

### 2026-05-24 - Goal 16B real MoGe probe blocked by missing MoGe import/model setup

Exact failure: `python -m homebrain.tools.run_owned_geometry_probe --frames
data\inbox\room_walk_001\frames --out runs\goal16b_moge_real_probe --camera
front_rgb --fps 10 --teacher moge --backend real --owned-or-license-approved
--max-frames 20 --device cuda` wrote
`runs\goal16b_moge_real_probe\result.json` with
`status=BLOCKED_MISSING_TEACHER_SETUP`. The scene-teacher step reported: `real
MoGe scene adapter failed: MoGe is not installed or importable. Install the
official MoGe package/check out the official MoGe repository in this environment,
or set HOMEBRAIN_MOGE_DIR so Python can import moge.model.v2.MoGeModel. Then
provide a local checkpoint path, set HOMEBRAIN_MOGE_MODEL_ID, or set
HOMEBRAIN_MOGE_ALLOW_DOWNLOAD=1 to allow the default MoGe-2 small model.`

Likely cause: HomeBrain now has an official adapter at
`homebrain.teachers.moge_official_adapter:run_scene_teacher`, but this workspace
does not have an importable MoGe package/check out exposing
`moge.model.v2.MoGeModel`. No operator-approved checkpoint or model source is
configured either.

Minimal next repair action: install or clone the official MoGe package outside
HomeBrain's run path or set `HOMEBRAIN_MOGE_DIR` to an importable checkout, then
set one model source: `HOMEBRAIN_MOGE_MODEL_ID`, `HOMEBRAIN_MOGE_ALLOW_DOWNLOAD=1`
for `Ruicheng/moge-2-vits-normal`, or `HOMEBRAIN_MOGE_CHECKPOINT` / `--checkpoint`
for a local checkpoint. Then rerun the Goal 16B probe command. Do not tune
policies, scorers, or memory-action goals until real teacher signal exists and
passes audit beyond review-only.

Command output summary: the probe imported `20` owned frames with `0` image load
errors and `owned_or_license_approved=true`. No fake fallback was used. No real
SceneTeacherPack exists, so SceneTeacherPack QA, signal audit, and visual review
were intentionally skipped.

## Historical / Non-active Blockers

The entries below are retained for audit context. They are not the active
production path. Goal 16A freezes policy/scorer/memory-action work until the
active real scene-teacher blocker above is repaired.

## 2026-05-24 - Goal 10B TUM Pioneer robot-frame BEV blocked by missing transform semantics

Exact failure: `python -m homebrain.geometry.robot_rgbd_to_bev --log runs/tum_freiburg2_pioneer_slam_route --out runs/tum_freiburg2_pioneer_slam_route/geometry/robot_rgbd_bev` refused to write robot-frame BEV: `camera_to_base transform is missing; rerun with --review-assumed-extrinsics only for review artifacts marked robot_frame_truth=false`.

Likely cause: the public TUM RGB-D `freiburg2_pioneer_slam` archive provides RGB, depth, groundtruth camera pose, and accelerometer samples, but the HomeBrain importer has no non-assumed camera-to-base transform and no robot/base pose semantics for the route. The imported groundtruth is preserved as dataset camera pose, not promoted to robot base pose.

Minimal next repair action: locate official TUM Pioneer camera-to-base/base-link calibration and robot/base pose semantics if they exist, or use a dataset/log with measured camera-to-base and base odometry. Do not use `--review-assumed-extrinsics` for action labels; if it is used for review-only geometry, keep `robot_frame_truth=false`, `action_supervision_ok=false`, and `control_safe=false`.

Command output summary: TUM setup downloaded a `1.515` GB archive under `max_download_gb=2.0` with `large_download_approval_record.approved=true`; route import wrote `runs/tum_freiburg2_pioneer_slam_route` with 300 RGB-D frames, 300 camera-pose events, `has_accelerometer=true`, `has_camera_to_base_transform=false`, `has_robot_base_pose=false`, `has_wheel_odometry=false`, and `has_commands=false`. No TUM robot-frame BEV, TUM Pioneer SpatialTrainPack, TUM Pioneer action sanity JSON, or TUM Pioneer ActionLabelPack examples were generated.

## 2026-05-24 - Goal 11B OpenLORIS `home1-1_5` route blocked by missing intrinsics

Exact failure: the Goal 11B nightly selected `home1-1_5`, but `robot_rgbd_to_bev` refused to write robot-frame BEV with `no robot-frame BEV frames written; first warning: frame_0_failed:association/frame is missing camera intrinsics; refusing robot-frame BEV`.

Likely cause: the staged OpenLORIS `home1-1_5` package/import path did not expose camera intrinsics for the first RGB-D association in the format HomeBrain requires. Goal 11B intentionally did not invent intrinsics or transforms.

Minimal next repair action: inspect `home1-1_5` calibration files and importer mapping for camera intrinsics. Only rerun BEV/action supervision after intrinsics, camera-to-base, and base pose/odom are present without assumptions; otherwise keep the route out of robot-frame action supervision.

Command output summary: Goal 11B still completed the hard route criterion with three fully evaluated robot-frame routes: `cafe1-1_2` 1200 frames, `office1-1_7` 809 frames, and `corridor1-1` 1200 frames. The failed `home1-1_5` route is recorded in `runs/goal11b_nightly_report.json` and `runs/goal11b_nightly_report.md`; OpenLORIS license review remains pending.

## 2026-05-24 - Goal 11B true route-out generalization concentration diagnosed

Exact failure: no benchmark-completion failure, but true leave-one-route-out retraining still crossed the Goal 11B collapse threshold on held-out `corridor1-1` and `office1-1_7`. Corridor held out reported `action_entropy=0.7871934753607142` and `dominant_action_fraction=0.8666666666666667`; office held out reported `dominant_action_fraction=0.7033374536464772`.

Likely cause: three evaluated OpenLORIS routes are still too few and scene-specific for robust route-out action generalization, even though the full-data v4 label pack and 4x2 sweeps no longer collapse globally.

Minimal next repair action: add more robot-frame routes that pass intrinsics/transform/action sanity, then rerun v4 balancing, true leave-one-route-out, and leave-one-scene-out. Do not promote the current scorer beyond replay/eval.

Command output summary: ActionLabelPack v4 QA passed with `action_entropy=2.393928609048208` and `dominant_action_fraction=0.3997569866342649`; best full-data model-BEV scorer reported `top1_action_agreement=0.7878315132605305`, `action_entropy=1.8101558477925317`, `dominant_action_fraction=0.5585023400936038`, and `unsafe_selected_rate=0.0`. The residual route-out concentration is explicitly recorded in `collapse_diagnosis` in the Goal 11B report.

## 2026-05-24 - Goal 12A SpatialMemoryNet v1 memory-benefit gate not passed

Exact failure: `python -m homebrain.train.eval_spatial_v1 --checkpoint runs\goal12a_spatial_memory_v1_poc\checkpoint.pt --dataset runs\goal11b_nightly\spatial_packs\openloris_cafe1_1_2_spatial_pack --features runs\goal11b_nightly\routes\openloris_cafe1_1_2_route\teacher_artifacts\dino --out runs\goal12a_spatial_memory_v1_eval.json --split val --window-length 2 --batch-size 32 --device cpu --baseline-checkpoint-v0 runs\goal11b_nightly\training\A_spatial_dino_bev_pose\seed_17\checkpoint.pt --report-json runs\goal12a_spatial_memory_v1_report.json --report-md runs\goal12a_spatial_memory_v1_report.md` wrote `memory_benefit_pass=false` with failure reason `current_bev_regressed_beyond_tolerance`.

Likely cause: the v1 checkpoint is a deliberately tiny 2-step PoC trained on one OpenLORIS route, while the comparison baseline is the much stronger Goal 11B v0 checkpoint trained in the full sweep. The v1 memory plumbing works and improves unknown reduction/temporal consistency, but its current-frame BEV head is undertrained and fused-memory IoU is not better than the v0 current-frame baseline.

Minimal next repair action: train SpatialMemoryNet v1 for a real sweep on multi-route temporal windows, compare against both v0 and a memory-disabled v1 baseline, tune the memory fusion loss/update alpha, and keep `memory_benefit_pass=false` until fused IoU or temporal consistency improves without current-BEV regression beyond tolerance.

Command output summary: v1 PoC train on OpenLORIS `cafe1-1_2` reported `train_loss_start=1.5515371986797877`, `train_loss_end=1.0521866317306245`, `val_loss=1.2494444052378337`, `memory_warp_valid_fraction=0.5`, and `memory_reset_fraction=0.5`. Eval reported `current_bev_iou_or_proxy=0.15507882038752238`, `fused_memory_bev_iou_or_proxy=0.14774751861890156`, `unknown_reduction_vs_current=0.2683714876572291`, `temporal_reprojection_consistency_iou=0.8004292050997416`, and v0 baseline `current_bev_iou_or_proxy=0.5605437725782394`. v1 modeld and replay each wrote 1200 BrainOutputEvents with `bad=0`, `resets=1`, `valid_warps=1199`, `cmd_vel=null`, and all safety flags `control_safe=false`, `replay_only=true`, `not_executed=true`, `product_training_approved=false`.

Resolution update: Goal 12B repaired the current-BEV parity blocker by adding unknown-prior memory initialization, conservative update-mask semantics, route pose/odom replay warp defaults, and shape-safe v0 warm start for the v1 encoder/current BEV head. The multi-route window1 memory-disabled eval matched the v0 baseline exactly on the same validation split (`0.6922143800184131` vs `0.6922143800184131`, `current_bev_parity_pass=true`). The window4 route-pose memory eval reported `fused_memory_bev_iou_or_proxy=0.9757767224311829`, `fused_memory_iou_delta=0.2297759104147553`, `temporal_consistency_delta=0.9620874587694804`, `update_mask_coverage_mean=0.1667717546224594`, `memory_overwrite_fraction=0.1193264801055193`, `pose_warp_source=route_pose_labels`, `valid_warp_fraction=0.75`, and `memory_benefit_pass=true`. No Goal 12B pass-gate blocker remains, but residual risks remain: OpenLORIS license review is pending, route-out metrics are per-source eval proxies rather than retrained leave-one-route-out models, and all artifacts remain replay/eval only with no `cmd_vel`.

## 2026-05-24 - Goal 13A memory-to-trajectory action benefit gate failed

Exact failure: `python -m homebrain.tools.goal13a_memory_policy_shadow_eval --batch-size 64` wrote `memory_action_benefit_pass=false`.

Likely cause: SpatialMemoryNetV1 fused memory improves some BEV uncertainty/unknown proxies but the transparent candidate scorer still collapses to one dominant action on the OpenLORIS validation routes. Memory does not improve oracle/future-motion agreement enough to overcome collapse, and it worsens the `corridor1-1` true route-out fold.

Minimal next repair action: do not train a learned memory-aware trajectory scorer yet. Inspect candidate-score distributions and BEV channels for current vs memory decisions, then add more robot-frame route diversity or recalibrate the transparent scorer so the Goal 11B collapse threshold is not crossed before rerunning Goal 13A.

Command output summary: normal eval covered `472` frame occurrences. V1 current reported `action_entropy=0.19897238107381565`, `dominant_action_fraction=0.972457627118644`, `unsafe_selected_rate=0.0`, `collision_proxy_rate=0.0`, `unknown_penalty_mean=0.01476968648069996`, `agreement_with_oracle_action=0.9766949152542372`, and `stop_fraction=0.0`. V1 memory reported `action_entropy=0.17555724310992266`, `dominant_action_fraction=0.9766949152542372`, `unsafe_selected_rate=0.0`, `collision_proxy_rate=0.0`, `unknown_penalty_mean=0.0125395347506313`, `agreement_with_oracle_action=0.972457627118644`, and `stop_fraction=0.0`. Memory-vs-current changed actions on `0.00847457627118644` of normal frames, improved `0.00211864406779661`, worsened `0.006355932203389831`, and had `memory_oracle_score_mean_delta=-0.0012516724861274286`. Route-out deltas were `cafe1-1_2=0.0016560370235119814`, `corridor1-1=-0.010013379889019429`, and `office1-1_7=0.0`. Failure reasons: normal, hidden-cell, and occlusion aggregate v1 memory crossed Goal 11B collapse thresholds; v1 memory worsened at least one true route-out fold.

## 2026-05-24 - Goal 13B action collapse root cause diagnosed

Exact failure: `python -m homebrain.policies.audit_goal13a_collapse --goal13a-decisions runs/goal13a_memory_policy_shadow_eval_decisions.jsonl --out-json runs/goal13b_policy_collapse_audit.json --out-md runs/goal13b_policy_collapse_audit.md --contact-sheet runs/goal13b_policy_collapse_worst.ppm` classified the Goal 13A collapse with primary root cause `oracle_labels_collapsed`. The Goal 13A action-benefit blocker remains active.

Likely cause: the transparent oracle/action-label path is itself collapsed and tie-order dominated. In normal mode, oracle actions are `rotate_left=450` and `straight_short=22` across 472 frames (`action_entropy=0.2718188297480972`, `dominant_action_fraction=0.9533898305084746`) while future-motion labels are diverse (`entropy=2.2840547173754913`, dominant label fraction `0.2711864406779661`) and oracle/future agreement is only `0.01694915254237288`. Exact top1-vs-top2 tie fractions are `0.9533898305084746` for oracle, `0.972457627118644` for v1 current, and `0.9766949152542372` for v1 memory; reversing candidate order changes the same fractions of decisions. Memory also changes scores but not actions on `0.3326271186440678` of frames, while action changes occur on only `0.00847457627118644`.

Minimal next repair action: repair action-label generation and candidate/scorer tie behavior before training any learned memory-aware scorer. After oracle labels are no longer collapsed and reversed candidate order no longer dominates, rerun Goal 13A and then Goal 13B. Only then decide whether SpatialMemory/data repair is still needed for `memory_delta_too_small_for_action`.

Command output summary: Goal 13B wrote `runs/goal13b_policy_collapse_audit.json`, `runs/goal13b_policy_collapse_audit.md`, and `runs/goal13b_policy_collapse_worst.ppm`. Root-cause buckets present were `oracle_labels_collapsed`, `candidate_set_too_weak`, `scorer_tie_break_dominates`, `model_bev_collapsed`, and `memory_delta_too_small_for_action`; absent buckets were `risk_or_unknown_dominates`, `coverage_gain_not_discriminative`, `route_out_data_too_small`, and `implementation_bug_suspected`. Safety flags remain `replay_only=true`, `not_executed=true`, `control_safe=false`, and `product_training_approved=false`; no `cmd_vel`, raw PWM, training, teacher, ROS, sim, or hardware control was added.

Resolution update: Goal 14 replaced the oracle label source with ActionLabelPack v5 future-motion behavior-cloning labels. The v5 label pack is non-collapsed and the reused audit on v5 decisions reports `oracle_labels_collapsed=false`, so this specific root cause is resolved for the v5 path.

## 2026-05-24 - Goal 14 v5 scorer memory-action benefit still blocked

Exact failure: `python -m homebrain.tools.goal13a_memory_policy_shadow_eval --batch-size 64 --action-label-pack runs\goal14_action_label_pack_v5 --scorer-checkpoint runs\goal14_trajectory_scorer_v1\checkpoint.pt ...` wrote `memory_action_benefit_pass=false`.

Likely cause: v5 labels are non-collapsed, but the learned scorer still maps v0/v1 model BEVs to the same action on the Goal 13A shadow frames. The reused audit now reports primary root cause `memory_delta_too_small_for_action` with `model_bev_collapsed=true`, not `oracle_labels_collapsed`.

Minimal next repair action: widen robot-frame data before more memory work or policy claims. Add more diverse robot-frame routes/logs, then rebuild v5 labels and train/evaluate the scorer under route-out/scene-out splits.

Command output summary: v5 QA passed with `action_entropy=2.3334583564283053`, `dominant_action_fraction=0.4644091223220456`, and synthetic-oracle agreement `0.0`. V5 scorer eval beat random with `top1_action_agreement=0.49568221070811747` and `distribution_collapse_flag=false`. Goal 14 shadow eval had normal memory action changed fraction `0.0` and no future-motion agreement gain; audit primary root cause was `memory_delta_too_small_for_action`.

## 2026-05-24 - Goal 15A optional real VGGT setup not configured

Exact failure: optional real-backend check
`python -m homebrain.teachers.run_scene_teacher --teacher vggt --backend real --log runs\goal15a_scene_teacher_fake_route --out runs\goal15a_scene_teacher_fake_route\teacher_artifacts\scene_v0_real_optional`
reported `VGGT scene teacher unavailable: Real VGGT backend requires a local
external/vggt checkout or --model-dir. HomeBrain does not clone repositories or
download weights during teacher runs.`

Likely cause: this workspace does not have a local `external/vggt` checkout,
local checkpoint, or `HOMEBRAIN_VGGT_ADAPTER` callable configured. Goal 15A
intentionally made real foundation-model backends optional and kept tests on the
fake backend.

Minimal next repair action: install or clone the selected VGGT/MoGe/SAM2 teacher
outside normal tests, place checkpoints locally, set `HOMEBRAIN_VGGT_DIR`,
`HOMEBRAIN_VGGT_CHECKPOINT`, and `HOMEBRAIN_VGGT_ADAPTER=module:function` (or
pass `--model-dir`/`--checkpoint`), then rerun the real scene teacher on an owned
or license-approved route.

Command output summary: fake SceneTeacherPack v0 verification completed and full
pytest passed. The optional real VGGT unavailability is not a Goal 15A failure.
All fake and review BEV artifacts remain `replay_only=true`,
`not_executed=true`, `control_safe=false`, `product_training_approved=false`,
and no `cmd_vel` or raw PWM was emitted.

## 2026-05-24 - Goal 15B optional real MoGe/VGGT owned-route run not configured

Exact failure: optional real-backend checks on the Goal 15B owned fixture did not
run real foundation geometry inference. `python -m
homebrain.teachers.run_scene_teacher --teacher moge --backend real --log
runs\goal15b_scene_teacher_signal_fake_route --out
runs\goal15b_scene_teacher_signal_fake_route\teacher_artifacts\moge_scene_v0_real_optional`
reported that real MoGe requires a local `external/moge` checkout or
`HOMEBRAIN_MOGE_DIR`, or `HOMEBRAIN_MOGE_ADAPTER=module:function`, and that
HomeBrain does not clone repositories during teacher runs. `python -m
homebrain.teachers.run_scene_teacher --teacher vggt --backend real --log
runs\goal15b_scene_teacher_signal_fake_route --out
runs\goal15b_scene_teacher_signal_fake_route\teacher_artifacts\vggt_scene_v0_real_optional`
reported that real VGGT requires a local `external/vggt` checkout or
`--model-dir`, and that HomeBrain does not clone repositories or download
weights during teacher runs.

Likely cause: this workspace has owned-looking inbox frames under
`data/inbox/room_walk_001/frames`, but it has no local `external/moge` checkout,
no local `external/vggt` checkout, no configured `HOMEBRAIN_MOGE_ADAPTER`,
`HOMEBRAIN_MOGE_DIR`, `HOMEBRAIN_MOGE_CHECKPOINT`,
`HOMEBRAIN_VGGT_ADAPTER`, `HOMEBRAIN_VGGT_DIR`, or
`HOMEBRAIN_VGGT_CHECKPOINT`.

Minimal next repair action: install the selected MoGe or VGGT repo/checkpoint
outside HomeBrain's teacher run path, set the matching environment variables or
pass `--model-dir` and `--checkpoint`, and provide
`HOMEBRAIN_MOGE_ADAPTER=module:function` or
`HOMEBRAIN_VGGT_ADAPTER=module:function` when the local package does not expose a
HomeBrain-compatible callable. Re-import the owned inbox frames with
`--owned-or-license-approved`, then rerun the real backend and
`audit_scene_teacher_signal`.

Command output summary: fake Goal 15B MoGe SceneTeacherPack and signal audit
passed on the tiny fixture with `next_allowed_use=review_only`. No real MoGe or
VGGT artifacts were written. This is optional setup, not a Goal 15B fake-backend
verification failure.
