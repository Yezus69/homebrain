# EVALS.md — Metrics and Gates

HomeBrain is not allowed to become a cool demo with no scorecard.

## Gate 0: log/replay spine

Required before ML work:
- `pytest -q` passes
- dummy log generation works
- deterministic replay works
- eval writes JSON metrics
- `CURRENT_STATUS.md` updated

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

## Gate 1: teacher artifact pipeline

Required before student training:
- teacher interface works with mocks
- artifacts can be written and loaded
- visualization is generated
- license audit updated
- no missing-weight crash unless user explicitly requested real run

Teacher metrics:
```text
teacher_artifact_count
teacher_mock_used
artifact_load_success
visualization_written
license_audit_updated
frames_with_teacher_artifacts
missing_artifact_count
artifact_shape_error_count
artifact_determinism_pass
depth_frame_count
depth_missing_count
depth_nan_count
depth_nonpositive_count
depth_shape_error_count
```

## Gate 2: SpatialMemoryNet v0

Required before trajectory learning:
- model can overfit a tiny dataset
- training script runs
- eval script writes metrics
- replay integration emits BEV/pose/uncertainty
- DINO-family features are frozen offline teacher artifacts, not a runtime dependency
- `trainable_candidate=true` and `structurally_trainable=true` do not imply robot control safety

Model metrics:
```text
train_loss_start
train_loss_end
loss_reduction_ratio
val_loss
bev_loss
bev_iou_or_proxy
pose_delta_rmse
uncertainty_calibration_proxy
inference_fps
```

Current DINO feature teacher verification:
```bash
python -m homebrain.teachers.run_teacher --teacher dino --backend fake --log runs/dummy_route --out runs/dummy_route/teacher_artifacts/dino_fake
python -m homebrain.tools.setup_dino_teacher --model-id dinov2_vits14 --device cuda
python -m homebrain.teachers.run_teacher --teacher dino --backend real --device cuda --model-id dinov2_vits14 --log runs/room_walk_001_route_short60 --out runs/room_walk_001_route_short60/teacher_artifacts/dino --image-size 224
```

DINO per-frame artifacts:
```text
patch_features.npy
cls_feature.npy
metadata.json
teacher_manifest.json
```

DINO manifest requirements:
```text
teacher_name=dino
model_id/model_path
feature_shape
license_review_status=pending_human_review
runtime_dependency=false
```

Current SpatialMemoryNet v0 commands:
```bash
python -m homebrain.train.train_spatial_v0 --dataset runs/room_walk_001_da3_stable_spatial_pack_short60 --features runs/room_walk_001_route_short60/teacher_artifacts/dino --out runs/spatial_v0_overfit --max-steps 300 --tiny-overfit
python -m homebrain.train.eval_spatial_v0 --checkpoint runs/spatial_v0_overfit/checkpoint.pt --dataset runs/room_walk_001_da3_stable_spatial_pack_short60 --out runs/spatial_v0_overfit_eval.json
python -m homebrain.brain.modeld --log runs/room_walk_001_route_short60 --checkpoint runs/spatial_v0_overfit/checkpoint.pt --features runs/room_walk_001_route_short60/teacher_artifacts/dino --out runs/spatial_v0_modeld_replay_short60
python -m homebrain.replay.replayd --log runs/room_walk_001_route_short60 --checkpoint runs/spatial_v0_overfit/checkpoint.pt --features runs/room_walk_001_route_short60/teacher_artifacts/dino --out runs/spatial_v0_replayed_short60
python -m homebrain.brain.visualize_spatial_outputs --log runs/spatial_v0_modeld_replay_short60 --out runs/spatial_v0_modeld_replay_short60_viz
```

Current Goal 7 outcome: real DINOv2-small artifacts exist for the short60 route with `feature_shape=[16,16,384]`; tiny overfit trained on 8 examples and wrote `runs/spatial_v0_overfit/checkpoint.pt`; train loss dropped from `0.6873307228088379` to `0.013037266675382853` with `loss_reduction_ratio=0.9810320326987496`. Full-pack eval is a generalization check only for this tiny overfit and reported `val_loss=2.3066179419402033`, `bev_iou_or_proxy=0.6176380563061684`, `pose_delta_rmse=null`, and `inference_fps=74.37446661472443`. Modeld and replayd load the checkpoint and emit BEV/pose/uncertainty `BrainOutputEvent`s with `representation_pretraining_only=true` and `control_safe=false`.

## Gate 2.1: SpatialMemoryNet v0 real train/val source comparison

Required before trajectory scoring:
- deterministic sequence-aware train/val/review splits
- no direct train/val adjacency for contiguous frame windows
- multi-pack dataset manifests
- TUM RGB-D pose-delta labels with masks and a documented non-robot frame convention
- real DINO features for every training example
- DA3-only, TUM-only, and DA3+TUM mixed train/val runs
- per-source and combined eval JSON
- modeld/replayd checkpoint load on real routes
- prediction-vs-label contact sheets

Dataset manifest examples:
```bash
python -m homebrain.train.train_spatial_v0 --dataset-manifest runs/spatial_v0_goal7b_manifests/da3_only.json --out runs/spatial_v0_goal7b_da3_only --max-steps 60 --batch-size 8
python -m homebrain.train.train_spatial_v0 --dataset-manifest runs/spatial_v0_goal7b_manifests/tum_only.json --out runs/spatial_v0_goal7b_tum_only --max-steps 60 --batch-size 8
python -m homebrain.train.train_spatial_v0 --dataset-manifest runs/spatial_v0_goal7b_manifests/da3_tum_mixed.json --out runs/spatial_v0_goal7b_da3_tum_mixed --max-steps 60 --batch-size 8
```

Current Goal 7B results:
```text
DA3-only: train_loss_start=0.6888465762138367, train_loss_end=0.07643128633499145, val_loss=0.155940979719162, bev_iou_or_proxy=0.7961926609277725, pose_delta_rmse=null
TUM-only: train_loss_start=0.6987095204266635, train_loss_end=0.1375339532440359, val_loss=0.17899657785892487, bev_iou_or_proxy=0.6982371807098389, pose_delta_rmse=0.027870545917272272
DA3+TUM mixed: train_loss_start=0.6956273503601551, train_loss_end=0.14402997074648738, val_loss=0.19234523177146912, bev_iou_or_proxy=0.7420604333281517, pose_delta_rmse=0.04466951437836996
Mixed per-source val: DA3 val_loss=0.2110961526632309, TUM val_loss=0.1939407934745153
```

Pose label convention:
```text
pose_label_frame=camera_relative_dataset_pose
transform=inv(T_dataset_camera_current) @ T_dataset_camera_next
components=[camera_relative_x_m, camera_relative_z_m_stored_as_dy, relative_yaw_about_camera_y_rad]
not_robot_odometry=true
control_safe=false
```

Replay/modeld verification:
```bash
python -m homebrain.brain.modeld --log runs/room_walk_001_route_short60 --checkpoint runs/spatial_v0_goal7b_da3_tum_mixed/checkpoint.pt --features runs/room_walk_001_route_short60/teacher_artifacts/dino --out runs/spatial_v0_goal7b_modeld_short60
python -m homebrain.replay.replayd --log runs/room_walk_001_route_short60 --checkpoint runs/spatial_v0_goal7b_da3_tum_mixed/checkpoint.pt --features runs/room_walk_001_route_short60/teacher_artifacts/dino --out runs/spatial_v0_goal7b_replayed_short60
python -m homebrain.brain.modeld --log runs/tum_freiburg1_xyz_route --checkpoint runs/spatial_v0_goal7b_da3_tum_mixed/checkpoint.pt --features runs/tum_freiburg1_xyz_route/teacher_artifacts/dino --out runs/spatial_v0_goal7b_modeld_tum
python -m homebrain.replay.replayd --log runs/tum_freiburg1_xyz_route --checkpoint runs/spatial_v0_goal7b_da3_tum_mixed/checkpoint.pt --features runs/tum_freiburg1_xyz_route/teacher_artifacts/dino --out runs/spatial_v0_goal7b_replayed_tum
```

All Goal 7B checkpoints and BrainOutputEvents remain `representation_pretraining_only=true` and `control_safe=false`. There is still no trajectory scorer.

## Gate 3: trajectory scorer

Required before control integration:
- candidate trajectories generated deterministically
- scorer ranks candidates
- unsafe candidates can be penalized
- coverage/risk/debug outputs visible in replay
- every decision artifact is marked `replay_only=true`, `control_safe=false`, and `not_executed=true`
- no raw PWM or hardware-control command is emitted

Trajectory metrics:
```text
candidate_count
selected_candidate_id
selected_risk_score
selected_coverage_gain_proxy
selected_unknown_penalty
selected_uncertainty_penalty
risky_candidate_fraction
stop_selected_fraction
coverage_memory_cells_seen
trajectory_eval_runtime_ms
```

Current Goal 8 commands:
```bash
python -m homebrain.policies.synthetic_fixture --out runs/goal8_synthetic_policy_fixture --count 6
python -m homebrain.policies.run_trajectory_scorer --log runs/goal8_synthetic_policy_fixture --bev-source labels --out runs/goal8_policy_synthetic
python -m homebrain.policies.run_trajectory_scorer --log runs/spatial_v0_goal7b_modeld_short60 --bev-source model --out runs/goal8_policy_modeld_short60
python -m homebrain.policies.run_trajectory_scorer --log runs/spatial_v0_goal7b_modeld_tum --bev-source model --out runs/goal8_policy_modeld_tum
python -m homebrain.policies.run_trajectory_scorer --log runs/room_walk_001_route_short60 --checkpoint runs/spatial_v0_goal7b_da3_tum_mixed/checkpoint.pt --features runs/room_walk_001_route_short60/teacher_artifacts/dino --bev-source model --out runs/goal8_policy_checkpoint_short60 --device cpu --max-frames 5
```

Current Goal 8 outputs:
```text
trajectory_decisions.jsonl
trajectory_eval.json
trajectory_overlay_contact_sheet.ppm
trajectory_policy_manifest.json
```

Current Goal 8 results:
```text
Synthetic fixture: frame_count=6, candidate_count=9, selected_candidate_id=straight_medium, risky_candidate_fraction=0.2222222222222222, selected_risk_score=0.0, selected_coverage_gain_proxy=3.5, stop_selected_fraction=0.0, coverage_memory_cells_seen=40
Short60 modeld: frame_count=60, candidate_count=9, selected_candidate_id=stop, risky_candidate_fraction=1.0, selected_risk_score=1.0, selected_coverage_gain_proxy=0.0, stop_selected_fraction=1.0, coverage_memory_cells_seen=966
TUM modeld: frame_count=120, candidate_count=9, selected_candidate_id=stop, risky_candidate_fraction=0.9064814814814814, selected_risk_score=0.8938543225328127, selected_coverage_gain_proxy=0.36666666666666664, stop_selected_fraction=0.8916666666666667, coverage_memory_cells_seen=932
Checkpoint+features smoke: frame_count=5, candidate_count=9, selected_candidate_id=stop, risky_candidate_fraction=1.0, selected_risk_score=1.0, stop_selected_fraction=1.0, coverage_memory_cells_seen=1002
```

Gate interpretation: Goal 8 trajectory scoring is replay/debug/eval only. Stop-heavy decisions on current SpatialMemoryNet outputs are a model/data signal, not a control-safety claim.

## Gate 3.1: stop-heavy audit and ActionLabelPack v0

Required before learned trajectory scoring:
- stop-heavy behavior is audited before any learned scorer is trained
- model-BEV policy behavior is compared with label-BEV policy behavior where accepted packs exist
- controlled indoor-like BEV grids exist for deterministic action supervision
- ActionLabelPack v0 stores per-candidate expert proxy scores, not executed controls
- QA reports label balance, source distribution, deterministic hash, and safety flags
- every action-label artifact is marked `replay_only=true`, `not_executed=true`, and `control_safe=false`
- no control-safe claim, raw PWM, ROS/Nav2/Isaac/Habitat, SAM, NoMaD, or ViNT integration is added

Stop audit metrics:
```text
stop_selected_fraction
selected_motion_fraction
risky_candidate_fraction
occupied_hit_rate
unknown_hit_rate
uncertainty_penalty_mean
coverage_gain_mean
candidate_footprint_block_rate
bev_channel_histograms
likely_root_causes
```

ActionLabelPack QA metrics:
```text
example_count
candidate_count_mean
selected_stop_fraction
selected_motion_fraction
collision_positive_rate
coverage_gain_mean
action_entropy
source_distribution
source_selected_distribution
source_selected_stop_fraction
source_selected_motion_fraction
deterministic_hash
```

Current Goal 9A commands:
```bash
python -m homebrain.policies.run_trajectory_scorer --log runs/room_walk_001_da3_stable_spatial_pack_short60 --bev-source labels --out runs/goal9_policy_labels_da3
python -m homebrain.policies.run_trajectory_scorer --log runs/tum_freiburg1_xyz_rgbd_truth_spatial_pack --bev-source labels --out runs/goal9_policy_labels_tum
python -m homebrain.policies.audit_stop_heavy --policy runs/goal8_policy_modeld_short60 --out runs/goal9_stop_audit_short60 --labels runs/room_walk_001_da3_stable_spatial_pack_short60 --modeld runs/spatial_v0_goal7b_modeld_short60
python -m homebrain.policies.audit_stop_heavy --policy runs/goal8_policy_modeld_tum --out runs/goal9_stop_audit_tum --labels runs/tum_freiburg1_xyz_rgbd_truth_spatial_pack --modeld runs/spatial_v0_goal7b_modeld_tum
python -m homebrain.policies.generate_controlled_bev_maps --out runs/goal9_controlled_bev_maps --meters-per-cell 0.05
python -m homebrain.policies.build_action_label_pack --source runs/goal9_controlled_bev_maps --source runs/room_walk_001_da3_stable_spatial_pack_short60 --source runs/tum_freiburg1_xyz_rgbd_truth_spatial_pack --out runs/goal9_action_label_pack_v0
python -m homebrain.policies.qa_action_label_pack --pack runs/goal9_action_label_pack_v0 --out runs/goal9_action_label_pack_v0_qa.json
```

Current Goal 9A results:
```text
Short60 model-BEV audit: stop_selected_fraction=1.0, selected_motion_fraction=0.0, risky_candidate_fraction=1.0, occupied_hit_rate=1.0, unknown_hit_rate=0.004166666666666667, candidate_footprint_block_rate=1.0. Likely causes: most motion candidates marked risky, candidate footprints overlap occupied/risky cells, little/no coverage gain, dense occupied/risky BEV.
Short60 model-vs-DA3-label comparison: overlap_count=59, agreement_fraction=1.0, model_stop_selected_fraction=1.0, label_stop_selected_fraction=1.0.
TUM model-BEV audit: stop_selected_fraction=0.8916666666666667, selected_motion_fraction=0.10833333333333334, risky_candidate_fraction=0.9064814814814818, occupied_hit_rate=0.9083333333333333, unknown_hit_rate=1.0, candidate_footprint_block_rate=1.0. Likely causes: motion candidates marked risky, occupied/risky hits, unknown hits, dense occupied/risky BEV.
TUM model-vs-label comparison: overlap_count=120, agreement_fraction=0.8916666666666667, model_stop_selected_fraction=0.8916666666666667, label_stop_selected_fraction=0.8583333333333333, model_stop_label_motion_fraction=0.03333333333333333.
Label-BEV scorer: DA3 stable pack stop_selected_fraction=1.0; TUM RGB-D truth pack stop_selected_fraction=0.8583333333333333.
ActionLabelPack QA: example_count=1299, candidate_count_mean=9.0, selected_stop_fraction=0.123941493456505, selected_motion_fraction=0.876058506543495, collision_positive_rate=0.2613121204345223, coverage_gain_mean=32.44256265503379, action_entropy=2.0299437511026057, action_label_pack_qa_pass=true. Controlled open_room selected_stop_fraction=0.0 and selected_motion_fraction=1.0; DA3 reviewed source selected_stop_fraction=1.0; TUM reviewed source selected_stop_fraction=0.85.
```

Gate interpretation: Goal 9A is an audit and deterministic action-labeling gate only. Stop-heavy model and label behavior is not hidden, and ActionLabelPack v0 is replay-only supervision for later experiments, not a learned policy or control evidence.

## Gate 3.2: robot-frame BEV/action sanity repair

Required before learned trajectory scoring:
- BEV action contract documents channels, origin, footprint, frame types, and action truth flags
- DA3/phone and TUM/public RGB-D BEVs are not treated as robot-frame action truth by default
- BEV action sanity reports center/footprint/corridor/candidate block reasons per source and frame
- derived traversability views preserve raw labels and remain `derived=true`, `review_required=true`, and `control_safe=false`
- scorer calibration sweeps only transparent knobs and refuses configs that fail controlled open/blocked gates
- ActionLabelPack v1 includes `source_weight` and `action_supervision_ok` and excludes/flags bad action frames
- old-vs-calibrated policy comparison marks stop-heavy reviewed/modeld sources as geometry-only for action learning

BEV action sanity metrics:
```text
robot_center_blocked_rate
footprint_blocked_rate
forward_corridor_free_rate
occupied_ratio_near_robot
unknown_ratio_near_robot
risky_ratio_near_robot
candidate_block_reason_counts
origin_frame_status_distribution
action_supervision_ok_fraction
```

Current Goal 9B commands:
```bash
python -m homebrain.policies.audit_bev_action_sanity --source runs/goal9_controlled_bev_maps --out runs/goal9b_action_sanity_controlled
python -m homebrain.policies.audit_bev_action_sanity --source runs/room_walk_001_da3_stable_spatial_pack_short60 --out runs/goal9b_action_sanity_da3
python -m homebrain.policies.audit_bev_action_sanity --source runs/tum_freiburg1_xyz_rgbd_truth_spatial_pack --out runs/goal9b_action_sanity_tum
python -m homebrain.policies.build_traversability_view --source runs/room_walk_001_da3_stable_spatial_pack_short60 --out runs/goal9b_traversability_da3
python -m homebrain.policies.build_traversability_view --source runs/tum_freiburg1_xyz_rgbd_truth_spatial_pack --out runs/goal9b_traversability_tum
python -m homebrain.policies.sweep_scorer_config --source runs/goal9_controlled_bev_maps --source runs/room_walk_001_da3_stable_spatial_pack_short60 --source runs/tum_freiburg1_xyz_rgbd_truth_spatial_pack --out runs/goal9b_scorer_sweep
python -m homebrain.policies.build_action_label_pack --source runs/goal9_controlled_bev_maps --source runs/room_walk_001_da3_stable_spatial_pack_short60 --source runs/tum_freiburg1_xyz_rgbd_truth_spatial_pack --out runs/goal9b_action_label_pack_v1 --pack-version 1
python -m homebrain.policies.qa_action_label_pack --pack runs/goal9b_action_label_pack_v1 --out runs/goal9b_action_label_pack_v1_qa.json
python -m homebrain.policies.compare_scorer_configs --source runs/room_walk_001_da3_stable_spatial_pack_short60 --source runs/tum_freiburg1_xyz_rgbd_truth_spatial_pack --source runs/spatial_v0_goal7b_modeld_short60 --source runs/spatial_v0_goal7b_modeld_tum --config runs/goal9b_scorer_sweep/scorer_sweep.json --out runs/goal9b_policy_comparison
```

Current Goal 9B results:
```text
Controlled action sanity: action_supervision_ok_fraction=0.8089285714285714, open_room action_supervision_ok_fraction=1.0, robot_center_blocked_rate=0.0, footprint_blocked_rate=0.19107142857142856.
DA3 action sanity: action_supervision_ok_fraction=0.0, robot_center_blocked_rate=1.0, footprint_blocked_rate=1.0, forward_corridor_free_rate=0.0, origin_frame_status=not_robot_frame_truth_phone_teacher_geometry.
TUM action sanity: action_supervision_ok_fraction=0.0, robot_center_blocked_rate=1.0, footprint_blocked_rate=1.0, forward_corridor_free_rate=0.0, origin_frame_status=not_robot_frame_truth_public_rgbd_camera_pose.
Scorer sweep selected config: unknown_weight=0.8, risk_weight=12.0, obstacle_threshold=0.35, footprint_radius_cells=3, obstacle_inflation_cells=1, stop_bias=10.0.
Scorer sweep gates: controlled_open_motion_rate=1.0, blocked_map_stop_rate=1.0, reviewed_motion_rate=0.09497206703910614, collision_proxy_rate=0.0, stop_fraction=0.41878367975365666.
ActionLabelPack v1 QA: example_count=906, excluded_frame_count=393, selected_stop_fraction=0.18322295805739514, selected_motion_fraction=0.8167770419426048, action_label_pack_qa_pass=true.
Old vs calibrated comparison: DA3 labels old/calibrated stop_fraction=1.0/1.0; TUM labels=0.8583333333333333/0.8583333333333333; short60 modeld=1.0/1.0; TUM modeld=0.8916666666666667/0.9. All four are marked geometry_only for action learning.
```

Gate interpretation: Goal 9B repairs the action contract and filtering path, not the real BEV labels themselves. Controlled maps are usable replay-only action-supervision proxies after sanity filtering; current DA3/TUM reviewed labels and modeld BEVs remain geometry-only for action learning until robot-frame origin/footprint semantics are repaired.

## Gate 3.3: public robot-frame dataset bridge

Required before learned trajectory scoring:
- public robot-mounted dataset setup/import adapters exist and record official source URLs plus license review status
- imported robot datasets write normal HomeBrain route logs with only available sensors, never faked sensors
- robot-frame BEV projection requires depth, intrinsics, camera-to-base transform, and robot/base pose or odometry evidence
- assumed extrinsics require explicit review override and must keep `robot_frame_truth=false` and `action_supervision_ok=false`
- public robot-mounted frames enter ActionLabelPack v2 only after `audit_bev_action_sanity` passes
- DA3/phone and TUM camera-pose geometry-only frames remain excluded from action supervision
- every action/policy/comparison output remains `replay_only=true`, `not_executed=true`, and `control_safe=false`

Public robot-frame commands:
```bash
python -m homebrain.datasets.setup_openloris_scene --out data/public/openloris_scene --sequence cafe1-1 --download --max-download-gb 2
python -m homebrain.datasets.setup_tum_rgbd --out data/public/tum_rgbd --sequence freiburg2_pioneer_slam --download --max-download-gb 0.25
python -m homebrain.policies.build_action_label_pack --source runs/goal9_controlled_bev_maps --source runs/room_walk_001_da3_stable_spatial_pack_short60 --source runs/tum_freiburg1_xyz_rgbd_truth_spatial_pack --out runs/goal10a_action_label_pack_v2 --pack-version 2
python -m homebrain.policies.qa_action_label_pack --pack runs/goal10a_action_label_pack_v2 --out runs/goal10a_action_label_pack_v2_qa.json
```

Robot-frame bridge implementation outputs:
```text
homebrain.datasets.setup_openloris_scene
homebrain.datasets.openloris_to_route
homebrain.geometry.robot_rgbd_to_bev
homebrain.policies.compare_robot_frame_action_sources
ActionLabelPack v2 schema: homebrain.action_label_pack.v2
```

Current Goal 10A results:
```text
OpenLORIS setup discovered the official Hugging Face package list and selected package/cafe1-1_2-package.tar, size 6.95 GB. The bounded setup blocked before download at max_download_gb=2.00 and wrote data/public/openloris_scene/openloris_scene_setup_status.json.
TUM Pioneer fallback resolved freiburg2_pioneer_slam to a 1.51 GB archive. The bounded setup blocked before download at max_download_gb=0.25 and wrote data/public/tum_rgbd/freiburg2_pioneer_slam/dataset_metadata.json.
ActionLabelPack v2 QA: example_count=906, excluded_frame_count=393, selected_stop_fraction=0.18322295805739514, selected_motion_fraction=0.8167770419426048, action_entropy=1.7035377080351073, action_label_pack_qa_pass=true.
Excluded geometry-only sources in v2: room_walk_001_route_short60=59, tum_freiburg1_xyz_route=120.
Full pytest: 51 passed.
```

Gate interpretation: Goal 10A adds the bridge and verifies the contract path with synthetic robot-frame fixtures, but no real public robot-mounted frames were imported under the bounded download caps. The next gate should stage/download a robot-mounted sequence or reduce the adapter to a smaller officially available package before learned scorer v0.

## Gate 3.4: real public robot-mounted action sanity

Required before learned trajectory scoring:
- large public dataset downloads require an explicit `--allow-large-download` approval record in setup metadata
- TUM/OpenLORIS imports preserve only available RGB/depth/pose/odom/IMU fields and masks, without faking camera-to-base, base pose, wheel odometry, commands, or IMU channels
- `robot_rgbd_to_bev` may run only when depth, intrinsics, camera-to-base, and base pose/odom semantics are present; missing transforms must produce an exact blocker instead of assumed truth
- real robot-frame public frames enter ActionLabelPack v3 only after `audit_bev_action_sanity` passes
- DA3/phone and TUM camera-pose geometry-only sources remain excluded or marked geometry-only for action learning
- every output remains `replay_only=true`, `not_executed=true`, and `control_safe=false`

Goal 10B commands:
```bash
python -m homebrain.datasets.setup_tum_rgbd --out data/public/tum_rgbd --sequence freiburg2_pioneer_slam --download --max-download-gb 2.0 --allow-large-download
python -m homebrain.datasets.tum_rgbd_to_route --source data/public/tum_rgbd/freiburg2_pioneer_slam --out runs/tum_freiburg2_pioneer_slam_route --max-frames 300
python -m homebrain.geometry.robot_rgbd_to_bev --log runs/tum_freiburg2_pioneer_slam_route --out runs/tum_freiburg2_pioneer_slam_route/geometry/robot_rgbd_bev
python -m homebrain.datasets.setup_openloris_scene --out data/public/openloris_scene --sequence cafe1-1_2 --download --max-download-gb 8.0 --allow-large-download
python -m homebrain.datasets.setup_openloris_scene --out data/public/openloris_scene --sequence cafe1-1_2 --package-file data/public/openloris_scene/_downloads/cafe1-1_2-package.tar --max-download-gb 8.0 --allow-large-download
python -m homebrain.datasets.openloris_to_route --source data/public/openloris_scene/cafe1-1_2 --out runs/openloris_cafe1_2_route --max-frames 300
python -m homebrain.geometry.robot_rgbd_to_bev --log runs/openloris_cafe1_2_route --out runs/openloris_cafe1_2_route/geometry/robot_rgbd_bev
python -m homebrain.data.pack_spatial_dataset --log runs/openloris_cafe1_2_route --bev runs/openloris_cafe1_2_route/geometry/robot_rgbd_bev --out runs/openloris_cafe1_2_robot_frame_spatial_pack
python -m homebrain.policies.audit_bev_action_sanity --source runs/openloris_cafe1_2_robot_frame_spatial_pack --out runs/goal10b_action_sanity_openloris_cafe1_2
python -m homebrain.policies.run_trajectory_scorer --log runs/openloris_cafe1_2_robot_frame_spatial_pack --bev-source labels --out runs/goal10b_policy_openloris_cafe1_2
python -m homebrain.policies.build_action_label_pack --source runs/goal9_controlled_bev_maps --source runs/openloris_cafe1_2_robot_frame_spatial_pack --source runs/room_walk_001_da3_stable_spatial_pack_short60 --source runs/tum_freiburg1_xyz_rgbd_truth_spatial_pack --out runs/goal10b_action_label_pack_v3 --pack-version 3
python -m homebrain.policies.qa_action_label_pack --pack runs/goal10b_action_label_pack_v3 --out runs/goal10b_action_label_pack_v3_qa.json
python -m homebrain.policies.compare_robot_frame_action_sources --source runs/openloris_cafe1_2_robot_frame_spatial_pack --source runs/room_walk_001_da3_stable_spatial_pack_short60 --source runs/tum_freiburg1_xyz_rgbd_truth_spatial_pack --label openloris_cafe1_2_robot_frame --label da3_phone_geometry_only --label tum_freiburg1_xyz_geometry_only --out runs/goal10b_robot_frame_source_comparison
```

Current Goal 10B results:
```text
TUM Pioneer downloaded 1.515 GB under max_download_gb=2.0 with explicit large-download approval, imported 300 RGB-D frames plus camera-pose events and accelerometer provenance, then correctly refused robot-frame BEV because camera_to_base and robot_base_pose semantics are missing. No assumed extrinsics were used.
OpenLORIS cafe1-1_2 downloaded/staged the 6.954 GB package under max_download_gb=8.0, extracted nested .7z archives with local 7-Zip, imported 300 RGB-D/IMU/odom/pose frames, and produced robot-frame BEV using dataset intrinsics, camera-to-base, and base pose/odom evidence.
OpenLORIS robot BEV validation: bev_frame_count=300, bev_missing_count=0, bev_shape_error_count=0, bev_nan_count=0, free_ratio_mean=0.0283203125, obstacle_ratio_mean=0.076953125, unknown_ratio_mean=0.8947265625, control_safe=false.
OpenLORIS action sanity: action_supervision_ok_fraction=1.0, robot_center_blocked_rate=0.0, footprint_blocked_rate=0.0, forward_corridor_free_rate=0.0, replay_only=true, not_executed=true, control_safe=false.
OpenLORIS label-BEV policy: frame_count=300, selected_candidate_id=rotate_left, risky_candidate_fraction=0.00037037037037037035, stop_selected_fraction=0.0, selected_risk_score=0.0, control_safe=false.
ActionLabelPack v3 QA: example_count=1206, OpenLORIS included=300, excluded_frame_count=393, excluded geometry-only sources room_walk_001_route_short60=59 and tum_freiburg1_xyz_route=120, selected_stop_fraction=0.13764510779436154, selected_motion_fraction=0.8623548922056384, action_entropy=2.0890729289226004, action_label_pack_qa_pass=true.
Source comparison: OpenLORIS action_supervision_ok_fraction=1.0; DA3 phone geometry=0.0; TUM freiburg1 XYZ geometry-only=0.0.
```

Gate interpretation: Goal 10B produces the first real public robot-mounted frames that pass the robot-frame action sanity contract, but they remain replay-only and `control_safe=false`. Learned scorer v0 is allowed only as a local research/replay experiment if OpenLORIS license risk is accepted; otherwise collect/stage owned robot-frame logs with measured camera-to-base and base odom/commands first.

## Gate 3.5: OpenLORIS robot-frame learned scorer v0

Required before any control-facing trajectory integration:
- real DINO features align with every OpenLORIS frame used by the robot-frame SpatialTrainPack
- SpatialMemoryNet refresh trains/evals on OpenLORIS robot-frame BEV and reports per-source metrics
- TrajectoryScorerNet v0 trains only from ActionLabelPack v3 action-supervision examples, not DA3/TUM geometry-only excluded frames
- scorer eval runs on both oracle label BEV and refreshed model-predicted BEV
- modeld/replayd can emit learned candidate trajectory scores when both spatial and scorer checkpoints are supplied
- every learned-scorer artifact remains `replay_only=true`, `not_executed=true`, `control_safe=false`, and `product_training_approved=false`
- distribution collapse is reported, not hidden

Goal 11A commands:
```bash
python -m homebrain.tools.setup_dino_teacher --model-id dinov2_vits14 --device cuda
python -m homebrain.teachers.run_teacher --teacher dino --backend real --device cuda --model-id dinov2_vits14 --log runs/openloris_cafe1_2_route --out runs/openloris_cafe1_2_route/teacher_artifacts/dino --image-size 224
python -m homebrain.train.train_spatial_v0 --dataset runs/openloris_cafe1_2_robot_frame_spatial_pack --features runs/openloris_cafe1_2_route/teacher_artifacts/dino --out runs/goal11a_spatial_openloris_refresh --max-steps 80 --batch-size 8 --device cuda
python -m homebrain.train.eval_spatial_v0 --checkpoint runs/goal11a_spatial_openloris_refresh/checkpoint.pt --dataset runs/openloris_cafe1_2_robot_frame_spatial_pack --features runs/openloris_cafe1_2_route/teacher_artifacts/dino --out runs/goal11a_spatial_openloris_refresh_eval.json --split val --batch-size 16 --device cuda
python -m homebrain.policies.train_trajectory_scorer_v0 --action-pack runs/goal10b_action_label_pack_v3 --source-name cafe1-1_2 --out runs/goal11a_trajectory_scorer_v0 --max-steps 200 --batch-size 32 --device cuda
python -m homebrain.policies.eval_trajectory_scorer_v0 --checkpoint runs/goal11a_trajectory_scorer_v0/checkpoint.pt --action-pack runs/goal10b_action_label_pack_v3 --source-name cafe1-1_2 --bev-source oracle --out runs/goal11a_trajectory_scorer_openloris_oracle_all_eval.json --viz-out runs/goal11a_trajectory_scorer_openloris_oracle_all_viz --split all --device cuda
python -m homebrain.brain.modeld --log runs/openloris_cafe1_2_route --checkpoint runs/goal11a_spatial_openloris_refresh/checkpoint.pt --features runs/openloris_cafe1_2_route/teacher_artifacts/dino --trajectory-scorer-checkpoint runs/goal11a_trajectory_scorer_v0/checkpoint.pt --out runs/goal11a_modeld_openloris_spatial_scorer --device cuda
python -m homebrain.replay.replayd --log runs/openloris_cafe1_2_route --checkpoint runs/goal11a_spatial_openloris_refresh/checkpoint.pt --features runs/openloris_cafe1_2_route/teacher_artifacts/dino --trajectory-scorer-checkpoint runs/goal11a_trajectory_scorer_v0/checkpoint.pt --out runs/goal11a_replayed_openloris_spatial_scorer --device cuda
python -m homebrain.policies.eval_trajectory_scorer_v0 --checkpoint runs/goal11a_trajectory_scorer_v0/checkpoint.pt --action-pack runs/goal10b_action_label_pack_v3 --source-name cafe1-1_2 --bev-source model --modeld runs/goal11a_modeld_openloris_spatial_scorer --out runs/goal11a_trajectory_scorer_openloris_model_bev_all_eval.json --viz-out runs/goal11a_trajectory_scorer_openloris_model_bev_all_viz --split all --device cuda
python -m homebrain.eval.run_eval --log runs/goal11a_replayed_openloris_spatial_scorer --out runs/goal11a_replayed_openloris_spatial_scorer_eval.json
python -m pytest -q
```

Current Goal 11A results:
```text
DINO OpenLORIS: frame_count=300, source_frame_count=300, backend=real, feature_shape=[16,16,384], mock=false, real_perception=true, runtime_dependency=false, control_safe=false.
Spatial refresh train: train_loss_start=0.7155767210892269, train_loss_end=0.04073466932667153, loss_reduction_ratio=0.9430743509030497, val_loss=0.11438632508118947, bev_iou_or_proxy=0.4798779853309194, pose_delta_rmse=null, uncertainty_calibration_proxy=0.03969770980377992, inference_fps=148.49809030822888, robot_supervision_grade=public_robot_frame_geometry.
Spatial standalone eval: val_loss=0.11612379550933838, bev_iou_or_proxy=0.47885076587166014, pose_delta_rmse=null, uncertainty_calibration_proxy=0.039988345156113304, inference_fps=59.89621184407719, failure_flags=["eval_loss_much_higher_than_train_loss"].
TrajectoryScorerNet v0 train: OpenLORIS train/val examples=240/60, train_loss_start=2.2950071692466736, train_loss_end=0.2735627815127373, val_loss=0.27464908361434937, val_top1_action_agreement=1.0, val_rank_correlation_or_proxy=0.778611111111111, beats_random=true.
Oracle label-BEV all-OpenLORIS eval: top1_action_agreement=1.0, selected_motion_fraction=1.0, selected_stop_fraction=0.0, action_entropy=0.0, dominant_action_fraction=1.0, rank_correlation_or_proxy=0.7823888888888874, collision_proxy_rate=0.0, coverage_gain_mean=21.0, unsafe_selected_rate=0.0, distribution_collapse_flag=true.
Model-BEV all-OpenLORIS eval: top1_action_agreement=1.0, selected_motion_fraction=1.0, selected_stop_fraction=0.0, action_entropy=0.0, dominant_action_fraction=1.0, rank_correlation_or_proxy=0.7044444444444451, collision_proxy_rate=0.0, coverage_gain_mean=21.0, unsafe_selected_rate=0.0, distribution_collapse_flag=true.
Replay with spatial+scorer checkpoint wrote 300 scored BrainOutputEvents with bad_flags=0; replay eval reported replay_determinism_pass=true.
Full pytest: 55 passed.
```

Gate interpretation: Goal 11A proves a replay-only learned scorer can train, save/load, eval on oracle/model BEV, and flow through modeld/replay. It is not a control policy. The OpenLORIS expert labels for the 300-frame subset collapse to `straight_short`, and the learned scorer inherits that collapse while reporting `distribution_collapse_flag=true`. OpenLORIS remains `CC BY-ND 4.0` with `pending_human_review`, so these artifacts are local research/replay outputs only.

## Gate 4: real-video spatial output

Required before hardware integration:
- real indoor video can be ingested
- model outputs BEV/risk/uncertainty overlays
- failure cases are logged
- performance is measured

Real-video metrics:
```text
video_frames_processed
overlay_written
avg_inference_fps
uncertain_frame_rate
dynamic_risk_event_count
manual_review_notes_present
```

## Standard verification commands

After Goal 0 these should exist or be created by Codex:

```bash
pytest -q
python -m homebrain.replay.generate_dummy_log --out runs/dummy_route
python -m homebrain.replay.replayd --log runs/dummy_route --out runs/replayed_route
python -m homebrain.eval.run_eval --log runs/dummy_route --out runs/dummy_eval.json
python -m homebrain.teachers.run_teacher --teacher mock --log runs/dummy_route --out runs/dummy_route/teacher_artifacts/mock_teacher
python -m homebrain.teachers.visualize_artifacts --artifacts runs/dummy_route/teacher_artifacts/mock_teacher --out runs/mock_teacher_viz
python -m homebrain.eval.run_eval --log runs/dummy_route --teacher-artifacts runs/dummy_route/teacher_artifacts/mock_teacher --out runs/dummy_eval_with_teacher.json
python -m homebrain.teachers.run_teacher --teacher depth_pro --backend fake --log runs/dummy_route --out runs/dummy_route/teacher_artifacts/depth_pro_fake
python -m homebrain.teachers.visualize_artifacts --artifacts runs/dummy_route/teacher_artifacts/depth_pro_fake --out runs/depth_pro_fake_viz
python -m homebrain.eval.run_eval --log runs/dummy_route --teacher-artifacts runs/dummy_route/teacher_artifacts/depth_pro_fake --out runs/dummy_eval_with_depth_pro_fake.json
```

As new features are added, Codex must update this file with exact current commands.

## Current teacher artifact validation

The mock teacher writes deterministic synthetic artifacts only:

```text
depth.npy
depth_confidence.npy
dense_features.npy
dynamic_mask.npy
bev_preview.npy
metadata.json
teacher_manifest.json
```

All mock teacher manifests and metadata must include `mock: true`, `synthetic: true`, and `real_perception: false`. These metrics validate artifact availability and format only; they are not perception/model performance metrics.

## Current Depth Pro teacher validation

The Depth Pro teacher is optional and defaults to `--backend real`. Real Depth Pro requires the external Apple `ml-depth-pro` package and local checkpoints installed by the operator. HomeBrain does not download weights automatically, and the normal test suite uses only `--backend fake`.

Real Depth Pro per-frame artifacts:

```text
depth_m.npy
depth_confidence.npy
focallength_px.npy
bev_preview.npy
metadata.json
teacher_manifest.json
```

Real Depth Pro manifests must include `teacher_name=depth_pro`, `mock=false`, `synthetic=false`, `real_perception=true`, `license_review_status=pending_human_review`, and `dependency_status`. Fake Depth Pro backend artifacts are for tests only and must be explicitly marked `mock: true`, `synthetic: true`, and `real_perception: false`.

Depth-specific eval metrics:

```text
depth_frame_count
depth_missing_count
depth_nan_count
depth_nonpositive_count
depth_shape_error_count
```

Depth Pro outputs are offline teacher artifacts for training/evaluation. They are not control-safe runtime dependencies or direct navigation labels.

## Gate 1.6: depth-to-BEV weak geometry labels

Depth Pro depth can be projected into local egocentric BEV weak labels for future SpatialMemoryNet training and trajectory scoring. This gate does not train a student model and does not claim traversability or control safety.

Required outputs:
```text
bev_free.npy
bev_obstacle.npy
bev_unknown.npy
bev_floor_candidate.npy
bev_height.npy
bev_confidence.npy
metadata.json
bev_manifest.json
```

Every BEV manifest and frame metadata must include:
```text
weak_label=true
control_safe=false
```

Required commands:
```bash
python -m homebrain.geometry.run_depth_to_bev --log runs/room_walk_001_route_short60 --depth-artifacts runs/room_walk_001_route_short60/teacher_artifacts/depth_pro --camera-config configs/camera/phone_robot_height_guess.json --out runs/room_walk_001_route_short60/geometry/depth_pro_bev
python -m homebrain.geometry.visualize_bev --bev runs/room_walk_001_route_short60/geometry/depth_pro_bev --out runs/room_walk_001_bev_viz_short60
python -m homebrain.geometry.validate_bev --bev runs/room_walk_001_route_short60/geometry/depth_pro_bev --out runs/room_walk_001_bev_eval_short60.json
```

Geometry eval metrics:
```text
bev_frame_count
bev_missing_count
bev_shape_error_count
bev_nan_count
free_ratio_mean
obstacle_ratio_mean
unknown_ratio_mean
confidence_mean
temporal_jitter_mean
```

Camera-config sweeps may report sanity/stability scores, but those scores are not ground truth and must not be used as control-safety evidence.

## Gate 1.7: BEV QA and SpatialTrainPack v0

Depth-to-BEV weak geometry labels can be packed into deterministic reviewed training-data candidates. This gate does not add new teachers, train SpatialMemoryNet, or claim label quality. QA must either expose acceptable structural metrics or quarantine low-quality labels.

Required package outputs:
```text
manifest.json
examples/*.npz
```

Each example must carry:
```text
frame_id
timestamp_ns
rgb_ref / rgb_path
bev_free
bev_obstacle
bev_unknown
bev_confidence
optional bev_height
optional bev_floor_candidate
provenance
weak_label=true
control_safe=false
camera_config_hash
teacher_manifest_hash
split=train|val|review
```

Required commands:
```bash
python -m homebrain.data.pack_spatial_dataset --log runs/room_walk_001_route_short60 --bev runs/room_walk_001_route_short60/geometry/depth_pro_bev --out runs/room_walk_001_spatial_pack_short60
python -m homebrain.data.qa_spatial_dataset --dataset runs/room_walk_001_spatial_pack_short60 --out runs/room_walk_001_spatial_pack_qa_short60.json
python -m homebrain.data.visualize_spatial_dataset --dataset runs/room_walk_001_spatial_pack_short60 --out runs/room_walk_001_spatial_pack_viz_short60
```

QA metrics:
```text
example_count
manifest_example_count
missing_count
shape_error_count
nan_count
weak_label_false_count
control_safe_true_count
free_ratio_mean/min/max
obstacle_ratio_mean/min/max
unknown_ratio_mean/min/max
confidence_mean/min/median/max
confidence_nonzero_ratio_mean
label_density_mean/min/median
observed_ratio_mean
visible_confidence_positive_ratio_mean
source_observed_ratio_mean
low_confidence_frame_count
empty_label_frame_count
label_overlap_cell_count
label_sum_error_cell_count
temporal_jitter_mean/p95
temporal_visible_jitter_mean/p95
temporal_label_flicker_mean/p95
trainable_candidate
quarantine_reasons
```

Gate interpretation:
- `missing_count`, `shape_error_count`, and `nan_count` must be zero for structural pass.
- `weak_label=true` and `control_safe=false` must remain true for every example.
- `trainable_candidate=false` is the correct result when confidence, observed/visible coverage, label density, or temporal stability is poor.
- Full-grid unknown ratio alone is not sufficient evidence for either acceptance or quarantine.
- Contact sheets are review aids only and are not ground-truth overlays.

## Gate 1.8: DA3 teacher and self-calibration QA

Depth Anything 3 can be used as an offline geometry/self-calibration teacher for real indoor routes or phone video. This gate does not train SpatialMemoryNet, does not mark phone video as robot-frame metric truth, and does not produce control-safe labels.

Setup command:
```bash
python -m homebrain.tools.setup_da3_teacher --external-dir external/depth-anything-3 --venv external/venvs/da3 --model-id depth-anything/DA3-SMALL --download
```

Fake backend verification:
```bash
python -m homebrain.teachers.run_teacher --teacher da3 --backend fake --log runs/dummy_route --out runs/dummy_route/teacher_artifacts/da3_fake
python -m homebrain.geometry.qa_self_calibration --log runs/dummy_route --teacher-artifacts runs/dummy_route/teacher_artifacts/da3_fake --out runs/dummy_da3_self_calib_qa.json
```

Real backend example:
```bash
python -m homebrain.teachers.run_teacher --teacher da3 --backend real --device cpu --model-id depth-anything/DA3-SMALL --max-frames 60 --window-size 10 --stride 1 --log runs/room_walk_001_route_short60 --out runs/room_walk_001_route_short60/teacher_artifacts/da3
python -m homebrain.geometry.qa_self_calibration --log runs/room_walk_001_route_short60 --teacher-artifacts runs/room_walk_001_route_short60/teacher_artifacts/da3 --out runs/room_walk_001_da3_self_calib_qa_short60.json
```

DA3 per-frame artifacts when available:
```text
depth.npy
confidence.npy
intrinsics.npy
extrinsics.npy
metadata.json
teacher_manifest.json
```

Manifest and metadata requirements:
```text
license_review_status=pending_human_review
calibration_class=teacher_estimated
intrinsics_source=teacher_estimated
extrinsics_source=teacher_estimated
pose_source=teacher_estimated_camera_pose
scale_source=teacher_relative_not_metric
not_robot_frame_truth=true
control_safe=false
```

Self-calibration QA metrics:
```text
frame_count
depth_valid_ratio
confidence_mean
intrinsics_valid_ratio
pose_valid_ratio
pose_jump_outlier_count
temporal_depth_consistency
floor_plane_found_ratio
floor_plane_stability
scale_source
promotable_to_weak_bev
quarantine_reasons
```

Optional weak BEV command:
```bash
python -m homebrain.geometry.da3_to_weak_bev --log runs/room_walk_001_route_short60 --teacher-artifacts runs/room_walk_001_route_short60/teacher_artifacts/da3 --qa runs/room_walk_001_da3_self_calib_qa_short60.json --out runs/room_walk_001_route_short60/geometry/da3_weak_bev
```

Weak BEV is written only when QA passes, unless `--force-review` is supplied. Any DA3-derived weak BEV must remain:
```text
weak_label=true
control_safe=false
calibration_class=teacher_estimated
not_robot_frame_truth=true
trainable_for=geometry_pretrain_only
```

Current short60 DA3 QA outcome: structurally valid real DA3 artifacts exist, but weak BEV promotion is quarantined because `pose_jump_outlier_count=1`.

## Gate 1.9: DA3 pose windows and RGB-D truth anchor

DA3 pose diagnostics and stable-window selection can unlock weak geometry labels without pretending a route-level pose outlier disappeared. This gate does not train SpatialMemoryNet and does not mark phone video or public RGB-D data as robot-frame control truth.

Required DA3 diagnostics:
```bash
python -m homebrain.geometry.inspect_pose_sequence --log runs/room_walk_001_route_short60 --teacher-artifacts runs/room_walk_001_route_short60/teacher_artifacts/da3 --out runs/room_walk_001_da3_pose_inspect_short60
python -m homebrain.geometry.select_stable_windows --log runs/room_walk_001_route_short60 --teacher-artifacts runs/room_walk_001_route_short60/teacher_artifacts/da3 --out runs/room_walk_001_da3_stable_windows_short60.json --min-window 20
```

Required DA3 window-gated weak BEV and pack:
```bash
python -m homebrain.geometry.da3_to_weak_bev --log runs/room_walk_001_route_short60 --teacher-artifacts runs/room_walk_001_route_short60/teacher_artifacts/da3 --qa runs/room_walk_001_da3_self_calib_qa_short60.json --window-spec runs/room_walk_001_da3_stable_windows_short60.json --out runs/room_walk_001_route_short60/geometry/da3_weak_bev_stable_windows_short60
python -m homebrain.geometry.validate_bev --bev runs/room_walk_001_route_short60/geometry/da3_weak_bev_stable_windows_short60 --out runs/room_walk_001_da3_weak_bev_stable_windows_short60_eval.json
python -m homebrain.data.pack_spatial_dataset --log runs/room_walk_001_route_short60 --bev runs/room_walk_001_route_short60/geometry/da3_weak_bev_stable_windows_short60 --teacher-artifacts runs/room_walk_001_route_short60/teacher_artifacts/da3 --out runs/room_walk_001_da3_stable_spatial_pack_short60
python -m homebrain.data.qa_spatial_dataset --dataset runs/room_walk_001_da3_stable_spatial_pack_short60 --out runs/room_walk_001_da3_stable_spatial_pack_qa_short60.json
```

Public RGB-D truth-anchor commands:
```bash
python -m homebrain.datasets.setup_tum_rgbd --out data/public/tum_rgbd --sequence freiburg1_xyz --download
python -m homebrain.datasets.tum_rgbd_to_route --source data/public/tum_rgbd/freiburg1_xyz --out runs/tum_freiburg1_xyz_route --max-frames 120
python -m homebrain.geometry.rgbd_truth_to_bev --log runs/tum_freiburg1_xyz_route --out runs/tum_freiburg1_xyz_route/geometry/rgbd_truth_bev
python -m homebrain.geometry.validate_bev --bev runs/tum_freiburg1_xyz_route/geometry/rgbd_truth_bev --out runs/tum_freiburg1_xyz_rgbd_truth_bev_eval.json
python -m homebrain.data.pack_spatial_dataset --log runs/tum_freiburg1_xyz_route --bev runs/tum_freiburg1_xyz_route/geometry/rgbd_truth_bev --out runs/tum_freiburg1_xyz_rgbd_truth_spatial_pack
python -m homebrain.data.qa_spatial_dataset --dataset runs/tum_freiburg1_xyz_rgbd_truth_spatial_pack --out runs/tum_freiburg1_xyz_rgbd_truth_spatial_pack_qa.json
```

Geometry comparison command:
```bash
python -m homebrain.geometry.compare_geometry_sources --log runs/room_walk_001_route_short60 --a runs/room_walk_001_route_short60/geometry/da3_weak_bev_stable_windows_short60 --b runs/room_walk_001_route_short60/geometry/depth_pro_bev --out runs/room_walk_001_geometry_compare_da3_depthpro_short60.json
```

Gate metrics:
```text
pose_jump_outlier_count
outlier_deltas[].previous_frame_id/current_frame_id
accepted_window_count
excluded_frame_ids
bev_missing_count
bev_shape_error_count
bev_nan_count
example_count
missing_count
shape_error_count
nan_count
control_safe_true_count
valid_mask_disagreement_mean
occupancy_disagreement_mean
```

Current Goal 6B outcome: short60 DA3 pose diagnostics identify exactly one outlier delta, frame 19 to frame 20. Stable-window selection accepts windows `0..19` and `21..59`, producing a 59-example DA3 weak-label pack with zero missing/shape/nan errors. TUM `freiburg1_xyz` setup/import/truth-BEV produces a 120-example geometry-anchor pack with zero missing/shape/nan errors. All artifacts remain `control_safe=false` and TUM license review remains `pending_human_review`.

## Gate 1.5: image-sequence imported routes

Real indoor image folders can be imported into ordinary HomeBrain segment logs with frame artifacts and explicit missing-sensor metadata.

Ingest command:
```bash
python -m homebrain.ingest.image_sequence --frames data/inbox/room_walk/frames --out runs/room_walk_route --camera front_rgb --fps 10
```

Optional sampling:
```bash
python -m homebrain.ingest.image_sequence --frames data/inbox/room_walk/frames --out runs/room_walk_route_stride2 --camera front_rgb --fps 10 --stride 2 --max-frames 100
```

Imported-route metrics:
```text
imported_frame_count
image_load_error_count
timestamp_interval_error_count
missing_sensor_notice_count
```

These metrics are emitted by `homebrain.eval.run_eval` when `route_metadata.json` identifies `source_type=image_sequence`. `missing_sensor_notice_count` should normally be 3 for image-only imports: IMU, wheel odometry, and commands are unavailable rather than faked.

Current verification commands for an imported route:
```bash
python -m homebrain.ingest.image_sequence --frames data/inbox/room_walk/frames --out runs/room_walk_route --camera front_rgb --fps 10
python -m homebrain.replay.replayd --log runs/room_walk_route --out runs/room_walk_replayed
python -m homebrain.teachers.run_teacher --teacher mock --log runs/room_walk_route --out runs/room_walk_route/teacher_artifacts/mock_teacher
python -m homebrain.teachers.visualize_artifacts --artifacts runs/room_walk_route/teacher_artifacts/mock_teacher --out runs/room_walk_mock_teacher_viz
python -m homebrain.eval.run_eval --log runs/room_walk_route --teacher-artifacts runs/room_walk_route/teacher_artifacts/mock_teacher --out runs/room_walk_eval_with_teacher.json
python -m homebrain.teachers.run_teacher --teacher depth_pro --backend real --log runs/room_walk_route --out runs/room_walk_route/teacher_artifacts/depth_pro
python -m homebrain.teachers.visualize_artifacts --artifacts runs/room_walk_route/teacher_artifacts/depth_pro --out runs/room_walk_depth_pro_viz
python -m homebrain.eval.run_eval --log runs/room_walk_route --teacher-artifacts runs/room_walk_route/teacher_artifacts/depth_pro --out runs/room_walk_eval_with_depth_pro.json
```
