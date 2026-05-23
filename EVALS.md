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

## Gate 3: trajectory scorer

Required before control integration:
- candidate trajectories generated deterministically
- scorer ranks candidates
- unsafe candidates can be penalized
- coverage/risk/debug outputs visible in replay

Trajectory metrics:
```text
candidate_count
selected_candidate_id
risk_score
coverage_gain_proxy
uncertainty_penalty
trajectory_eval_runtime_ms
```

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
