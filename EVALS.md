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

Model metrics:
```text
train_loss
val_loss
bev_loss
bev_iou_or_proxy
pose_delta_rmse
uncertainty_calibration_proxy
inference_fps
```

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
