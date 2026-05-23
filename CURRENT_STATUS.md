# CURRENT_STATUS.md

Codex must update this file at the end of every goal.

## Current objective

Goal 6A DA3 teacher setup/run and self-calibration QA completed; DA3 weak BEV promotion is quarantined pending pose-outlier review.

## Last completed goal

Goal 6A: Depth Anything 3 cloned/installed/downloaded as an offline teacher, fake and real DA3 artifacts produced, self-calibration QA added, and weak BEV promotion gated/quarantined.

## Current implementation status

- `homebrain` Python package exists.
- Goal 0 schemas, segment log writer/reader, deterministic replay, dummy model output, and eval metrics exist.
- Dummy `modeld` output is explicitly marked with `mock: true` and is not model performance.
- `homebrain.teachers` exists with a teacher base interface, deterministic `.npy` artifact helpers, a mock teacher, run CLI, visualization CLI, manifest loader, and artifact validator.
- Mock teacher artifacts are explicitly marked `mock: true`, `synthetic: true`, and `real_perception: false`.
- Eval can optionally validate teacher artifacts without changing Goal 0 metrics when no teacher path is supplied.
- `homebrain.ingest` exists with an image-sequence CLI that imports `.jpg`, `.jpeg`, `.png`, `.pgm`, and `.ppm` frames into normal route logs.
- Imported image routes write copied frame artifacts, `FrameEvent` records, and `route_metadata.json` with source provenance, fps, frame count, dimensions when consistent, explicit missing sensor notices, and `user_owned_or_license_unknown=true`.
- Eval emits imported-route metrics when `route_metadata.json` has `source_type=image_sequence`.
- Optional `homebrain.teachers.depth_pro_teacher` exists for Depth Pro geometry teacher artifacts.
- Teacher registry supports `--teacher mock` and `--teacher depth_pro`.
- Real Depth Pro is optional and fails clearly if dependencies/checkpoints are unavailable; HomeBrain does not download weights.
- Fake Depth Pro backend is explicitly test-only and marked `mock: true`, `synthetic: true`, and `real_perception: false`.
- Depth Pro artifacts write `depth_m.npy`, `depth_confidence.npy`, `focallength_px.npy`, `bev_preview.npy`, and `metadata.json` per frame.
- Eval emits Depth Pro validation metrics: `depth_frame_count`, `depth_missing_count`, `depth_nan_count`, `depth_nonpositive_count`, and `depth_shape_error_count`.
- Depth Pro visualization writes previews for depth/confidence/BEV artifacts and records focal length as scalar metadata.
- Existing real room-walk import `runs/room_walk_001_route` has 350 frames; Goal 3.1 created a bounded 60-frame route at `runs/room_walk_001_route_short60`.
- Real Depth Pro ran from the local ignored `external/ml-depth-pro` install with local checkpoint `external/ml-depth-pro/checkpoints/depth_pro.pt` and CUDA.
- Real Depth Pro artifacts from Goal 3.1 are structurally usable as offline geometry inputs for the next depth-to-BEV prototype, but they are not production-approved or control-safe.
- Real Depth Pro did not emit model confidence in this run, so `depth_confidence.npy` is the documented HomeBrain heuristic.
- `homebrain.geometry` exists with camera config loading, pinhole depth-to-point projection, BEV rasterization, depth-to-BEV CLI, BEV visualization CLI, and BEV validation/eval CLI.
- Default assumed camera config exists at `configs/camera/phone_robot_height_guess.json`.
- Depth-to-BEV consumes `depth_m.npy`, `focallength_px.npy`, and optional `depth_confidence.npy`; missing principal point is recorded as an image-center assumption.
- BEV artifacts write `bev_free.npy`, `bev_obstacle.npy`, `bev_unknown.npy`, `bev_floor_candidate.npy`, `bev_height.npy`, `bev_confidence.npy`, and `metadata.json` per frame, plus route-level `bev_manifest.json`.
- BEV manifests and frame metadata are explicitly marked `weak_label=true` and `control_safe=false`.
- Geometry eval emits `bev_frame_count`, `bev_missing_count`, `bev_shape_error_count`, `bev_nan_count`, `free_ratio_mean`, `obstacle_ratio_mean`, `unknown_ratio_mean`, `confidence_mean`, and `temporal_jitter_mean`.
- Real short60 and full 350-frame route BEV artifacts exist under each route's `geometry/depth_pro_bev/` directory.
- `homebrain.data` exists with `pack_spatial_dataset`, `qa_spatial_dataset`, and `visualize_spatial_dataset` CLIs.
- SpatialTrainPack v0 writes deterministic `.npz` examples with BEV labels, confidence, optional height/floor candidate arrays, RGB references, provenance, split, camera-config hash, teacher-manifest hash, `weak_label=true`, and `control_safe=false`.
- Spatial QA emits structural counts, label/confidence/visibility density metrics, temporal jitter/flicker metrics, `trainable_candidate`, and `quarantine_reasons`.
- Current short60 SpatialTrainPack is structurally clean but quarantined as low-quality: confidence, label density, and observed/visible coverage are too low for training.
- `homebrain.tools.setup_da3_teacher` exists and explicitly clones DA3 to `external/depth-anything-3`, creates `external/venvs/da3`, installs dependencies, downloads/caches `depth-anything/DA3-SMALL`, and writes ignored `external/da3_setup_status.json`.
- `homebrain.teachers.da3_teacher` exists with fake and real backends; fake is test-only and marked `mock=true`, `synthetic=true`, `real_perception=false`.
- DA3 teacher artifacts write `depth.npy`, `confidence.npy`, `intrinsics.npy`, `extrinsics.npy`, and `metadata.json` per frame when available.
- DA3 manifests include model id/path, repo commit, dependency status, `license_review_status=pending_human_review`, `calibration_class=teacher_estimated`, `scale_source=teacher_relative_not_metric`, `not_robot_frame_truth=true`, and `control_safety=not_control_safe_training_teacher_only`.
- `homebrain.ingest.video` exists and imports video frames via cv2 or ffmpeg without faking IMU, wheel odometry, commands, intrinsics, pose, scale, or gravity/floor truth.
- Imported image/video routes now include calibration metadata fields: `calibration_class`, `intrinsics_source`, `extrinsics_source`, `pose_source`, `scale_source`, and `gravity_floor_source`.
- `homebrain.geometry.qa_self_calibration` exists and writes DA3/self-calibration QA metrics plus contact sheets.
- `homebrain.geometry.da3_to_weak_bev` exists and refuses weak BEV promotion unless QA passes or `--force-review` is supplied.
- Real DA3 ran on `runs/room_walk_001_route_short60` using the ignored external DA3 venv on CPU.
- Real DA3 artifacts are structurally valid, but self-calibration QA quarantined weak BEV promotion due one pose jump outlier.
- No student ML yet.

## Commands that should work

```bash
pytest -q
python -m homebrain.replay.generate_dummy_log --out runs/dummy_route
python -m homebrain.replay.replayd --log runs/dummy_route --out runs/replayed_route
python -m homebrain.eval.run_eval --log runs/dummy_route --out runs/dummy_eval.json
python -m homebrain.teachers.run_teacher --teacher mock --log runs/dummy_route --out runs/dummy_route/teacher_artifacts/mock_teacher
python -m homebrain.teachers.visualize_artifacts --artifacts runs/dummy_route/teacher_artifacts/mock_teacher --out runs/mock_teacher_viz
python -m homebrain.eval.run_eval --log runs/dummy_route --teacher-artifacts runs/dummy_route/teacher_artifacts/mock_teacher --out runs/dummy_eval_with_teacher.json
python -m homebrain.ingest.image_sequence --frames data/inbox/room_walk/frames --out runs/room_walk_route --camera front_rgb --fps 10
python -m homebrain.replay.replayd --log runs/room_walk_route --out runs/room_walk_replayed
python -m homebrain.teachers.run_teacher --teacher mock --log runs/room_walk_route --out runs/room_walk_route/teacher_artifacts/mock_teacher
python -m homebrain.teachers.visualize_artifacts --artifacts runs/room_walk_route/teacher_artifacts/mock_teacher --out runs/room_walk_mock_teacher_viz
python -m homebrain.eval.run_eval --log runs/room_walk_route --teacher-artifacts runs/room_walk_route/teacher_artifacts/mock_teacher --out runs/room_walk_eval_with_teacher.json
python -m homebrain.teachers.run_teacher --teacher depth_pro --backend fake --log runs/dummy_route --out runs/dummy_route/teacher_artifacts/depth_pro_fake
python -m homebrain.teachers.visualize_artifacts --artifacts runs/dummy_route/teacher_artifacts/depth_pro_fake --out runs/depth_pro_fake_viz
python -m homebrain.eval.run_eval --log runs/dummy_route --teacher-artifacts runs/dummy_route/teacher_artifacts/depth_pro_fake --out runs/dummy_eval_with_depth_pro_fake.json
python -m homebrain.teachers.run_teacher --teacher depth_pro --backend real --log runs/room_walk_route --out runs/room_walk_route/teacher_artifacts/depth_pro
python -m homebrain.ingest.image_sequence --frames data/inbox/room_walk_001/frames --out runs/room_walk_001_route_short60 --camera front_rgb --fps 10 --max-frames 60
.\external\ml-depth-pro\.venv\Scripts\python.exe -m homebrain.teachers.run_teacher --teacher depth_pro --backend real --device cuda --log runs\room_walk_001_route_short60 --out runs\room_walk_001_route_short60\teacher_artifacts\depth_pro
python -m homebrain.teachers.visualize_artifacts --artifacts runs/room_walk_001_route_short60/teacher_artifacts/depth_pro --out runs/room_walk_001_depth_pro_viz_short60
python -m homebrain.eval.run_eval --log runs/room_walk_001_route_short60 --teacher-artifacts runs/room_walk_001_route_short60/teacher_artifacts/depth_pro --out runs/room_walk_001_eval_with_depth_pro_short60.json
python -m homebrain.geometry.run_depth_to_bev --log runs/room_walk_001_route_short60 --depth-artifacts runs/room_walk_001_route_short60/teacher_artifacts/depth_pro --camera-config configs/camera/phone_robot_height_guess.json --out runs/room_walk_001_route_short60/geometry/depth_pro_bev
python -m homebrain.geometry.visualize_bev --bev runs/room_walk_001_route_short60/geometry/depth_pro_bev --out runs/room_walk_001_bev_viz_short60
python -m homebrain.geometry.validate_bev --bev runs/room_walk_001_route_short60/geometry/depth_pro_bev --out runs/room_walk_001_bev_eval_short60.json
python -m homebrain.data.pack_spatial_dataset --log runs/room_walk_001_route_short60 --bev runs/room_walk_001_route_short60/geometry/depth_pro_bev --out runs/room_walk_001_spatial_pack_short60
python -m homebrain.data.qa_spatial_dataset --dataset runs/room_walk_001_spatial_pack_short60 --out runs/room_walk_001_spatial_pack_qa_short60.json
python -m homebrain.data.visualize_spatial_dataset --dataset runs/room_walk_001_spatial_pack_short60 --out runs/room_walk_001_spatial_pack_viz_short60
```

## Metrics snapshot

Latest Goal 3 fake Depth Pro eval from `runs/dummy_eval_with_depth_pro_fake.json`:

```json
{
  "artifact_determinism_pass": true,
  "artifact_load_success": true,
  "artifact_shape_error_count": 0,
  "brain_output_count": 6,
  "depth_frame_count": 6,
  "depth_missing_count": 0,
  "depth_nan_count": 0,
  "depth_nonpositive_count": 0,
  "depth_shape_error_count": 0,
  "dropped_frame_count": 0,
  "eval_runtime_sec": 0.004258,
  "event_count": 24,
  "event_ordering_error_count": 0,
  "frame_count": 6,
  "frames_with_teacher_artifacts": 6,
  "missing_artifact_count": 0,
  "replay_determinism_pass": true,
  "teacher_artifact_count": 24,
  "teacher_manifest_frame_count": 6,
  "teacher_mock_used": true
}
```

Latest Goal 3.1 real Depth Pro eval from `runs/room_walk_001_eval_with_depth_pro_short60.json`:

```json
{
  "artifact_load_success": true,
  "artifact_shape_error_count": 0,
  "depth_frame_count": 60,
  "depth_missing_count": 0,
  "depth_nan_count": 0,
  "depth_nonpositive_count": 0,
  "depth_shape_error_count": 0,
  "frame_count": 60,
  "frames_with_teacher_artifacts": 60,
  "teacher_artifact_count": 240,
  "teacher_manifest_frame_count": 60,
  "teacher_mock_used": false
}
```

Latest Goal 4 short60 BEV eval from `runs/room_walk_001_bev_eval_short60.json`:

```json
{
  "bev_frame_count": 60,
  "bev_load_success": true,
  "bev_manifest_frame_count": 60,
  "bev_missing_count": 0,
  "bev_nan_count": 0,
  "bev_shape_error_count": 0,
  "confidence_mean": 0.05978431813418865,
  "control_safe": false,
  "free_ratio_mean": 0.012500000000000002,
  "obstacle_ratio_mean": 0.0663984375,
  "temporal_jitter_mean": 0.026459216101694914,
  "unknown_ratio_mean": 0.9211015625000002,
  "weak_label": true
}
```

Latest Goal 4 full-route BEV eval from `runs/room_walk_001_bev_eval_full.json`:

```json
{
  "bev_frame_count": 350,
  "bev_load_success": true,
  "bev_manifest_frame_count": 350,
  "bev_missing_count": 0,
  "bev_nan_count": 0,
  "bev_shape_error_count": 0,
  "confidence_mean": 0.15065131975321233,
  "control_safe": false,
  "free_ratio_mean": 0.03238169642857142,
  "obstacle_ratio_mean": 0.03787812499999998,
  "temporal_jitter_mean": 0.03962974570200571,
  "unknown_ratio_mean": 0.9297401785714291,
  "weak_label": true
}
```

Latest Goal 5 short60 SpatialTrainPack QA from `runs/room_walk_001_spatial_pack_qa_short60.json`:

```json
{
  "example_count": 60,
  "missing_count": 0,
  "shape_error_count": 0,
  "nan_count": 0,
  "free_ratio_mean": 0.012500000000000002,
  "obstacle_ratio_mean": 0.0663984375,
  "unknown_ratio_mean": 0.9211015625000002,
  "confidence_mean": 0.05978431813418865,
  "confidence_nonzero_ratio_mean": 0.059875000000000005,
  "label_density_mean": 0.0788984375,
  "observed_ratio_mean": 0.08698958333333336,
  "visible_confidence_positive_ratio_mean": 0.059875000000000005,
  "low_confidence_frame_count": 60,
  "empty_label_frame_count": 0,
  "temporal_visible_jitter_mean": 0.27240900394193374,
  "temporal_label_flicker_mean": 0.29610244463242247,
  "trainable_candidate": false,
  "quality_status": "quarantined_low_quality",
  "quarantine_reasons": [
    "mean_confidence_below_0.20",
    "low_confidence_frame_fraction_above_0.20",
    "label_density_below_0.10",
    "observed_visible_ratio_below_0.10"
  ]
}
```

## Known blockers

No external setup blockers. DA3 weak BEV promotion for `runs/room_walk_001_route_short60` is intentionally blocked by QA until `pose_jump_outliers_present` is reviewed; see `BLOCKERS.md`.

## Rules for future updates

Every update must include:

```text
Goal attempted:
Files changed:
Commands run:
Test result:
Artifacts created:
Metrics:
Blockers:
Risks:
Next recommended goal:
```

Do not delete previous useful status. Append concise updates below.

---

## Status log

### 000 — repo memory initialized

Goal attempted: create persistent project context for Codex.
Files changed: docs only.
Commands run: none.
Test result: not applicable.
Artifacts created: repo memory files.
Metrics: none.
Blockers: implementation not started.
Risks: Codex may drift unless it reads AGENTS.md and updates this file every run.
Next recommended goal: Goal 0 from `CODEX_GOALS.md`.

### 001 - Goal 0 deterministic log/replay/eval spine

Goal attempted: build the minimal HomeBrain foundation for typed events, deterministic segment logs, replay, dummy brain output, eval metrics, and tests.
Files changed: added `homebrain/__init__.py`, `homebrain/messages/__init__.py`, `homebrain/messages/schema.py`, `homebrain/replay/__init__.py`, `homebrain/replay/segment_log.py`, `homebrain/replay/generate_dummy_log.py`, `homebrain/replay/replayd.py`, `homebrain/brain/__init__.py`, `homebrain/brain/modeld.py`, `homebrain/eval/__init__.py`, `homebrain/eval/run_eval.py`, and tests under `tests/`.
Commands run: `pytest -q` with `%APPDATA%\Python\Python310\Scripts` added to PATH; `python -m pytest -q`; `python -m homebrain.replay.generate_dummy_log --out runs/dummy_route`; `python -m homebrain.replay.replayd --log runs/dummy_route --out runs/replayed_route`; `python -m homebrain.eval.run_eval --log runs/dummy_route --out runs/dummy_eval.json`.
Test result: pass, 5 tests passed. Initial bare `pytest -q` failed only because `pytest.exe` was installed outside PATH; the package itself was available and the PATH-adjusted command passed.
Artifacts created: `runs/dummy_route/manifest.json`, `runs/dummy_route/events.jsonl`, six deterministic raw RGB frame artifacts, `runs/replayed_route/manifest.json`, `runs/replayed_route/events.jsonl`, copied replay frame artifacts, and `runs/dummy_eval.json`.
Metrics: `event_count=24`, `frame_count=6`, `dropped_frame_count=0`, `event_ordering_error_count=0`, `replay_determinism_pass=true`, `brain_output_count=6`, `eval_runtime_sec=0.004262`.
Blockers: none.
Risks: dummy brain outputs are mock-only placeholders and must not be interpreted as model performance; this environment may need the Python user Scripts directory on PATH for bare `pytest`.
Next recommended goal: Goal 1 teacher artifact interface and deterministic mock teacher.

### 002 - Goal 1 teacher artifact interface and deterministic mock teacher

Goal attempted: implement a clean teacher artifact layer that can later support real foundation-model teachers while only providing deterministic mock/synthetic outputs in this goal.
Files changed: added `homebrain/teachers/__init__.py`, `homebrain/teachers/base.py`, `homebrain/teachers/artifacts.py`, `homebrain/teachers/mock_teacher.py`, `homebrain/teachers/run_teacher.py`, `homebrain/teachers/visualize_artifacts.py`, and `tests/test_teachers.py`; updated `homebrain/eval/run_eval.py`, `tests/test_eval.py`, `EVALS.md`, `LICENSE_AUDIT.md`, and `CURRENT_STATUS.md`.
Commands run: `python -m pytest -q`; `python -m homebrain.replay.generate_dummy_log --out runs/dummy_route`; `python -m homebrain.replay.replayd --log runs/dummy_route --out runs/replayed_route`; `python -m homebrain.eval.run_eval --log runs/dummy_route --out runs/dummy_eval.json`; `python -m homebrain.teachers.run_teacher --teacher mock --log runs/dummy_route --out runs/dummy_route/teacher_artifacts/mock_teacher`; `python -m homebrain.teachers.visualize_artifacts --artifacts runs/dummy_route/teacher_artifacts/mock_teacher --out runs/mock_teacher_viz`; `python -m homebrain.eval.run_eval --log runs/dummy_route --teacher-artifacts runs/dummy_route/teacher_artifacts/mock_teacher --out runs/dummy_eval_with_teacher.json`; `python -m homebrain.teachers.run_teacher --teacher mock --log runs/dummy_route --out runs/dummy_route/teacher_artifacts/mock_teacher_repeat`; byte-compare script for repeated mock teacher artifacts.
Test result: pass, 10 tests passed.
Artifacts created: `runs/dummy_route/teacher_artifacts/mock_teacher/teacher_manifest.json`, per-frame mock teacher artifact directories containing `depth.npy`, `depth_confidence.npy`, `dense_features.npy`, `dynamic_mask.npy`, `bev_preview.npy`, and `metadata.json`; `runs/mock_teacher_viz/visualization_manifest.json`; PGM/PPM previews under `runs/mock_teacher_viz/`; `runs/dummy_eval_with_teacher.json`; repeat determinism artifact tree under `runs/dummy_route/teacher_artifacts/mock_teacher_repeat/`.
Metrics: `event_count=24`, `frame_count=6`, `dropped_frame_count=0`, `event_ordering_error_count=0`, `replay_determinism_pass=true`, `brain_output_count=6`, `teacher_artifact_count=30`, `teacher_mock_used=true`, `artifact_load_success=true`, `frames_with_teacher_artifacts=6`, `missing_artifact_count=0`, `artifact_shape_error_count=0`, `artifact_determinism_pass=true`, `teacher_manifest_frame_count=6`, `eval_runtime_sec=0.004416`. Repeated mock teacher run produced 37 byte-identical files.
Blockers: none.
Risks: mock depth/features/masks/BEV are deterministic synthetic interface artifacts only and must not be used as real perception labels or model performance; all real teacher candidates remain license-unverified.
Next recommended goal: Goal 2 real indoor video/image ingestion, or a small Goal 1.1 cleanup if artifact schema docs should be promoted into `SCHEMA.md`.

### 003 - Goal 2 real indoor image-sequence ingestion

Goal attempted: implement a lean ingestion path that turns real indoor image sequences into normal HomeBrain route logs while keeping missing IMU, wheel odometry, commands, and intrinsics explicitly unavailable.
Files changed: added `homebrain/ingest/__init__.py`, `homebrain/ingest/metadata.py`, `homebrain/ingest/image_sequence.py`, and `tests/test_ingest_image_sequence.py`; updated `homebrain/messages/schema.py`, `homebrain/eval/run_eval.py`, `EVALS.md`, `DATA_STRATEGY.md`, `SCHEMA.md`, and `CURRENT_STATUS.md`.
Commands run: `python -m pytest -q`; PowerShell fixture generation for `runs/goal2_tiny_room_walk_frames` (first byte-expression attempt emitted nonfatal PowerShell errors, then corrected and rewrote the three PGM frames); `python -m homebrain.ingest.image_sequence --frames runs/goal2_tiny_room_walk_frames --out runs/goal2_room_walk_route --camera front_rgb --fps 10`; `python -m homebrain.replay.replayd --log runs/goal2_room_walk_route --out runs/goal2_room_walk_replayed`; `python -m homebrain.teachers.run_teacher --teacher mock --log runs/goal2_room_walk_route --out runs/goal2_room_walk_route/teacher_artifacts/mock_teacher`; `python -m homebrain.teachers.visualize_artifacts --artifacts runs/goal2_room_walk_route/teacher_artifacts/mock_teacher --out runs/goal2_room_walk_mock_teacher_viz`; `python -m homebrain.eval.run_eval --log runs/goal2_room_walk_route --teacher-artifacts runs/goal2_room_walk_route/teacher_artifacts/mock_teacher --out runs/goal2_room_walk_eval_with_teacher.json`; `python -m homebrain.replay.generate_dummy_log --out runs/dummy_route`; `python -m homebrain.replay.replayd --log runs/dummy_route --out runs/replayed_route`; `python -m homebrain.eval.run_eval --log runs/dummy_route --out runs/dummy_eval.json`; `python -m homebrain.teachers.run_teacher --teacher mock --log runs/dummy_route --out runs/dummy_route/teacher_artifacts/mock_teacher`; `python -m homebrain.teachers.visualize_artifacts --artifacts runs/dummy_route/teacher_artifacts/mock_teacher --out runs/mock_teacher_viz`; `python -m homebrain.eval.run_eval --log runs/dummy_route --teacher-artifacts runs/dummy_route/teacher_artifacts/mock_teacher --out runs/dummy_eval_with_teacher.json`.
Test result: pass, 14 tests passed.
Artifacts created: `runs/goal2_tiny_room_walk_frames/*.pgm`; `runs/goal2_room_walk_route/manifest.json`, `events.jsonl`, `route_metadata.json`, and copied frame artifacts; `runs/goal2_room_walk_replayed/`; `runs/goal2_room_walk_route/teacher_artifacts/mock_teacher/`; `runs/goal2_room_walk_mock_teacher_viz/visualization_manifest.json` and preview PGM/PPM files; `runs/goal2_room_walk_eval_with_teacher.json`; refreshed dummy route, replay, teacher, visualization, and eval artifacts under `runs/`.
Metrics: imported route eval reported `event_count=3`, `frame_count=3`, `imported_frame_count=3`, `image_load_error_count=0`, `timestamp_interval_error_count=0`, `missing_sensor_notice_count=3`, `dropped_frame_count=0`, `event_ordering_error_count=0`, `replay_determinism_pass=true`, `brain_output_count=3`, `teacher_artifact_count=15`, `teacher_mock_used=true`, `artifact_load_success=true`, `frames_with_teacher_artifacts=3`, `missing_artifact_count=0`, `artifact_shape_error_count=0`, `artifact_determinism_pass=true`, `teacher_manifest_frame_count=3`, `eval_runtime_sec=0.001932`. Refreshed dummy eval with teacher reported `event_count=24`, `frame_count=6`, `brain_output_count=6`, `teacher_artifact_count=30`, and `artifact_load_success=true`.
Blockers: none.
Risks: image dimensions are parsed with lean standard-library header readers, not full image decoding; this avoids OpenCV/Pillow but catches header-level load errors only. Imported image routes still have no real odometry, IMU, command, or calibration data, by design.
Next recommended goal: run a real self-collected room walk through the importer and mock teacher, then decide whether Goal 3 should begin with teacher-backed dataset loading or a small Goal 2.1 around richer route-source metadata/provenance.

### 004 - Goal 3 optional Depth Pro geometry teacher

Goal attempted: add optional Depth Pro geometry teacher support that can run real Depth Pro when dependencies/checkpoints are installed, while keeping tests dependency-light through an explicit fake backend.
Files changed: added `homebrain/teachers/depth_pro_teacher.py`, `homebrain/teachers/registry.py`, `tests/test_depth_pro_teacher.py`, and `docs/TEACHER_SETUP.md`; updated `homebrain/teachers/__init__.py`, `homebrain/teachers/artifacts.py`, `homebrain/teachers/run_teacher.py`, `homebrain/teachers/visualize_artifacts.py`, `EVALS.md`, `LICENSE_AUDIT.md`, and `CURRENT_STATUS.md`.
Commands run: `python -m pytest tests\test_depth_pro_teacher.py -q`; `python -m pytest -q`; `python -m homebrain.replay.generate_dummy_log --out runs/dummy_route`; `python -m homebrain.replay.replayd --log runs/dummy_route --out runs/replayed_route`; `python -m homebrain.eval.run_eval --log runs/dummy_route --out runs/dummy_eval.json`; `python -m homebrain.teachers.run_teacher --teacher mock --log runs/dummy_route --out runs/dummy_route/teacher_artifacts/mock_teacher`; `python -m homebrain.teachers.visualize_artifacts --artifacts runs/dummy_route/teacher_artifacts/mock_teacher --out runs/mock_teacher_viz`; `python -m homebrain.eval.run_eval --log runs/dummy_route --teacher-artifacts runs/dummy_route/teacher_artifacts/mock_teacher --out runs/dummy_eval_with_teacher.json`; `python -m homebrain.teachers.run_teacher --teacher depth_pro --backend fake --log runs/dummy_route --out runs/dummy_route/teacher_artifacts/depth_pro_fake`; `python -m homebrain.teachers.visualize_artifacts --artifacts runs/dummy_route/teacher_artifacts/depth_pro_fake --out runs/depth_pro_fake_viz`; `python -m homebrain.eval.run_eval --log runs/dummy_route --teacher-artifacts runs/dummy_route/teacher_artifacts/depth_pro_fake --out runs/dummy_eval_with_depth_pro_fake.json`.
Test result: pass, 18 tests passed. Targeted Depth Pro tests also passed, 4 tests passed.
Artifacts created: refreshed `runs/dummy_route/`, `runs/replayed_route/`, and `runs/dummy_eval.json`; refreshed mock artifacts under `runs/dummy_route/teacher_artifacts/mock_teacher/`, visualization under `runs/mock_teacher_viz/`, and eval at `runs/dummy_eval_with_teacher.json`; fake Depth Pro artifacts under `runs/dummy_route/teacher_artifacts/depth_pro_fake/`, visualization under `runs/depth_pro_fake_viz/`, and eval at `runs/dummy_eval_with_depth_pro_fake.json`.
Metrics: fake Depth Pro eval reported `event_count=24`, `frame_count=6`, `brain_output_count=6`, `teacher_artifact_count=24`, `teacher_mock_used=true`, `artifact_load_success=true`, `frames_with_teacher_artifacts=6`, `missing_artifact_count=0`, `artifact_shape_error_count=0`, `artifact_determinism_pass=true`, `teacher_manifest_frame_count=6`, `depth_frame_count=6`, `depth_missing_count=0`, `depth_nan_count=0`, `depth_nonpositive_count=0`, `depth_shape_error_count=0`. Mock teacher eval still reported `teacher_artifact_count=30`, `artifact_load_success=true`, and `frames_with_teacher_artifacts=6`.
Blockers: none.
Risks: real Depth Pro was not executed because this environment has no confirmed local Depth Pro dependency/checkpoint setup; Apple Depth Pro license status remains `pending_human_review`; fake backend output is test-only and not real perception; Depth Pro depth is an offline teacher signal and not control-safe.
Next recommended goal: after human license review and local checkpoint setup, run real Depth Pro on a short imported room route and compare artifact/eval summaries against the fake backend before using depth artifacts for student-training data.

### 005 - Goal 3.1 real Depth Pro room-walk smoke test

Goal attempted: run the already-implemented real Depth Pro backend on a short real imported room-walk route, visualize artifacts, evaluate them, and decide whether the real geometry artifacts are usable for the next depth-to-BEV step.
Files changed: updated `CURRENT_STATUS.md` only. Existing uncommitted edits in `.gitignore`, `docs/TEACHER_SETUP.md`, `homebrain/teachers/depth_pro_teacher.py`, and `tests/test_depth_pro_teacher.py` were present before this run and were not reverted.
Commands run: `python -m pytest -q`; `python -m homebrain.ingest.image_sequence --frames data/inbox/room_walk_001/frames --out runs/room_walk_001_route_short60 --camera front_rgb --fps 10 --max-frames 60`; `.\external\ml-depth-pro\.venv\Scripts\python.exe -m homebrain.teachers.run_teacher --teacher depth_pro --backend real --device cuda --log runs\room_walk_001_route_short60 --out runs\room_walk_001_route_short60\teacher_artifacts\depth_pro`; `python -m homebrain.teachers.visualize_artifacts --artifacts runs/room_walk_001_route_short60/teacher_artifacts/depth_pro --out runs/room_walk_001_depth_pro_viz_short60`; `python -m homebrain.eval.run_eval --log runs/room_walk_001_route_short60 --teacher-artifacts runs/room_walk_001_route_short60/teacher_artifacts/depth_pro --out runs/room_walk_001_eval_with_depth_pro_short60.json`; artifact stats one-liner over the real Depth Pro `.npy` files.
Test result: pass, 20 tests passed. Real Depth Pro teacher, visualization, and eval commands all exited 0.
Artifacts created: `runs/room_walk_001_route_short60/`; `runs/room_walk_001_route_short60/teacher_artifacts/depth_pro/teacher_manifest.json`; per-frame real Depth Pro `depth_m.npy`, `depth_confidence.npy`, `focallength_px.npy`, `bev_preview.npy`, and `metadata.json` for 60 frames; `runs/room_walk_001_depth_pro_viz_short60/visualization_manifest.json`; per-frame PGM previews under `runs/room_walk_001_depth_pro_viz_short60/`; `runs/room_walk_001_eval_with_depth_pro_short60.json`; teacher stdout/stderr logs at `runs/room_walk_001_depth_pro_teacher.out.log` and `runs/room_walk_001_depth_pro_teacher.err.log`.
Metrics: eval reported `event_count=60`, `frame_count=60`, `imported_frame_count=60`, `image_load_error_count=0`, `timestamp_interval_error_count=0`, `missing_sensor_notice_count=3`, `replay_determinism_pass=true`, `brain_output_count=60`, `artifact_load_success=true`, `frames_with_teacher_artifacts=60`, `missing_artifact_count=0`, `artifact_shape_error_count=0`, `teacher_artifact_count=240`, `teacher_manifest_frame_count=60`, `teacher_mock_used=false`, `depth_frame_count=60`, `depth_missing_count=0`, `depth_nan_count=0`, `depth_nonpositive_count=0`, and `depth_shape_error_count=0`. Artifact summary found depth shape `[1080, 1920]`, BEV preview shape `[16, 16]`, finite depth ratio `1.0`, depth median range `0.7821933031082153..1.4078009128570557` meters, depth max range `1.3864846229553223..2.9322547912597656` meters, confidence mean range `0.9991635084152222..0.9997700452804565`, and focal length range `2102.803955078125..2180.53564453125` px. Visualization manifest reported `visualization_written=true` and `frame_count=60`.
Blockers: none. `BLOCKERS.md` was not created or updated.
Risks: Apple Depth Pro remains `pending_human_review` and is not production-approved; outputs are monocular offline teacher geometry and are not control-safe navigation labels; no ground-truth calibration or metric-depth validation exists for this room walk; route is image-only with no IMU, wheel odometry, commands, or intrinsics; `depth_confidence.npy` came from the HomeBrain heuristic because this Depth Pro prediction exposed only `depth` and `focallength_px`; real backend artifacts are nondeterministic (`artifact_determinism_pass=false`) and should not be used as a determinism gate.
Next recommended goal: build the smallest depth-to-BEV prototype that consumes real Depth Pro `depth_m.npy` plus focal length metadata from this short route, writes explicit geometry/occupancy preview artifacts, and evaluates shape/load/finiteness without claiming traversability or control safety.

### 006 - Goal 4 depth-to-BEV weak geometry labels

Goal attempted: convert Depth Pro per-frame depth into explicit local egocentric BEV weak labels for future SpatialMemoryNet training and trajectory scoring, without student training or control-safety claims.
Files changed: added `homebrain/geometry/__init__.py`, `homebrain/geometry/camera_config.py`, `homebrain/geometry/depth_to_points.py`, `homebrain/geometry/bev_projector.py`, `homebrain/geometry/run_depth_to_bev.py`, `homebrain/geometry/visualize_bev.py`, `homebrain/geometry/validate_bev.py`, `configs/camera/phone_robot_height_guess.json`, and `tests/test_geometry_bev.py`; updated `ARCHITECTURE.md`, `DATA_STRATEGY.md`, `EVALS.md`, and `CURRENT_STATUS.md`.
Commands run: `python -m pytest tests\test_geometry_bev.py -q`; `python -m pytest -q`; `python -m homebrain.geometry.run_depth_to_bev --log runs/room_walk_001_route_short60 --depth-artifacts runs/room_walk_001_route_short60/teacher_artifacts/depth_pro --camera-config configs/camera/phone_robot_height_guess.json --out runs/room_walk_001_route_short60/geometry/depth_pro_bev --sweep-report runs/room_walk_001_bev_sweep_report.json`; `python -m homebrain.geometry.visualize_bev --bev runs/room_walk_001_route_short60/geometry/depth_pro_bev --out runs/room_walk_001_bev_viz_short60`; `python -m homebrain.geometry.validate_bev --bev runs/room_walk_001_route_short60/geometry/depth_pro_bev --out runs/room_walk_001_bev_eval_short60.json`; `.\external\ml-depth-pro\.venv\Scripts\python.exe -m homebrain.teachers.run_teacher --teacher depth_pro --backend real --device cuda --log runs\room_walk_001_route --out runs\room_walk_001_route\teacher_artifacts\depth_pro`; `python -m homebrain.eval.run_eval --log runs/room_walk_001_route --teacher-artifacts runs/room_walk_001_route/teacher_artifacts/depth_pro --out runs/room_walk_001_eval_with_depth_pro_full.json`; `python -m homebrain.geometry.run_depth_to_bev --log runs/room_walk_001_route --depth-artifacts runs/room_walk_001_route/teacher_artifacts/depth_pro --camera-config configs/camera/phone_robot_height_guess.json --out runs/room_walk_001_route/geometry/depth_pro_bev`; `python -m homebrain.geometry.validate_bev --bev runs/room_walk_001_route/geometry/depth_pro_bev --out runs/room_walk_001_bev_eval_full.json`; `python -m homebrain.geometry.visualize_bev --bev runs/room_walk_001_route/geometry/depth_pro_bev --out runs/room_walk_001_bev_viz_full`.
Test result: pass. Targeted geometry tests: 6 passed. Full suite: 26 passed.
Artifacts created: short60 BEV artifacts under `runs/room_walk_001_route_short60/geometry/depth_pro_bev/`; short60 BEV previews under `runs/room_walk_001_bev_viz_short60/`; short60 BEV eval at `runs/room_walk_001_bev_eval_short60.json`; camera-config sweep report at `runs/room_walk_001_bev_sweep_report.json`; full-route real Depth Pro artifacts under `runs/room_walk_001_route/teacher_artifacts/depth_pro/`; full-route Depth Pro eval at `runs/room_walk_001_eval_with_depth_pro_full.json`; full-route BEV artifacts under `runs/room_walk_001_route/geometry/depth_pro_bev/`; full-route BEV eval at `runs/room_walk_001_bev_eval_full.json`; full-route BEV previews under `runs/room_walk_001_bev_viz_full/`; full teacher logs at `runs/room_walk_001_depth_pro_teacher_full.out.log` and `runs/room_walk_001_depth_pro_teacher_full.err.log`.
Metrics: short60 BEV eval reported `bev_frame_count=60`, `bev_missing_count=0`, `bev_shape_error_count=0`, `bev_nan_count=0`, `free_ratio_mean=0.012500000000000002`, `obstacle_ratio_mean=0.0663984375`, `unknown_ratio_mean=0.9211015625000002`, `confidence_mean=0.05978431813418865`, `temporal_jitter_mean=0.026459216101694914`, `weak_label=true`, and `control_safe=false`. Sweep report processed 4 variants over 60 frames; base sanity score was `0.004592084094796972`, and all variants had `frame_error_count=0`. Full Depth Pro eval reported `depth_frame_count=350`, `depth_missing_count=0`, `depth_nan_count=0`, `depth_nonpositive_count=0`, `depth_shape_error_count=0`, `frames_with_teacher_artifacts=350`, and `teacher_mock_used=false`. Full-route BEV eval reported `bev_frame_count=350`, `bev_missing_count=0`, `bev_shape_error_count=0`, `bev_nan_count=0`, `free_ratio_mean=0.03238169642857142`, `obstacle_ratio_mean=0.03787812499999998`, `unknown_ratio_mean=0.9297401785714291`, `confidence_mean=0.15065131975321233`, `temporal_jitter_mean=0.03962974570200571`, `weak_label=true`, and `control_safe=false`.
Blockers: none. `BLOCKERS.md` was not created or updated.
Risks: camera intrinsics/extrinsics are assumed rather than calibrated; short60 and full-route BEV manifests record principal point as image-center assumed; Depth Pro remains offline teacher-only with Apple license still pending human review; BEV free/obstacle/unknown labels are weak geometry labels, not ground truth traversability and not control-safe; temporal jitter is a raw egocentric frame-to-frame sanity metric without odometry alignment; real Depth Pro artifacts remain nondeterministic and should not be used as a determinism gate.
Next recommended goal: calibrate or estimate camera intrinsics/extrinsics and add a reviewed BEV dataset packer that samples these weak labels into SpatialMemoryNet-ready training examples with explicit weak-label provenance.

### 007 - Goal 5 BEV QA + SpatialTrainPack v0

Goal attempted: turn existing DepthPro-to-BEV short60 outputs into a deterministic reviewed training-data package, or quarantine them honestly if label quality is poor; no new teachers and no student training.
Files changed: added `homebrain/data/__init__.py`, `homebrain/data/spatial_dataset.py`, `homebrain/data/pack_spatial_dataset.py`, `homebrain/data/qa_spatial_dataset.py`, `homebrain/data/visualize_spatial_dataset.py`, `tests/test_data_spatial_dataset.py`, and `docs/EXPERT_ACTION_DATA_PLAN.md`; updated `EVALS.md` and `CURRENT_STATUS.md`.
Commands run: required context reads of `AGENTS.md`, `PROJECT_BRIEF.md`, `CURRENT_STATUS.md`, `EVALS.md`, `DECISIONS.md`, `DATA_STRATEGY.md`, `ARCHITECTURE.md`, and `MODEL_SPEC.md`; `python -m pytest tests\test_data_spatial_dataset.py -q`; `python -m pytest -q`; `python -m homebrain.data.pack_spatial_dataset --log runs/room_walk_001_route_short60 --bev runs/room_walk_001_route_short60/geometry/depth_pro_bev --out runs/room_walk_001_spatial_pack_short60`; `python -m homebrain.data.qa_spatial_dataset --dataset runs/room_walk_001_spatial_pack_short60 --out runs/room_walk_001_spatial_pack_qa_short60.json`; `python -m homebrain.data.visualize_spatial_dataset --dataset runs/room_walk_001_spatial_pack_short60 --out runs/room_walk_001_spatial_pack_viz_short60`; `python -m homebrain.data.pack_spatial_dataset --log runs/room_walk_001_route_short60 --bev runs/room_walk_001_route_short60/geometry/depth_pro_bev --out runs/room_walk_001_spatial_pack_short60_repeat`; determinism hash-tree compare over the two package directories. One inline determinism one-liner failed due PowerShell quoting and was rerun successfully via stdin Python.
Test result: pass. Targeted data tests: 3 passed. Full suite: 29 passed. Required pack, QA, and visualization commands exited 0. Determinism compare reported `deterministic_match=True` over 61 files.
Artifacts created: `runs/room_walk_001_spatial_pack_short60/manifest.json`; 60 deterministic `.npz` examples under `runs/room_walk_001_spatial_pack_short60/examples/`; QA report `runs/room_walk_001_spatial_pack_qa_short60.json`; visualization manifest and contact sheets under `runs/room_walk_001_spatial_pack_viz_short60/` (`first_frames.ppm`, `median_confidence_frames.ppm`, `worst_confidence_frames.ppm`); repeat deterministic package under `runs/room_walk_001_spatial_pack_short60_repeat/`.
Metrics: QA reported `example_count=60`, `missing_count=0`, `shape_error_count=0`, `nan_count=0`, `weak_label_false_count=0`, `control_safe_true_count=0`, `free_ratio_mean=0.012500000000000002`, `obstacle_ratio_mean=0.0663984375`, `unknown_ratio_mean=0.9211015625000002`, `confidence_mean=0.05978431813418865`, `confidence_nonzero_ratio_mean=0.059875000000000005`, `label_density_mean=0.0788984375`, `observed_ratio_mean=0.08698958333333336`, `visible_confidence_positive_ratio_mean=0.059875000000000005`, `low_confidence_frame_count=60`, `empty_label_frame_count=0`, `temporal_jitter_mean=0.026459216101694914`, `temporal_visible_jitter_mean=0.27240900394193374`, `temporal_label_flicker_mean=0.29610244463242247`, `trainable_candidate=false`, and `quality_status=quarantined_low_quality`. Quarantine reasons were `mean_confidence_below_0.20`, `low_confidence_frame_fraction_above_0.20`, `label_density_below_0.10`, and `observed_visible_ratio_below_0.10`.
Blockers: none. `BLOCKERS.md` was not created or updated.
Risks: the current short60 package is structurally usable for review but not a training candidate; camera intrinsics/extrinsics remain assumed, labels are weak DepthPro-derived geometry rather than ground truth traversability, confidence and visible coverage are low, temporal visible jitter/flicker is notable in reviewed cells, and all examples remain `weak_label=true` and `control_safe=false`.
Next recommended goal: perform camera calibration or collect a calibrated RGB-D/pose reference sequence, then rerun BEV projection and SpatialTrainPack QA before any SpatialMemoryNet training; optionally add a small human review notes file for the quarantined short60 contact sheets.

### 008 - Goal 6A DA3 teacher + self-calibration QA

Goal attempted: clone/install/download Depth Anything 3 as an offline geometry teacher, add DA3 fake/real teacher support, add video ingest, add calibration metadata, run self-calibration QA, and gate optional weak BEV without training SpatialMemoryNet.
Files changed: `.gitignore`; `homebrain/teachers/artifacts.py`, `homebrain/teachers/da3_teacher.py`, `homebrain/teachers/registry.py`, `homebrain/teachers/run_teacher.py`, `homebrain/teachers/__init__.py`; `homebrain/tools/__init__.py`, `homebrain/tools/setup_da3_teacher.py`; `homebrain/ingest/image_sequence.py`, `homebrain/ingest/video.py`; `homebrain/geometry/qa_self_calibration.py`, `homebrain/geometry/da3_to_weak_bev.py`; `homebrain/eval/run_eval.py`; `tests/test_da3_teacher_qa.py`; `docs/TEACHER_SETUP.md`; `EVALS.md`; `LICENSE_AUDIT.md`; `BLOCKERS.md`; `CURRENT_STATUS.md`.
Commands run: `python -m pytest -q`; `python -m homebrain.replay.generate_dummy_log --out runs/dummy_route`; `python -m homebrain.teachers.run_teacher --teacher da3 --backend fake --log runs/dummy_route --out runs/dummy_route/teacher_artifacts/da3_fake`; `python -m homebrain.geometry.qa_self_calibration --log runs/dummy_route --teacher-artifacts runs/dummy_route/teacher_artifacts/da3_fake --out runs/dummy_da3_self_calib_qa.json`; `python -m homebrain.tools.setup_da3_teacher --external-dir external/depth-anything-3 --venv external/venvs/da3 --model-id depth-anything/DA3-SMALL --download`; `.\external\venvs\da3\Scripts\python.exe -m homebrain.teachers.run_teacher --teacher da3 --backend real --device cpu --model-id depth-anything/DA3-SMALL --max-frames 60 --window-size 10 --stride 1 --log runs\room_walk_001_route_short60 --out runs\room_walk_001_route_short60\teacher_artifacts\da3`; `python -m homebrain.teachers.run_teacher --teacher da3 --backend real --device cpu --model-id depth-anything/DA3-SMALL --max-frames 1 --window-size 1 --stride 1 --log runs/room_walk_001_route_short60 --out runs/room_walk_001_route_short60/teacher_artifacts/da3_auto_smoke`; `python -m homebrain.geometry.qa_self_calibration --log runs/room_walk_001_route_short60 --teacher-artifacts runs/room_walk_001_route_short60/teacher_artifacts/da3 --out runs/room_walk_001_da3_self_calib_qa_short60.json`; `python -m homebrain.eval.run_eval --log runs/room_walk_001_route_short60 --teacher-artifacts runs/room_walk_001_route_short60/teacher_artifacts/da3 --out runs/room_walk_001_eval_with_da3_short60.json`; `python -m homebrain.geometry.da3_to_weak_bev --log runs/room_walk_001_route_short60 --teacher-artifacts runs/room_walk_001_route_short60/teacher_artifacts/da3 --qa runs/room_walk_001_da3_self_calib_qa_short60.json --out runs/room_walk_001_route_short60/geometry/da3_weak_bev`.
Test result: pass, 32 tests passed. DA3 setup command exited successfully and wrote `external/da3_setup_status.json`. Real DA3 inference completed on 60 frames with CPU Torch; xFormers emitted a CUDA-extension warning but inference still succeeded. Weak BEV command correctly refused promotion because QA did not pass.
Artifacts created: ignored DA3 clone at `external/depth-anything-3`; ignored DA3 venv at `external/venvs/da3`; ignored model cache at `external/models/da3/depth-anything__DA3-SMALL`; ignored setup status `external/da3_setup_status.json`; fake DA3 artifacts under `runs/dummy_route/teacher_artifacts/da3_fake/`; fake QA report `runs/dummy_da3_self_calib_qa.json` and contact sheets; real DA3 artifacts under `runs/room_walk_001_route_short60/teacher_artifacts/da3/`; auto-reexec smoke artifacts under `runs/room_walk_001_route_short60/teacher_artifacts/da3_auto_smoke/`; real DA3 eval `runs/room_walk_001_eval_with_da3_short60.json`; real QA report `runs/room_walk_001_da3_self_calib_qa_short60.json` and contact sheets. No DA3 weak BEV or SpatialTrainPack was created because QA blocked promotion.
Metrics: DA3 setup status reported `success=true`, repo commit `41736238f5bced4debf3f2a12375d2466874866d`, model `download_status=downloaded_or_cached`, `depth_anything_3_api=ok`, `torch_version=2.12.0+cpu`, and `torch_cuda_available=false`. Fake DA3 QA reported `frame_count=6`, `depth_valid_ratio=1.0`, `confidence_mean=0.8299999435742696`, `intrinsics_valid_ratio=1.0`, `pose_valid_ratio=1.0`, `temporal_depth_consistency=0.9873461872339249`, `floor_plane_found_ratio=1.0`, `floor_plane_stability=0.9783915989100933`, `promotable_to_weak_bev=false`, and quarantine reasons `mock_or_synthetic_teacher`, `real_perception_false`. Real DA3 eval reported `frame_count=60`, `teacher_artifact_count=240`, `frames_with_teacher_artifacts=60`, `artifact_load_success=true`, `depth_frame_count=60`, `depth_missing_count=0`, `depth_nan_count=0`, `depth_nonpositive_count=0`, `depth_shape_error_count=0`, and `teacher_mock_used=false`. Real self-calibration QA reported `depth_valid_ratio=1.0`, `confidence_mean=1.0`, `intrinsics_valid_ratio=1.0`, `pose_valid_ratio=1.0`, `pose_jump_outlier_count=1`, `temporal_depth_consistency=0.9791352233683722`, `floor_plane_found_ratio=1.0`, `floor_plane_stability=0.9764022380113602`, `scale_source=teacher_relative_not_metric`, `promotable_to_weak_bev=false`, and quarantine reason `pose_jump_outliers_present`.
Blockers: real DA3 setup/run is not blocked. DA3 weak BEV promotion is blocked by QA due one pose jump outlier; recorded in `BLOCKERS.md`.
Risks: DA3 license/model use remains `pending_human_review`; DA3-SMALL outputs are relative teacher geometry, not robot-frame metric truth; the route is image-only with no robot IMU/wheel/command/intrinsics truth; xFormers CUDA extensions are unavailable in the CPU venv; weak BEV was intentionally not generated, so there is no DA3 SpatialTrainPack to train on.
Next recommended goal: inspect the DA3 pose outlier and either collect a smoother calibrated route or add a pose-window review/repair step before rerunning QA and attempting weak BEV promotion.
