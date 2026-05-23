# CURRENT_STATUS.md

Codex must update this file at the end of every goal.

## Current objective

Goal 3 optional real Depth Pro geometry teacher implemented and verified with a fake test backend.

## Last completed goal

Goal 3: optional Depth Pro teacher wrapper, artifact validation, visualization, setup docs, and tests.

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

## Known blockers

None yet.

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
