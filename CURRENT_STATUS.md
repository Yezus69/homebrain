# CURRENT_STATUS.md

Codex must update this file at the end of every goal.

## Current objective

Goal 1 teacher artifact interface and deterministic mock teacher implemented and verified.

## Last completed goal

Goal 1: teacher artifact interface and deterministic mock teacher.

## Current implementation status

- `homebrain` Python package exists.
- Goal 0 schemas, segment log writer/reader, deterministic replay, dummy model output, and eval metrics exist.
- Dummy `modeld` output is explicitly marked with `mock: true` and is not model performance.
- `homebrain.teachers` exists with a teacher base interface, deterministic `.npy` artifact helpers, a mock teacher, run CLI, visualization CLI, manifest loader, and artifact validator.
- Mock teacher artifacts are explicitly marked `mock: true`, `synthetic: true`, and `real_perception: false`.
- Eval can optionally validate teacher artifacts without changing Goal 0 metrics when no teacher path is supplied.
- No ML yet.

## Commands that should work

```bash
pytest -q
python -m homebrain.replay.generate_dummy_log --out runs/dummy_route
python -m homebrain.replay.replayd --log runs/dummy_route --out runs/replayed_route
python -m homebrain.eval.run_eval --log runs/dummy_route --out runs/dummy_eval.json
python -m homebrain.teachers.run_teacher --teacher mock --log runs/dummy_route --out runs/dummy_route/teacher_artifacts/mock_teacher
python -m homebrain.teachers.visualize_artifacts --artifacts runs/dummy_route/teacher_artifacts/mock_teacher --out runs/mock_teacher_viz
python -m homebrain.eval.run_eval --log runs/dummy_route --teacher-artifacts runs/dummy_route/teacher_artifacts/mock_teacher --out runs/dummy_eval_with_teacher.json
```

## Metrics snapshot

Latest Goal 1 eval from `runs/dummy_eval_with_teacher.json`:

```json
{
  "artifact_determinism_pass": true,
  "artifact_load_success": true,
  "artifact_shape_error_count": 0,
  "brain_output_count": 6,
  "dropped_frame_count": 0,
  "eval_runtime_sec": 0.004416,
  "event_count": 24,
  "event_ordering_error_count": 0,
  "frame_count": 6,
  "frames_with_teacher_artifacts": 6,
  "missing_artifact_count": 0,
  "replay_determinism_pass": true,
  "teacher_artifact_count": 30,
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
