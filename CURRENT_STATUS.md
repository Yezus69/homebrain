# CURRENT_STATUS.md

Codex must update this file at the end of every goal.

## Current objective

Goal 0 log/replay/eval spine implemented and verified.

## Last completed goal

Goal 0: deterministic log/replay/eval spine.

## Current implementation status

- `homebrain` Python package exists.
- Goal 0 schemas, segment log writer/reader, deterministic replay, dummy model output, and eval metrics exist.
- Dummy `modeld` output is explicitly marked with `mock: true` and is not model performance.
- No ML yet.

## Commands that should work

```bash
pytest -q
python -m homebrain.replay.generate_dummy_log --out runs/dummy_route
python -m homebrain.replay.replayd --log runs/dummy_route --out runs/replayed_route
python -m homebrain.eval.run_eval --log runs/dummy_route --out runs/dummy_eval.json
```

## Metrics snapshot

Latest Goal 0 eval from `runs/dummy_eval.json`:

```json
{
  "brain_output_count": 6,
  "dropped_frame_count": 0,
  "eval_runtime_sec": 0.004262,
  "event_count": 24,
  "event_ordering_error_count": 0,
  "frame_count": 6,
  "replay_determinism_pass": true
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
