# AGENTS.md - HomeBrain Repo Instructions

This file is repo memory for future Codex work. Keep it short.

## Mission

Build HomeBrain: a proof-of-concept indoor robot brain for a future
camera/IR/IMU/wheel-encoder floor-cleaning robot. The near-term proof is
software-only: replay real indoor data, use open-weight teachers offline, train
a smaller spatial-memory student, and score cleaning trajectories well enough
to justify hardware work.

## Read First

Before coding, read:

1. `PROJECT_BRIEF.md`
2. `ARCHITECTURE.md`
3. `CURRENT_STATUS.md`
4. The user's current request or issue

Read `EVALS.md`, `DECISIONS.md`, `LICENSE_AUDIT.md`, or deeper docs only when
the task touches those areas.

## Working Rules

- No fake success. Mock, synthetic, weak, replay-only, and POC artifacts must say so.
- Logs, replay, evals, and deterministic artifacts come before bigger models.
- Open-weight models and public datasets are allowed for local POC training and
  eval when provenance is recorded. They are not automatically control-safe,
  redistributable, or product-approved.
- Use foundation models as offline teachers first. Runtime should trend toward a
  smaller student model.
- Use candidate trajectories or `cmd_vel`; never arbitrary raw PWM.
- Prefer explicit spatial memory, BEV, coverage, uncertainty, and route
  provenance over hidden-only state.
- Keep modules small and directly tied to logging, replay, teachers, geometry,
  spatial memory, trajectory scoring, eval, or later hardware integration.
- Delete stale context instead of adding more instructions.

## Completion

For implementation tasks, leave proof:

- run focused tests, and full `python -m pytest -q` when feasible;
- update `CURRENT_STATUS.md` briefly with objective, files changed, commands,
  pass/fail result, artifacts, risks, and next recommended step;
- update `BLOCKERS.md` only for active blockers that still need action.

## Dependency Policy

Allowed by default: Python stdlib, numpy, pytest, dataclasses or pydantic-style
validation, opencv-python when image/video IO is needed, and torch when ML work
requires it. Avoid ROS/Nav2/Isaac/Habitat/web-scale frameworks unless the user
explicitly asks.

## Product Reality

Target homes are messy: humans, pets, cords, boxes, rugs, chair legs, dark
rooms, reflections, moved furniture, and temporary blocked paths. Do not build
only for clean static maps.
