# AGENTS.md — Instructions for Codex

This file is the repo-level memory. Read it before doing any work.

## Mission

Build HomeBrain: a production-directed indoor robot brain for a future camera/IR/IMU/wheel-encoder floor-cleaning robot.

The brain must eventually maintain spatial memory, predict traversability/risk, choose useful cleaning trajectories, and support deterministic replay/evaluation. It must not become a toy demo.

## Always read first

Before coding, read:
1. `PROJECT_BRIEF.md`
2. `CURRENT_STATUS.md`
3. `EVALS.md`
4. `DECISIONS.md`
5. The specific goal prompt or issue

If context was compacted, recover from those files instead of guessing.

## Non-negotiable principles

1. **Logs/replay/evals before models.** No serious ML code until deterministic log/replay/eval works.
2. **No fake success.** If a model is mocked, label it as a mock. Do not report mock metrics as real model performance.
3. **Every goal must leave proof.** Tests, commands, generated artifacts, and `CURRENT_STATUS.md` updates are mandatory.
4. **Keep the stack lean.** Avoid ROS, Nav2, Isaac, Habitat, web-scale training frameworks, and giant dependencies unless a goal explicitly asks for them.
5. **Open models are teachers first.** Big foundation models should produce features/pseudo-labels offline. The runtime brain should be a smaller distilled student.
6. **Candidate trajectories, not raw PWM.** The learned brain may score/select trajectories or emit `cmd_vel`; it must not jump straight to arbitrary motor PWM.
7. **Structured memory beats hidden-only memory.** Use explicit local/global spatial memory representations, not only transformer context.
8. **Production path matters.** Prefer deterministic data formats, typed schemas, repeatable commands, and clear dependency boundaries.
9. **Delete complexity.** Do not add a module unless it directly supports logging, replay, teacher labeling, spatial memory, trajectory scoring, evaluation, or later hardware integration.
10. **When stuck, reduce scope, do not thrash.** Implement the smallest useful subset that keeps the project moving and document the blocker.

## Required completion behavior for every goal

At the end of every Codex goal:
- Run the verification commands requested by the goal.
- Update `CURRENT_STATUS.md` with:
  - objective attempted
  - files changed
  - commands run
  - pass/fail results
  - artifacts created
  - metrics observed
  - blockers/risks
  - recommended next goal
- If a goal fails, create or update `BLOCKERS.md` with:
  - exact failure
  - likely cause
  - minimal next repair action
  - command output summary
- Do not claim completion unless verification passes.

## Dependency policy

Initial allowed dependencies:
- Python standard library
- numpy
- pydantic or dataclasses-based validation
- pytest
- rich or tqdm only if helpful
- opencv-python only when image/video I/O is actually needed
- torch only when the specific goal introduces ML

Do not add large dependencies casually.

## Repo shape to prefer

Use the package name `homebrain`.

Prefer:
```text
homebrain/
  messages/
  replay/
  eval/
  teachers/
  brain/
  policies/
  data/
  scripts/
tests/
runs/
```

Generated data goes under `runs/` and should be gitignored.

## Engineering style

- Type hints for public interfaces.
- Small modules.
- Deterministic tests.
- CLI entry points for every important workflow.
- Clear errors over silent fallbacks.
- If external model weights are missing, the wrapper may fall back to a mock only when explicitly configured, and artifacts must say `mock: true`.

## Product reality

The target homes are messy and dynamic:
- humans
- pets
- cords
- boxes
- rugs
- chair legs
- dark rooms
- reflective floors
- moved furniture
- temporarily blocked paths

Do not build only for clean static maps.
