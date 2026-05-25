# AGENTS.md - HomeBrain Repo Instructions

This file is repo memory for future Codex work. Keep it short.

## Mission

Build HomeBrain: a production-directed indoor robot brain for a future
camera/IR/IMU/wheel-encoder floor-cleaning robot. The near-term proof is still
software-only, but every serious change should move toward a deployable runtime:
replay real indoor robot data as if it were live, use open-weight teachers only
offline, train a smaller spatial-memory student, and score bounded cleaning
trajectories well enough to justify hardware work.

## Read First

Before coding, read:

1. `PROJECT_BRIEF.md`
2. `ARCHITECTURE.md`
3. `CURRENT_STATUS.md`
4. The user's current request or issue

Read `EVALS.md`, `DECISIONS.md`, `LICENSE_AUDIT.md`, or deeper docs only when
the task touches those areas.

## Default Priority

When scope is open, choose the work that most directly connects the system into
a real robot brain:

1. Connect existing modules end-to-end before adding new isolated modules.
2. Prefer real robot or robot-mounted public data over mock, synthetic, or toy
   examples. OpenLORIS-style route-heldout replay is the current baseline.
3. Prefer online memory, BEV, trajectory scoring, and runtime APIs over
   offline-only reports.
4. Prefer evals that expose leakage, collapse, latency, and route generalization
   over demo scripts.

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
- Avoid scaffolding-only PRs. A broad change should improve a real-data gate,
  connect two runtime pieces, or remove an active blocker.
- Treat the Goal25 OpenLORIS route-heldout milestone as the minimum real-data
  baseline unless a newer baseline is documented.
- Runtime code must clearly separate online inputs from labels, future frames,
  route ground truth, teacher outputs, and other eval-only data.
- Foundation-model teachers are for dataset creation, supervision, and audits.
  The runtime should trend toward a teacher-free student control tick.
- Delete stale context instead of adding more instructions.

## Latest Real-Data Milestone

Current route-heldout proof:
`runs/goal25_openloris_route_heldout_milestone/milestone_report.json`.

Reproduce with:

```text
python -m homebrain.tools.run_openloris_route_heldout_milestone --out runs\goal25_openloris_route_heldout_milestone --sequences cafe1-1_2,corridor1-1,office1-1_7 --heldout-sequence corridor1-1 --max-frames 96 --spatial-steps 30 --future-steps 30 --runtime-max-frames 48 --max-spatial-folds 2
```

Do not present it as control-safe. It uses real public OpenLORIS robot data,
rejects degenerate FutureBEV action-label groups, emits bounded `cmd_vel`
proposals only, and still has a policy-collapse blocker.

## Completion

For implementation tasks, leave proof:

- run focused tests, and full `python -m pytest -q` when feasible;
- update `CURRENT_STATUS.md` briefly with objective, files changed, commands,
  pass/fail result, artifacts, risks, and next recommended step;
- update `BLOCKERS.md` only for active blockers that still need action.
- for broad autonomous work, record the exact real-data artifact path and the
  metric that improved or regressed.

## Dependency Policy

Allowed by default: Python stdlib, numpy, pytest, dataclasses or pydantic-style
validation, opencv-python when image/video IO is needed, and torch when ML work
requires it. Avoid ROS/Nav2/Isaac/Habitat/web-scale frameworks unless the user
explicitly asks.

## Product Reality

Target homes are messy: humans, pets, cords, boxes, rugs, chair legs, dark
rooms, reflections, moved furniture, and temporary blocked paths. Do not build
only for clean static maps.
