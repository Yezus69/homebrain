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
- Treat the Goal26 OpenLORIS route-heldout runtime milestone as the minimum real-data
  baseline unless a newer baseline is documented.
- Runtime code must clearly separate online inputs from labels, future frames,
  route ground truth, teacher outputs, and other eval-only data.
- Foundation-model teachers are for dataset creation, supervision, and audits.
  The runtime should trend toward a teacher-free student control tick.
- Delete stale context instead of adding more instructions.

## Latest Real-Data Milestone

Current route-heldout proof:
`runs/goal26_scene_runtime_brain_milestone/milestone_report.json`.

Reproduce with:

```text
python -m homebrain.tools.run_openloris_route_heldout_milestone --out runs\goal26_scene_runtime_brain_milestone --sequences cafe1-1_2,corridor1-1,office1-1_7 --heldout-sequence corridor1-1 --max-frames 96 --spatial-steps 30 --future-steps 30 --runtime-max-frames 96 --runtime-feature-source direct_rgbd --max-spatial-folds 2 --future-rollout-selection-mode guided_transparent
```

Do not present it as control-safe. It uses real public OpenLORIS robot data,
connects `Brain.step(...)`, emits bounded `cmd_vel` proposals only, and writes
scene-memory artifacts. It is not the requested robot brain yet: raw FutureBEV
selection is still collapsed, accepted action diversity uses a hand-weighted
guided selector with temporal diversity, future free/occupied prediction is
zero, scene unknown reduction regressed, and pose metrics are odometry-proxy
only.

## Active Required Goal

When asked to continue the robot-brain work, build Goal27: a real-data learned
scene-level runtime brain that improves Goal26 instead of repeating it.

Goal27 acceptance is hard:

- no fake, mock, synthetic, generated, or random data for milestone metrics;
- no classical SLAM/Nav2 stack as the core brain;
- accepted runtime uses `Brain.step(...)` or equivalent online tick semantics;
- accepted runtime does not use teacher outputs, future labels, route ground
  truth, oracle BEV, or future frames during the control tick;
- accepted direct RGB-D/RGB-IR path is trained/evaluated as a student, not only
  a patch-stat adapter into a DINO-trained model;
- accepted action diversity must come from learned scoring/FutureBEV/policy
  outputs, not `guided_transparent`, temporal diversity priors, randomization,
  or hand-authored alternation;
- scene memory must improve measured physical geometry:
  `unknown_reduction_vs_current>0`, `coverage_memory_cells_seen>0`, and nonzero
  free/occupied future metrics when labels exist;
- pose/localization metrics must identify whether they are learned, odometry
  proxy, independent ground truth eval, or leakage ablation.

If these gates fail, keep iterating or mark the run blocked with exact artifact
paths. Do not call the milestone accepted.

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
