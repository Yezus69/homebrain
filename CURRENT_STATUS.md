# CURRENT_STATUS.md

Keep this file short. It should explain where the repo is now, not preserve old
command history.

## Current State

HomeBrain is a deterministic replay/eval and offline-teacher stack for a
software-only floor-cleaning robot brain POC.

Implemented:

- typed event schemas and deterministic segment logs;
- image/video and public RGB-D route ingestion;
- replay/model/eval CLIs;
- teacher artifacts for DINO, DA3, Depth Pro-style depth, MoGe/VGGT-style scene
  geometry, plus fake backends for tests;
- geometry-to-BEV and SpatialTrainPack builders;
- SpatialMemoryNet v0/v1 with explicit local BEV memory;
- fixed candidate trajectories, transparent scorer, learned scorer, future
  motion action labels, and closed-loop replay reports.

Not implemented:

- real robot runtime process;
- synchronized owned RGB/IR/IMU/wheel/command logs;
- hardware controller, watchdog, recovery, docking, or physical safety gate;
- robust dynamic-risk labels;
- strong free-space traversability labels;
- global home-scale coverage memory;
- a policy good enough to drive a real cleaning robot.

## Current Policy

Open-weight models and public datasets such as OpenLORIS are allowed for local
POC training/eval when provenance is recorded. This no longer blocks useful
experiments. It still does not imply product approval, redistribution rights, or
control safety.

Replay/control safety remains:

```text
replay_only=true
not_executed=true
control_safe=false
raw_pwm_emitted=false
```

## Latest Known Model/Policy Result

Goal 22A best checkpoint:

```text
runs/goal22a_overnight_runtime_scorer_repair/train/model_current_all_features_seed21_pair_constrained_search2/checkpoint.pt
```

Observed result: non-collapsed closed-loop replay on cafe/office/corridor, but
still narrow. It mainly chooses `straight_medium` and `arc_right_small`.

Useful metrics from the last accepted report:

```text
learned_logs_collapsed = 0/6
max_dominant_fraction = 0.73053152039555
mean_entropy = 0.9406909534104936
weighted_v5_agreement = 0.25760193503800966
cmd_vel_non_null_count = 0
```

## Last Completed Goal

Goal 23B: cleanup stale Codex goal/history docs, relax local POC dataset/model
policy, and keep the repo focused on robot-useful code.

## Goal 23B Proof

Objective attempted: remove stale Codex prompt/history clutter, reduce default
context, and relax public-dataset/open-weight model restrictions for local POC
work without claiming product/control safety.

Files changed: deleted old goal prompt/history docs; trimmed `AGENTS.md`,
`README.md`, `EVALS.md`, `LICENSE_AUDIT.md`, `BLOCKERS.md`, and
`CURRENT_STATUS.md`; updated dataset/license policy code and scene-teacher audit
logic; adjusted related docs/tests.

Commands run: required first docs were read; repo files were listed with
`rg --files`/PowerShell; policy and license references were searched with `rg`;
`python -m py_compile homebrain\tools\audit_scene_teacher_signal.py
homebrain\datasets\usage_policy.py homebrain\datasets\openloris_scene.py
homebrain\datasets\openloris_to_route.py tests\test_goal15b_scene_teacher_signal.py`;
`python -m pytest tests\test_goal15b_scene_teacher_signal.py
tests\test_goal12c_hard_validation.py -q`; full `python -m pytest -q`;
`git diff --stat`; final `git diff --check` and `git status --short`.

Pass/fail results: py_compile passed; focused tests passed with `20 passed in
5.68s`; full pytest passed with `120 passed in 179.78s`; `git diff --check`
passed. PowerShell tool output still appends a non-project `-NoProfile` warning
after commands.

Artifacts created: none.

Metrics observed: tracked diff removes about `4093` old lines and adds about
`455` focused lines; old Codex prompt/archive docs are gone; Markdown files now
list as 17 concise docs; OpenLORIS-style POC usage is allowed when
`poc_training_eval_allowed=true`.

Blockers/risks: safety flags still matter. POC data/model permission must not be
misread as product approval or permission to execute on hardware.

Recommended next goal: run a route-diversity POC loop on more public
robot-mounted routes and widen learned policy action diversity before adding
hardware integration.
