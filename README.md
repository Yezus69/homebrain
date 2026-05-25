# HomeBrain

HomeBrain is a production-directed indoor floor-cleaning robot brain project.
The repo is still software-only, but every useful change should move it toward
a real robot runtime that can perceive, remember, choose, and explain safe
cleaning motion from camera/IR/IMU/wheel data.

The target robot eventually has RGB camera(s), IR camera, IMU, wheel encoders,
differential drive, and no lidar/GPS dependency. Hardware integration is later,
but the software should already be shaped like a robot brain: live-compatible
sensor events in, online spatial memory update, candidate trajectory selection,
bounded `cmd_vel` proposal out, and deterministic logs/evals around every run.

## Development Mandate

Codex should not spend compute on scaffolding that only creates empty commands,
placeholder files, or future plans. The highest-value work connects existing
modules into an end-to-end runtime path and proves it on real robot or
robot-mounted public data.

Meaningful progress usually means at least one of these:

- stronger real-data ingestion, calibration, QA, or route-heldout coverage;
- better online spatial memory from current sensors without future leakage;
- a runtime student path that reduces teacher/precomputed-feature dependence;
- non-collapsed candidate trajectory selection with safer failure behavior;
- measurable latency, safety, collapse, and route-out reports;
- code that makes the replay path closer to a live robot process.

Do not claim product readiness yet. Do build production-directed interfaces,
tests, and artifacts that make product readiness less speculative.

## Current Spine

The pieces are meant to be connected, not admired in isolation:

```text
real route / future live sensors
  FrameEvent + ImuEvent + Wheel/Odom/Command context
        |
        v
log/replay spine
  deterministic segment logs, route metadata, artifact provenance
        |
        v
offline real-data supervision
  OpenLORIS/TUM/owned route import, RGB-D/scene geometry, BEV packs,
  DINO/DA3/MoGe-style teachers for training only
        |
        v
runtime student
  SpatialMemoryNetV1, direct RGB-D student slices, pose/memory/update masks,
  FutureBEVRolloutV1 and candidate scoring
        |
        v
robot decision surface
  fixed candidate trajectories, transparent/learned scores, bounded cmd_vel
  proposals, stop/recovery reasons, no raw PWM
        |
        v
eval gates
  route-heldout replay, latency, collapse audit, unsafe selection, BEV/future
  metrics, visual failure cases
```

Module map:

- `homebrain/messages`: typed event contracts for replay and future live sensors.
- `homebrain/replay`: deterministic segment logs and replay-as-live execution.
- `homebrain/runtime`: online-style route replay wrappers and runtime reports.
- `homebrain/eval`: scorecards, closed-loop replay reports, latency/collapse views.
- `homebrain/teachers`: offline feature/depth/scene teacher artifacts.
- `homebrain/geometry`: RGB-D, depth, point-map, calibration, and BEV conversion.
- `homebrain/data` and `homebrain/train`: SpatialTrainPack, rollout packs, and student training.
- `homebrain/brain`: SpatialMemoryNet v0/v1, direct runtime slices, model outputs.
- `homebrain/policies`: candidate trajectories, scorers, labels, collapse/sanity audits.
- `homebrain/tools`: operational real-data pipelines and validation commands.

## Latest Real-Data Milestone

Route-heldout OpenLORIS replay milestone command:

```text
python -m homebrain.tools.run_openloris_route_heldout_milestone --out runs\goal25_openloris_route_heldout_milestone --sequences cafe1-1_2,corridor1-1,office1-1_7 --heldout-sequence corridor1-1 --max-frames 96 --spatial-steps 30 --future-steps 30 --runtime-max-frames 48 --max-spatial-folds 2
```

Main report:
`runs/goal25_openloris_route_heldout_milestone/milestone_report.json`

Artifacts include SpatialMemoryNetV1 checkpoints, a FutureBEVRolloutV1
checkpoint, route-heldout runtime reports, and a visual failure contact sheet.
The result is real public OpenLORIS replay evidence only:
`replay_only=true`, `control_safe=false`, `raw_pwm_emitted=false`.

Known result: heldout corridor replay produced 96 non-empty decisions with
p50/p95 latency `28.2573/31.700175 ms`, zero unsafe selected rate, and zero
route-pose leakage, but both future rollout and runtime action selection
collapsed to one candidate (`action_entropy=0.0`, dominant fraction `1.0`).

That collapse is the current high-return blocker. The next strong Codex run
should improve real-data route diversity, label quality, runtime-BEV policy
training, or fallback/recovery logic, then prove the improvement with the same
route-heldout and latency gates.

## What To Build Next

Prefer work that makes the existing stack run more like a robot:

- extend `run_openloris_route_heldout_milestone` or successor tools to more
  real robot-mounted routes with route/scene-heldout splits;
- improve robot-frame BEV/free-space quality instead of hiding weak labels;
- train/evaluate non-collapsed `FutureBEVRolloutV1` or trajectory scoring on
  runtime BEVs, not only oracle labels;
- grow the direct RGB-D runtime student so deployment does not require
  precomputed DINO features at every control tick;
- add an explicit online `Brain.step(...)` style interface that owns memory
  state, accepts current sensor events, and returns candidate decisions plus a
  bounded `cmd_vel` proposal;
- add watchdog/recovery decision outputs as replay-only proposals before any
  physical hardware claim;
- keep every artifact inspectable, deterministic, and honest about safety.

Avoid new placeholder packages, empty CLIs, unconnected model classes,
synthetic-data milestones, or broad rewrites that do not improve a real-data
gate.

## Read This First

1. `PROJECT_BRIEF.md`
2. `ARCHITECTURE.md`
3. `CURRENT_STATUS.md`
4. The relevant module or test for your change

Use `EVALS.md` for current gates and `LICENSE_AUDIT.md` for the local POC data
policy. Old Codex goal prompts and historical command transcripts were removed;
they were clutter, not architecture.
