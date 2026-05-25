# PROJECT_BRIEF.md - HomeBrain North Star

## One-sentence goal

Build a mostly-neural, production-directed indoor robot brain that can
eventually run on a consumer vacuum/mop robot using only cameras, IR camera,
IMU, and wheel encoders in real homes.

## What the brain must answer

At runtime, the robot must continuously answer:

```text
Where am I relative to what I have seen?
What floor is free, occupied, risky, or unknown?
What changed since I last saw this area?
What have I already cleaned?
Where should I go next to clean useful area?
What will likely happen if I take this trajectory?
How confident am I?
Should I stop, slow down, recover, or continue?
```

## Target system

```text
sensor events
  RGB camera(s), IR later, IMU, wheel encoders, previous command
        |
log/replay spine
        |
foundation-model teachers, mostly offline
        |
small Spatial Predictive Memory student
        |
heads:
  pose_delta
  local BEV traversability
  dynamic risk
  coverage memory update
  candidate trajectory scores
  future rollout
  uncertainty / stop reason
        |
trajectory selection
        |
cmd_vel / low-level controller
```

## Current phase

Software-only, production-directed runtime POC.

No robot hardware is required yet. The valuable artifact is a system that can
ingest real replayable indoor robot routes and run them through the same shape
of loop a robot would use live: sensor event, memory update, BEV/risk/coverage
prediction, bounded candidate scoring, `cmd_vel` proposal, eval report.

Goal26 is the current real-data connector baseline: OpenLORIS route-heldout
replay, offline DINO/geometry supervision, SpatialMemoryNetV1, direct RGB-D
runtime features, FutureBEV rollout, `Brain.step(...)`, bounded `cmd_vel`
proposals, latency/leakage reports, and scene-memory artifacts. It proves the
pieces can run together, but it is not the requested robot brain yet because
the accepted action diversity used a hand-weighted guided selector, raw
FutureBEV selection still collapsed, scene memory did not reduce unknown area,
future free/occupied prediction was zero, and pose was odometry-proxy only.

## What Codex Should Optimize For

- connect existing ingestion, teachers, packs, memory, rollout, scoring, and
  runtime into one measurable path;
- replace offline-only dependencies with direct online inputs where possible;
- improve real route-heldout metrics, especially action diversity, label
  quality, latency, leakage checks, and memory benefit;
- make every new broad module callable from replay-as-live runtime or remove it;
- leave inspectable artifacts and status updates that a future hardware run can
  reuse.

## What we are not building first

- no perfect home simulator
- no ROS/Nav2 core
- no brittle SLAM as the brain
- no giant VLA as the main policy
- no pixels-to-PWM black box
- no RL-first approach
- no fake demo without metrics

## Next Real Milestone

A learned scene-level robot-brain milestone on multiple route-heldout public
robot routes.

The milestone is complete only when:

- replay uses real robot or robot-mounted public data only;
- the online runtime path consumes current sensor slices, previous command,
  calibration, odom/IMU/wheel where available, and prior memory;
- future frames, future labels, route ground truth, and teacher outputs are not
  available to the control tick;
- SpatialMemoryNet/current BEV/fused memory/FutureBEV or successor heads feed a
  bounded candidate trajectory selector;
- accepted action diversity comes from learned scoring or learned FutureBEV
  outputs, not hand-weighted diversity priors, `guided_transparent`,
  randomization, or alternation;
- action entropy and dominant-action gates show the learned policy is not
  collapsed;
- scene memory improves measured physical geometry:
  `unknown_reduction_vs_current>0`, `coverage_memory_cells_seen>0`, and
  nonzero free/occupied future metrics when labels exist;
- direct RGB-D/RGB-IR runtime is a trained student path, not only a patch-stat
  adapter into a DINO-trained model;
- pose/localization metrics identify whether they are learned, odometry proxy,
  independent ground-truth eval, or leakage ablation;
- p50/p95 latency, unsafe selections, stop/recovery reasons, and leakage flags
  are written to JSON;
- tests pass and `CURRENT_STATUS.md` records the exact artifact path.

## Why this order

If replay/eval is not solid, every later ML result is anecdotal. The project
must be measurable from day one, and each new piece must make the replay loop
closer to a live robot loop.
