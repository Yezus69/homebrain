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

Goal25 is the current real-data baseline: OpenLORIS route-heldout replay,
offline DINO/geometry supervision, SpatialMemoryNetV1, FutureBEV rollout, and
runtime reports. It proves the stack can create real artifacts and avoid
route-pose leakage, but it is not control-safe because the runtime policy
collapsed to one candidate.

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

A non-collapsed real-data runtime brain on multiple route-heldout public robot
routes.

The milestone is complete only when:

- replay uses real robot or robot-mounted public data only;
- the online runtime path consumes current sensor slices, previous command,
  calibration, odom/IMU/wheel where available, and prior memory;
- future frames, future labels, route ground truth, and teacher outputs are not
  available to the control tick;
- SpatialMemoryNet/current BEV/fused memory/FutureBEV or successor heads feed a
  bounded candidate trajectory selector;
- action entropy and dominant-action gates show the policy is not collapsed;
- p50/p95 latency, unsafe selections, stop/recovery reasons, and leakage flags
  are written to JSON;
- tests pass and `CURRENT_STATUS.md` records the exact artifact path.

## Why this order

If replay/eval is not solid, every later ML result is anecdotal. The project
must be measurable from day one, and each new piece must make the replay loop
closer to a live robot loop.
