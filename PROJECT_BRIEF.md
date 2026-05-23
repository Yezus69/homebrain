# PROJECT_BRIEF.md — HomeBrain North Star

## One-sentence goal

Build a mostly-neural, production-directed indoor robot brain that can eventually run on a consumer vacuum/mop robot using only cameras, IR camera, IMU, and wheel encoders in real homes.

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
        ↓
log/replay spine
        ↓
foundation-model teachers, mostly offline
        ↓
small Spatial Predictive Memory student
        ↓
heads:
  pose_delta
  local BEV traversability
  dynamic risk
  coverage memory update
  candidate trajectory scores
  future rollout
  uncertainty / stop reason
        ↓
trajectory selection
        ↓
cmd_vel / low-level controller
```

## Current phase

Software only.

No robot hardware is required yet. The first valuable artifact is a system that can ingest replayable indoor visual sequences and emit spatial memory, risk, coverage, and candidate motion decisions.

## What we are not building first

- no perfect home simulator
- no ROS/Nav2 core
- no brittle SLAM as the brain
- no giant VLA as the main policy
- no pixels-to-PWM black box
- no RL-first approach
- no fake demo without metrics

## First real milestone

A deterministic log/replay/eval spine with a dummy brain.

The first milestone is complete only when:
- logs can be generated
- logs can be replayed deterministically
- brain outputs are recorded
- eval metrics are written
- tests pass
- `CURRENT_STATUS.md` is updated

## Why this order

If replay/eval is not solid, every later ML result is anecdotal. The project must be measurable from day one.
