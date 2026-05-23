# MODEL_SPEC.md — SpatialMemoryNet v0

## Purpose

SpatialMemoryNet v0 is the first trainable HomeBrain student.

It should learn useful local spatial state from image/odometry/IMU streams and teacher artifacts.

It is not a giant VLA and not an RL agent.

## Inputs

Minimum:
- RGB image or precomputed visual features
- timestamp
- previous action / command
- wheel odometry delta
- optional IMU delta/window

Later:
- IR image
- previous memory state
- keyframe descriptors
- global memory crop

## Outputs

Minimum v0:
- local BEV grid
  - free
  - occupied
  - unknown
  - traversable
  - risky
- pose delta
  - dx
  - dy
  - dyaw
  - confidence
- uncertainty
  - scalar confidence
  - optional uncertainty grid

Later:
- dynamic risk grid
- coverage update grid
- candidate trajectory scores
- future rollout predictions

## Preferred v0 architecture

Keep it small:

```text
image/features
  ↓
visual adapter
  ↓
temporal fusion with wheel/IMU/action
  ↓
latent local memory grid
  ↓
heads:
  BEV decoder
  pose_delta head
  uncertainty head
```

Trainable parameter target: 10M–80M, not hundreds of millions.

## Losses

v0:
- BEV occupancy/traversability loss
- pose delta regression loss
- uncertainty calibration proxy
- temporal consistency loss if sequence labels exist

## Tiny overfit test

Before claiming training works:
- create a tiny synthetic or teacher-artifact dataset
- train until loss decreases significantly
- verify outputs have correct shapes
- save checkpoint
- run checkpoint in replay/modeld

## Runtime constraints

- Must run from a checkpoint inside `modeld`.
- Must emit valid `BrainOutputEvent`.
- Must support CPU/mocked mode for tests.
- GPU usage is allowed only in training/inference goals.
