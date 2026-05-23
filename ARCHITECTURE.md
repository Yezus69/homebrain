# ARCHITECTURE.md — HomeBrain Architecture

## Design summary

HomeBrain is a small production-directed robot-brain stack. It uses large open foundation models as teachers and trains a smaller runtime student to maintain spatial memory and choose useful cleaning trajectories.

The architecture is intentionally not a giant monolith. It is a lean dataflow with one primary learned brain and explicit logging/evaluation.

## Runtime processes

Initial software-only processes:

```text
loggerd   writes route/segment logs
replayd   deterministically replays logs
modeld    runs dummy model first, then HomeBrain model
evald     computes metrics and writes reports
```

Later hardware processes:

```text
sensord   reads camera/IMU/wheel encoder data
controld  sends cmd_vel or wheel targets to motor controller
debugd    visualization only
```

Do not build hardware daemons before replay/model/eval are useful.

## Primary dataflow

```text
FrameEvent + ImuEvent + WheelEvent + CommandEvent
        ↓
ordered segment log
        ↓
replayd
        ↓
modeld
        ↓
BrainOutputEvent
        ↓
evald
```

## Brain model: SpatialMemoryNet

Inputs:
- RGB image frame(s)
- optional IR frame later
- IMU window
- wheel odometry delta/window
- previous command/action
- previous memory state

State:
- local egocentric latent grid
- sparse/global coverage memory
- keyframe/place descriptors later

Outputs:
- `pose_delta`: dx, dy, dyaw, confidence/covariance
- `local_bev`: free/occupied/unknown/traversable/risky grid
- `dynamic_risk`: human/pet/moving-object risk map
- `coverage_update`: cleaned/seen/reachable/uncertain cells
- `trajectory_scores`: scores for candidate trajectories
- `future_rollout`: predicted pose/BEV/risk under action sequence
- `uncertainty`: confidence, stop reason, recovery suggestion
- `cmd_vel`: optional selected velocity command

## Foundation teachers

Teacher models are allowed to be heavy. Runtime student should be small.

Candidate teachers:
- dense visual feature model, e.g. DINO-family
- geometry/depth model, e.g. Depth Anything / Depth Pro / MoGe / VGGT-family
- segmentation/video mask model, e.g. SAM-family
- navigation-prior model, e.g. GNM / ViNT / NoMaD

Teacher outputs become pseudo-labels or features. They are not automatically product runtime dependencies.

## Control philosophy

Use candidate trajectories.

```text
candidate trajectories
        ↓
trajectory scorer
        ↓
select best safe/useful candidate
        ↓
execute short chunk
        ↓
replan
```

Avoid direct arbitrary motor PWM output.

## Memory philosophy

Use explicit spatial memory:
- local egocentric BEV
- global coverage grid
- dynamic obstacle decay
- uncertainty map
- keyframe descriptors later

Do not rely only on hidden recurrent/transformer state.

## Simulation philosophy

Simulation is useful for causal labels, collision/recovery cases, and sanity checks. It is not the source of truth for real-home robustness.

Use sim as a plug-in, not as the core dependency.

## Anti-goals

Do not spend early time on:
- perfect simulator realism
- ROS graph plumbing
- complex SLAM pipeline as the brain
- huge end-to-end policy
- beautiful 3D viewer before useful BEV/risk/coverage outputs
- non-measurable demos
