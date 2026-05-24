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

## Offline geometry labels

Depth teacher artifacts can be converted into explicit local egocentric BEV weak labels before any student training. The current depth-to-BEV path consumes Depth Pro `depth_m.npy` plus `focallength_px.npy`, applies an explicit camera-height/pitch/roll/yaw config, and writes per-frame local grids:

```text
depth_m + focallength_px + camera config
        ->
pinhole point projection
        ->
robot-frame floor/obstacle/unknown BEV weak labels
```

These BEV labels are training/evaluation artifacts only. Manifests and frame metadata must include `weak_label=true` and `control_safe=false`. Missing calibration is handled by documented assumptions, not silent defaults.

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

Current SpatialMemoryNet v1 implements the first local egocentric memory path:
per-frame DINO features produce current BEV logits, a persistent BEV memory
state is warped by robot-relative SE(2) pose deltas, and current observations
are fused into memory with explicit missing-pose behavior (`reset`, `no_warp`,
or `masked_update`). The v1 modeld/replay path resets memory on sequence
boundaries and writes current BEV, memory BEV, uncertainty, observed-mask, and
memory debug artifacts. This is still representation pretraining and replay
evaluation only; it is not a control-safe planner and it emits no `cmd_vel`.

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
