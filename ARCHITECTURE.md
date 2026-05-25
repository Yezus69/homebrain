# ARCHITECTURE.md - HomeBrain Architecture

## Design summary

HomeBrain is a production-directed robot-brain stack for a future indoor
floor-cleaning robot. It uses real replayable robot data, open-weight teachers
offline, and a smaller runtime student to maintain spatial memory and choose
useful bounded cleaning trajectories.

The architecture is intentionally not a giant monolith. It is a lean dataflow
with one primary learned brain, explicit spatial memory, deterministic replay,
and eval artifacts that can later map onto hardware.

## Connected Stack

The useful path is:

```text
real route or live sensors
  RGB/depth or RGB/IR, IMU, wheel/odom, previous command, calibration
        |
log/replay spine
  ordered segment logs, route splits, provenance, deterministic replay
        |
offline supervision
  open-weight teachers, RGB-D geometry, weak BEV labels, future-motion labels
        |
runtime student
  direct sensor encoder, SpatialMemoryNet, explicit BEV/memory/uncertainty
        |
decision surface
  candidate trajectories, FutureBEV/risk/coverage scoring, cmd_vel proposal
        |
eval and safety gates
  latency, leakage, action collapse, unsafe selections, artifact QA
```

New broad work should connect into this path. A model class, CLI, dataset, or
report that cannot be used by replay-as-live runtime needs a clear reason to
exist.

## Runtime processes

Current software-only processes:

```text
loggerd   writes route/segment logs
replayd   deterministically replays logs as online events
modeld    runs HomeBrain models with persistent memory state
evald     computes metrics and writes reports
```

Later hardware processes:

```text
sensord   reads camera/IR/IMU/wheel encoder data
braind    owns Brain.step(...), memory state, trajectory scoring, stop reasons
controld  sends cmd_vel or wheel targets to motor controller
watchdogd monitors heartbeat, collision/risk stops, and recovery limits
debugd    visualization only
```

Do not build hardware daemons before replay/model/eval and policy POC evidence
are useful. Do build software APIs that mirror the future live loop: one sensor
tick in, one memory update and bounded command proposal out.

## Primary dataflow

```text
FrameEvent + ImuEvent + WheelEvent + CommandEvent
        |
ordered segment log
        |
replayd
        |
Brain.step(...) / modeld
        |
BrainOutputEvent
        |
evald
```

## Runtime Input Boundary

Allowed online inputs:

- current RGB/depth/IR frame or synchronized sensor window;
- IMU, wheel odometry, route odom, and previous command when present;
- calibration and documented sensor assumptions;
- prior memory state and previous model outputs.

Training/eval-only inputs:

- future frames, future labels, future route motion;
- route ground truth or route pose if not available to the deployed robot;
- teacher outputs such as DINO, Depth Pro, MoGe, VGGT, SAM, or navigation-prior
  features;
- oracle BEV, collision labels, or human-cleaned annotations.

Any intentional use of eval-only information in runtime must be marked as an
ablation with leakage metrics. Production-directed runtime should trend toward
a teacher-free direct sensor student at the control tick.

## Brain model: SpatialMemoryNet

Inputs:

- RGB image frame(s)
- optional depth/IR frame later
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

Teacher models are allowed to be heavy. They are for dataset creation,
supervision, audits, and ablations. Runtime student should be small and should
not require teacher features every control tick.

Candidate teachers:

- dense visual feature model, e.g. DINO-family
- geometry/depth model, e.g. Depth Anything / Depth Pro / MoGe / VGGT-family
- segmentation/video mask model, e.g. SAM-family
- navigation-prior model, e.g. GNM / ViNT / NoMaD

Teacher outputs become pseudo-labels or training features. Public datasets and
open weights are allowed for local POC work when provenance is recorded. They
are not automatically product runtime dependencies, redistributable artifacts,
or control-safety evidence.

## Offline geometry labels

Depth teacher artifacts can be converted into explicit local egocentric BEV weak
labels before any student training. The current depth-to-BEV path consumes Depth
Pro `depth_m.npy` plus `focallength_px.npy`, applies an explicit
camera-height/pitch/roll/yaw config, and writes per-frame local grids:

```text
depth_m + focallength_px + camera config
        |
pinhole point projection
        |
robot-frame floor/obstacle/unknown BEV weak labels
```

These BEV labels are training/evaluation artifacts only. Manifests and frame
metadata must include `weak_label=true` and `control_safe=false`. Missing
calibration is handled by documented assumptions, not silent defaults.

## Control philosophy

Use candidate trajectories.

```text
candidate trajectories
        |
trajectory scorer
        |
select best safe/useful candidate
        |
execute short chunk
        |
replan
```

Avoid direct arbitrary motor PWM output. Runtime may emit bounded `cmd_vel`
proposals only when the report also records replay/control-safety flags.

Current action-labeling status: ActionLabelPack v5 can derive replay-only
behavior-cloning candidate labels from robot-frame dataset future motion. The
older synthetic coverage/risk label path remains available for ablation, but it
is no longer treated as the action oracle after Goal 13B showed it collapsed.
FutureBEV rollout can score candidates, but Goal26 only avoided a one-action
runtime report by adding `guided_transparent`, a hand-weighted selector with a
temporal diversity prior. That does not satisfy the robot-brain goal. Accepted
runtime policy diversity must come from learned scoring or learned FutureBEV
outputs.

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
state is initialized as semantic unknown, warped by robot-relative SE(2) pose
deltas, and fused with current observations through explicit update masks. When
trusted observation masks are unavailable, v1 derives a conservative mask from
current BEV confidence/unknown rather than trusting the full grid. Training and
replay report update-mask coverage and memory-overwrite fraction.

The default v1 modeld/replay warp source is route pose or odometry when the log
contains it. Predicted pose warp is an explicit ablation, not the default. If
route pose/odom is unavailable, behavior is explicit through the configured
missing-pose mode (`reset`, `no_warp`, or `masked_update`). The v1 modeld/replay
path resets memory on sequence boundaries and writes current BEV, memory BEV,
uncertainty, update-mask, and memory debug artifacts. This is still
representation pretraining and replay evaluation only.

Goal26 added a first `Brain.step(...)` path that owns memory across replayed
sensor ticks and can run without precomputed DINO at runtime. It is still not a
complete physical robot brain: the direct RGB-D path is a patch-stat adapter
into the existing feature interface, scene memory did not reduce unknown area,
and pose was odometry-proxy only. The next runtime step is a trained direct
RGB-D/RGB-IR student whose scene memory improves measured physical geometry and
whose future rollout predicts nonzero free/occupied/unknown evolution for
candidate actions.

## Integration priorities

Prefer work that:

- extends the Goal26 route-heldout runtime milestone to more real routes and
  stronger route/scene splits;
- improves robot-frame BEV/free-space labels and QA;
- trains direct RGB-D or RGB/IR runtime students that remove precomputed
  teacher-feature dependency at the control tick;
- connects SpatialMemoryNet/current BEV/fused memory/FutureBEV outputs into a
  learned non-collapsed candidate selector;
- makes scene memory measurably useful:
  `unknown_reduction_vs_current>0`, `coverage_memory_cells_seen>0`, and nonzero
  future free/occupied metrics when labels exist;
- adds watchdog, stop, recovery, uncertainty, or fallback behavior in replay
  with measurable artifacts.

Avoid work that only adds:

- empty CLIs;
- unconnected model classes;
- hand-weighted action diversity that is counted as learned policy success;
- synthetic-only milestones;
- broad rewrites that do not improve a real-data gate;
- dashboards or viewers before the underlying BEV/risk/coverage outputs are
  useful.

## Simulation philosophy

Simulation is useful for causal labels, collision/recovery cases, and sanity
checks. It is not the source of truth for real-home robustness.

Use sim as a plug-in, not as the core dependency.

## Anti-goals

Do not spend early time on:

- perfect simulator realism
- ROS graph plumbing
- complex SLAM pipeline as the brain
- huge end-to-end policy
- beautiful 3D viewer before useful BEV/risk/coverage outputs
- non-measurable demos
