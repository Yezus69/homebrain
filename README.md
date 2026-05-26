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

Goal26 is the latest committed real-data connector milestone:

```text
python -m homebrain.tools.run_openloris_route_heldout_milestone --out runs\goal26_scene_runtime_brain_milestone --sequences cafe1-1_2,corridor1-1,office1-1_7 --heldout-sequence corridor1-1 --max-frames 96 --spatial-steps 30 --future-steps 30 --runtime-max-frames 96 --runtime-feature-source direct_rgbd --max-spatial-folds 2 --future-rollout-selection-mode guided_transparent
```

Main report:
`runs/goal26_scene_runtime_brain_milestone/milestone_report.json`

Goal26 connected OpenLORIS ingestion, replay-as-live, `Brain.step(...)`,
SpatialMemoryNetV1 memory, direct RGB-D runtime features, FutureBEV scoring,
bounded `cmd_vel` proposals, latency/leakage reports, and scene-memory
visuals. It is real public OpenLORIS replay evidence only:
`replay_only=true`, `control_safe=false`, `raw_pwm_emitted=false`,
`teacher_runtime_dependency=false`.

Goal26 is not the requested robot brain yet. Treat it as plumbing and a baseline
to improve, not as completion:

- the apparent action diversity came from `guided_transparent`, a hand-weighted
  selector with a temporal diversity prior;
- raw FutureBEV action selection was still collapsed (`action_entropy=0.0`,
  dominant fraction `1.0`);
- future free/occupied IoU proxies were `0.0 / 0.0`;
- scene memory had negative unknown reduction (`-0.3163`), so it did not yet
  prove useful remembered physical geometry;
- pose metrics used an odometry proxy, not independent learned localization.

## Active Required Goal

The next Codex run must build the user-requested scene-level robot-brain
milestone, not another connector milestone.

Goal27 must produce one connected learned real-data runtime system, not a set of
separate reports. The system must replay real routes as live sensor ticks and
maintain a persistent `SceneState` or equivalent world model that is updated and
used every tick.

The required vertical slice is:

```text
real public robot route
  -> replay-as-live sensor tick
  -> Brain.step(...)
  -> pose estimate inside persistent scene memory
  -> local + scene BEV physical geometry
  -> future state rollout for candidate trajectories
  -> learned trajectory score / bounded cmd_vel proposal
  -> route-level metrics + visual proof
```

The accepted runtime must answer and save evidence for all five questions:

```text
Where does the robot think it is in the remembered scene?
What physical floor geometry does it believe is free/occupied/unknown/risky?
What parts of the scene has it already seen or covered?
What does it predict will happen over the next few states for each candidate?
Which bounded trajectory/cmd_vel does it choose, and why?
```

Goal27 must satisfy the original hard constraints:

- real public robot or robot-mounted data only for milestone metrics;
- no classical SLAM stack as the core brain;
- online `Brain.step(...)` consumes only current/past sensor data, prior memory,
  calibration, previous command, and odom/IMU/wheel when available;
- teacher outputs, future frames, future labels, route ground truth, and oracle
  BEV are training/eval-only;
- persistent scene-level memory must improve physical geometry, not only save a
  visualization;
- direct RGB-D/RGB-IR runtime student must be trained/evaluated, not just a
  patch-stat adapter into a DINO-trained model;
- near-future prediction must learn free/occupied/unknown/risk evolution for
  candidate actions;
- accepted action diversity must come from learned scoring or learned
  FutureBEV/policy outputs, not from `guided_transparent`,
  temporal-diversity penalties, randomization, or hand-authored alternation;
- accepted pose/localization metrics must say whether they are independent
  ground truth, odometry proxy, learned pose, or leakage ablation.
- the scene map must be used by trajectory scoring or future rollout; saving a
  map that the policy ignores is not enough.

Goal27 does not pass unless the main report shows all of these:

```text
real_public_data_only=true
runtime_api_step_count>0
scene_memory_used_for_policy=true
scene_memory_artifact_exists=true
scene_pose_trace_artifact_exists=true
physical_geometry_visual_exists=true
future_state_visual_exists=true
teacher_runtime_dependency=false
future_or_groundtruth_runtime_dependency=false
route_pose_leakage_ablation_fraction=0.0
raw_pwm_emitted=false
control_safe=false
accepted_policy_uses_handcrafted_diversity_prior=false
accepted_policy_uses_guided_transparent=false
action_entropy>0.0
dominant_action_fraction<1.0
future_free_iou_or_proxy>0.0
future_occupied_iou_or_proxy>0.0
unknown_reduction_vs_current>0.0
coverage_memory_cells_seen>0
latency_step_p95_ms<=100.0
```

If any of those fail, Codex must keep iterating or mark the run blocked with the
exact artifact proving why. It must not call the run accepted.

Do not accept:

- a command that only stages data, trains a model, or writes offline reports;
- a runtime that creates scene memory but does not use it for decisions;
- pose numbers that are only odometry-vs-odometry while presented as
  localization;
- future prediction that only predicts unknown and has zero free/occupied
  signal;
- action diversity from hand-written penalties instead of learned policy
  outputs;
- a visual artifact that does not show scene geometry, pose trace, selected
  path, and future-state prediction.

## What To Build Next

Prefer work that turns Goal26 plumbing into real physical understanding:

- train a real direct RGB-D/RGB-IR student on real route-heldout data;
- improve BEV/free/occupied labels and QA until physical geometry is visible
  and measured;
- make scene memory reduce unknown area and preserve free/occupied structure;
- replace `guided_transparent` with learned non-collapsed scoring;
- evaluate all accepted runtime metrics on at least two routes and preferably
  three scenes;
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
