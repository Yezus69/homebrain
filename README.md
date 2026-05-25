# HomeBrain

HomeBrain is a software-first proof of concept for an indoor floor-cleaning
robot brain.

The target robot eventually has RGB camera(s), IR camera, IMU, wheel encoders,
differential drive, and no lidar/GPS dependency. Hardware integration is later;
the current value is a replayable system that can ingest indoor data, run
open-weight teacher models offline, train spatial memory, and score candidate
cleaning trajectories.

## Current Spine

- `homebrain/messages`: typed event schemas.
- `homebrain/replay`: deterministic segment logs and replay.
- `homebrain/eval`: scorecards and closed-loop replay reports.
- `homebrain/teachers`: offline feature/depth/scene teacher artifacts.
- `homebrain/geometry`: RGB-D, depth, point-map, and BEV conversion.
- `homebrain/data` and `homebrain/train`: SpatialTrainPack and student training.
- `homebrain/brain`: SpatialMemoryNet v0/v1 and replay model outputs.
- `homebrain/policies`: candidate trajectories, scorers, labels, and audits.
- `homebrain/tools`: narrow operational probes and validation commands.

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

## Read This First

1. `PROJECT_BRIEF.md`
2. `ARCHITECTURE.md`
3. `CURRENT_STATUS.md`
4. The relevant module or test for your change

Use `EVALS.md` for current gates and `LICENSE_AUDIT.md` for the local POC data
policy. Old Codex goal prompts and historical command transcripts were removed;
they were clutter, not architecture.
