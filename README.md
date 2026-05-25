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

## Read This First

1. `PROJECT_BRIEF.md`
2. `ARCHITECTURE.md`
3. `CURRENT_STATUS.md`
4. The relevant module or test for your change

Use `EVALS.md` for current gates and `LICENSE_AUDIT.md` for the local POC data
policy. Old Codex goal prompts and historical command transcripts were removed;
they were clutter, not architecture.
