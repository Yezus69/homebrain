HomeBrain is a replay-first, mostly-neural robot-brain stack for a future low-speed indoor vacuum/mop robot.

The core architecture is one coherent learned loop, not many unrelated model demos.

## Canonical robot-brain loop

```text
Open-weight teacher farm, offline only
  depth / geometry / segmentation / visual features / dynamic masks / navigation priors
        |
        v
Dataset and artifact builder
  public robot routes + RGB-D + pose/odom + weak video data
        |
        v
Direct student encoder
  RGB/RGB-D/IR later + IMU/wheel/previous action
        |
        v
Online SceneState
  pose estimate
  local BEV
  scene BEV memory
  uncertainty
  coverage
  dynamic object decay
        |
        v
Action-conditioned neural world model
  predicts future free/occupied/unknown/risk for each candidate trajectory
        |
        v
Learned candidate scorer
  collision risk
  dynamic risk
  unknown exposure
  coverage gain
  progress
  recovery/stop reason
        |
        v
Hard safety envelope
  bounded replay-only cmd_vel proposal