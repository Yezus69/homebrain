# HomeBrain

HomeBrain is a replay-first, mostly-neural software brain for a future low-speed indoor vacuum/mop robot.

The target robot eventually uses:
- RGB camera(s),
- IR camera later,
- IMU,
- wheel encoders,
- differential drive,
- no lidar/GPS dependency.

The current phase is software-only. HomeBrain does not control real hardware and is not product-safe.

## Architecture

The core loop is:

sensor/log replay
-> offline open-weight teachers for labels/features
-> direct runtime student
-> online SceneState and BEV memory
-> future BEV/risk rollout
-> bounded candidate trajectory scoring
-> replay-only cmd_vel proposal
-> eval reports and visual proof

## Current truth

See:
- `CURRENT_STATUS.md` for latest accepted result and active blockers.
- `EVALS.md` for gates and metrics.
- `DECISIONS.md` for architecture decisions.model policy.
- `AGENTS.md` for Codex instructions.

## Safety

Current outputs are replay artifacts only:

```text
replay_only=true
not_executed=true
control_safe=false
raw_pwm_emitted=false