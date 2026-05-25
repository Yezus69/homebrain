# REQUIREMENTS.md

## Product-direction requirements

The future robot is an indoor vacuum/mop-class device.

Sensor assumptions:
- camera(s)
- IR camera later
- IMU
- wheel encoders
- no lidar
- no GPS
- no ultrasonic requirement

Behavior assumptions:
- must operate in messy homes
- must handle dynamic humans/pets/objects conservatively
- must maintain coverage memory
- must know when it is uncertain
- must stop or recover when confused
- must be compatible with low-cost edge deployment eventually

## Phase 0 requirements: log/replay/eval spine

Must implement:
- typed event schemas
- deterministic segment logs
- replay by timestamp
- dummy model that emits valid `BrainOutputEvent`
- metrics writer
- tests
- CLI commands

Must not implement:
- large ML dependencies
- ROS
- simulator
- teacher models
- robot control daemon

## Phase 1 requirements: teacher lab

Must implement:
- teacher plug-in interface
- mock teacher for tests
- at least one real image/depth/feature teacher wrapper if weights are available
- teacher artifact format
- visualization
- license audit updates

Must preserve:
- deterministic logs
- replay compatibility
- no fake metrics

## Phase 2 requirements: SpatialMemoryNet v0

Must implement:
- small trainable student
- dataset loader for teacher artifacts
- local BEV head
- pose delta head
- uncertainty head
- overfit test on tiny dataset
- replay integration

Must not:
- rely on expert human demos
- train giant models
- block on perfect simulator

## Phase 3 requirements: trajectory scoring

Must implement:
- candidate trajectory generator
- scoring interface
- policy that selects from candidates
- risk/coverage/uncertainty scoring
- replay visualization

Must not:
- emit arbitrary raw motor PWM
- hide decisions without debug output

## Nonfunctional requirements

- Reproducible commands.
- Tests must run locally.
- Generated artifacts live under `runs/`.
- Dependency additions must be justified.
- Implementation work updates `CURRENT_STATUS.md` briefly with proof and risks.
- Every important output must be inspectable from replay.
