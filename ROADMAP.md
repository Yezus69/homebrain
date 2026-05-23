# ROADMAP.md

## Phase 0 — Repo memory and log/replay/eval spine

Goal:
- deterministic logs
- replay
- dummy model
- eval metrics

No ML.

## Phase 1 — Teacher lab

Goal:
- teacher plug-in interface
- mock teacher
- geometry/feature/mask teacher wrappers
- teacher artifacts
- visualization
- license audit

## Phase 2 — SpatialMemoryNet v0

Goal:
- small student
- BEV/pose/uncertainty outputs
- overfit tiny data
- replay integration

## Phase 3 — Trajectory scoring

Goal:
- candidate generator
- scorer
- selected candidate
- coverage/risk/uncertainty debug output

## Phase 4 — Real-video replay

Goal:
- ingest indoor video
- process frames
- show overlays
- log failure modes
- measure FPS/uncertainty/dynamic risk

## Phase 5 — Synthetic control data

Goal:
- plug-in simulator only where useful
- action-conditioned labels
- recovery/collision cases
- trajectory scorer improvement

## Phase 6 — Hardware integration

Goal:
- sensord
- controld
- real camera/IMU/wheel encoder logs
- same replay/eval stack
