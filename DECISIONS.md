# DECISIONS.md

Architecture decisions live here. Codex may add entries, but must not silently reverse an existing decision.

## D001 — No ROS/Nav2 as core

Decision: Do not use ROS/Nav2 as the core architecture.

Reason: The project needs an openpilot-like log/replay/model/eval spine. ROS can become a compatibility bridge later.

Status: active.

## D002 — Do not bet on a perfect simulator

Decision: Simulation is a plug-in for causal labels and stress cases, not the core truth source.

Reason: Real homes are too dynamic and messy to simulate perfectly. Real video and public indoor datasets must be first-class.

Status: active.

## D003 — Foundation models are teachers first

Decision: Large open-weight models are used primarily to create features/pseudo-labels. Runtime student should be smaller.

Reason: Production hardware cannot depend on giant expensive models for every control tick.

Status: active.

## D004 — Candidate trajectories over raw motor commands

Decision: The learned brain scores/selects short candidate trajectories or `cmd_vel`, not arbitrary raw PWM.

Reason: Candidate-based action is safer, more debuggable, and easier to evaluate.

Status: active.

## D005 — Explicit spatial memory

Decision: Maintain explicit local/global spatial memory: BEV, coverage, uncertainty, dynamic object decay.

Reason: Hidden model state alone is not reliable for hours-long cleaning or debugging.

Status: active.

## D006 — Evals gate development

Decision: No major module counts as complete without tests, replay output, metrics, and `CURRENT_STATUS.md`.

Reason: Prevent toy demos and context drift.

Status: active.

## D007 - POC use is allowed, product claims wait

Decision: Public datasets and open-weight/open-source models may be used for
local proof-of-concept training/eval when provenance and limits are recorded.
They must not be called product-approved, redistributable, runtime-required, or
control-safe until later review says so.

Reason: HomeBrain needs to prove the approach before hardware or owned robot
data exists. Over-blocking local POC work slows the core learning loop.

Status: active.

## D008 - Runtime control tick has a strict input boundary

Decision: A production-directed runtime tick may consume current sensors,
calibration, previous command, odom/IMU/wheel data when available, and prior
memory. Future frames, future labels, route ground truth, teacher outputs, and
oracle BEV are training/eval-only unless an artifact is explicitly marked as an
ablation with leakage metrics.

Reason: The repo must train from public robot data and heavy open-weight
teachers without accidentally building a runtime that cannot deploy on a real
robot.

Status: active.

## D009 - Connect before adding scaffolds

Decision: When Codex has open scope, it should improve the connected
replay-as-live path before adding standalone modules: ingestion, teachers,
packs, spatial memory, future rollout, trajectory scoring, runtime reports, and
eval gates should compose into one measurable robot-brain loop.

Reason: The highest return now is turning existing pieces into a real-time
software brain, not adding more disconnected POC surfaces.

Status: active.

## D010 - Heuristic diversity is not learned policy success

Decision: A runtime report cannot be accepted as a learned robot-brain policy
success if action diversity comes from `guided_transparent`, hand-weighted
transparent scoring, temporal diversity penalties, randomization, or
hand-authored alternation. Those modes are allowed only as baselines or
ablations. Accepted policy diversity must come from learned scoring, learned
FutureBEV outputs, or another trained student evaluated route-heldout.

Reason: Goal26 connected the system, but the non-collapsed action distribution
was created by a hand-weighted selector while raw FutureBEV still collapsed.
The project needs real physical prediction and learned action choice, not a
metric workaround.

Status: active.

## D011 - Scene memory must be in the control path

Decision: Whole-scene memory only counts for the robot-brain milestone if it is
updated online and consumed by future rollout or trajectory scoring. A saved
scene map, contact sheet, or post-hoc visualization is useful debug evidence
but cannot satisfy the scene-level brain requirement by itself.

Reason: The robot needs physical understanding to traverse. A map artifact that
does not affect pose, future prediction, or action choice is not a deployable
robot brain.

Status: active.
