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
