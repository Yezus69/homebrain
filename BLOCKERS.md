# BLOCKERS.md

Only active blockers live here. Historical command transcripts were removed.

## Active Blockers

- Production deployment is still blocked. There is no real robot hardware loop,
  synchronized owned RGB/IR/IMU/wheel/command log, controller daemon, watchdog,
  docking/recovery system, hardware safety review, or physical test evidence.
- Public-data runtime coverage is no longer the Goal28 caveat-repair blocker:
  `runs/goal28_caveat_repair_three_scene_market_v1/milestone_report.json`
  passes with three heldout OpenLORIS scenes. The remaining blocker is product
  readiness, not the earlier two-route coverage gap.

## Active Risks

- Goal28 caveat repair passes its original local hard gates and the stricter
  user-requested caveat gates on three heldout OpenLORIS scenes with online
  `SceneState` inside `Brain.step(...)`, scene-memory-conditioned FutureBEV
  `safe_argmin` selection, and repair artifact
  `runs/goal28_caveat_repair_three_scene_market_v1/milestone_report.json`.
  This is still replay-only evidence, not a deployable robot brain.
- Goal28 caveat repair now uses `pose_metric_source=learned_visual_memory` via
  bounded odom-plus-visual correction and has two routes with independent pose
  reference, but it is not a hardware localization proof.
- Behavior-cloning labels are no longer all stop in the repair run
  (`stop=45, straight_medium=5, straight_short=102` at `0.4 s` horizon), but the
  BC scorer is still not used as the runtime selector; runtime acceptance still
  comes from learned FutureBEV scoring over bounded candidates.
- The online-style `Brain.step(...)` API owns persistent scene state across
  replay ticks, but it is not yet a hardware daemon and has not been validated
  with synchronized owned RGB/IR/IMU/wheel/command logs.
- Traversability/free-space labels are weak. MoGe/OpenLORIS evidence is more
  useful for obstacle/unknown geometry than free-space proof.
- Dynamic risk supervision for people, pets, cords, and moved objects is still
  mostly planned, not solved.
- Public-route coverage is improved but still narrow for product claims. The
  repair run evaluates `corridor1-1`, `office1-1_7`, and `market1-1_3`; Market
  action diversity is close to collapse (`dominant_action_fraction=0.96875`),
  so broader data is still needed before hardware work.
- `cafe1-1_2` remains rejected as a runtime-heldout policy route for this
  repair because `runs/goal28_caveat_repair_three_route_probe` failed when all
  cafe FutureBEV examples were rejected as degenerate.
- Public datasets and open-weight models are allowed for local POC work, but
  derived artifacts are not product-approved or redistributable by default.

## Minimal Next Repair

Build Goal29 around hardware-directed runtime readiness and broader robustness:

1. collect or stage synchronized owned robot RGB/IR/IMU/wheel/command logs;
2. add a replay-compatible watchdog/recovery/safety envelope before any hardware
   run;
3. keep staging more calibrated public or owned robot-frame routes beyond
   corridor/office/market;
4. generate/reuse teacher geometry and DINO features only for offline labels,
   QA, and distillation;
5. improve robot-frame BEV/free-space QA before trusting action labels;
6. keep Goal28's online SceneState, learned visual pose correction,
   non-degenerate BC labels, scene-memory policy input, zero unsafe
   selected motion, nonzero future free/occupied metrics, and p95 latency under
   100 ms across more heldout routes;
7. improve scene-memory physical geometry and direct RGB-D student quality on
   route-heldout real data;
8. keep all outputs replay-only until hardware safety exists.

Do not clear this risk by reporting `guided_transparent`, handcrafted diversity,
or odometry-vs-odometry localization as robot-brain success. The next accepted
repair must preserve Goal28's online SceneState, learned scene-conditioned
selection, learned visual pose correction, non-degenerate BC labels, positive
unknown reduction, nonzero future free/occupied metrics, low latency, explicit
pose metric provenance, at least three route-heldout real-data runtime routes,
and a credible path toward hardware safety.
