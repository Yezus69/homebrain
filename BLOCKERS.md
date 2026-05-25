# BLOCKERS.md

Only active blockers live here. Historical command transcripts were removed.

## Active Blockers

None that prevent local POC development.

## Active Risks

- No real robot hardware loop exists yet: no synchronized RGB/IR/IMU/wheel logs,
  no controller daemon, no watchdog, and no physical safety gate.
- Current learned trajectory scorer is replay-only and still narrow. Goal 22A
  repaired total collapse in older closed-loop replay, but Goal25 route-heldout
  FutureBEV/runtime selection collapsed to one candidate. Route-level action
  diversity is not strong enough for a real robot.
- There is no online `Brain.step(...)` deployment interface yet that owns memory
  across live sensor ticks and cleanly separates online inputs from labels,
  future frames, route ground truth, and teacher outputs.
- Traversability/free-space labels are weak. MoGe/OpenLORIS evidence is more
  useful for obstacle/unknown geometry than free-space proof.
- Dynamic risk supervision for people, pets, cords, and moved objects is still
  mostly planned, not solved.
- Public datasets and open-weight models are allowed for local POC work, but
  derived artifacts are not product-approved or redistributable by default.

## Minimal Next Repair

Build a broader POC data loop around public robot-mounted routes and local
owned videos:

1. stage more OpenLORIS or comparable public robot-frame routes;
2. generate/reuse teacher geometry and DINO features;
3. rebuild v5/future-motion labels;
4. improve robot-frame BEV/free-space QA before trusting action labels;
5. train/evaluate runtime-BEV policy with route-level action-diversity gates;
6. connect the trained memory/rollout/scorer path through an online-style
   `Brain.step(...)` API;
7. keep all outputs replay-only until hardware safety exists.
