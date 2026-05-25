# BLOCKERS.md

Only active blockers live here. Historical command transcripts were removed.

## Active Blockers

None that prevent local POC development.

## Active Risks

- No real robot hardware loop exists yet: no synchronized RGB/IR/IMU/wheel logs,
  no controller daemon, no watchdog, and no physical safety gate.
- Current learned trajectory scorer is replay-only and still narrow. Goal 22A
  repaired total collapse, but route-level action diversity is not strong enough
  for a real robot.
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
4. train/evaluate runtime-BEV policy with route-level action-diversity gates;
5. keep all outputs replay-only until hardware safety exists.
