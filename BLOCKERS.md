# BLOCKERS.md

Only active blockers live here. Historical command transcripts were removed.

## Active Blockers

None that prevent local POC development.

## Active Risks

- No real robot hardware loop exists yet: no synchronized RGB/IR/IMU/wheel logs,
  no controller daemon, no watchdog, and no physical safety gate.
- Current learned trajectory scorer is replay-only and still narrow. Goal26
  repaired the Goal25 one-action collapse in the accepted direct RGB-D heldout
  runtime path, but it uses a hand-weighted transparent/FutureBEV guided
  selector with a temporal diversity prior. This is not yet a learned,
  route-general control policy strong enough for a real robot.
- The online-style `Brain.step(...)` API now exists for replay and owns memory
  across ticks, but it is not yet a hardware daemon and has not been validated
  with synchronized owned RGB/IR/IMU/wheel/command logs.
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
5. train/evaluate a learned runtime-BEV policy that keeps Goal26's
   route-level action-diversity gate without a hand-weighted diversity prior;
6. improve scene-memory unknown/observed-cell metrics and direct RGB-D student
   quality on route-heldout real data;
7. keep all outputs replay-only until hardware safety exists.
