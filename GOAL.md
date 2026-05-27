/goal Repair the Goal29 fixture slice so it is honest, smaller, and actually connected to the runtime world-model path.

Read AGENTS.md, CURRENT_STATUS.md, EVALS.md, and GOAL.md only. Then inspect the changed Goal29 code.

Objectives:

1. Rename the current result conceptually from “Goal29 accepted” to “Goal29a fixture probe”.
   - Do not call it the latest robot-brain baseline.
   - Keep Goal28 as the latest route-heldout public-data robot-brain baseline.
   - CURRENT_STATUS.md must clearly say:
     accepted_goal29a_fixture_probe=true
     accepted_goal29_route_heldout=false
     accepted_goal29_robot_brain=false

2. Make the runtime FutureBEV scoring path actually consume online-available memory/history/uncertainty.
   - Extend score_local_bev_with_future_rollout(...) to accept memory_bev, bev_history, uncertainty_map.
   - In the SpatialMemoryV1/SceneState runtime path, pass scene/memory/history/uncertainty when available.
   - Preserve fallback behavior if unavailable.
   - Add tests proving the runtime helper changes output when memory/history/risk changes.

3. Add a no-leakage runtime test stronger than key-name checks.
   - It should prove runtime scoring cannot access future_bev, future_risk, candidate_oracle_cost, teacher artifacts, or route ground truth.
   - The test should fail if future labels are accidentally threaded into decide_trajectory.

4. Fix report acceptance semantics.
   - Fixture report may set accepted_fixture_probe=true.
   - It must not set accepted robot-brain milestone true.
   - Gate H should be “exercised” not “improved” unless a replay log demonstrates watchdog/stop/recovery metrics.

5. If the report path is referenced in CURRENT_STATUS.md, either commit the report artifact or mark it as local/uncommitted.
   - Do not reference non-existent primary artifacts.

6. Reduce slop where possible.
   - Merge goal-specific fixture/eval helpers into existing future_bev rollout test/eval structure if practical.
   - If not practical, add comments/docstrings explaining why these files remain separate.
   - Do not add new broad files.

Required tests:
- existing Goal29 tests pass,
- full pytest if feasible,
- new test that memory/history/uncertainty reaches FutureBEV runtime scoring,
- new test that future labels cannot reach runtime decision,
- new test that report distinguishes fixture_probe from robot_brain acceptance.

Safety invariants:
replay_only=true
not_executed=true
control_safe=false
raw_pwm_emitted=false
hardware_validated=false

Do not add new capabilities until this repair is done.