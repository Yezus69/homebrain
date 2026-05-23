# CODEX_GOALS.md

Paste one goal at a time into Codex. Do not start the next goal until the previous goal passes and `CURRENT_STATUS.md` is updated.

## How to recover after a failed goal

Paste this into Codex:

```text
/goal Recover the last failed HomeBrain goal.

First read AGENTS.md, PROJECT_BRIEF.md, CURRENT_STATUS.md, EVALS.md, DECISIONS.md, and BLOCKERS.md if it exists.

Objective:
Make the smallest useful repair that moves the project back toward the active goal without broad refactors.

Requirements:
1. Identify the exact failure from CURRENT_STATUS.md, test output, or BLOCKERS.md.
2. Fix the root cause or reduce the scope to a smaller useful subset.
3. Re-run the verification commands.
4. Update CURRENT_STATUS.md with the repair, commands, results, artifacts, and next recommended goal.
5. If the original goal remains blocked, update BLOCKERS.md with a minimal next action.

Do not add large dependencies. Do not rewrite unrelated code. Do not claim success unless verification passes.
```

---

## Goal 0 — Build the log/replay/eval spine

```text
/goal Build HomeBrain Goal 0: deterministic log/replay/eval spine.

Before coding, read:
- AGENTS.md
- PROJECT_BRIEF.md
- ARCHITECTURE.md
- REQUIREMENTS.md
- EVALS.md
- SCHEMA.md
- DECISIONS.md
- CURRENT_STATUS.md

Objective:
Create the minimal production-directed foundation for HomeBrain: typed events, segment logs, deterministic replay, dummy model output, eval metrics, and tests.

Hard constraints:
- Do not add ROS, Nav2, Habitat, Isaac, or large ML dependencies.
- Do not add torch yet.
- Do not build a simulator.
- Do not create fake model performance claims.
- Keep the implementation lean and testable.

Implement:
1. Python package `homebrain`.
2. Event schema for FrameEvent, ImuEvent, WheelEvent, CommandEvent, BrainOutputEvent, EvalEvent, SegmentManifest.
3. Segment log writer and reader.
4. Deterministic dummy log generator:
   - `python -m homebrain.replay.generate_dummy_log --out runs/dummy_route`
5. Deterministic replay command:
   - `python -m homebrain.replay.replayd --log runs/dummy_route --out runs/replayed_route`
6. Dummy `modeld` behavior that consumes replayed events and emits valid BrainOutputEvent records.
7. Eval command:
   - `python -m homebrain.eval.run_eval --log runs/dummy_route --out runs/dummy_eval.json`
8. Metrics:
   - event_count
   - frame_count
   - dropped_frame_count
   - event_ordering_error_count
   - replay_determinism_pass
   - brain_output_count
   - eval_runtime_sec
9. Tests:
   - schema serialization/deserialization
   - log write/read round trip
   - replay determinism
   - eval metrics
10. `.gitignore` for generated `runs/`, caches, checkpoints.
11. Update EVALS.md if actual commands differ.
12. Update CURRENT_STATUS.md at the end.

Stopping condition:
- `pytest -q` passes.
- dummy log generation works.
- replay works.
- eval writes JSON metrics.
- CURRENT_STATUS.md is updated with commands, results, artifacts, and next recommended goal.
```

---

## Goal 1 — Add teacher artifact interface, no heavy model dependency yet

```text
/goal Build HomeBrain Goal 1: teacher artifact interface and mock teacher.

Before coding, read AGENTS.md, CURRENT_STATUS.md, EVALS.md, LICENSE_AUDIT.md, DATA_STRATEGY.md, MODEL_SPEC.md.

Objective:
Add a teacher-lab interface that can run foundation-model teachers later, but for now works with a deterministic mock teacher and produces artifact files suitable for student training.

Hard constraints:
- Do not require external model weights.
- Do not install giant ML dependencies yet unless necessary.
- Mock outputs must be clearly labeled `mock: true`.
- Do not report mock teacher metrics as real performance.
- Preserve Goal 0 commands.

Implement:
1. `homebrain.teachers` package.
2. Teacher plugin interface:
   - input: frame sequence or segment log
   - output: deterministic artifact directory
3. Mock teacher that emits:
   - depth array
   - confidence array
   - simple dense feature array
   - metadata.json with `mock: true`
4. Teacher run command:
   - `python -m homebrain.teachers.run_teacher --teacher mock --log runs/dummy_route --out runs/mock_teacher_artifacts`
5. Teacher artifact loader.
6. Visualization command:
   - `python -m homebrain.teachers.visualize_artifacts --artifacts runs/mock_teacher_artifacts --out runs/mock_teacher_preview`
7. Tests for:
   - artifact writing/loading
   - deterministic mock teacher output
   - visualization output existence
8. Update LICENSE_AUDIT.md with a section saying real teacher dependencies are not yet added.
9. Update EVALS.md with teacher commands/metrics.
10. Update CURRENT_STATUS.md.

Stopping condition:
- Goal 0 tests still pass.
- mock teacher command works.
- visualization command creates an artifact.
- CURRENT_STATUS.md updated.
```

---

## Goal 2 — Add first real image/video ingestion path

```text
/goal Build HomeBrain Goal 2: real indoor video/image ingestion.

Before coding, read AGENTS.md, CURRENT_STATUS.md, DATA_STRATEGY.md, SCHEMA.md, EVALS.md.

Objective:
Allow HomeBrain to ingest a user-provided indoor video or image folder into the same segment-log format used by replay/eval.

Hard constraints:
- Keep dependencies minimal.
- If OpenCV is added, justify it in CURRENT_STATUS.md.
- Preserve all previous commands/tests.
- Do not add ML yet.

Implement:
1. CLI to convert image folder to segment log:
   - `python -m homebrain.data.ingest_images --images <folder> --out runs/image_route`
2. CLI to convert video to segment log if OpenCV is available:
   - `python -m homebrain.data.ingest_video --video <path> --out runs/video_route`
3. Generate FrameEvent records with stable timestamps.
4. Store frames or data references deterministically.
5. Eval/replay must work on ingested routes.
6. Tests using tiny generated images.
7. Update EVALS.md and CURRENT_STATUS.md.

Stopping condition:
- image ingestion works.
- replay works on ingested route.
- eval works on ingested route.
- tests pass.
```

---

## Goal 3 — SpatialMemoryNet v0 with tiny overfit test

```text
/goal Build HomeBrain Goal 3: SpatialMemoryNet v0.

Before coding, read AGENTS.md, CURRENT_STATUS.md, MODEL_SPEC.md, EVALS.md, DATA_STRATEGY.md.

Objective:
Implement the first small trainable student model that consumes teacher artifacts/log data and predicts BEV-like local spatial outputs, pose delta, and uncertainty.

Hard constraints:
- Keep model small.
- Torch may be added only if needed and must be documented.
- No giant pretrained models in this goal.
- Must include a tiny overfit test.
- Must integrate with modeld/replay enough to emit BrainOutputEvent from a checkpoint or mock checkpoint.

Implement:
1. `homebrain.brain` package.
2. SpatialMemoryNet v0 with:
   - simple visual/feature adapter
   - temporal/action/odom fusion
   - BEV head
   - pose_delta head
   - uncertainty head
3. Dataset loader for mock teacher artifacts.
4. Training command:
   - `python -m homebrain.brain.train_spatial_memory --artifacts runs/mock_teacher_artifacts --out runs/spatial_memory_v0`
5. Eval command:
   - `python -m homebrain.brain.eval_spatial_memory --checkpoint runs/spatial_memory_v0/checkpoint.pt --artifacts runs/mock_teacher_artifacts --out runs/spatial_memory_v0_eval.json`
6. Tiny overfit test.
7. Replay/modeld integration:
   - modeld can use dummy mode or checkpoint mode.
8. Metrics:
   - train_loss
   - bev_loss
   - pose_loss
   - uncertainty_loss
   - inference_runtime_ms
9. Update EVALS.md and CURRENT_STATUS.md.

Stopping condition:
- model overfits tiny data enough to show decreasing loss.
- tests pass.
- eval writes metrics.
- replay/modeld can emit BrainOutputEvent using the model path or a clearly documented fallback.
```

---

## Goal 4 — Candidate trajectory generator and scorer

```text
/goal Build HomeBrain Goal 4: candidate trajectory generator and scorer.

Before coding, read AGENTS.md, CURRENT_STATUS.md, ARCHITECTURE.md, MODEL_SPEC.md, EVALS.md.

Objective:
Add candidate-based motion decision logic. The brain should generate short differential-drive candidate trajectories and score them using BEV/risk/coverage/uncertainty proxies.

Hard constraints:
- Do not output raw motor PWM.
- Do not require a real robot.
- Do not require a simulator.
- Keep scoring debuggable.

Implement:
1. `homebrain.policies` package.
2. Candidate trajectory generator for differential-drive actions:
   - forward slow/medium
   - left/right arcs
   - rotate left/right
   - reverse
   - stop
3. Trajectory scoring function using available BEV/uncertainty/proxy coverage data.
4. BrainOutputEvent includes:
   - candidate_trajectories
   - selected_trajectory_id
   - cmd_vel
   - debug score breakdown
5. Replay visualization or JSON report showing selected candidates over time.
6. Tests:
   - deterministic candidate generation
   - unsafe/risky candidate penalization
   - stop candidate selected under high uncertainty
7. Update EVALS.md and CURRENT_STATUS.md.

Stopping condition:
- tests pass.
- replay output includes selected trajectory and debug scores.
- eval writes trajectory metrics.
```

---

## Goal 5 — Architecture alignment review

```text
/goal Perform a HomeBrain architecture alignment review and repair pass.

Before coding, read every root doc:
- AGENTS.md
- PROJECT_BRIEF.md
- ARCHITECTURE.md
- REQUIREMENTS.md
- EVALS.md
- CURRENT_STATUS.md
- DECISIONS.md
- DATA_STRATEGY.md
- MODEL_SPEC.md
- FAILURE_MODES.md
- LICENSE_AUDIT.md

Objective:
Check whether the current codebase still matches the HomeBrain mission. Repair small drift. Do not add new features unless they are required to restore alignment.

Review:
1. Are all documented commands still correct?
2. Does CURRENT_STATUS.md reflect reality?
3. Are generated artifacts gitignored?
4. Are there fake metrics or mislabeled mocks?
5. Did any dependency violate the dependency policy?
6. Does every module support log/replay/eval?
7. Are tests meaningful?
8. Is there any toy-only code pretending to be production path?

Deliverables:
1. Update CURRENT_STATUS.md with the review result.
2. Update EVALS.md commands if needed.
3. Update DECISIONS.md only if a real decision changed.
4. Add BLOCKERS.md if there are unresolved issues.
5. Make minimal code/doc fixes required for tests and command correctness.

Stopping condition:
- tests pass.
- standard commands run.
- review notes are in CURRENT_STATUS.md.
```
