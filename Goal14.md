/goal Build HomeBrain Goal 14: behavior-cloning action labels from dataset future motion, replace the collapsed synthetic-oracle label path, and prove non-collapsed training is reachable on current data.

Read first: AGENTS.md, PROJECT_BRIEF.md, ARCHITECTURE.md, CURRENT_STATUS.md (note Goal 13B finding: primary root cause `oracle_labels_collapsed`, oracle/future-motion agreement = 0.017), BLOCKERS.md, EVALS.md, docs/BEV_ACTION_CONTRACT.md, homebrain/policies/candidate_trajectories.py, homebrain/policies/trajectory_scorer.py, homebrain/policies/build_action_label_pack.py, homebrain/policies/qa_action_label_pack.py, homebrain/policies/trajectory_scorer_net_v0.py, homebrain/policies/train_trajectory_scorer_v0.py, homebrain/policies/eval_trajectory_scorer_v0.py, homebrain/policies/audit_goal13a_collapse.py, homebrain/tools/goal13a_memory_policy_shadow_eval.py.

Objective:
Goal 13B proved the action-label collapse is upstream of the model: the synthetic-oracle scorer ties on ~95% of frames and disagrees with actual dataset future motion 98.3% of the time. Replace the synthetic-oracle label source with a behavior-cloning label source derived from dataset future trajectory. Build a non-collapsed ActionLabelPack v5, retrain TrajectoryScorerNet v0 on it, and rerun Goal 13A using v5 labels so we can finally separate "memory does not help" from "labels were never asking memory to help."

Hard constraints:
- Do not train new spatial memory models. Do not add new teachers. Do not add ROS/Nav2/Isaac/Habitat/sim. Do not emit cmd_vel. Do not output raw PWM. Do not promote any artifact to control_safe.
- All outputs remain replay_only=true, not_executed=true, control_safe=false, product_training_approved=false.
- License-review status flags must remain accurate but do not block work; this is local PoC.
- Do not delete the existing synthetic-oracle ActionLabelPack v3/v4 builder paths; keep them callable for ablation.
- Keep changes minimal. No broad refactors of memory, geometry, or replay code.

Implement:

1. Add homebrain.policies.future_motion_action_labels with a deterministic future-motion → candidate-id matcher:
   - Input per frame t: PoseEvent / OdomEvent sequence on the route, and the candidate trajectory set from candidate_trajectories.py.
   - Compute the robot's actual relative SE(2) trajectory from frame t over a configurable future horizon (default: same horizon used by candidate trajectories, e.g., t to t+H seconds), in the robot frame at t.
   - For each candidate c, compute a deterministic distance between the candidate's integrated poses and the dataset's actual integrated poses over H. Use a weighted sum: position L2 (meters) + yaw L1 (radians) * configurable weight (default 0.5 m/rad). Both candidate and ground-truth trajectories must be resampled to the same time grid before comparison.
   - The matched candidate_id is the argmin distance. Also record: the L2/yaw distances to top-3 candidates, the gap between best and second-best (margin), and a `bc_label_confidence` derived from the margin.
   - Mark frames with insufficient future horizon, missing pose data, or near-stationary motion as `bc_label_valid=false` with explicit reason strings: `future_horizon_truncated`, `pose_missing`, `stationary_below_threshold`. Do not invent labels for those frames.

2. Add homebrain.policies.build_action_label_pack_v5 (new module; do NOT mutate v3/v4 paths):
   - Same inputs as ActionLabelPack v4 (routes + candidate set + BEV/scorer context for sanity flags), but the selected_by_expert field is now sourced from future_motion_action_labels, not from the synthetic scorer.
   - Preserve existing per-candidate sanity columns (collision, near_collision, unknown, coverage, smoothness) for debug — those stay computed by the synthetic scorer — but the SELECTED_ACTION_ID supervision target comes from the future-motion match.
   - Filter to frames where `bc_label_valid=true` AND `action_supervision_ok=true` AND robot_frame_truth=true. Record per-source counts of included/excluded frames and reasons.
   - Write `pack_version=5`, `label_source=future_motion_behavior_cloning`, and `not_synthetic_expert=true` in the manifest. Preserve `replay_only=true`, `control_safe=false`.

3. Add homebrain.policies.qa_action_label_pack_v5 (or extend qa_action_label_pack with a --pack-version 5 path):
   - Report: selected_action_distribution, action_entropy, dominant_action_fraction, per-route action distribution, bc_label_confidence histogram, fraction of frames excluded by reason, and crucially the agreement between v5 labels and the existing v4 synthetic-oracle labels on overlapping frames.
   - Pass gate: action_entropy >= 1.5 (well above the Goal 11B collapse threshold) AND no single action exceeds 50% of frames AND bc_label_confidence_mean above a configurable floor.
   - Fail gate clearly if behavior cloning labels are themselves collapsed on current data; if so, that is a real data-diversity finding, not a label-source bug.

4. Train TrajectoryScorerNet v1 on ActionLabelPack v5:
   - New training entry point: homebrain.policies.train_trajectory_scorer_v1 — same architecture as v0, same input plumbing, only the label source differs. Save to runs/goal14_trajectory_scorer_v1/checkpoint.pt.
   - Train ON OpenLORIS cafe1-1_2 + office1-1_7 + corridor1-1 with the same v4 split discipline (sequence-aware contiguous blocks with embargo).
   - Evaluation entry point: homebrain.policies.eval_trajectory_scorer_v1.
   - Report standard metrics PLUS: agreement with future_motion label on val, agreement with synthetic-oracle label on val (expected to be ~0.02 if BC labels worked), distribution_collapse_flag, per-route held-out top-1 agreement.

5. Rerun Goal 13A using v5-trained scorer:
   - Extend homebrain.tools.goal13a_memory_policy_shadow_eval to accept a `--scorer-checkpoint` flag that overrides the transparent scorer with the learned v1 scorer trained on v5 labels.
   - Write a parallel report at runs/goal14_memory_policy_shadow_eval_v5_report.{json,md} and decisions at runs/goal14_memory_policy_shadow_eval_v5_decisions.jsonl.
   - The gate is now: under non-collapsed v5 labels, does v1 memory change v1 current decisions on a meaningful fraction of frames (target >= 5%), and does memory improve agreement with future_motion label on held-out frames?

6. Run homebrain.policies.audit_goal13a_collapse on the new decisions JSONL so the Goal 13B audit is reused 1:1 against v5 results. Expected: oracle_labels_collapsed should now be false; if memory still doesn't help, root cause moves to memory_delta_too_small_for_action or model_bev_collapsed, which is the *real* question.

7. Tests:
   - Synthetic future-motion test: hand-built pose sequences that should match exactly one candidate id; verify deterministic matching, margin computation, and bc_label_valid flags including all three exclusion reasons.
   - ActionLabelPack v5 test: synthetic route data with diverse future motion produces non-collapsed v5 labels and the QA pass gate fires.
   - Comparison test: v4 vs v5 agreement on the same synthetic frames should reflect the constructed disagreement (i.e., the test should construct frames where the synthetic scorer would pick rotate_left but the future motion is straight_short, and verify v5 picks straight_short).
   - Determinism test: rerunning the v5 build produces byte-identical pack files.
   - Safety-flag test: every v5 example and the scorer v1 outputs carry replay_only=true, not_executed=true, control_safe=false.

8. Repo hygiene (mandatory, do not skip):
   - Move the per-goal "Status log" and "Goal completion log" entries older than Goal 12 from CURRENT_STATUS.md into docs/status_archive/2026-05_goals_00_to_11.md. Preserve content byte-for-byte; only the location changes.
   - Move resolved-blocker sections older than Goal 12 from BLOCKERS.md into docs/blockers_archive/2026-05_resolved.md.
   - Update AGENTS.md so the required Codex read-list explicitly excludes archive paths.
   - After this archive pass, CURRENT_STATUS.md must be under 250 lines and BLOCKERS.md must be under 100 lines.

Stopping condition:
- py_compile passes.
- Targeted Goal 14 tests pass.
- Full pytest passes.
- ActionLabelPack v5 exists with QA report and passes the non-collapse gate, OR honestly fails it with a clear data-diversity finding logged to BLOCKERS.md.
- TrajectoryScorerNet v1 checkpoint exists with eval JSON.
- Goal 14 memory-policy shadow eval and audit artifacts exist.
- CURRENT_STATUS.md and BLOCKERS.md are archived/trimmed as specified.
- Updated docs accurately describe whether memory helps a non-collapsed scorer.

Reporting:
In CURRENT_STATUS.md "Recommended next goal", state explicitly which of the following the v5 results imply: (a) memory now helps — proceed to multi-source data widening (ScanNet/Habitat/SCAND); (b) memory still does not help on non-collapsed labels — current data is the bottleneck, recommend widening data before any further memory work; (c) v5 labels themselves collapsed — single-platform data is too narrow for behavior cloning, recommend widening data before any further policy work. Do not pick the optimistic option unless metrics support it.