# CURRENT_STATUS.md

Codex must update this file at the end of every goal. Historical goal logs older
than Goal 12 are archived under `docs/status_archive/` and are not part of the
default read-list.

## Current objective

Goal 14 completed: replace the collapsed synthetic-oracle action-label path with
future-motion behavior-cloning labels, build ActionLabelPack v5, train/evaluate a
v5 TrajectoryScorerNet entry point, rerun Goal 13A with the v5 scorer, and reuse
the Goal 13B audit.

## Last completed goal

Goal 14: behavior-cloning action labels from dataset future motion. The v5 label
pack is non-collapsed, but memory still does not improve learned scorer decisions
on current OpenLORIS-only data.

## Current implementation status

- `homebrain` package contains deterministic log/replay/eval, teacher artifacts,
  image/video ingest, geometry-to-BEV, SpatialTrainPack, spatial v0/v1 training,
  trajectory scoring, and policy audit tools.
- Runtime/control constraints remain intact: no ROS/Nav2/Isaac/Habitat/sim was
  added, no raw PWM is emitted, and current policy artifacts keep
  `replay_only=true`, `not_executed=true`, `control_safe=false`, and
  `product_training_approved=false`.
- SpatialMemoryNet v1 has explicit persistent BEV memory, route-pose/odom warp
  defaults, conservative update masks, hard validation, and true route-out folds.
  Goal 12C showed useful memory in spatial eval, but not control-safe behavior.
- The old synthetic coverage/risk action oracle remains callable for ablation in
  v3/v4 builders, but Goal 13B showed it collapsed and disagreed with real future
  motion.
- Goal 14 added `homebrain.policies.future_motion_action_labels` and
  `build_action_label_pack_v5`; v5 labels are matched by resampled relative
  future trajectories and invalid frames are explicitly labeled
  `future_horizon_truncated`, `pose_missing`, or `stationary_below_threshold`.
- Goal 14 added TrajectoryScorerNet v1 train/eval entry points that reuse the v0
  architecture/input plumbing with v5 labels and class-balanced loss.
- Goal 14 updated Goal 13A shadow eval with `--scorer-checkpoint`; in scorer mode
  `oracle_bev` is the v5 behavior-cloning label oracle, while v0/v1 current and
  memory BEVs are scored by the learned checkpoint.
- Active blockers are tracked in `BLOCKERS.md`; resolved old blockers are archived
  under `docs/blockers_archive/`.

## Goal completion log

### 024 - Goal 14 future-motion behavior-cloning action labels

Objective attempted: build behavior-cloning action labels from dataset future
motion, replace the collapsed synthetic-oracle label source for the new v5 path,
prove non-collapsed labels are reachable on current data, train/evaluate a v5
scorer, rerun Goal 13A with the v5 scorer, reuse the Goal 13B audit, and trim
status/blocker archives.

Files changed: added `homebrain/policies/future_motion_action_labels.py`,
`homebrain/policies/build_action_label_pack_v5.py`,
`homebrain/policies/qa_action_label_pack_v5.py`,
`homebrain/policies/train_trajectory_scorer_v1.py`,
`homebrain/policies/eval_trajectory_scorer_v1.py`, and
`tests/test_goal14_behavior_cloning_labels.py`; updated
`homebrain/policies/build_action_label_pack.py`,
`homebrain/policies/qa_action_label_pack.py`,
`homebrain/policies/trajectory_scorer_net_v0.py`,
`homebrain/tools/goal13a_memory_policy_shadow_eval.py`, `AGENTS.md`,
`ARCHITECTURE.md`, `EVALS.md`, `BLOCKERS.md`, and `CURRENT_STATUS.md`; added
archives under `docs/status_archive/` and `docs/blockers_archive/`.

Commands run: required context reads; py_compile for new/modified Goal 14 modules;
targeted tests `python -m pytest tests\test_goal14_behavior_cloning_labels.py
tests\test_goal13a_memory_policy_shadow_eval.py
tests\test_goal11a_trajectory_scorer_v0.py -q`; real v5 build/QA; v1 scorer
train/eval; Goal 13A v5 shadow eval; Goal 13B audit on v5 decisions.

Pass/fail results: targeted tests passed with `11 passed`. ActionLabelPack v5 QA
passed the non-collapse gate. TrajectoryScorerNet v1 eval wrote metrics and kept
all safety flags false. Goal 13A v5 shadow eval intentionally failed the memory
action-benefit gate; this is now a data/model bottleneck finding, not a label
collapse.

Artifacts created: `runs/goal14_action_label_pack_v5/`,
`runs/goal14_action_label_pack_v5_qa.json`,
`runs/goal14_trajectory_scorer_v1/checkpoint.pt`,
`runs/goal14_trajectory_scorer_v1_eval.json`,
`runs/goal14_trajectory_scorer_v1_eval_predictions.jsonl`,
`runs/goal14_trajectory_scorer_v1_viz/`,
`runs/goal14_memory_policy_shadow_eval_v5_report.json`,
`runs/goal14_memory_policy_shadow_eval_v5_report.md`,
`runs/goal14_memory_policy_shadow_eval_v5_decisions.jsonl`,
`runs/goal14_memory_policy_shadow_eval_v5_worst.ppm`,
`runs/goal14_policy_collapse_audit_v5.json`,
`runs/goal14_policy_collapse_audit_v5.md`, and
`runs/goal14_policy_collapse_worst_v5.ppm`.

Metrics observed: v5 QA reported `example_count=2894`,
`action_entropy=2.3334583564283053`, `dominant_action_fraction=0.4644091223220456`,
`bc_label_confidence_mean=0.25870618115853394`, excluded frames `315`
(`future_horizon_truncated=216`, `stationary_below_threshold=99`), and
v5/synthetic-oracle agreement `0.0`. V1 scorer val reported
`top1_action_agreement=0.49568221070811747`, `beats_random=true`,
`agreement_with_synthetic_oracle_label=0.0`, `distribution_collapse_flag=false`,
and selected distribution `straight_medium=416`, `stop=79`,
`arc_right_medium=61`, `straight_short=23`. Goal 13A v5 shadow eval reported
normal memory action changed fraction `0.0`, current and memory future-motion
agreement both `0.18764302059496568`, and `memory_action_benefit_pass=false`.
The reused audit reported `oracle_labels_collapsed=false` and primary root cause
`memory_delta_too_small_for_action`, with `model_bev_collapsed=true`.

Blockers/risks: OpenLORIS remains local PoC only and not product-training
approved. The learned v5 scorer still collapses on model-BEV shadow decisions,
even though labels are non-collapsed. All artifacts remain replay/eval only,
not executed, not control-safe, and no `cmd_vel` or raw PWM was emitted.

Recommended next goal: option (b). Memory still does not help on non-collapsed
labels, so current data/model signal is the bottleneck. Widen robot-frame data
before further memory work or policy claims: add more diverse owned/public
robot-frame logs, rebuild v5 labels, and rerun scorer plus route-out/scene-out
shadow eval.

### 023 - Goal 13B replay-only Goal 13A collapse diagnosis

Summary: diagnosed the old Goal 13A collapse as `oracle_labels_collapsed` with
tie/order dominance; this is resolved only for the new v5 path. Artifacts:
`runs/goal13b_policy_collapse_audit.json`, `.md`, and `.ppm`.

### 022 - Goal 13A SpatialMemoryV1 memory-to-trajectory shadow evaluation

Summary: transparent scorer memory-action gate failed before v5 labels. This
remains useful as the old synthetic-oracle baseline. Artifacts:
`runs/goal13a_memory_policy_shadow_eval_report.json`, `.md`, decisions JSONL,
and contact sheet.

### 021 - Goal 12C OpenLORIS PoC policy and SpatialMemoryV1 hard validation

Summary: hard v1 spatial-memory validation passed for replay/eval representation
pretraining, including deployment-style, hidden-cell, occlusion, pose ablation,
and true leave-one-route-out checks. Artifacts:
`runs/goal12c_spatial_memory_v1_hard_validation_report.json` and `.md`.

### 020 - Goal 12B SpatialMemoryNetV1 parity and memory-sanity repair

Summary: repaired v1 current-BEV parity and showed route-pose memory benefit in
spatial eval. Artifacts: `runs/goal12b_spatial_memory_v1_parity_report.json`
and `.md`.

### 019 - Goal 12A SpatialMemoryNetV1 temporal egocentric memory

Summary: added the first explicit v1 temporal-memory infrastructure. The tiny
PoC failed the memory-benefit gate against the stronger v0 baseline, which led
to Goal 12B.
