GOAL: Build Passive Dynamic Future-Risk Pack v0 — an end-to-end data, training, evaluation, and runtime-integration path that turns passive real-world video sequences into action-conditioned FutureBEV rollout training examples, then proves on held-out dynamic scenes that HomeBrain predicts future collision/risk for candidate trajectories better than static/copy-forward baselines.

Hard constraints:
- Do not use a real robot.
- Do not use simulation.
- Do not emit hardware commands.
- Do not require classical SLAM as a mandatory dependency.
- Do not update README/docs for acceptance.
- Runtime must not receive future labels, teacher artifacts, oracle costs, or ground-truth poses.
- Keep all runtime outputs replay_only=true, control_safe=false, raw_pwm_emitted=false, hardware_validated=false.

Implement:

1. Add a new passive dynamic scene pack builder:
   - Create `homebrain/train/passive_dynamic_future_risk_pack.py`.
   - Input: a directory of frame sequences with optional sidecar files:
     - RGB frames.
     - Optional depth maps.
     - Optional camera/ego poses.
     - Optional semantic masks/tracks from external open models.
     - Optional dynamic object masks.
   - Output: a FutureBEVRollout-compatible pack with:
     - current_bev_free / occupied / unknown / traversable / risky
     - memory_bev_*
     - bev_history
     - uncertainty_map
     - future_free / future_occupied / future_unknown
     - future_risk
     - future_valid_mask
     - candidate_footprint_masks
     - candidate_collision
     - candidate_future_collision
     - candidate_horizon_future_risk
     - candidate_oracle_cost
     - candidate_valid_mask
   - The pack must use the existing `FutureBEVRolloutDataset` without breaking old packs.

2. Add a pluggable teacher interface:
   - Create `homebrain/teachers/passive_dynamic_teacher.py`.
   - Define a deterministic schema for teacher sidecars:
     - depth
     - dynamic_mask
     - semantic_mask
     - track_id_mask
     - camera_pose_delta, optional
     - confidence
   - Include a simple built-in deterministic teacher for tests:
     - moving blob crossing path
     - person/pet-like dynamic obstacle
     - occluder appearing/disappearing
     - ego-camera pan/translation
   - Do not require heavy external models in CI.
   - The production path should allow sidecars from open models, but tests should run with deterministic fixtures.

3. Add route-heldout evaluation:
   - Create `homebrain/eval/eval_passive_dynamic_future_risk.py`.
   - Train `FutureBEVRolloutV1` on one or more passive dynamic routes.
   - Evaluate on held-out routes/scenes.
   - Compare against:
     - current BEV copy-forward
     - static memory copy-forward
     - transparent scorer
     - stop-only
     - unknown-is-dangerous scorer
   - Report:
     - future_risk_iou_or_proxy
     - candidate_future_risk_mse
     - candidate_risk_ranking_accuracy
     - unsafe_selected_rate
     - future collision recall on moving obstacles
     - action entropy / dominant_action_fraction
     - improvement_vs_copy_forward
     - improvement_vs_static_memory

4. Runtime integration:
   - Ensure `decide_trajectory` can consume the trained FutureBEV rollout checkpoint using only:
     - current LocalBev
     - memory_bev
     - bev_history
     - uncertainty_map
     - patch_features / sensor_mask if available
   - Add an artifact field that explains why the selected candidate changed:
     - selected_candidate_id_before_future_risk
     - selected_candidate_id_after_future_risk
     - candidate_dynamic_future_risk
     - candidate_static_risk
     - future_risk_avoidance_reason
   - The runtime path must reject or ignore future labels and teacher-only fields.

5. Tests:
   - Add tests proving:
     - pack generation is deterministic
     - pack examples load through `FutureBEVRolloutDataset`
     - moving obstacle crossing a straight trajectory produces higher future risk for straight candidates than stop/turn candidates
     - held-out fixture eval beats copy-forward on future risk
     - runtime selection changes when future dynamic risk blocks the transparent scorer’s preferred path
     - no future labels, oracle costs, ground-truth poses, or teacher artifacts enter runtime scoring
     - all safety flags remain replay_only/not_executed/control_safe=false/raw_pwm_emitted=false/hardware_validated=false

Acceptance:
- A single command can generate the passive dynamic pack, train a tiny FutureBEV model, evaluate it, and write a JSON report.
- The report must keep accepted_robot_brain=false, but may set accepted_passive_dynamic_future_risk_probe=true only if:
  - held-out candidate_future_risk_mse beats copy-forward and static-memory baselines,
  - moving-obstacle collision recall improves over copy-forward,
  - unsafe_selected_rate is 0 on deterministic held-out dynamic fixtures,
  - dominant_action_fraction < 0.95,
  - runtime leakage tests pass.