GOAL: Goal30 — Build Counterfactual Dynamic BEV World Model V0

Read first:
- README.md
- ARCHITECTURE.md
- DECISIONS.md
- EVALS.md
- GOAL.md
- homebrain/brain/direct_bev_student_v0.py
- homebrain/runtime/direct_bev_runtime.py
- homebrain/train/real_rgbd_route_bev_pack.py
- homebrain/eval/eval_real_rgbd_brain_slice.py
- existing candidate trajectory / scorer / SceneState modules

Purpose:
Build the missing world-model layer between LocalBev/SceneState and candidate trajectory scoring.

The new module must learn:
  recent local BEV history + ego motion history + previous action
  -> future local BEV occupancy/risk/flow at multiple horizons
  -> candidate trajectory risk scores

This must be replay-only. It must not control hardware.

Do not create another disconnected demo.
Do not add markdown-only progress.
Do not use future frames, teacher labels, target BEV, oracle candidate costs, route ground truth, or teacher masks at runtime.
Do not require classical SLAM at runtime.
Do not depend on a giant VLM at runtime.
Do not emit raw PWM or hardware commands.

Create these modules:

1. Dynamic world-model pack builder

Create:
  homebrain/train/dynamic_bev_world_pack_v0.py

Command:
  python -m homebrain.train.dynamic_bev_world_pack_v0 \
    --input-pack artifacts/real_rgbd_route_bev_pack \
    --out artifacts/dynamic_bev_world_pack_v0 \
    --history-frames 6 \
    --future-horizons-sec 0.5,1.0,2.0,3.0 \
    --candidate-set default_low_speed_indoor \
    --max-examples 50000

Input:
  Existing RealRGBDRouteBEVPack examples if available.

Build training windows:
  history:
    current/past local BEV channels:
      free
      occupied
      unknown
      traversable
      risky
      dynamic_residual_risk
      uncertainty
    pose_delta history
    previous action if available
    observed mask / sensor mask

  future teacher-only targets:
    future_occupied[h]
    future_free[h]
    future_unknown[h]
    future_risky[h]
    future_dynamic_risk[h]
    future_flow_xy[h] if derivable
    future_valid_mask[h]

  candidate targets:
    candidate_id
    candidate_trajectory_xytheta
    candidate_collision_risk[h]
    candidate_dynamic_risk[h]
    candidate_unknown_exposure[h]
    candidate_total_teacher_risk
    candidate_safe_label
    candidate_rank_target

Important:
  Candidate labels are created by overlaying the robot footprint along each candidate trajectory
  against future teacher BEV occupancy/risk.
  This is allowed offline only.
  This gives counterfactual labels for candidates that were not actually executed.

Manifest must include:
  package_type = DynamicBEVWorldPackV0
  source_pack_sha256
  route_heldout_split_basis
  train_route_ids
  val_route_ids
  history_frames
  future_horizons_sec
  candidate_count
  real_source_route_count
  real_source_frame_count
  dynamic_positive_frame_count
  candidate_positive_risk_fraction
  synthetic_or_fixture
  no_future_labels_at_runtime = true
  replay_only = true
  control_safe = false

Missing-data behavior:
  If input pack is missing:
    exit 0
    write a report JSON
    accepted_dynamic_bev_world_pack = false
    reason includes missing_real_rgbd_route_bev_pack
    do not invent data

Fixture behavior:
  Tiny fixture packs may be used for tests but must have:
    synthetic_or_fixture = true
    accepted_dynamic_bev_world_pack = false

2. Counterfactual world model

Create:
  homebrain/brain/counterfactual_dynamic_bev_world_model_v0.py

Class:
  CounterfactualDynamicBEVWorldModelV0

Inputs:
  bev_history: [B, T, C, H, W]
  pose_delta_history: [B, T, D]
  previous_action_history: optional [B, T, A]
  candidate_trajectories: [B, K, P, 3]
  sensor_mask: optional

Outputs:
  future_occupied_logits: [B, HN, H, W]
  future_free_logits
  future_unknown_logits
  future_risky_logits
  future_dynamic_risk_logits
  future_flow_xy: [B, HN, 2, H, W]
  future_uncertainty_logits
  candidate_risk_logits: [B, K]
  candidate_dynamic_risk_logits: [B, K]
  candidate_unknown_exposure_logits: [B, K]
  candidate_score: [B, K]
  latent_memory_debug optional

Architecture:
  Keep it small enough for two 4090s:
    BEV encoder: small CNN / ConvNeXt-like blocks
    temporal memory: ConvGRU or small temporal transformer
    future decoder: multi-horizon U-Net-ish decoder
    candidate scorer:
      deterministic footprint/risk pooling from predicted future BEV
      plus tiny MLP for smooth learned scoring
  CPU smoke mode must work.
  AMP training must work.
  No giant VLM dependency.

3. Training

Create:
  homebrain/train/train_counterfactual_dynamic_bev_world_model_v0.py

Command:
  python -m homebrain.train.train_counterfactual_dynamic_bev_world_model_v0 \
    --pack artifacts/dynamic_bev_world_pack_v0 \
    --out artifacts/counterfactual_dynamic_bev_world_model_v0 \
    --device cuda \
    --batch-size 32 \
    --max-steps 20000 \
    --amp

Losses:
  current/future occupancy:
    BCE + Dice/Focal for occupied/free/unknown/risky/dynamic_risk

  dynamic flow:
    Charbonnier/EPE only on valid dynamic occupied cells

  candidate risk:
    BCE for candidate_safe_label / candidate_risk
    pairwise ranking loss so safer candidates rank above risky candidates
    optional soft target regression to teacher total risk

  uncertainty:
    Brier or NLL-like calibration proxy
    penalize confident wrong free-space

  temporal consistency:
    static occupancy should be equivariant under ego-motion
    unknown space should not become free without evidence

Loss report must separately include:
  loss_future_occupied
  loss_future_dynamic_risk
  loss_future_flow
  loss_candidate_bce
  loss_candidate_rank
  loss_uncertainty
  loss_equivariance
  loss_total

Checkpoint metadata must include:
  model_type = CounterfactualDynamicBEVWorldModelV0
  pack_sha256
  train_route_ids
  val_route_ids
  future_horizons_sec
  candidate_count
  no_future_labels_used_at_runtime = true
  no_teacher_fields_at_runtime = true
  replay_only = true
  not_executed = true
  control_safe = false
  raw_pwm_emitted = false
  hardware_validated = false

4. Evaluation

Create:
  homebrain/eval/eval_counterfactual_dynamic_bev_world_model_v0.py

Command:
  python -m homebrain.eval.eval_counterfactual_dynamic_bev_world_model_v0 \
    --checkpoint artifacts/counterfactual_dynamic_bev_world_model_v0/checkpoint.pt \
    --pack artifacts/dynamic_bev_world_pack_v0 \
    --split val \
    --out artifacts/counterfactual_dynamic_bev_world_model_v0/eval.json

Compare against baselines:
  static_copy_baseline
  previous_frame_copy_baseline
  constant_flow_baseline if flow labels exist
  center_prior_baseline
  transparent_teacher_current_bev_scorer

Metrics:
  future_occupied_auprc_by_horizon
  future_occupied_iou_by_horizon
  future_dynamic_risk_auprc_by_horizon
  future_dynamic_risk_f1_by_horizon
  future_flow_epe_by_horizon
  uncertainty_brier_or_ece_proxy
  candidate_risk_auroc
  candidate_risk_auprc
  candidate_rank_spearman
  candidate_top1_high_risk_rate
  candidate_top3_contains_safe_rate
  improvement_vs_static_copy
  improvement_vs_previous_frame_copy
  improvement_vs_center_prior
  route_heldout_count
  runtime_field_leakage_passed

Acceptance can be true only if:
  synthetic_or_fixture = false
  real_source_route_count >= 3
  real_source_frame_count >= 1000
  route-heldout split is true
  no train route appears in val
  val dynamic_positive_frame_count > 0
  future_dynamic_risk_auprc beats previous_frame_copy by >= 0.05 absolute
  future_occupied_auprc beats static_copy by >= 0.05 absolute
  candidate_risk_auroc >= 0.70
  candidate_top1_high_risk_rate is lower than static_copy scorer
  runtime_field_leakage_passed = true
  no_future_labels_used_at_runtime = true
  no_teacher_fields_at_runtime = true
  replay_only = true
  not_executed = true
  control_safe = false
  raw_pwm_emitted = false
  hardware_validated = false

If thresholds fail:
  write eval.json
  accepted_counterfactual_dynamic_bev_world_model_v0 = false
  list failed checks
  do not fake success

5. Runtime candidate scorer

Create:
  homebrain/runtime/counterfactual_world_model_scorer.py

Expose:
  load_counterfactual_world_model(checkpoint, device)
  predict_future_bev_from_scene_state(...)
  score_candidates_with_world_model(...)

Runtime input may contain only:
  SceneState / LocalBev history
  current/past predicted BEV
  pose_delta history
  previous action
  candidate trajectories
  checkpoint

Runtime must not accept:
  target_bev
  future_bev
  future labels
  teacher masks
  route ground truth
  candidate_oracle_cost
  ground_truth_global_trajectory
  future frames

Runtime output:
  candidate scores
  future risk debug maps
  replay-only decision debug artifact

Must include debug flags:
  no_future_labels_used_at_runtime = true
  no_teacher_fields_at_runtime = true
  replay_only = true
  not_executed = true
  control_safe = false
  raw_pwm_emitted = false
  hardware_validated = false

6. Brain-slice integration

Modify or add:
  homebrain/eval/eval_dynamic_world_model_brain_slice_v0.py

It should run:
  RealRGBD/DirectBEV or teacher LocalBev history
  -> SceneState memory
  -> CounterfactualDynamicBEVWorldModelV0
  -> candidate scoring
  -> replay-only selected candidate artifact

Report:
  selected_candidate_id distribution
  stop rate
  high-risk selected rate
  unknown-exposure selected rate
  candidate change rate vs old scorer
  cases where world model prevented risky selection
  cases where world model made selection worse
  no runtime leakage

7. Tests

Add:
  tests/test_dynamic_bev_world_pack_v0.py
  tests/test_counterfactual_dynamic_bev_world_model_v0.py
  tests/test_counterfactual_world_model_runtime_leakage.py
  tests/test_counterfactual_world_model_eval_v0.py
  tests/test_dynamic_world_model_brain_slice_v0.py

Tests must prove:
  1. Missing input pack exits cleanly with accepted=false.
  2. Tiny fixture pack cannot pass real acceptance.
  3. Candidate risk labels are generated by footprint overlap with future BEV.
  4. Train/val route overlap fails.
  5. Model forward pass works on CPU with tiny tensors.
  6. Training smoke run writes checkpoint metadata and loss report.
  7. Eval compares against static_copy and previous_frame_copy baselines.
  8. Runtime function signatures reject target_bev, future labels, teacher masks, oracle costs, route ground truth, and future frames.
  9. Brain-slice eval writes selected candidate artifacts and safety flags.
  10. Existing tests still pass.

Definition of done:
  - One end-to-end world-model pack/train/eval/runtime path exists.
  - The model predicts multi-horizon future BEV/risk/flow.
  - Candidate risk is learned/evaluated from counterfactual footprint overlap.
  - The runtime scorer can plug into the existing candidate decision stack.
  - The result is useful even when no real data exists, because it reports exactly what is missing.
  - The result is useful when real Bonn/TUM/OpenLORIS-style data exists, because it can train and compare to strong baselines.
  - The repo moves from current-frame BEV perception toward actual dynamic navigation intelligence.