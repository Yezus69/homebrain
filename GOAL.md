GOAL: Goal31 — Unify the brain into one HomeBrainNet with staged training

Read first:
- AGENTS.md, ARCHITECTURE.md, EVALS.md, DECISIONS.md
- homebrain/brain/direct_bev_student_v0.py
- homebrain/brain/spatial_memory_v1.py
- homebrain/brain/counterfactual_dynamic_bev_world_model_v0.py
- homebrain/brain/modeld.py  (this is the runtime loop, NOT a model — do not duplicate it)
- homebrain/datasets/openloris_to_route.py, datasets/tum_rgbd_to_route.py, data/tum_rgbd_route.py
- homebrain/train/real_rgbd_route_bev_pack.py, train/dynamic_bev_world_pack_v0.py
- homebrain/policies/runtime_decision.py, policies/candidate_trajectories.py

Purpose:
Replace four separately-trained encoders with ONE shared-backbone HomeBrainNet that outputs:
  camera pose delta, BEV occupancy, metric depth, dynamic occupancy + BEV flow,
  and (via the existing world-model head) action-conditioned multi-horizon future + candidate scores.
This is a consolidation, not a new demo. Reuse existing heads as submodules. Retire or warm-start
from the old per-module paths; do not leave five parallel encoders.

STEP 0 — Audit before code (required, write artifacts/goal31/audit.md):
  Produce a table: each desired output -> which existing module/head provides it ->
  shared vs duplicated encoder -> what is missing. Do NOT write model code until this exists.
  If the audit shows an existing module already suffices for a head, REUSE it, do not rewrite.

Hard constraints (unchanged invariants — see EVALS.md):
  replay_only=true, not_executed=true, control_safe=false, raw_pwm_emitted=false, hardware_validated=false
  Runtime must NOT consume: target_bev, future_bev, future labels, teacher masks, route ground truth,
  candidate_oracle_cost, ground_truth_global_trajectory, future frames.
  Runtime must NOT require a heavy teacher (DINO/MoGe/VGGT) per tick.
  No classical SLAM at runtime. No raw PWM. CPU smoke must pass. AMP must work. Fits two 4090s.

1. Datasets — add the missing one, keep one route format
   Create homebrain/datasets/bonn_rgbd.py, bonn_rgbd_to_route.py, setup_bonn_rgbd.py
   mirroring the TUM importer. Bonn RGB-D Dynamic has RGB-D + groundtruth trajectory + people.
   All of OpenLORIS / TUM / Bonn must import into the SAME route format and feed the SAME packs.
   Missing-data behavior (mandatory): if a dataset root is absent, exit 0, write a report with
   accepted=false and reason missing_<dataset>_root. Never invent data. Fixtures must set
   synthetic_or_fixture=true and accepted=false.

2. Unified model
   Create homebrain/brain/homebrain_net_v0.py
   Class: HomeBrainNetV0  (config dataclass with bev_shape, image_size, hidden, horizons, etc.)
   ONE shared RGB-D encoder (promote DirectBEVStudentV0's encoder). Heads:
     pose_delta:        reuse SpatialMemoryNetV1.pose_head design (3-DoF x,y,yaw)
     bev_occupancy:     {free,occupied,unknown,traversable} + uncertainty
     metric_depth:      dense image-space depth (new head)
     dynamic:           dynamic-occupancy logits + BEV flow_xy (2ch)
     memory:            reuse SpatialMemoryNetV1 warp_memory_se2 + fusion (submodule)
     world_model:       reuse CounterfactualDynamicBEVWorldModelV0 over BEV history (submodule)
   Each head independently disable-able. Forward must return a dict and run on CPU with tiny tensors.
   Keep the file < 400 lines by composing existing submodules; do not re-implement them.

3. Labels for the new heads (offline, weak, flagged)
   Extend real_rgbd_route_bev_pack.py (do not fork it) to also emit:
     target_metric_depth (from sensor depth) + valid mask
     target_dynamic_occupancy + target_bev_flow_xy (ego-motion-compensated residual between
       consecutive BEV occupancy; valid only on moving cells) — reuse passive_dynamic_teacher logic
     pose_delta_next is teacher-only as today
   Mark all new labels weak_label=true, product_training_approved=false. Keep RUNTIME_ALLOWED_FIELDS
   vs TEACHER_ONLY_FIELDS separation; depth/flow/dynamic targets are TEACHER_ONLY.

4. Staged trainer
   Create homebrain/train/train_homebrain_net_v0.py with --stage {perception,memory,world,joint}.
     perception: occupancy + depth + dynamic + flow + pose, per-frame real_rgbd pack
     memory:     freeze perception, train memory fusion on temporal pack
     world:      freeze perception+memory, train world-model head on dynamic_bev_world_pack_v0
     joint:      unfreeze all, small LR fine-tune
   Warm-start from existing DirectBEV / SpatialMemoryV1 / Counterfactual checkpoints when present
   (reuse warm_start_current_bev_from_v0 pattern). Per-stage loss_report must break out every loss.
   Checkpoint metadata: model_type=HomeBrainNetV0, stage, pack_sha256, train/val_route_ids,
   future_horizons_sec, plus all five safety flags + no_*_at_runtime flags.

5. Evaluation with baselines (reuse EVALS.md gates)
   Create homebrain/eval/eval_homebrain_net_v0.py, split=val, route-heldout.
   Per-head metrics + baselines:
     pose:      ATE-RMSE, RPE vs groundtruth trajectory; baseline = identity + odom-integration
     depth:     AbsRel / RMSE on valid pixels; baseline = median-plane / constant depth
     occupancy: AUPRC/IoU by class; baseline = static-copy
     dynamic:   AUPRC/F1; flow EPE on dynamic cells; baseline = previous_frame_copy, zero_flow
     world:     future AUPRC/IoU/EPE by horizon; candidate AUROC, rank Spearman, top1-high-risk-rate;
                baselines = static_copy, previous_frame_copy, center_prior, transparent_scorer
   Acceptance can be true ONLY if (else write accepted=false + failed checks, do not fake):
     synthetic_or_fixture=false, >=3 real routes, >=1000 real frames, route-heldout with no overlap,
     val dynamic_positive_frame_count>0, runtime_field_leakage_passed=true,
     and each head beats its named baseline by the EVALS.md margins; pose ATE better than odom-only.

6. Runtime integration (no new orchestrator)
   Make HomeBrainNetV0 loadable inside the EXISTING modeld.py Brain.step path
   (add a checkpoint-name branch like the SpatialMemoryNetV1 branch). It must feed
   runtime_decision.decide_trajectory unchanged. Same debug/safety flags. No second runtime loop.

7. Tests (tests/test_homebrain_net_v0*.py, test_bonn_rgbd*.py)
   1. Missing dataset root -> exit 0, accepted=false.
   2. Fixture pack cannot pass real acceptance.
   3. Bonn importer maps to the shared route format with provenance + missing-sensor notices.
   4. HomeBrainNetV0 forward runs on CPU tiny tensors; each head toggle works.
   5. Train/val route overlap fails acceptance.
   6. Each stage writes checkpoint metadata + per-loss report; warm-start copies matching keys only.
   7. Eval compares against the named baselines for every head.
   8. Runtime path rejects all forbidden teacher/future fields.
   9. Existing tests still pass; net new lines stay within AGENTS.md budget (retire/warm-start old paths).

Definition of done:
  - One HomeBrainNetV0 with a shared encoder produces pose + occupancy + depth + dynamic/flow + future.
  - It trains in stages on real OpenLORIS/TUM/Bonn routes and reports honestly when data is missing.
  - Every head beats a named baseline on route-heldout val, or acceptance is false with reasons.
  - It plugs into the existing modeld.Brain + runtime_decision stack with safety flags intact.
  - The repo has fewer parallel encoders than before, not more.
