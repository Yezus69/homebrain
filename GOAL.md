You are working in the Yezus69/homebrain repo.

Read ARCHITECTURE.md first. The task is to implement the missing architecture stage:
Dataset/artifact builder -> Direct student encoder -> Online SceneState/local BEV.

Do not solve this by adding another synthetic fixture, another mock teacher report, or another markdown-only milestone.

GOAL:
Build RealRGBDRouteBEVStudentV0: an end-to-end real-data path that ingests real RGB-D route sequences in TUM/Bonn RGB-D format, builds weak local-BEV supervision offline, trains a direct sensor-to-BEV student model, evaluates on held-out real routes, and exposes the trained student as a runtime LocalBev provider for the existing SceneState / FutureBEV / trajectory decision stack.

Primary dataset target:
- Bonn RGB-D Dynamic Dataset format.
- Also support generic TUM RGB-D format because Bonn uses the same sequence structure.
- Do not download datasets automatically.
- Require the user to pass a local dataset root.
- If no real dataset root is provided, tests may still run with tiny synthetic smoke data, but the real acceptance flag must remain false.

Hard constraints:
1. No synthetic, deterministic, generated, mock, fixture, or randomly generated sequence may set accepted_real_rgbd_route_bev_student=true.
2. Unit tests may use tiny fixtures only for schema/smoke tests. They must not be used for acceptance metrics.
3. Runtime must not consume:
   - target_bev
   - future_* labels
   - candidate_oracle_cost
   - teacher masks
   - ground-truth global trajectory
   - future frames
   - route-level oracle data
4. Offline pack building may use depth, camera intrinsics, timestamps, and provided ground-truth poses from the real dataset to create weak labels.
5. Runtime may consume only:
   - current RGB frame
   - optional current depth frame
   - optional previous frame
   - relative pose delta / odom-like delta if available
   - previous action if available
   - model checkpoint
6. No classical SLAM runtime dependency.
7. Do not emit hardware commands.
8. Keep safety flags:
   - replay_only=true
   - not_executed=true
   - control_safe=false
   - raw_pwm_emitted=false
   - hardware_validated=false
9. Do not edit README.md, GOAL.md, or ARCHITECTURE.md for acceptance. Code, tests, and JSON metrics are the deliverables.
10. If real route data is missing, the command must exit cleanly with accepted_real_rgbd_route_bev_student=false and a clear reason. Do not silently fall back to fixtures.

Implement these modules:

1. Real route loader

Create:
- homebrain/data/tum_rgbd_route.py

It must parse TUM/Bonn-style sequences:
- rgb.txt
- depth.txt
- groundtruth.txt
- associated RGB/depth pairs
- timestamps
- camera intrinsics
- optional sequence metadata

Expose:
- RouteFrame
- RouteSequence
- load_tum_rgbd_sequence(...)
- associate_rgb_depth_pose(...)
- route_id / split_unit_id handling

Requirements:
- Deterministic frame ordering.
- Robust to missing depth or pose entries.
- Route-level split support. Never split train/val by adjacent frames from the same route.
- Source manifest must record dataset_name, route_id, frame count, used frame count, skipped frame count, and hashes for a bounded sample of source files.

2. Offline RGB-D-to-BEV weak teacher

Create:
- homebrain/teachers/rgbd_bev_teacher.py

Build weak BEV labels from real RGB-D data:
- Project depth into a local robot-centric BEV grid using camera intrinsics and configurable camera extrinsics.
- Produce:
  - current_bev_free
  - current_bev_occupied
  - current_bev_unknown
  - current_bev_traversable
  - current_bev_risky
  - uncertainty_map
  - dynamic_residual_risk
- Use simple ray carving for free/unknown/occupied.
- Use temporal residuals across registered nearby frames to mark dynamic risk:
  - warp previous BEV into current frame using pose_delta
  - mark cells that appear/disappear inconsistently as dynamic_residual_risk
  - do not require SLAM; use provided dataset poses only in offline teacher mode
- Store all labels as weak_label=true and product_training_approved=false.

Do not use this teacher at runtime.

3. Real RGB-D BEV pack builder

Create:
- homebrain/train/real_rgbd_route_bev_pack.py

Command:
python -m homebrain.train.real_rgbd_route_bev_pack \
  --dataset bonn_rgbd_dynamic \
  --input /path/to/bonn/sequences \
  --out artifacts/real_rgbd_route_bev_pack \
  --train-routes route_a,route_b \
  --val-routes route_c \
  --frame-stride 5 \
  --grid-shape 64x64 \
  --meters-per-cell 0.05

Pack schema:
- package_type: RealRGBDRouteBEVPack
- real_dataset_required: true
- synthetic_or_fixture: false
- source_family: real_rgbd_route
- route_held_out_split_basis: route_id
- examples:
  - rgb_path
  - depth_path
  - timestamp_ns
  - route_id
  - split
  - pose_delta_prev
  - pose_delta_next_teacher_only
  - camera_intrinsics
  - rgb_tensor or path reference
  - depth_tensor or path reference
  - target_current_bev_*
  - target_uncertainty_map
  - target_dynamic_residual_risk
  - teacher_only_fields clearly marked
  - runtime_allowed_fields clearly listed
- manifest must include:
  - real_source_frame_count
  - real_source_route_count
  - dynamic_positive_cell_fraction
  - heldout_route_ids
  - train_route_ids
  - no_fixture_data_used=true

Acceptance guard:
- If route_count < 3, accepted_real_rgbd_route_bev_pack=false.
- If val route has zero dynamic-positive frames, accepted_real_rgbd_route_bev_pack=false.
- If any route appears in both train and val, fail.

4. DirectBEV student model

Create:
- homebrain/brain/direct_bev_student_v0.py

Model:
- DirectBEVStudentV0
- Input:
  - RGB image tensor, resized
  - optional depth tensor
  - sensor_mask
  - pose_delta_prev
  - optional previous action vector
- Output:
  - bev_logits for free/occupied/unknown/traversable/risky
  - uncertainty_logits
  - dynamic_risk_logits
  - optional compact feature map for FutureBEV rollout

Keep it small enough for two 4090s:
- default image size 160x96 or 224x128
- small CNN or ConvNeXt-like blocks
- no giant VLM dependency
- mixed precision support
- CPU smoke mode must work

Do not make the model depend on future labels.

5. Training

Create:
- homebrain/train/train_direct_bev_student_v0.py

Command:
python -m homebrain.train.train_direct_bev_student_v0 \
  --pack artifacts/real_rgbd_route_bev_pack \
  --out artifacts/direct_bev_student_v0 \
  --device cuda \
  --batch-size 32 \
  --max-steps 20000 \
  --amp

Losses:
- BCE/Dice for occupied/free/unknown/traversable/risky
- weighted BCE/Focal loss for dynamic_risk due class imbalance
- uncertainty calibration loss
- route-heldout validation every N steps

Checkpoint metadata must include:
- route ids used for train/val
- source dataset name
- pack manifest sha256
- model config
- input modality flags
- no_future_labels_used=true
- no_teacher_fields_at_runtime=true
- replay_only=true
- control_safe=false

6. Evaluation

Create:
- homebrain/eval/eval_direct_bev_student_v0.py

Command:
python -m homebrain.eval.eval_direct_bev_student_v0 \
  --checkpoint artifacts/direct_bev_student_v0/checkpoint.pt \
  --pack artifacts/real_rgbd_route_bev_pack \
  --split val \
  --out artifacts/direct_bev_student_v0/eval_real_route.json

Metrics:
- occupied_iou
- free_iou
- unknown_iou
- traversable_iou
- risky_iou
- dynamic_risk_auprc
- dynamic_risk_f1_at_0_5
- uncertainty_ece_or_proxy
- route_heldout_count
- dynamic_positive_frame_count
- improvement_vs_center_prior
- improvement_vs_previous_frame_copy
- improvement_vs_static_memory_copy
- runtime_field_leakage_passed

Acceptance:
accepted_real_rgbd_route_bev_student=true only if all are true:
- pack manifest says synthetic_or_fixture=false
- real_source_route_count >= 3
- real_source_frame_count >= 1000
- train/val split is route-heldout
- heldout route has dynamic_positive_frame_count > 0
- occupied_iou >= 0.35
- risky_iou >= 0.20
- dynamic_risk_auprc beats previous_frame_copy baseline by at least 0.05 absolute
- dynamic_risk_f1_at_0_5 >= 0.20
- runtime_field_leakage_passed=true
- no_future_labels_used=true
- no_teacher_fields_at_runtime=true
- safety flags remain replay_only/not_executed/control_safe=false/raw_pwm_emitted=false/hardware_validated=false

If thresholds are not met, still write the report, but accepted_real_rgbd_route_bev_student=false. Do not fake success.

7. Runtime integration

Create:
- homebrain/runtime/direct_bev_runtime.py

Expose:
- load_runtime_direct_bev_student(checkpoint, device)
- predict_local_bev_from_rgbd(...)
- direct_bev_to_local_bev(...)

Integrate lightly with existing decision stack:
- Add an optional path that lets modeld or runtime_decision consume LocalBev produced by DirectBEVStudentV0.
- Do not break existing LocalBev or FutureBEV rollout APIs.
- Add debug artifact fields:
  - direct_bev_student_checkpoint
  - direct_bev_student_checkpoint_sha256
  - direct_bev_student_source_dataset
  - direct_bev_runtime_inputs
  - direct_bev_teacher_fields_used_at_runtime=false
  - direct_bev_future_labels_used_at_runtime=false
  - direct_bev_replay_only=true
  - direct_bev_control_safe=false

8. End-to-end real route brain-slice eval

Create:
- homebrain/eval/eval_real_rgbd_brain_slice.py

This should run:
real RGB-D frame -> DirectBEVStudentV0 -> LocalBev -> existing candidate scorer / future rollout if checkpoint supplied -> replay-only trajectory decision artifact.

Report:
- selected_candidate_id distribution
- stop rate
- high-risk selected rate using teacher labels offline for eval only
- agreement with teacher-BEV transparent scorer
- cases where DirectBEV risk changed candidate choice
- no runtime leakage

This eval may use teacher labels only after runtime decision is produced, never as runtime input.

9. Tests

Add tests:
- tests/test_tum_rgbd_route_loader.py
- tests/test_rgbd_bev_teacher.py
- tests/test_direct_bev_student_v0.py
- tests/test_real_rgbd_route_bev_pack.py
- tests/test_direct_bev_runtime_leakage.py

Tests may create tiny fake TUM-format sequences for parsing/smoke tests, but:
- any report generated from fake data must have accepted_real_rgbd_route_bev_student=false
- test must assert fake/synthetic data cannot pass real acceptance
- test must assert train/val route leakage fails
- test must assert runtime function signatures do not accept target_bev, future labels, oracle cost, teacher masks, or ground-truth global route
- test must assert runtime output safety flags remain false/not executed
- test must assert DirectBEV output can be converted into LocalBev and scored by existing trajectory scorer

Definition of done:
- New real-data pack builder exists.
- New DirectBEV student exists.
- New trainer exists.
- New evaluator exists.
- New runtime LocalBev adapter exists.
- Existing tests pass.
- New tests pass.
- A real-data command path is documented in code argparse help and JSON report fields, not by editing markdown.
- If no real data is present, the repo still passes tests but reports accepted_real_rgbd_route_bev_student=false.
- If real Bonn/TUM-format dynamic data is provided and thresholds pass, the report may set accepted_real_rgbd_route_bev_student=true.