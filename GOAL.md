GOAL: Goal31 — Semantic Floor-Hazard Channel (bev_hazard) V0

Read first:
- AGENTS.md
- docs/BEV_ACTION_CONTRACT.md  (this goal adds a new channel + new semantics)
- EVALS.md  (Gate C BEV label quality, Gate F learned candidate scoring)
- homebrain/teachers/run_teacher.py
- homebrain/teachers/registry.py
- homebrain/teachers/base.py
- homebrain/teachers/mock_teacher.py
- homebrain/geometry/da3_to_weak_bev.py
- homebrain/geometry/scene_teacher_to_bev.py
- homebrain/geometry/bev_projector.py
- homebrain/geometry/camera_config.py
- homebrain/train/real_rgbd_route_bev_pack.py
- homebrain/brain/direct_bev_student_v0.py
- homebrain/train/train_direct_bev_student_v0.py
- homebrain/eval/eval_direct_bev_student_v0.py
- homebrain/runtime/direct_bev_runtime.py
- homebrain/runtime/counterfactual_world_model_scorer.py

Purpose:
Add a semantic floor-hazard layer to the BEV stack so the brain can avoid thin,
near-flat, geometrically-invisible floor objects that depth->occupancy provably
misses: cables, power cords, chargers, socks, shoes, small toys, pet waste,
liquid spills, rug fringes.

Today these objects are projected as FREE floor by every depth/RGB-D teacher,
because they have ~no height. They are a recognition problem, not a geometry
problem. This goal closes that gap with the existing teacher-farm pattern.

The pipeline is:
  open-vocabulary detection/segmentation teacher (offline only)
  -> weak per-frame hazard masks with provenance
  -> robot-frame BEV ground-plane projection into a new bev_hazard channel
  -> fused into the existing route BEV pack
  -> learned as a prediction head on the existing direct BEV student
  -> consumed by candidate scoring as an additional risk source

This must be replay-only. It must not control hardware.

Do not:
- depend on the open-vocabulary detector at runtime (no_teacher_fields_at_runtime).
- silently convert hazard cells into occupied cells. A hazard cell stays
  geometrically free and is flagged as do-not-traverse. The whole point is the
  free-but-unsafe case. Collapsing it into occupied destroys the signal and the
  free-space metrics.
- treat teacher hazard masks as ground truth. They are weak_label=true.
- download model weights in the normal test suite.
- fork real_rgbd_route_bev_pack.py. Extend it. (Code budget: modify existing paths.)
- add a new CLI. Hazards run through the existing run_teacher.py dispatch.
- count a fixture-only or synthetic-only run as accepted.
- use raw PWM or emit hardware commands.

License note (you care about this):
Prefer an Apache-2.0 open-vocabulary detector for the teacher (e.g.
Grounding-DINO; SAM-family for mask refinement). Record license_review_status in
the teacher manifest exactly like the DA3 / Depth Pro teachers. Flag any
AGPL-encumbered detector (e.g. Ultralytics-based YOLO-World) for review and do
not promote it past POC without an explicit LICENSE_AUDIT entry.

Create / modify these modules:

1. Open-vocabulary hazard teacher

Create:
  homebrain/teachers/hazard_teacher.py
Register in:
  homebrain/teachers/registry.py   (so run_teacher.py --teacher hazard dispatches)

Config:
  configs/hazard/prompts.yaml
    hazard ontology + open-vocab prompts, e.g.:
      cable, power_cord, charging_cable, phone_charger,
      sock, shoe, small_toy, pet_waste, liquid_spill, rug_fringe, cloth_on_floor
    per-class min confidence
    per-class default risk weight

Real backend:
  Wrap an Apache-2.0 open-vocab detector (boxes) + optional SAM-family mask
  refinement, isolated under external/ if weights are heavy (mirror the DA3 /
  Depth Pro external-venv pattern). Auto-reexec allowed; never auto-download in tests.

Fake backend:
  Mirror mock_teacher.py. Deterministic boxes/masks for fixtures. Marked
  mock=true, synthetic=true, real_perception=false. QA must quarantine it and
  must not promote it to weak BEV.

Command:
  python -m homebrain.teachers.run_teacher \
    --teacher hazard --backend real \
    --device cuda \
    --prompts configs/hazard/prompts.yaml \
    --log runs/room_walk_001_route \
    --out runs/room_walk_001_route/teacher_artifacts/hazard

Per-frame artifacts:
  hazard_masks.npy        (per-class probability or binary mask)
  hazard_boxes.json       (class, score, box)
  hazard_confidence.npy
  metadata.json           (model_name, backend, version/checkpoint,
                           license_review_status, weak_label=true,
                           control_safe=false, not_robot_frame_truth=true)

Route manifest fields:
  teacher_name = hazard
  backend
  model_name
  license_review_status
  weak_label = true
  control_safe = false

2. Hazard -> BEV projection + fusion

Create:
  homebrain/geometry/hazard_to_bev.py
  (mirror da3_to_weak_bev.py / scene_teacher_to_bev.py; reuse bev_projector.py
   and camera_config.py. If projection logic duplicates existing code, factor the
   shared part into bev_projector.py instead of copying.)

Behavior:
  Project floor-contacting hazard detections onto the robot-frame BEV grid using
  the known camera-to-base extrinsics + ground-plane (inverse-perspective)
  mapping. The floor is a known plane at known height, so floor-level hazards get
  metric placement without trusting monocular metric depth.

  Output channel:
    bev_hazard   (probability per cell)
    hazard_valid_mask  (cells with usable projection / in-FOV)

  Hard rule:
    bev_hazard does NOT modify bev_free / bev_obstacle / bev_unknown geometry.
    A hazard cell remains free in the occupancy channels and is flagged hazard.

  QA gating (mirror da3_to_weak_bev):
    Refuse to write bev_hazard on bad/missing extrinsics unless --review-assumed-extrinsics.
    Forced outputs stay weak_label=true, control_safe=false,
    robot_frame_truth=false, trainable_for=hazard_pretrain_only.

3. Pack integration

Modify:
  homebrain/train/real_rgbd_route_bev_pack.py

Add optional bev_hazard + hazard_valid_mask channels when hazard artifacts exist.
If hazard artifacts are absent:
  omit the channel, set has_hazard_channel=false, do not invent hazard data.

Manifest additions:
  has_hazard_channel
  hazard_source_model
  hazard_license_review_status
  hazard_positive_frame_count
  hazard_positive_cell_fraction
  hazard_weak_label = true

4. Student hazard head

Modify:
  homebrain/brain/direct_bev_student_v0.py
    add a small bev_hazard prediction head (no new heavy deps, keep under size budget)
  homebrain/train/train_direct_bev_student_v0.py
    add hazard loss: BCE + Dice/Focal, masked to hazard_valid_mask
    skip hazard loss cleanly when has_hazard_channel=false

So the runtime student predicts bev_hazard directly from RGB(-D)+state. The
open-vocab teacher is offline-only.

Loss report must separately include:
  loss_hazard_bce
  loss_hazard_dice
  loss_hazard_total

Checkpoint metadata:
  hazard_head = true
  hazard_trained = (true only if has_hazard_channel and positive frames > 0)
  no_teacher_fields_at_runtime = true
  replay_only = true
  control_safe = false
  raw_pwm_emitted = false
  hardware_validated = false

5. Candidate-scorer integration

Modify:
  homebrain/runtime/counterfactual_world_model_scorer.py
  (and the candidate footprint/risk pooling it uses)

Add a deterministic hazard penalty: a candidate footprint overlapping high
bev_hazard cells increases candidate risk, same pooling style as existing
footprint-risk. Add debug field candidate_hazard_exposure[k].

Runtime input may add only:
  predicted bev_hazard (from the student)
Runtime must NOT accept:
  hazard_boxes / hazard_masks from the teacher
  the open-vocab detector itself
  any teacher hazard artifact at control tick

6. Evaluation

Create:
  homebrain/eval/eval_bev_hazard_v0.py

Command:
  python -m homebrain.eval.eval_bev_hazard_v0 \
    --checkpoint artifacts/direct_bev_student_v0/checkpoint.pt \
    --pack artifacts/real_rgbd_route_bev_pack \
    --split val \
    --out artifacts/eval_bev_hazard_v0/eval.json

Baselines (required):
  depth_occupancy_only        # the killer baseline: marks thin hazards as FREE
  risky_channel_only          # current risk without the hazard channel
  previous_accepted_checkpoint

Metrics:
  hazard_auprc
  hazard_f1
  hazard_recall_thin_floor_objects        # recall specifically on flat/thin hazards
  free_space_false_positive_rate          # must NOT balloon (don't turn floor into hazard)
  candidate_hazard_exposure_mean
  candidate_hazard_exposure_vs_depth_only_scorer
  route_heldout_count
  per_route_hazard_metric_json
  worst_route_hazard_recall
  runtime_field_leakage_passed
  weak_label_provenance_present

Acceptance can be true only if:
  synthetic_or_fixture = false
  real_source_route_count >= 1
  real frames with verified hazard instances exist (hand-verified positives)
  hazard_positive_frame_count > 0
  route-heldout split is true (no train route in val)
  hazard_recall_thin_floor_objects beats depth_occupancy_only by a clear margin
    (set threshold; depth_occupancy_only recall on thin hazards is ~0 by construction)
  free_space_false_positive_rate <= threshold
  candidate_hazard_exposure_mean < depth-only scorer
  runtime_field_leakage_passed = true
  no_teacher_fields_at_runtime = true
  replay_only = true
  not_executed = true
  control_safe = false
  raw_pwm_emitted = false
  hardware_validated = false

This eval reports hazard detection quality against weak labels. It does NOT
claim the hazard labels are ground truth. It claims only: the hazard channel
flags thin floor objects that depth->occupancy misses, without destroying
free-space, and the scorer avoids them.

If thresholds fail:
  write eval.json
  accepted_bev_hazard_v0 = false
  list failed checks
  do not fake success

7. Tests

Add:
  tests/test_hazard_teacher_fake.py
  tests/test_hazard_to_bev.py
  tests/test_real_rgbd_route_bev_pack_hazard_channel.py
  tests/test_direct_bev_student_hazard_head.py
  tests/test_hazard_scorer_runtime_leakage.py
  tests/test_eval_bev_hazard_v0.py

Tests must prove:
  1. Fake hazard backend is marked fake and cannot pass real acceptance.
  2. Missing hazard artifacts -> pack sets has_hazard_channel=false and omits the channel cleanly.
  3. hazard_to_bev places a fixture floor detection into bev_hazard at the correct
     robot-frame cells via the camera-to-base ground-plane projection.
  4. A hazard cell stays geometrically FREE (bev_free/bev_obstacle/bev_unknown unchanged).
  5. Student forward pass with the hazard head works on CPU with tiny tensors.
  6. Training smoke run writes hazard loss in the report and hazard_trained metadata.
  7. Thin-cable fixture: depth marks the cell free, bev_hazard flags it, and the
     candidate scorer avoids the candidate whose footprint crosses it (its risk /
     candidate_hazard_exposure rises above a clear candidate).
  8. Runtime scorer signature rejects teacher hazard_boxes / hazard_masks and the detector.
  9. Train/val route overlap fails.
  10. Existing tests still pass.

Definition of done:
- One end-to-end hazard path exists: teacher -> hazard_to_bev -> pack -> student
  head -> candidate scorer -> eval, integrated into the existing direct BEV
  student and candidate scorer, not a disconnected demo.
- The runtime student predicts bev_hazard with no open-vocab detector dependency.
- Eval proves the hazard channel catches thin floor hazards that depth->occupancy
  misses, without ballooning free-space false positives.
- The candidate scorer avoids hazard cells (candidate_hazard_exposure drops vs the
  depth-only scorer).
- docs/BEV_ACTION_CONTRACT.md is updated to document the bev_hazard channel and its
  free-but-do-not-traverse semantics, and its required flags
  (weak_label, control_safe=false, hazard_pretrain_only).
- All replay/hardware safety invariants stay conservative.

The one input you must supply for real acceptance:
At least one real route with hand-verified hazard frames. Cheapest path: take a
phone/stick-cam room_walk capture (like data/inbox/room_walk_001) with a power
cable, a charger, and a sock laid on the floor, ingest it as a route, run the
hazard teacher, and hand-verify a few dozen positive frames. That single capture
is enough to flip this goal from fixture-only to accepted.