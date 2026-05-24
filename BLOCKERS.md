# BLOCKERS.md

## 2026-05-23 - Goal 6A DA3 weak BEV promotion quarantined

Exact failure: `python -m homebrain.geometry.da3_to_weak_bev --log runs/room_walk_001_route_short60 --teacher-artifacts runs/room_walk_001_route_short60/teacher_artifacts/da3 --qa runs/room_walk_001_da3_self_calib_qa_short60.json --out runs/room_walk_001_route_short60/geometry/da3_weak_bev` refused promotion because QA reported `quarantine_reasons=["pose_jump_outliers_present"]`.

Likely cause: DA3 estimated camera pose has one temporal jump outlier on the handheld short60 route. The route remains image-only/phone-like and is not robot-frame metric truth.

Minimal next repair action: inspect DA3 pose deltas around the outlier, rerun QA on a smoother/shorter route window or improve the pose-jump heuristic, then promote only if QA passes or use `--force-review` for manual review artifacts that remain `control_safe=false`.

Command output summary: real DA3 setup and inference succeeded, structural eval passed with `frames_with_teacher_artifacts=60`, `artifact_load_success=true`, and `depth_frame_count=60`; self-calibration QA reported `depth_valid_ratio=1.0`, `confidence_mean=1.0`, `intrinsics_valid_ratio=1.0`, `pose_valid_ratio=1.0`, `pose_jump_outlier_count=1`, `temporal_depth_consistency=0.9791352233683722`, and `promotable_to_weak_bev=false`.

Resolution update: Goal 6B identified the exact short60 outlier as the delta from frame 19 to frame 20 (`translation_delta=0.27583223581314087`) and selected accepted windows `0..19` and `21..59`. The route-level QA quarantine remains valid, but window-gated DA3 weak BEV and a 59-example SpatialTrainPack now exist without using `--force-review`; artifacts remain `control_safe=false`.

## 2026-05-23 - Goal 6B DA3 CUDA auto setup blocked

Exact failure: `python -m homebrain.tools.setup_da3_teacher --cuda-auto` could not produce a successful CUDA DA3 one-frame smoke test.

Likely cause: torch installed but torch.cuda.is_available() is false.

Minimal next repair action: install/verify an NVIDIA driver visible to `nvidia-smi`, then rerun DA3 setup with `--cuda-auto`; CPU DA3 remains acceptable when documented.

Command output summary: see `external/da3_setup_status.json` field `cuda_auto` for the probe, pip install attempt, and smoke command details.

Resolution update: rerunning setup after forcing a CUDA wheel reinstall selected `https://download.pytorch.org/whl/cu126`, installed `torch=2.12.0+cu126`, reported `torch.cuda.is_available()=true`, and passed the DA3 one-frame CUDA smoke. This CUDA setup blocker is resolved as of the latest `external/da3_setup_status.json`.

## 2026-05-23 - Goal 7 DINO SpatialMemoryNet v0 status

Exact failure: none active. DINOv2-small setup, real short60 feature extraction, tiny SpatialMemoryNet v0 overfit, eval, modeld checkpoint inference, replayd checkpoint inference, visualization, and pytest all completed.

Likely cause: not applicable.

Minimal next repair action: not applicable. The next useful action is broader training/eval beyond the deliberate 8-example overfit subset.

Command output summary: `python -m homebrain.tools.setup_dino_teacher --model-id dinov2_vits14 --device cuda` wrote `external/dino_setup_status.json` with `success=true`; real DINO wrote 60 frame feature artifacts under `runs/room_walk_001_route_short60/teacher_artifacts/dino`; `python -m pytest -q` reported 36 passed. Residual risk: the checkpoint is an overfit representation-pretraining artifact only, not a navigation or control-safe model.

## 2026-05-23 - Goal 7B SpatialMemoryNet v0 train/val source comparison status

Exact failure: none active. Pytest passed, deterministic split repeat manifests matched byte-for-byte for both accepted packs, real TUM DINO extraction completed, DA3-only/TUM-only/DA3+TUM train/eval runs completed, and the mixed checkpoint loaded in modeld/replayd on short60 and TUM routes.

Likely cause: not applicable.

Minimal next repair action: not applicable. The next useful action is either trajectory scoring if this baseline is accepted as sufficient, or data/model repair if the current validation losses and source gap are considered too weak.

Command output summary: `python -m pytest -q` reported 37 passed. DA3-only val loss was `0.155940979719162`; TUM-only val loss was `0.17899657785892487` with `pose_delta_rmse=0.027870545917272272`; mixed val loss was `0.19234523177146912`, with mixed per-source val losses DA3 `0.2110961526632309` and TUM `0.1939407934745153`. Replay evals reported `replay_determinism_pass=true`, short60 `brain_output_count=120` over 60 frames, and TUM `brain_output_count=240` over 120 frames. Residual risk: all outputs are representation-pretraining-only, not control-safe, and no trajectory scorer exists.

## 2026-05-23 - Goal 8 replay-only trajectory scorer status

Exact failure: none active. Candidate generation, heuristic scoring, replay-local coverage memory, policy artifact writing, overlays, synthetic fixture scoring, short60 modeld scoring, TUM modeld scoring, checkpoint+features smoke scoring, and pytest all completed.

Likely cause: not applicable.

Minimal next repair action: not applicable. The next useful action is learned trajectory labels from deterministic algorithmic coverage/risk supervision on controlled grids or reviewed BEV packs.

Command output summary: `python -m pytest -q` reported 42 passed. Synthetic fixture scoring selected `straight_medium` with `selected_risk_score=0.0`, `selected_coverage_gain_proxy=3.5`, and `stop_selected_fraction=0.0`. Short60 modeld scoring selected `stop` for all 60 frames with `risky_candidate_fraction=1.0`; TUM modeld scoring selected stop for most frames with `stop_selected_fraction=0.8916666666666667`. Every Goal 8 decision JSONL line was marked `replay_only`, `control_safe=false`, and `not_executed`. Residual risk: current modeld BEV outputs are stop-heavy under the heuristic scorer, and all artifacts remain debug/eval only rather than control-safe behavior.

## 2026-05-23 - Goal 9A stop-heavy audit and ActionLabelPack v0 status

Exact failure: none active. Stop-heavy audit, label-BEV policy comparisons, controlled BEV generation, ActionLabelPack v0 build, ActionLabelPack QA, and pytest all completed.

Likely cause: not applicable as a goal failure. The audit did confirm the underlying stop-heavy behavior is real: current model-BEV and reviewed label-BEV sources frequently mark candidate footprints risky/blocked.

Minimal next repair action: not applicable for Goal 9A completion. The next useful action is either train a learned trajectory scorer from the new ActionLabelPack with the stop-heavy reviewed-source caveat, or repair perception/scorer thresholds before training.

Command output summary: `python -m pytest -q` reported 45 passed. Short60 audit reported `stop_selected_fraction=1.0`, `risky_candidate_fraction=1.0`, `occupied_hit_rate=1.0`, and `candidate_footprint_block_rate=1.0`; DA3 label comparison also selected stop for every overlapping frame. TUM audit reported `stop_selected_fraction=0.8916666666666667`, `occupied_hit_rate=0.9083333333333333`, `unknown_hit_rate=1.0`, and `candidate_footprint_block_rate=1.0`; TUM label BEV selected stop on `0.8583333333333333` of frames. ActionLabelPack QA reported `example_count=1299`, `selected_motion_fraction=0.876058506543495`, `selected_stop_fraction=0.123941493456505`, `action_label_pack_qa_pass=true`, and controlled open-room `source_selected_stop_fraction=0.0`. Residual risk: reviewed real BEV sources remain stop-heavy under the deterministic heuristic, while controlled-grid labels are simplified and not control-safe.

## 2026-05-23 - Goal 9B robot-frame BEV/action sanity repair status

Exact failure: none active. BEV action contract, action sanity audit, derived traversability view builder, transparent scorer sweep, ActionLabelPack v1 filtering/QA, old-vs-calibrated comparison, and pytest all completed.

Likely cause: not applicable as a goal failure. The audit confirmed the stop-heavy issue is due to source semantics and local BEV structure: DA3/phone and TUM/public RGB-D labels are not robot-frame action truth, and their robot center/footprint regions are structurally blocked under the action contract.

Minimal next repair action: repair or regenerate real BEV labels with correct robot-origin/footprint semantics, or collect true robot-frame action logs. Until then, keep DA3/TUM reviewed labels and modeld BEVs geometry-only for action learning.

Command output summary: `python -m pytest -q` reported 49 passed. Controlled action sanity reported `action_supervision_ok_fraction=0.8089285714285714` and open-room `action_supervision_ok_fraction=1.0`. DA3 and TUM action sanity both reported `action_supervision_ok_fraction=0.0`, `robot_center_blocked_rate=1.0`, and `footprint_blocked_rate=1.0`. The scorer sweep selected a transparent config with controlled gates passing: `controlled_open_motion_rate=1.0` and `blocked_map_stop_rate=1.0`; reviewed motion remained low at `0.09497206703910614`. ActionLabelPack v1 QA passed with `example_count=906`, `excluded_frame_count=393`, and all outputs `replay_only=true`, `not_executed=true`, `control_safe=false`. Policy comparison marked DA3 labels, TUM labels, short60 modeld, and TUM modeld as `geometry_only` for action learning.

## 2026-05-24 - Goal 10A public robot-frame dataset setup blocked by bounded downloads

Exact failure: `python -m homebrain.datasets.setup_openloris_scene --out data\public\openloris_scene --sequence cafe1-1 --download --max-download-gb 2` discovered the official Hugging Face mirror but refused to download `package/cafe1-1_2-package.tar` because it is 6.95 GB, exceeding `max_download_gb=2.00`.

Likely cause: OpenLORIS-Scene is packaged as multi-GB tar archives; the bounded Codex setup intentionally avoids a large unattended download.

Minimal next repair action: stage/extract the OpenLORIS `cafe1-1_2` package manually or rerun setup with `--allow-large-download` when disk/network budget is approved, then run `python -m homebrain.datasets.openloris_to_route` followed by `python -m homebrain.geometry.robot_rgbd_to_bev`.

Command output summary: setup wrote `data/public/openloris_scene/openloris_scene_setup_status.json` with official source URLs, selected package size `6.954` GB, local free disk `118.925` GB, license `CC BY-ND 4.0`, and `license_review_status=pending_human_review`. No real OpenLORIS route, robot-frame BEV, or public robot-frame action labels were generated.

Fallback attempt: `python -m homebrain.datasets.setup_tum_rgbd --out data\public\tum_rgbd --sequence freiburg2_pioneer_slam --download --max-download-gb 0.25` resolved the TUM Pioneer robot-mounted archive but refused to download it because it is 1.51 GB, exceeding `max_download_gb=0.25`.

Fallback minimal next repair action: approve a larger bounded TUM Pioneer download or stage `rgbd_dataset_freiburg2_pioneer_slam.tgz`; then import it and keep it geometry-only unless robot/base pose and camera-to-base semantics pass the BEV action contract.
