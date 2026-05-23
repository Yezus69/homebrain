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
