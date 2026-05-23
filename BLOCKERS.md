# BLOCKERS.md

## 2026-05-23 - Goal 6A DA3 weak BEV promotion quarantined

Exact failure: `python -m homebrain.geometry.da3_to_weak_bev --log runs/room_walk_001_route_short60 --teacher-artifacts runs/room_walk_001_route_short60/teacher_artifacts/da3 --qa runs/room_walk_001_da3_self_calib_qa_short60.json --out runs/room_walk_001_route_short60/geometry/da3_weak_bev` refused promotion because QA reported `quarantine_reasons=["pose_jump_outliers_present"]`.

Likely cause: DA3 estimated camera pose has one temporal jump outlier on the handheld short60 route. The route remains image-only/phone-like and is not robot-frame metric truth.

Minimal next repair action: inspect DA3 pose deltas around the outlier, rerun QA on a smoother/shorter route window or improve the pose-jump heuristic, then promote only if QA passes or use `--force-review` for manual review artifacts that remain `control_safe=false`.

Command output summary: real DA3 setup and inference succeeded, structural eval passed with `frames_with_teacher_artifacts=60`, `artifact_load_success=true`, and `depth_frame_count=60`; self-calibration QA reported `depth_valid_ratio=1.0`, `confidence_mean=1.0`, `intrinsics_valid_ratio=1.0`, `pose_valid_ratio=1.0`, `pose_jump_outlier_count=1`, `temporal_depth_consistency=0.9791352233683722`, and `promotable_to_weak_bev=false`.
