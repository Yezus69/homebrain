# EVALS.md — Metrics and Gates

HomeBrain is not allowed to become a cool demo with no scorecard.

## Gate 0: log/replay spine

Required before ML work:
- `pytest -q` passes
- dummy log generation works
- deterministic replay works
- eval writes JSON metrics
- `CURRENT_STATUS.md` updated

Core metrics:
```text
event_count
frame_count
dropped_frame_count
event_ordering_error_count
replay_determinism_pass
brain_output_count
eval_runtime_sec
```

## Gate 1: teacher artifact pipeline

Required before student training:
- teacher interface works with mocks
- artifacts can be written and loaded
- visualization is generated
- license audit updated
- no missing-weight crash unless user explicitly requested real run

Teacher metrics:
```text
teacher_artifact_count
teacher_mock_used
artifact_load_success
visualization_written
license_audit_updated
```

## Gate 2: SpatialMemoryNet v0

Required before trajectory learning:
- model can overfit a tiny dataset
- training script runs
- eval script writes metrics
- replay integration emits BEV/pose/uncertainty

Model metrics:
```text
train_loss
val_loss
bev_loss
bev_iou_or_proxy
pose_delta_rmse
uncertainty_calibration_proxy
inference_fps
```

## Gate 3: trajectory scorer

Required before control integration:
- candidate trajectories generated deterministically
- scorer ranks candidates
- unsafe candidates can be penalized
- coverage/risk/debug outputs visible in replay

Trajectory metrics:
```text
candidate_count
selected_candidate_id
risk_score
coverage_gain_proxy
uncertainty_penalty
trajectory_eval_runtime_ms
```

## Gate 4: real-video spatial output

Required before hardware integration:
- real indoor video can be ingested
- model outputs BEV/risk/uncertainty overlays
- failure cases are logged
- performance is measured

Real-video metrics:
```text
video_frames_processed
overlay_written
avg_inference_fps
uncertain_frame_rate
dynamic_risk_event_count
manual_review_notes_present
```

## Standard verification commands

After Goal 0 these should exist or be created by Codex:

```bash
pytest -q
python -m homebrain.replay.generate_dummy_log --out runs/dummy_route
python -m homebrain.replay.replayd --log runs/dummy_route --out runs/replayed_route
python -m homebrain.eval.run_eval --log runs/dummy_route --out runs/dummy_eval.json
```

As new features are added, Codex must update this file with exact current commands.
