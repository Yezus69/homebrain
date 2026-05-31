GOAL: Goal32 — Checkpoint 3D Spatial Timeline Visualizer V0

Purpose:
Build a real debugging/inspection tool for HomeBrain checkpoints.

Given a trained checkpoint plus a route/pack/replay artifact, the tool must run the model in replay mode, extract the model’s spatial outputs, pose/camera information, BEV/memory channels, candidate trajectory info, and safety/debug info, then build an accumulated 3D/2.5D scene over time.

This must NOT be a per-frame screenshot generator.
This must build a stitched map over time.

A human should be able to open the output and understand:
- where the robot thinks it is,
- where the camera is looking,
- what the model thinks is free/occupied/unknown/risky/hazardous,
- how the local map accumulates over time,
- where the robot has been,
- what candidate trajectories were considered,
- what action/candidate was selected,
- why a stop/recovery happened,
- what model channels exist in the checkpoint,
- where uncertainty, stale memory, or pose problems occur.

This is an offline replay/debug tool only.
It must not control hardware.

Hard constraints:
- replay_only=true
- not_executed=true
- control_safe=false
- raw_pwm_emitted=false
- hardware_validated=false
- no hardware transport
- no motor command execution
- no raw PWM
- no future-frame leakage in incremental map mode
- no teacher/runtime leakage
- no pretending BEV-only data is dense true 3D
- no silent use of oracle pose or teacher labels as runtime truth
- no disconnected demo code
- every nontrivial behavior must have tests and machine-readable artifacts

Important honesty rule:
If the route has real depth/RGB-D, the visualizer may create a real point-cloud-style accumulated scene.
If the route only has BEV/model predictions, the visualizer must render an extruded 2.5D BEV map, not claim it is a dense 3D reconstruction.

Every artifact must explicitly record:
- geometry_source: one of depth_pointcloud, rgbd_pointcloud, bev_extruded_2p5d, fixture_synthetic
- pose_source: one of odom, route_pose, checkpoint_pose, fixture_pose, assumed_stationary
- pose_source_is_oracle: bool
- dense_3d_claimed: bool, must be false unless true depth/RGB-D geometry exists
- replay_only=true
- control_safe=false

Read first:
- README.md
- ARCHITECTURE.md
- EVALS.md
- docs/BEV_ACTION_CONTRACT.md
- homebrain/runtime/direct_bev_runtime.py
- homebrain/runtime/counterfactual_world_model_scorer.py
- homebrain/brain/direct_bev_student_v0.py
- homebrain/train/real_rgbd_route_bev_pack.py
- existing visualization/eval/artifact patterns, if any

Main CLI:
Create:

homebrain/viz/visualize_checkpoint_3d_v0.py

Command:

python -m homebrain.viz.visualize_checkpoint_3d_v0 \
  --checkpoint artifacts/direct_bev_student_v0/checkpoint.pt \
  --pack artifacts/real_rgbd_route_bev_pack \
  --route-id room_walk_001 \
  --out artifacts/checkpoint_3d_visualizer_v0/room_walk_001 \
  --device cpu \
  --max-frames 300 \
  --pose-source online \
  --write-html \
  --write-ply \
  --write-jsonl

The CLI must also support fixture mode:

python -m homebrain.viz.visualize_checkpoint_3d_v0 \
  --fixture stitched_room_tiny \
  --out artifacts/checkpoint_3d_visualizer_v0/fixture_stitched_room_tiny \
  --write-html \
  --write-ply \
  --write-jsonl

Outputs:
The output directory must contain:

1. summary.json
2. scene_timeline.jsonl
3. accumulated_scene.json
4. accumulated_map.ply
5. camera_trajectory.ply
6. viewer.html
7. frame_debug/
   - optional compact per-frame debug JSON files
8. manifest.json
9. visualizer_acceptance.json

summary.json must include:

{
  "accepted_checkpoint_3d_visualizer_v0": false,
  "checkpoint_path": "...",
  "route_id": "...",
  "frame_count": 0,
  "geometry_source": "...",
  "pose_source": "...",
  "pose_source_is_oracle": false,
  "dense_3d_claimed": false,
  "map_accumulates_over_time": true,
  "incremental_no_future_leakage_passed": true,
  "camera_pose_rendered": true,
  "robot_pose_rendered": true,
  "trajectory_rendered": true,
  "model_channels_rendered": [],
  "candidate_trajectories_rendered": false,
  "safety_debug_rendered": false,
  "artifacts": [],
  "warnings": [],
  "hard_failures": [],
  "safety": {
    "replay_only": true,
    "not_executed": true,
    "control_safe": false,
    "raw_pwm_emitted": false,
    "hardware_validated": false
  }
}

Implement these modules:

1. Scene types

Create:
homebrain/viz/checkpoint_3d_scene_types_v0.py

Define typed dataclasses:

TransformV0:
- frame_from: str
- frame_to: str
- matrix_4x4: list[list[float]]
- source: str
- timestamp_ns: int | None
- covariance: optional list
- is_assumed: bool
- is_oracle: bool

CameraFrustumV0:
- timestamp_ns
- camera_id
- T_map_camera: TransformV0
- intrinsics:
  fx, fy, cx, cy, width, height
- near_m
- far_m
- frustum_lines_map: list of 3D line segments
- valid: bool
- warnings: list[str]

RobotPoseVizV0:
- timestamp_ns
- T_map_base: TransformV0
- T_map_camera: optional TransformV0
- base_axes_lines_map
- footprint_polygon_map
- pose_confidence: float | None
- pose_source: str
- pose_source_is_oracle: bool

ModelChannelSummaryV0:
- name: str
- shape: list[int]
- min: float | None
- max: float | None
- mean: float | None
- valid_fraction: float | None
- source: str
- rendered: bool
- notes: list[str]

LocalBEVFrameV0:
- timestamp_ns
- frame_id
- T_map_base: TransformV0
- resolution_m
- origin_convention
- channels:
  free
  obstacle
  unknown
  traversable optional
  risky optional
  hazard optional
  confidence optional
  uncertainty optional
  memory_age optional
- channel_summaries: list[ModelChannelSummaryV0]
- rendered_cells_map: compact list or compressed representation
- valid: bool
- warnings: list[str]

AccumulatedMapStateV0:
- timestamp_ns
- frame_id
- map_resolution_m
- global_bounds_m
- free_cells
- obstacle_cells
- unknown_cells
- risky_cells
- hazard_cells
- stale_cells
- trajectory_points
- camera_frustums
- conflict_count
- stale_update_count
- observation_count
- geometry_source
- pose_source
- dense_3d_claimed

CandidateTrajectoryVizV0:
- timestamp_ns
- candidate_id
- points_map
- footprint_polygons_map optional
- score optional
- risk optional
- hazard_exposure optional
- unknown_exposure optional
- selected: bool
- vetoed: bool
- stop_reasons: list[str]

SafetyDebugVizV0:
- timestamp_ns
- proposed_cmd_vel optional
- safe_cmd_vel optional
- accepted: bool
- stop_reasons: list[str]
- recovery_proposal optional
- stale_sensor: bool | None
- high_uncertainty: bool | None
- high_risk: bool | None
- obstacle_too_close: bool | None
- replay_only: true
- not_executed: true

SceneTimelineFrameV0:
- schema_version = "Checkpoint3DSceneTimelineV0"
- route_id
- frame_id
- timestamp_ns
- robot_pose: RobotPoseVizV0
- camera_frustum: CameraFrustumV0 | None
- local_bev: LocalBEVFrameV0 | None
- accumulated_map_summary: dict
- candidate_trajectories: list[CandidateTrajectoryVizV0]
- safety_debug: SafetyDebugVizV0 | None
- model_channel_summaries: list[ModelChannelSummaryV0]
- warnings: list[str]
- safety flags

All dataclasses must serialize deterministically to JSON.

2. Checkpoint output adapter

Create:
homebrain/viz/checkpoint_output_adapter_v0.py

Purpose:
Load a checkpoint and route/pack, run the existing runtime/inference path, and normalize the outputs for visualization.

Must:
- prefer existing HomeBrain runtime APIs
- not duplicate model inference logic
- not invent channels
- gracefully handle missing checkpoint fields
- inspect checkpoint metadata when available
- infer model outputs from known names where possible:
  bev_free
  bev_obstacle
  bev_unknown
  bev_traversable
  bev_risky
  bev_hazard
  bev_confidence
  uncertainty
  future_free
  future_obstacle
  future_unknown
  future_risk
  candidate_scores
  candidate_debug
  safety_debug
- if a channel is missing, record it as missing, do not synthesize it
- if checkpoint loading fails, fixture mode tests must still pass, but real mode must fail honestly

Expose:

class Checkpoint3DOutputAdapterV0:
    def load_checkpoint(...)
    def iter_route_outputs(...)
    def extract_model_channel_summaries(...)
    def extract_local_bev(...)
    def extract_candidate_trajectories(...)
    def extract_safety_debug(...)

The adapter should return normalized typed objects, not raw tensors.

3. Pose and camera geometry

Create:
homebrain/viz/pose_camera_geometry_v0.py

Implement:
- build_T_map_base_from_pose_sequence(...)
- compose_transforms(...)
- invert_transform(...)
- transform_points(...)
- camera_frustum_from_intrinsics(...)
- robot_footprint_polygon(...)
- bev_cell_centers_to_robot_frame(...)
- robot_frame_to_map_frame(...)

Requirements:
- Use explicit coordinate frames:
  map
  odom if available
  base_link
  camera
  bev_grid
- Do not silently assume transforms.
- Missing camera-to-base extrinsics should warn and mark camera_pose_rendered=false unless fixture mode supplies them.
- If --review-assumed-extrinsics is used, mark is_assumed=true and accepted_checkpoint_3d_visualizer_v0=false for real data.
- If route pose is known to be ground truth/oracle, set pose_source_is_oracle=true and accepted=false unless run is explicitly marked debug_oracle_pose=true.
- Fixture tests may use fixture_pose without failing.

4. Accumulated 3D / 2.5D map builder

Create:
homebrain/viz/accumulated_scene3d_v0.py

Implement:

class AccumulatedScene3DV0:
    def __init__(resolution_m, decay_sec, max_points, geometry_source)
    def update_from_local_bev(frame: LocalBEVFrameV0)
    def update_from_depth_pointcloud(...)
    def snapshot(frame_id, timestamp_ns) -> AccumulatedMapStateV0
    def export_cells(...)
    def export_point_cloud(...)

Behavior:
- The map must accumulate over time.
- At frame k, the snapshot may use only frames <= k.
- Local robot-frame BEV cells must be transformed into map frame using T_map_base.
- If depth/RGB-D exists, project depth to point cloud using camera intrinsics/extrinsics, then transform to map frame.
- If no depth exists, create a 2.5D BEV visualization:
  - free cells: floor tiles at z=0
  - obstacle cells: vertical low columns
  - unknown cells: low-opacity or separate unknown layer
  - risky cells: floor overlay
  - hazard cells: floor overlay, not obstacle
  - stale cells: visually distinct layer
- Do not collapse hazard into obstacle.
- Do not mark unknown as free.
- Track conflicts:
  - same global cell seen as free and obstacle
  - same global cell seen as hazard and free is not a conflict; hazard may be free-but-unsafe
  - unknown should not overwrite known unless staleness policy says so
- Track age/staleness per global cell.
- Track coverage/visited path.
- Track camera frustums over time.
- Track robot trajectory over time.

Important:
The visualizer must have two modes:
1. incremental timeline mode:
   shows accumulated map up to the selected frame only
2. final map mode:
   shows the complete accumulated map after all frames

The HTML viewer must default to incremental timeline mode.

5. PLY/JSON/HTML exporters

Create:
homebrain/viz/export_checkpoint_3d_artifacts_v0.py

Implement:
- write_scene_timeline_jsonl(...)
- write_accumulated_scene_json(...)
- write_ply_point_cloud(...)
- write_ply_lines(...)
- write_viewer_html(...)

Artifact requirements:
- deterministic ordering
- stable numeric formatting
- stable sha256 in manifest
- no huge raw tensors in JSON by default
- support --max-points and --max-cells
- support gzip optionally, but default test artifacts should be plain text

viewer.html:
Must be a human-usable interactive 3D viewer.

Minimum UI:
- timeline slider
- play/pause button
- frame index and timestamp
- layer toggles:
  robot trajectory
  camera frustum
  local BEV
  accumulated free
  accumulated obstacle
  accumulated unknown
  accumulated risky
  accumulated hazard
  stale memory
  candidate trajectories
  selected candidate
  safety stops
  future predictions if available
- side panel showing:
  route_id
  frame_id
  timestamp
  pose source
  geometry source
  camera pose availability
  model channels present/missing
  selected candidate id
  candidate score/risk if available
  stop reasons if any
  warnings
  safety flags

Implementation guidance:
- Prefer a self-contained HTML artifact.
- Do not require a server for basic viewing.
- Do not require internet access for tests.
- It is acceptable for tests to validate the emitted HTML structure and embedded scene JSON rather than visually inspect WebGL.
- Use a simple embedded WebGL/canvas renderer or a minimal local JS renderer.
- Do not introduce a large web app framework.
- If optional plotting libraries are used, the code must degrade gracefully and tests must not depend on them.

6. Visual checkpoint eval

Create:
homebrain/eval/eval_checkpoint_3d_visualizer_v0.py

Command:

python -m homebrain.eval.eval_checkpoint_3d_visualizer_v0 \
  --viz-dir artifacts/checkpoint_3d_visualizer_v0/room_walk_001 \
  --out artifacts/checkpoint_3d_visualizer_v0/room_walk_001/eval.json

The eval must inspect generated artifacts and write:

{
  "accepted_checkpoint_3d_visualizer_v0": bool,
  "gates_improved": ["Gate A", "Gate D"],
  "artifact_count": int,
  "html_exists": bool,
  "scene_timeline_exists": bool,
  "accumulated_map_exists": bool,
  "ply_exists": bool,
  "frame_count": int,
  "map_accumulates_over_time": bool,
  "incremental_no_future_leakage_passed": bool,
  "camera_pose_rendered": bool,
  "robot_pose_rendered": bool,
  "trajectory_rendered": bool,
  "local_bev_rendered": bool,
  "accumulated_map_rendered": bool,
  "candidate_trajectories_rendered": bool,
  "safety_debug_rendered": bool,
  "geometry_source": "...",
  "pose_source": "...",
  "pose_source_is_oracle": false,
  "dense_3d_claimed": false,
  "model_channels_present": [],
  "model_channels_missing": [],
  "warnings": [],
  "hard_failures": [],
  "safety": {
    "replay_only": true,
    "not_executed": true,
    "control_safe": false,
    "raw_pwm_emitted": false,
    "hardware_validated": false
  }
}

Acceptance can be true only if:
- viewer.html exists
- scene_timeline.jsonl exists
- accumulated_scene.json exists
- at least one PLY exists
- frame_count > 1
- map_accumulates_over_time == true
- incremental_no_future_leakage_passed == true
- robot pose is rendered
- camera pose or explicit camera-pose-missing warning is rendered
- trajectory is rendered
- local BEV or depth geometry is rendered
- accumulated map is rendered
- geometry_source is explicit
- dense_3d_claimed is false unless real depth/RGB-D was used
- safety flags remain conservative
- no raw PWM
- no hardware transport
- no hardware-safe claim

Acceptance must be false if:
- the tool only writes per-frame outputs and no accumulated map
- it silently assumes camera extrinsics on real data
- it uses future frames to render the map at earlier timeline positions
- it marks BEV-only visualization as dense 3D
- it uses teacher artifacts at runtime without explicit debug overlay marking
- it claims control safety

7. Fixture generator

Create:
tests/checkpoint_3d_visualizer_fixtures.py

Add deterministic fixtures:

A. stitched_room_tiny
- 5 frames
- robot moves forward 0.25m per frame
- simple local BEV with free floor and one obstacle
- known T_map_base sequence
- known camera intrinsics/extrinsics
- expected accumulated obstacle position

B. turning_camera_frustum
- robot rotates in place
- camera frustum should rotate in map frame
- trajectory remains fixed but orientation changes

C. hazard_free_but_unsafe
- cell is free and hazard
- accumulated map must render both free and hazard
- hazard must not become obstacle

D. unknown_not_free
- unknown cells remain unknown
- unknown cannot be silently converted to free

E. stale_memory
- observed obstacle becomes stale after decay
- stale layer is visible and counted

F. missing_extrinsics_real_mode
- real-mode route missing camera-to-base
- must warn/fail acceptance, not silently render fake camera pose

G. oracle_pose_debug
- route pose marked oracle
- visualizer may render, but accepted=false unless fixture/debug mode

H. no_future_leakage
- obstacle appears only at frame 4
- timeline snapshot at frame 2 must not include it

I. candidate_trajectory_overlay
- three candidate trajectories
- one selected, one vetoed
- viewer JSON records selected/vetoed states

J. safety_stop_overlay
- high risk near footprint
- stop reason appears in timeline and side panel data

8. Tests

Add:

tests/test_checkpoint_3d_scene_types_v0.py
tests/test_pose_camera_geometry_v0.py
tests/test_accumulated_scene3d_v0.py
tests/test_checkpoint_3d_exporters_v0.py
tests/test_visualize_checkpoint_3d_cli_v0.py
tests/test_eval_checkpoint_3d_visualizer_v0.py

Tests must prove:

1. Scene timeline dataclasses serialize deterministically.
2. Transform composition and inversion are numerically sane.
3. Camera frustum lines are generated from intrinsics/extrinsics.
4. Robot footprint polygon transforms into map frame.
5. BEV cells transform from robot frame into map frame.
6. Accumulated map grows over multiple frames.
7. Timeline snapshot at frame k uses only frames <= k.
8. Fixture obstacle stitches into expected map position.
9. Hazard cells remain free-but-unsafe, not obstacle.
10. Unknown cells do not become free.
11. Stale memory is tracked.
12. Missing extrinsics warn/fail in real mode.
13. Oracle pose is explicitly marked.
14. Candidate trajectories are exported and selected/vetoed fields are preserved.
15. Safety stop reasons are exported.
16. viewer.html is generated and contains timeline/layer controls.
17. PLY output is deterministic.
18. summary.json and eval.json contain conservative safety flags.
19. accepted_checkpoint_3d_visualizer_v0 is false for bad/missing assumptions.
20. Existing tests still pass.

9. Integration rules

Do not create a disconnected toy visualizer.

Where practical:
- reuse existing runtime inference paths
- reuse existing BEV/action contract semantics
- reuse existing candidate trajectory representations
- reuse existing SceneState/memory outputs if present
- reuse existing pack/route loaders
- reuse existing safety/debug fields if present
- do not rewrite model code unless an adapter is enough
- do not invent fake model outputs
- do not require real checkpoint tests to pass in CI; use fixtures for tests
- real checkpoint mode should fail honestly when files are missing

10. Runtime leakage and debug overlays

The visualizer may support optional teacher/debug overlays:

--debug-teacher-overlay path/to/teacher_artifacts

But:
- default must be off
- runtime model view must not depend on teacher artifacts
- if teacher overlays are shown, mark:
  teacher_overlay_enabled=true
  overlay_is_runtime_truth=false
  accepted_checkpoint_3d_visualizer_v0=false for real model-progress claims
- the HTML side panel must visually separate:
  model prediction
  accumulated memory
  teacher/debug overlay
  route/oracle truth

11. Performance constraints

For large routes:
- support --max-frames
- support --stride
- support --max-points
- support --max-cells
- support voxel/grid downsampling
- do not dump full tensors into HTML by default
- write compact summaries
- keep fixture artifacts small

12. Definition of done

These commands must work:

python -m pytest -q tests/test_checkpoint_3d_scene_types_v0.py
python -m pytest -q tests/test_pose_camera_geometry_v0.py
python -m pytest -q tests/test_accumulated_scene3d_v0.py
python -m pytest -q tests/test_checkpoint_3d_exporters_v0.py
python -m pytest -q tests/test_visualize_checkpoint_3d_cli_v0.py
python -m pytest -q tests/test_eval_checkpoint_3d_visualizer_v0.py

Fixture command must work:

python -m homebrain.viz.visualize_checkpoint_3d_v0 \
  --fixture stitched_room_tiny \
  --out artifacts/checkpoint_3d_visualizer_v0/fixture_stitched_room_tiny \
  --write-html \
  --write-ply \
  --write-jsonl

Eval command must work:

python -m homebrain.eval.eval_checkpoint_3d_visualizer_v0 \
  --viz-dir artifacts/checkpoint_3d_visualizer_v0/fixture_stitched_room_tiny \
  --out artifacts/checkpoint_3d_visualizer_v0/fixture_stitched_room_tiny/eval.json

The fixture eval must include:

{
  "accepted_checkpoint_3d_visualizer_v0": true,
  "gates_improved": ["Gate A", "Gate D"],
  "map_accumulates_over_time": true,
  "incremental_no_future_leakage_passed": true,
  "camera_pose_rendered": true,
  "robot_pose_rendered": true,
  "trajectory_rendered": true,
  "accumulated_map_rendered": true,
  "geometry_source": "fixture_synthetic",
  "dense_3d_claimed": false,
  "safety": {
    "replay_only": true,
    "not_executed": true,
    "control_safe": false,
    "raw_pwm_emitted": false,
    "hardware_validated": false
  }
}

Do not update README with claims unless:
- fixture visualizer works,
- eval artifact exists,
- tests pass,
- safety flags remain conservative.

This goal is successful when a human can open viewer.html and inspect a stitched, time-accumulated 3D/2.5D replay of what the checkpoint believes about the scene, robot pose, camera pose, memory, hazards, candidates, and safety decisions.