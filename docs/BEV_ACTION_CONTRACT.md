# BEV Action Contract

HomeBrain BEV arrays are not automatically action supervision. A BEV frame can be useful for geometry pretraining while still being invalid for candidate-trajectory labels.

## Channels

Each action-audited BEV frame uses a 2D local grid with these channels:

- `bev_free`: probability or binary mask for observed floor/free space.
- `bev_obstacle`: probability or binary mask for occupied cells.
- `bev_unknown`: probability or binary mask for unobserved or untrusted cells.
- `bev_traversable`: optional derived probability for cells a robot-sized footprint may traverse.
- `bev_risky`: optional derived probability for cells that should penalize candidate motion.
- `bev_confidence`: optional label confidence. Low confidence does not make a frame control-safe.

All channels are interpreted as replay/eval data only unless a later hardware log explicitly provides robot-frame action truth. `control_safe=false` remains mandatory for current artifacts.

## Origin And Frame

Current candidate trajectories assume:

- The robot origin is the bottom-center grid cell.
- Forward motion decreases row index.
- Left motion increases column index according to the current candidate generator.
- Candidate footprints are rasterized from `cmd_vel_proxy` and are not hardware commands.

Frame types must be explicit:

- `controlled_robot_frame_proxy`: deterministic grid-world supervision for replay-only candidate-label tests.
- `phone_or_teacher_estimated_geometry`: DA3/phone/depth-teacher geometry; useful for geometry pretraining, not action truth.
- `public_rgbd_camera_pose_geometry`: public RGB-D pose/depth geometry such as TUM; useful for geometry/pose pretraining, not robot action truth.
- `model_prediction`: SpatialMemoryNet output; debug/eval only.
- `explicit_robot_frame_truth`: reserved for future robot logs with calibrated robot-frame pose/footprint evidence.

## Footprint And Semantics

The robot footprint is a disk in grid cells. A frame is structurally bad for action supervision when the robot center or current footprint is occupied, risky, unknown, or not sufficiently free. Motion candidates can be rejected for:

- `occupied`
- `risky`
- `unknown`
- `low_free`
- `empty_or_out_of_grid`

Blocked maps can still be valid action supervision if the robot footprint is sane and the blockage is a real scene condition that should select `stop`.

## Required Flags

Every BEV/action artifact must carry or derive:

- `geometry_pretrain_ok`: the frame may teach geometry labels.
- `pose_pretrain_ok`: the frame may teach pose labels.
- `action_supervision_ok`: the frame may teach candidate trajectory labels.
- `robot_frame_truth`: the frame is true in the robot action frame, or a controlled robot-frame proxy.
- `control_safe`: always `false` for current replay artifacts.

DA3/phone BEVs and TUM/public RGB-D labels are not automatically robot-frame action truth. They may be marked geometry-only for action learning even when their geometry labels are structurally valid.
