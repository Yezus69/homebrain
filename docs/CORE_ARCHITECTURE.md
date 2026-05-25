# HomeBrain Core Architecture

HomeBrain is organized around deterministic evidence, not demos. Active modules
must support one of the core robot-brain responsibilities below and must leave
typed artifacts that can be replayed, inspected, and evaluated.

## Active Spine

### Logs and replay

The log/replay spine is the first-class input and output boundary. Route logs
must preserve frame ids, timestamps, sensor metadata, route provenance, and any
available pose/odometry truth without inventing streams. Replay commands must
be deterministic and write auditable output events.

### Eval

Eval modules define the scorecard for every claim. A new behavior is not active
architecture until it has tests, command-line verification, JSON metrics, and a
clear pass/fail interpretation. Failed gates are evidence, not cleanup targets.

### Teacher artifacts

Foundation models are offline teachers. SceneTeacherPack, DINO/DA3/depth
artifacts, QA reports, and signal audits may provide features or pseudo-labels,
but they must carry provenance, mock/synthetic flags, scale status, and safety
flags. Teacher outputs are not robot-frame truth unless measured transforms and
pose/odometry evidence support that claim.

### Geometry and BEV

Geometry code owns camera/depth/point-map conversion into reviewable BEV
representations. Current MoGe robot-BEV projection evidence is diagnostic only:
obstacle/unknown signal exists, false-free-over-obstacle was measured as 0.0 on
the evaluated routes, but held-out free IoU was 0.0 and convention selection was
not universal. This is not product-training approval.

### Spatial memory and data packs

SpatialTrainPack and spatial-memory modules own explicit local/global spatial
state: free, occupied, unknown, confidence, pose labels, temporal windows, and
route/source splits. Data-pack builders must be deterministic and must keep
weak/replay/control-safety flags explicit.

### Trajectory scoring and policy eval

Policy modules score deterministic candidate trajectories and produce replay-only
decision artifacts. They may compare transparent scorers, learned scorers, and
future-motion labels, but they must not emit raw PWM and must keep
`control_safe=false` until a later hardware-safe gate exists.

## Historical and Audit Tools

A tool is historical or audit-scoped when it exists to reproduce one goal's
evidence, diagnose a failed gate, or aggregate already-written artifacts. These
tools may remain callable when they are the only source of a critical metric,
but they should be thin CLIs around shared artifact IO, spatial loading, and
visualization helpers.

Historical/audit tools should not become new architecture when they:

- encode one route, checkpoint, or goal-specific report shape;
- only summarize generated artifacts under `runs/`;
- exist to preserve a blocked or failed result;
- generate review contact sheets rather than new training labels;
- depend on public/research data that is not product-training approved.

Deletion is safe only when the metric is already preserved by a smaller active
command or by an immutable artifact referenced from status docs. Otherwise,
extract shared helpers first and keep the tool.

## Do Not Add Next Without Better Evidence

The next layer should not add new model training, MoGe-generated
SpatialTrainPack training, extra open-weight model wrappers, simulator work, or
ROS/Nav2 integration until the current evidence improves. In particular, MoGe
candidate generation must remain local/replay review-only until free-space
weakness and route convention disagreement are understood.

Prefer contraction before expansion: shared typed IO, deterministic spatial
loading, small visualization helpers, narrower CLIs, and status updates that
make blockers easy to resume.
