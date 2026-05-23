# DATA_STRATEGY.md

## Core idea

We do not have large expert robot demos and we do not have hardware yet.

Therefore, HomeBrain uses a teacher-student strategy:
1. Collect or load real indoor visual sequences.
2. Run large foundation teachers offline.
3. Save pseudo-labels and features.
4. Train a smaller SpatialMemoryNet student.
5. Evaluate using replay and measurable metrics.
6. Later attach real robot logs to the same pipeline.

## Data sources by purpose

### Real indoor visual distribution

Use public indoor datasets and self-collected phone/robot-height video to expose the model to real homes, clutter, lighting, furniture, cords, humans, pets, and motion blur.

Purpose:
- perception
- geometry priors
- dynamic object handling
- uncertainty detection

### Public RGB-D / geometry datasets

Purpose:
- image-to-depth / point-map supervision
- local BEV labels
- pose delta supervision
- place recognition/keyframes later

License must be audited before commercial use.

### Dynamic indoor datasets

Purpose:
- moving humans/pets/object-like dynamics
- do-not-permanently-map temporary obstacles
- dynamic risk labels

### Simulation

Use sim only where real videos cannot provide action-conditioned labels:
- candidate action outcomes
- coverage tasks
- collision/recovery pretraining
- synthetic rare cases

Do not try to build a perfect simulator before training useful components.

### Future robot logs

Once hardware exists, every run becomes data:
- normal coverage
- stuck events
- near-collisions
- low light
- moved furniture
- failed docking
- human intervention

## First data artifact format

For each sequence:

```text
sequence_id/
  manifest.json
  frames/
    000000.jpg
    ...
  events.jsonl.zst or events.jsonl.gz
  teachers/
    teacher_name/
      metadata.json
      depth.npy
      confidence.npy
      features.npy
      masks.npy
  eval/
    metrics.json
    overlay.mp4
```

Codex may choose a simpler compressed JSON/NPZ format initially, but the format must be deterministic and documented.

## Current image-sequence ingestion

Goal 2 adds a lean image-folder importer that writes normal HomeBrain route logs:

```bash
python -m homebrain.ingest.image_sequence --frames data/inbox/room_walk/frames --out runs/room_walk_route --camera front_rgb --fps 10
```

Accepted frame extensions are `.jpg`, `.jpeg`, `.png`, `.pgm`, and `.ppm`. Files are imported in deterministic sorted order, copied under the route `frames/` directory, and represented as `FrameEvent` records. The route also gets `route_metadata.json` with:

```text
source_type=image_sequence
source_path
camera_name
fps
frame_count
width/height when consistent and available
has_imu=false
has_wheel_odometry=false
has_commands=false
user_owned_or_license_unknown=true
```

Image-only imports must not synthesize IMU, wheel odometry, or command events. Missing sensors are recorded as unavailable in `route_metadata.json`, and camera intrinsics are marked missing on imported frame events.

Optional sampling:

```bash
python -m homebrain.ingest.image_sequence --frames data/inbox/room_walk/frames --out runs/room_walk_route_stride2 --camera front_rgb --fps 10 --stride 2 --max-frames 300
```

External phone-video conversion example:

```bash
ffmpeg -i phone_room_walk.mp4 -vf fps=10 data/inbox/room_walk/frames/%06d.jpg
```

`ffmpeg` is an external operator tool, not a HomeBrain Python dependency. Keep raw collected videos and derived frames out of git unless a tiny fixture is intentionally committed for tests.
